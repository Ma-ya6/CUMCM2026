"""九个互不相同的**决策模型**，统一接口 ``plan(StageContext) -> 购电向量``。

预测层、物理执行、经济结算三者全部固定（见 ``dispatch_core``），差别只在
"给定同一份预报，怎么定承诺购电量"。九个模型分属不同建模范式（标注按程序
实际行为，不由范式名称推断）：

| 代号 | 模型 | 范式 | 风险处理 |
| --- | --- | --- | --- |
| M1 | 确定性经济调度 LP | 确定性优化 | 不加安全余量，预报即真值 |
| M2 | 分位安全余量 LP | 确定性优化 | 逐时段历史残差分位作为**额外净负荷**加进功率平衡右端 |
| M3 | 两阶段随机规划 | 随机优化 | 9 条历史残差日整曲线等权场景，紧急购电为追索 |
| M4 | 均值–CVaR 随机规划 | 风险厌恶随机优化 | 目标含 $0.5\\cdot$期望 $+0.5\\cdot\\text{CVaR}(0.90)$ |
| M5 | 鲁棒优化 | 鲁棒优化 | 取累计残差最大的**单条**历史日曲线作最坏情形（非盒式集） |
| M6 | 精确链式结算 MILP | 混合整数优化 | 余量与 M2 同源，结算项用 0-1 变量精确刻画 |
| M7 | 规则型启发式 | 无优化 | 净负荷跟随 + 电价四分位充放，不显式建模不确定性 |
| M9 | 储能追索型 LP | 确定性优化 + 储能储备 | 整窗累计残差的单标量 $q=0.80$ 分位，仅抬升储电量下界 |
| M10 | 储能追索型 LP (q=0.70) | 确定性优化 + 储能储备 | 同 M9，分位数按开发期标定为 0.70 |

## 为什么 M2 与 M6 单独列出

模型3 的 LP 把链式结算的 $c\\min\\{x_{\\text{prev}},x\\}$ 锚定凸化成
$0.5cx+1.0ce$。M2 沿用该凸化，M6 用 0-1 变量精确刻画 $\\min$，两者共用同一份
分位余量，故 M2 与 M6 的差额即**凸化近似的代价**；在固定电价底座下实测两者
费用逐位相等（13,751,025.02 元），说明该储备形态下凸化未引入可测的近似代价。

## 共同的经济口径

购电承诺 $x$ 的成本按下单时刻的**边际**计，基准量取历史承诺的最小值
$c\\min_k x^{(k)}$，缺口以 $5c$ 紧急购电。调整加价按上游模型3 的写法施加：
$0.5c$ 记在 $x$ 上、$1.0c$ 记在**下调量** $(\\text{ref}-x)^+$ 上（即锚定凸化
$0.5cx+1.0ce_b$）。注意这与题面 $\\varphi=0.5cx+1.0c(x-r)^+$ 把 $1.0c$ 记在
**上调量**上的方向相反；主链 ``opt_core.stage_lp`` 用题面口径
（``settle="correct"``），本层九个范式则一律用上游口径。除 M7 外的八个模型
都是在这个口径下最小化，M7 不做最小化。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from common import (
    BREACH_RATE,
    DT,
    EMERGENCY_MULTIPLIER,
    E_MAX,
    E_MIN,
    OVERBUY_RATE,
    P_MAX_KW,
    Q2_MODEL_DIR,
    SLOTS_PER_HOUR,
)

import sys

if str(Q2_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(Q2_MODEL_DIR))

import deterministic_baseline as base  # noqa: E402

STEP_MAX = P_MAX_KW * DT
ETA_C = ETA_D = 0.90
BIG_M = 1e5


class DecisionModel:
    """决策模型的统一接口。"""

    code = "?"
    name = "?"
    paradigm = ""
    risk = ""

    def begin_day(self, day: int, start_energy: float) -> None:
        """每天开始时清空与当日有关的内部状态。"""

    def plan(self, ctx) -> np.ndarray:
        raise NotImplementedError


# --------------------------------------------------------------------------
# 通用单阶段 LP：M1–M5 共用（场景数 S=1 时退化为确定性模型）
# --------------------------------------------------------------------------
def solve_scenario_lp(
    load_fc: np.ndarray,
    pv_fc: np.ndarray,
    price: np.ndarray,
    start_energy: float,
    ref: np.ndarray | None,
    scenarios: np.ndarray,
    weights: np.ndarray,
    *,
    emergency: bool = True,
    cvar_lambda: float = 0.0,
    cvar_alpha: float = 0.90,
    force_end_soc: float | None = None,
    soc_reserve: float = 0.0,
    soc_reserve_up: float = 0.0,
) -> np.ndarray:
    """场景型两阶段 LP 的扩展式。

    第一阶段的量（购电 $x$、充放电 $u,v$、储电量 $s$、超购/违约量）对全部场景
    相同——它们是"此时此地"的决策；第二阶段只有紧急购电与弃光随场景变化，
    即"看到实际后的追索"。这符合本题物理：承诺必须在不确定消除前下达。

    ``scenarios`` 形状 ``(S, n)``，为该窗口的净负荷误差情景（kWh/Δt）。
    """
    n = len(price)
    s_count = len(scenarios)
    load = np.maximum(load_fc, 0.0) * DT
    pv = np.maximum(pv_fc, 0.0) * DT

    if emergency:
        i_x, i_u, i_v, i_s, i_eb, i_eo = 0, n, 2 * n, 3 * n, 4 * n, 5 * n
        i_e, i_w = 6 * n, 6 * n + s_count * n
        nvar = 6 * n + 2 * s_count * n
    else:
        i_x, i_u, i_v, i_s, i_eb, i_eo = 0, n, 2 * n, 3 * n, 4 * n, 5 * n
        i_e = None
        i_w = 6 * n
        nvar = 6 * n + s_count * n

    if cvar_lambda > 0.0:
        i_eta, i_z = nvar, nvar + 1
        nvar += 1 + s_count

    obj = np.zeros(nvar)
    front = (1.0 - cvar_lambda) if cvar_lambda > 0.0 else 1.0
    if ref is None:
        # 0:00 首次计划：没有"上一版承诺"，目标就是常规购电费 c·x
        obj[i_x : i_x + n] = front * price
    else:
        # 锚定边际：0.5c·x + 1.0c·eb。系数落在 i_eb 而非 i_eo 并非笔误：
        # 与下方等式 eb − eo = ref − x 配合，1.0c 于是记在下调量 (ref−x)⁺ 上，
        # 即上游 opt_core ``settle="legacy"`` 的口径（题面 φ 把 1.0c 记在
        # 上调量 (x−ref)⁺ 上，方向相反）。要复现冻结值就必须照抄这份矩阵。
        obj[i_x : i_x + n] = front * BREACH_RATE * price
        obj[i_eb : i_eb + n] = front * (OVERBUY_RATE - BREACH_RATE) * price
    obj[i_u : i_u + n] = 1e-7
    obj[i_v : i_v + n] = 1e-7
    obj[i_w : i_w + s_count * n] = 1e-8
    if force_end_soc is None:
        obj[i_s + n - 1] = -base.end_water_value(price)
    for s in range(s_count):
        if emergency:
            obj[i_e + s * n : i_e + (s + 1) * n] = (
                front * weights[s] * EMERGENCY_MULTIPLIER * price
            )
    if cvar_lambda > 0.0:
        obj[i_eta] = cvar_lambda
        obj[i_z : i_z + s_count] = cvar_lambda * weights / (1.0 - cvar_alpha)

    a_eq, b_eq = [], []
    for s in range(s_count):
        # 场景为净负荷误差（kW），换算成每时段电量后叠加
        net = load + scenarios[s] * DT - pv
        for t in range(n):
            row = np.zeros(nvar)
            row[i_x + t], row[i_u + t], row[i_v + t] = 1.0, -1.0, 1.0
            row[i_w + s * n + t] = -1.0
            if emergency:
                row[i_e + s * n + t] = 1.0
            a_eq.append(row)
            b_eq.append(net[t])

    for t in range(n):
        row = np.zeros(nvar)
        row[i_u + t], row[i_v + t], row[i_s + t] = -ETA_C, 1.0 / ETA_D, 1.0
        if t:
            row[i_s + t - 1] = -1.0
            b_eq.append(0.0)
        else:
            b_eq.append(start_energy)
        a_eq.append(row)

    if ref is not None:
        for t in range(n):
            # eb − eo = ref − x，配合上式把 1.0c 计在 (ref−x)⁺ 上。
            # 目标系数落在 i_eb（而非 i_eo）不是笔误：eb、eo 均为非负自由变量，
            # 罚 eb 即罚 (ref−x)⁺，与上式必须一一对应。
            row = np.zeros(nvar)
            row[i_eb + t], row[i_eo + t], row[i_x + t] = 1.0, -1.0, 1.0
            a_eq.append(row)
            b_eq.append(float(ref[t]))

    if force_end_soc is not None:
        end_bound = (float(np.clip(force_end_soc, E_MIN, E_MAX)),) * 2
    else:
        end_bound = (E_MIN, E_MAX)

    # 两条储备边界都必须是"爬坡可达"的，否则约束直接不可行：
    # 下界从起点按最大充电功率抬升，上界从满电按最大放电功率下降。
    band = (E_MAX - E_MIN) / 2.0
    r_lo = min(max(soc_reserve, 0.0), band)
    r_up = min(max(soc_reserve_up, 0.0), band)
    t_lo = (0 if start_energy >= E_MIN + r_lo
            else int(np.ceil((E_MIN + r_lo - start_energy) / (ETA_C * STEP_MAX))))
    t_up = int(np.ceil(r_up * ETA_D / STEP_MAX)) if r_up > 0 else 0
    soc_bounds = []
    for t in range(n):
        lo = (E_MIN + r_lo) if t >= t_lo else E_MIN
        hi = (E_MAX - r_up) if t >= t_up else E_MAX
        soc_bounds.append((lo, hi))
    if force_end_soc is not None:
        soc_bounds[-1] = end_bound
    bounds = (
        [(0.0, None)] * n
        + [(0.0, STEP_MAX)] * n
        + [(0.0, STEP_MAX)] * n
        + soc_bounds
        + [(0.0, None)] * n
        + [(0.0, None)] * n
    )
    if emergency:
        bounds += [(0.0, None)] * (s_count * n)
    bounds += [(0.0, None)] * (s_count * n)
    if cvar_lambda > 0.0:
        bounds += [(None, None)] + [(0.0, None)] * s_count

    a_ub, b_ub = [], []
    if cvar_lambda > 0.0:
        for s in range(s_count):
            # z_s − η − C_s ≥ 0
            row = np.zeros(nvar)
            row[i_z + s], row[i_eta] = 1.0, -1.0
            if ref is None:
                row[i_x : i_x + n] -= front * price
            else:
                row[i_x : i_x + n] -= front * BREACH_RATE * price
                row[i_eb : i_eb + n] -= front * (OVERBUY_RATE - BREACH_RATE) * price
            if emergency:
                row[i_e + s * n : i_e + (s + 1) * n] -= (
                    front * EMERGENCY_MULTIPLIER * price
                )
            a_ub.append(row)
            b_ub.append(0.0)
    a_ub_arr = np.asarray(a_ub) if a_ub else None
    b_ub_arr = np.asarray(b_ub) if b_ub else None

    result = linprog(
        obj, A_eq=np.asarray(a_eq), b_eq=np.asarray(b_eq),
        A_ub=a_ub_arr, b_ub=b_ub_arr, bounds=bounds, method="highs",
    )
    if not result.success:
        raise RuntimeError(f"场景 LP 失败（S={s_count}, n={n}）：{result.message}")
    return result.x[i_x : i_x + n]


# --------------------------------------------------------------------------
# M9 储能追索型 LP：余量不进口购电量，只进口储能储备
# --------------------------------------------------------------------------
class StorageReserveLP(DecisionModel):
    code = "M9"
    name = "储能追索型 LP"
    paradigm = "确定性优化 + 储能储备"
    risk = "购电按点预报下达，不确定性全交给储能储备带"

    QUANTILE = 0.80
    MIN_HISTORY = 20
    # 只保能量下界，不预留充电空间上界：多买 1 kWh 赔 c，放电造成缺口赔 4c，
    # 两侧代价 4:1 不对称，上界按同一分位设会强制放电、反而更贵（实测 +37.7 万）。
    USE_UPPER = False

    def reserve(self, ctx) -> float:
        """窗口累计净负荷残差的单标量分位数（保留时间相关性）。

        与逐时段分位叠加不同：这里只问"这一整段窗口总共会偏多少"，
        而不是"每个时段各自偏多少"，故不会把 144 个独立事件当成 144 个
        必然事件，也不再把多余电量硬塞进购电计划。

        q=0.80 由新闻vendor临界比给出：多买 1 kWh 付 c（储能已满，无处可去），
        少买 1 kWh 要付 4c（紧急购电 5c 减去本可付的 c），故 q*=4c/5c=0.80。
        数值与逐时段口径相同，改的是它作用的**对象**。
        """
        hist = ctx.residual_hist
        if len(hist) < self.MIN_HISTORY:
            return 0.0
        return float(np.quantile(hist.sum(axis=1) * DT, self.QUANTILE))

    def plan(self, ctx) -> np.ndarray:
        r = self.reserve(ctx)
        return solve_scenario_lp(
            ctx.load_fc, ctx.pv_fc, ctx.price, ctx.start_energy, ctx.ref,
            np.zeros((1, ctx.n)), np.ones(1), emergency=False,
            force_end_soc=ctx.force_end_soc,
            soc_reserve=r, soc_reserve_up=(r if self.USE_UPPER else 0.0),
        )


# --------------------------------------------------------------------------
# M1 确定性经济调度 LP
# --------------------------------------------------------------------------
class DeterministicLP(DecisionModel):
    code = "M1"
    name = "确定性经济调度 LP"
    paradigm = "确定性优化"
    risk = "不加安全余量，预报即真值"

    def plan(self, ctx) -> np.ndarray:
        zeros = np.zeros((1, ctx.n))
        return solve_scenario_lp(
            ctx.load_fc, ctx.pv_fc, ctx.price, ctx.start_energy, ctx.ref,
            zeros, np.ones(1), emergency=False, force_end_soc=ctx.force_end_soc,
        )


# --------------------------------------------------------------------------
# M2 分位安全余量 LP（模型3 口径）
# --------------------------------------------------------------------------
class QuantileLP(DecisionModel):
    code = "M2"
    name = "分位安全余量 LP"
    paradigm = "确定性优化"
    risk = "逐时段 q=0.80 历史残差分位作额外净负荷，加进功率平衡右端（模型3 口径）"

    def plan(self, ctx) -> np.ndarray:
        return solve_scenario_lp(
            ctx.load_fc, ctx.pv_fc, ctx.price, ctx.start_energy, ctx.ref,
            ctx.margin_ref[None, :], np.ones(1), emergency=False,
            force_end_soc=ctx.force_end_soc,
        )


# --------------------------------------------------------------------------
# M3 两阶段随机规划
# --------------------------------------------------------------------------
class StochasticLP(DecisionModel):
    code = "M3"
    name = "两阶段随机规划"
    paradigm = "随机优化"
    risk = "9 个真实历史残差日场景，等权期望，紧急购电为追索"

    SCENARIO_LEVELS = np.linspace(0.10, 0.90, 9)
    N_SCENARIOS = 9

    def plan(self, ctx) -> np.ndarray:
        scenarios, weights = self._draw(ctx)
        return solve_scenario_lp(
            ctx.load_fc, ctx.pv_fc, ctx.price, ctx.start_energy, ctx.ref,
            scenarios, weights, emergency=True, force_end_soc=ctx.force_end_soc,
        )

    def _draw(self, ctx) -> tuple[np.ndarray, np.ndarray]:
        """取真实的整条历史残差日曲线作为情景。

        逐时段分位数（``np.quantile(..., axis=0)``）是把 144 个时段的边际分布
        各自取分位后拼成一条曲线，等于同时施加 144 个机会约束，储备被高估约 3 倍；
        整行取样则保留日内时间相关性，情景才是真正可发生的日曲线。
        """
        m = ctx.residual_hist.shape[0]
        if m < 5:
            return np.zeros((1, ctx.n)), np.ones(1)
        s = min(self.N_SCENARIOS, m)
        idx = np.unique(np.linspace(0, m - 1, s).astype(int))
        scenarios = ctx.residual_hist[idx]
        return scenarios, np.full(len(scenarios), 1.0 / len(scenarios))


# --------------------------------------------------------------------------
# M4 均值–CVaR 随机规划
# --------------------------------------------------------------------------
class MeanCVaRLP(StochasticLP):
    code = "M4"
    name = "均值–CVaR 随机规划"
    paradigm = "风险厌恶随机优化"
    risk = "目标 = 0.5·期望 + 0.5·CVaR(0.90)，在目标里罚尾部而非在约束里留量"

    def __init__(self, lam: float = 0.5, alpha: float = 0.90) -> None:
        self.lam = lam
        self.alpha = alpha

    def plan(self, ctx) -> np.ndarray:
        scenarios, weights = self._draw(ctx)
        return solve_scenario_lp(
            ctx.load_fc, ctx.pv_fc, ctx.price, ctx.start_energy, ctx.ref,
            scenarios, weights, emergency=True,
            cvar_lambda=self.lam, cvar_alpha=self.alpha,
            force_end_soc=ctx.force_end_soc,
        )


# --------------------------------------------------------------------------
# M5 鲁棒优化
# --------------------------------------------------------------------------
class RobustLP(DecisionModel):
    code = "M5"
    name = "鲁棒优化"
    paradigm = "鲁棒优化"
    risk = "取历史累计残差最大的单条日曲线作最坏情形，对最坏情形求可行解"

    def plan(self, ctx) -> np.ndarray:
        if ctx.residual_hist.shape[0] < 5:
            worst = np.zeros(ctx.n)
        else:
            # 逐时段 ``np.max(axis=0)`` 会拼出一条历史上从未发生过的曲线
            # （各时段的极值不在同一天），同样高估需求；改为取累计残差最大的那一天。
            hist = ctx.residual_hist
            worst = hist[int(np.argmax(hist.sum(axis=1)))]
        return solve_scenario_lp(
            ctx.load_fc, ctx.pv_fc, ctx.price, ctx.start_energy, ctx.ref,
            worst[None, :], np.ones(1), emergency=False,
            force_end_soc=ctx.force_end_soc,
        )


# --------------------------------------------------------------------------
# M6 精确链式结算 MILP
# --------------------------------------------------------------------------
def solve_exact_chain_milp(
    load_fc: np.ndarray,
    pv_fc: np.ndarray,
    price: np.ndarray,
    start_energy: float,
    ref: np.ndarray | None,
    margin: np.ndarray,
    hist_min: np.ndarray | None,
    *,
    force_end_soc: float | None = None,
) -> np.ndarray:
    """把链式结算的目标**精确**写进优化，不做锚定凸化。

    第 $k$ 阶段的真实边际（历史承诺的最小值 $m_t$ 已知）为

    $$ \\varphi_t(x)=c_t\\min\\{m_t,x_t\\}+0.5c_t(x_{\\text{prev},t}-x_t)^+
       +1.5c_t(x_t-x_{\\text{prev},t})^+ .$$

    $\\min\\{m,x\\}$ 用 0-1 变量 $\\delta_t$ 线性化：$\\delta_t=1\\Rightarrow x_t\\le m_t$。
    由此得到真最优；模型3 的凸化只在一个特例下与它重合。
    """
    n = len(price)
    load = np.maximum(load_fc + margin, 0.0) * DT
    pv = np.maximum(pv_fc, 0.0) * DT

    i_x, i_u, i_v, i_s, i_eb, i_eo = 0, n, 2 * n, 3 * n, 4 * n, 5 * n
    i_wmin, i_delta = 6 * n, 7 * n
    i_w = 8 * n
    nvar = 9 * n

    obj = np.zeros(nvar)
    obj[i_u : i_u + n] = 1e-7
    obj[i_v : i_v + n] = 1e-7
    obj[i_w : i_w + n] = 1e-8
    if force_end_soc is None:
        obj[i_s + n - 1] = -base.end_water_value(price)
    if hist_min is None:
        obj[i_x : i_x + n] = price                       # 0:00 首次计划：c·x
    else:
        obj[i_wmin : i_wmin + n] = price                 # c·min{m,x}
        obj[i_eb : i_eb + n] = BREACH_RATE * price       # 0.5c·下调
        obj[i_eo : i_eo + n] = OVERBUY_RATE * price      # 1.5c·上调

    a_eq, b_eq = [], []
    for t in range(n):
        row = np.zeros(nvar)
        row[i_x + t], row[i_u + t], row[i_v + t], row[i_w + t] = 1.0, -1.0, 1.0, -1.0
        a_eq.append(row)
        b_eq.append(load[t] - pv[t])

        row = np.zeros(nvar)
        row[i_u + t], row[i_v + t], row[i_s + t] = -ETA_C, 1.0 / ETA_D, 1.0
        if t:
            row[i_s + t - 1] = -1.0
            b_eq.append(0.0)
        else:
            b_eq.append(start_energy)
        a_eq.append(row)

    if ref is not None:
        for t in range(n):
            row = np.zeros(nvar)
            row[i_eb + t], row[i_eo + t], row[i_x + t] = 1.0, -1.0, 1.0
            a_eq.append(row)
            b_eq.append(float(ref[t]))

    a_ub, b_ub = [], []
    if hist_min is not None:
        for t in range(n):
            m = float(hist_min[t])
            # wmin ≤ x_t
            row = np.zeros(nvar)
            row[i_wmin + t], row[i_x + t] = 1.0, -1.0
            a_ub.append(row); b_ub.append(0.0)
            # wmin ≤ m_t
            row = np.zeros(nvar)
            row[i_wmin + t] = 1.0
            a_ub.append(row); b_ub.append(m)
            # wmin ≥ x_t − M(1−δ)
            row = np.zeros(nvar)
            row[i_wmin + t], row[i_x + t], row[i_delta + t] = 1.0, -1.0, -BIG_M
            a_ub.append(row); b_ub.append(-BIG_M)
            # wmin ≥ m_t − M·δ  →  −wmin − M·δ ≤ −m_t
            row = np.zeros(nvar)
            row[i_wmin + t], row[i_delta + t] = -1.0, -BIG_M
            a_ub.append(row); b_ub.append(-m)

    if force_end_soc is not None:
        end_bound = (float(np.clip(force_end_soc, E_MIN, E_MAX)),) * 2
    else:
        end_bound = (E_MIN, E_MAX)
    bounds = (
        [(0.0, None)] * n
        + [(0.0, STEP_MAX)] * n
        + [(0.0, STEP_MAX)] * n
        + [(E_MIN, E_MAX)] * (n - 1)
        + [end_bound]
        + [(0.0, None)] * n
        + [(0.0, None)] * n
        + [(0.0, None)] * n
        + [(0.0, 1.0)] * n
        + [(0.0, None)] * n
    )
    integrality = np.zeros(nvar)
    integrality[i_delta : i_delta + n] = 1

    result = linprog(
        obj, A_eq=np.asarray(a_eq), b_eq=np.asarray(b_eq),
        A_ub=np.asarray(a_ub) if a_ub else None,
        b_ub=np.asarray(b_ub) if b_ub else None,
        bounds=bounds, integrality=integrality, method="highs",
    )
    if not result.success:
        raise RuntimeError(f"链式 MILP 失败（n={n}）：{result.message}")
    return result.x[i_x : i_x + n]


class ExactChainMILP(DecisionModel):
    code = "M6"
    name = "精确链式结算 MILP"
    paradigm = "混合整数优化"
    risk = "与 M2 同用 q=0.80 分位余量，但结算项用 0-1 变量精确刻画"

    def begin_day(self, day: int, start_energy: float) -> None:
        self._carry: np.ndarray | None = None
        self._minsofar: np.ndarray | None = None

    def plan(self, ctx) -> np.ndarray:
        hist_min = None if self._minsofar is None else self._minsofar[ctx.abs_start:]
        purchase = solve_exact_chain_milp(
            ctx.load_fc, ctx.pv_fc, ctx.price, ctx.start_energy, ctx.ref,
            ctx.margin_ref, hist_min, force_end_soc=ctx.force_end_soc,
        )
        if self._carry is None:
            self._carry = np.zeros(SLOTS_PER_HOUR * 24)
            self._minsofar = np.full(SLOTS_PER_HOUR * 24, np.inf)
        self._carry[ctx.abs_start:] = purchase
        self._minsofar[ctx.abs_start:] = np.minimum(
            self._minsofar[ctx.abs_start:], purchase
        )
        return purchase


# --------------------------------------------------------------------------
# M7 规则型启发式
# --------------------------------------------------------------------------
class RuleHeuristic(DecisionModel):
    code = "M7"
    name = "规则型启发式"
    paradigm = "无优化"
    risk = "凭规则留量：净负荷跟随 + 储能在电价高低四分位充放"

    POWER_FRACTION = 0.6

    def plan(self, ctx) -> np.ndarray:
        n = ctx.n
        price = ctx.price
        net = (ctx.load_fc - ctx.pv_fc) * DT          # kWh/Δt
        hi = np.quantile(price, 0.75)
        lo = np.quantile(price, 0.25)
        step = STEP_MAX * self.POWER_FRACTION

        energy = float(np.clip(ctx.start_energy, E_MIN, E_MAX))
        x = np.zeros(n)
        for t in range(n):
            charge = discharge = 0.0
            if price[t] <= lo:
                charge = min(step, max(0.0, (E_MAX - energy) / ETA_C))
            elif price[t] >= hi:
                discharge = min(step, max(0.0, (energy - E_MIN) * ETA_D))
            energy += ETA_C * charge - discharge / ETA_D
            x[t] = max(0.0, net[t] + charge - discharge)
        return x


def all_models() -> list[DecisionModel]:
    class StorageReserveLP70(StorageReserveLP):
        QUANTILE = 0.70
        code = "M10"
        name = "储能追索型 LP (q=0.70)"
        risk = "同 M9，q 按开发期标定为 0.70"

    return [DeterministicLP(), QuantileLP(), StochasticLP(), MeanCVaRLP(),
            RobustLP(), ExactChainMILP(), StorageReserveLP(),
            StorageReserveLP70(), RuleHeuristic()]
