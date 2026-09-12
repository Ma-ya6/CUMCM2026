# -*- coding: utf-8 -*-
"""优化版的纯计算内核：修正后的单阶段 LP + 储能储备 + 短时域滚动执行器。

本模块**只依赖 numpy / scipy**，不导入任何上游模块。四个问题目录各自
``import opt_core`` 即可——这样避免了 ``deterministic_baseline`` / ``seasonal``
在不同问题目录下的同名模块冲突，也保证了四个问题用的是**逐位相同**的优化内核。

对应 `优化建议.md` 的三条主线：

  §二.1  费用方向修正   —— ``settle="correct"``。上游 ``models.solve_scenario_lp``
                          把 1.0c 的附加费记在 $e_b=(r-x)^+$（下调量）上，
                          按题面应记在 $e_o=(x-r)^+$（上调量）上。
  §三.1  储能储备型 LP  —— ``reserve`` 参数。购电按点预测下达，风险改由
                          $E_t\\ge E_{\\min}+R$ 承担，不再逐时段叠加余量。
  §三.2  短时域滚动执行  —— ``execute_day_mpc``。把"有缺口就立即放电"的贪心
                          规则换成 H 步前瞻 LP：放电省下 5c 的紧急购电，但要
                          消耗储电；两者谁更划算由滚动窗口内的价格路径决定。

口径说明见各函数 docstring。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

# --------------------------------------------------------------------------
# 物理与经济常数（与上游 common.py / deterministic_baseline.py 逐位一致）
# --------------------------------------------------------------------------
DT = 1.0 / 6.0
N_SLOT = 144
SLOTS_PER_HOUR = 6
ETA_C = 0.90
ETA_D = 0.90
E_MIN = 1200.0
E_MAX = 10800.0
E_INITIAL = 6000.0
P_MAX_KW = 5000.0
STEP_MAX = P_MAX_KW * DT
EMERGENCY_MULTIPLIER = 5.0
BREACH_RATE = 0.5
OVERBUY_RATE = 1.5
BIG_M = 1.0e5          # 精确链式结算 MILP 的松弛常数（远大于单时段承诺量的量级）
ISSUE_HOURS = (0, 6, 12, 18)


def end_water_value(price: np.ndarray) -> float:
    """日末储电的水值 λ = η_d × 窗口平均电价（与上游逐位一致）。"""
    return float(ETA_D * np.mean(price))


# ==========================================================================
# 一、单阶段计划 LP（问题2/4-2 的 0:00 计划，问题3/4-3 的四个决策时刻）
# ==========================================================================
def stage_lp(
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    price: np.ndarray,
    start_energy: float,
    *,
    ref: np.ndarray | None = None,
    settle: str = "correct",
    reserve: float = 0.0,
    reserve_up: float = 0.0,
    force_end_soc: float | None = None,
    scenarios: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    emergency: bool = False,
    cvar_lambda: float = 0.0,
    cvar_alpha: float = 0.90,
    hist_min: np.ndarray | None = None,
) -> np.ndarray:
    """单阶段购电计划 LP，返回承诺购电量向量 $x$（kWh/时段）。

    变量（下标按块）：``i_x, i_u, i_v, i_s, i_eb, i_eo``，紧急购电 ``i_e`` 与
    弃光 ``i_w`` 按场景索引。

    ``settle`` 口径：
      · ``"correct"`` —— 题面口径 $\\varphi=0.5c\\,x+1.0c\\,(x-r)^+$
                          （去掉常数项 $0.5cr$），1.0c 记在 $e_o$ 上；
      · ``"legacy"``  —— 复刻上游 ``models.solve_scenario_lp`` 的写法，
                          1.0c 记在 $e_b$ 上（方向相反），只用于复现冻结值；
      · ``"none"``    —— 无上一版承诺（当日首次计划），目标就是 $c\\,x$。

    ``reserve`` 是储能储备 $R$（kWh），施加为 $E_t\\ge E_{\\min}+R$；
    ``reserve_up`` 是充电空间储备 $E_t\\le E_{\\max}-R_{up}$（默认 0，见 §三.1）。
    """
    n = len(price)
    load = np.maximum(load_kw, 0.0) * DT
    pv = np.maximum(pv_kw, 0.0) * DT

    if scenarios is None:
        scenarios = np.zeros((1, n))
        weights = np.ones(1)
    else:
        weights = np.asarray(weights, dtype=float)
    s_count = len(scenarios)

    i_x, i_u, i_v, i_s, i_eb, i_eo = 0, n, 2 * n, 3 * n, 4 * n, 5 * n
    if emergency:
        i_e = 6 * n
        i_w = 6 * n + s_count * n
    else:
        i_e = None
        i_w = 6 * n
    nvar = 6 * n + s_count * n + (s_count * n if emergency else 0)

    if cvar_lambda > 0.0:
        i_eta, i_z = nvar, nvar + 1
        nvar += 1 + s_count
    front = (1.0 - cvar_lambda) if cvar_lambda > 0.0 else 1.0

    obj = np.zeros(nvar)
    if settle == "none":
        obj[i_x : i_x + n] = front * price
    else:
        obj[i_x : i_x + n] = front * BREACH_RATE * price
        extra = front * (OVERBUY_RATE - BREACH_RATE) * price
        if settle == "correct":
            obj[i_eo : i_eo + n] = extra          # 1.0c 记在上调量 (x-r)+
        elif settle == "legacy":
            obj[i_eb : i_eb + n] = extra          # 上游写法：记在下调量 (r-x)+
        else:
            raise ValueError(f"settle 只能是 correct/legacy/none，收到 {settle!r}")

    obj[i_u : i_u + n] = 1e-7
    obj[i_v : i_v + n] = 1e-7
    obj[i_w : i_w + s_count * n] = 1e-8
    if force_end_soc is None:
        obj[i_s + n - 1] = -end_water_value(price)
    if emergency:
        for s in range(s_count):
            obj[i_e + s * n : i_e + (s + 1) * n] = (
                front * weights[s] * EMERGENCY_MULTIPLIER * price
            )
    if cvar_lambda > 0.0:
        obj[i_eta] = cvar_lambda
        obj[i_z : i_z + s_count] = cvar_lambda * weights / (1.0 - cvar_alpha)

    a_eq, b_eq = [], []
    for s in range(s_count):
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

    if settle != "none":
        if ref is None:
            raise ValueError("settle 非 none 时必须给出上一版承诺 ref")
        for t in range(n):
            row = np.zeros(nvar)
            row[i_eb + t], row[i_eo + t], row[i_x + t] = 1.0, -1.0, 1.0
            a_eq.append(row)
            b_eq.append(float(ref[t]))

    if force_end_soc is not None:
        end_bound = (float(np.clip(force_end_soc, E_MIN, E_MAX)),) * 2
    else:
        end_bound = (E_MIN, E_MAX)

    # 储备边界必须"爬坡可达"，否则直接不可行：下界从起点按最大充电功率抬升，
    # 上界从当前电量按最大放电功率下降（与上游同口径）。
    band = (E_MAX - E_MIN) / 2.0
    r_lo = min(max(reserve, 0.0), band)
    r_up = min(max(reserve_up, 0.0), band)
    t_lo = (0 if start_energy >= E_MIN + r_lo
            else int(np.ceil((E_MIN + r_lo - start_energy) / (ETA_C * STEP_MAX))))
    t_up = (0 if start_energy <= E_MAX - r_up
            else int(np.ceil((start_energy - (E_MAX - r_up)) * ETA_D / STEP_MAX)))
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
    bounds += [(0.0, None)] * (s_count * n)          # 弃光
    if emergency:
        bounds += [(0.0, None)] * (s_count * n)      # 紧急购电
    if cvar_lambda > 0.0:
        bounds += [(None, None)] + [(0.0, None)] * s_count

    a_ub, b_ub = [], []
    if cvar_lambda > 0.0:
        for s in range(s_count):
            row = np.zeros(nvar)
            row[i_z + s], row[i_eta] = 1.0, -1.0
            if settle == "none":
                row[i_x : i_x + n] -= front * price
            else:
                row[i_x : i_x + n] -= front * BREACH_RATE * price
                tgt = i_eo if settle == "correct" else i_eb
                row[tgt : tgt + n] -= front * (OVERBUY_RATE - BREACH_RATE) * price
            if emergency:
                row[i_e + s * n : i_e + (s + 1) * n] -= (
                    front * EMERGENCY_MULTIPLIER * price
                )
            a_ub.append(row)
            b_ub.append(0.0)

    # --- 精确链式结算：补上历史最小承诺项 $-c\\,(m-x)^+$（凹项，需 0-1 变量）---
    if hist_min is not None:
        if settle == "none":
            raise ValueError("hist_min 只在 settle != none（有上一版承诺）时才有意义")
        if cvar_lambda > 0.0:
            # CVaR 情景费用（上方 a_ub）仍用近似目标 $\varphi$，而期望目标已改为
            # 真实增量 $\Delta F$，两者不同口径；本组合从未跑过，直接禁止以免误用。
            raise ValueError(
                "hist_min（精确链式结算）与 cvar_lambda>0 的组合未实现："
                "CVaR 约束内的情景费用仍是近似目标，与精确期望目标不同口径")
        # 真实阶段增量（去掉与 $x$ 无关的常数 $-c\\,m$）：
        #   $\\Delta F=c\\min(m,x)+0.5c\\,(r-x)^+ +1.5c\\,(x-r)^+$.
        # 直接按此改写目标，不再在 $\\varphi=0.5cx+1.0c(x-r)^+$ 上打补丁。
        obj[i_x : i_x + n] = 0.0
        obj[i_eb : i_eb + n] = front * BREACH_RATE * price      # $0.5c$
        obj[i_eo : i_eo + n] = front * OVERBUY_RATE * price     # $1.5c$
        m_hist = np.asarray(hist_min, dtype=float)
        i_d, i_z = nvar, nvar + n
        nvar += 2 * n
        obj = np.concatenate([obj, np.zeros(2 * n)])
        obj[i_d : i_d + n] = -front * price
        bounds = tuple(bounds) + ((0.0, None),) * n + ((0.0, 1.0),) * n
        for t in range(n):
            mt = max(float(m_hist[t]), 1e-9)   # $d_t\le m_t$，故可用 $m_t$ 作紧的 big-M
            r1 = np.zeros(nvar)
            r1[i_d + t], r1[i_z + t] = 1.0, -mt
            a_ub.append(r1)
            b_ub.append(0.0)
            r2 = np.zeros(nvar)
            r2[i_d + t], r2[i_x + t], r2[i_z + t] = 1.0, 1.0, BIG_M
            a_ub.append(r2)
            b_ub.append(float(m_hist[t]) + BIG_M)

    if hist_min is not None:
        def _pad(rows: list) -> np.ndarray:
            out = np.zeros((len(rows), nvar))
            for i, r in enumerate(rows):
                out[i, : len(r)] = r
            return out

        a_eq_np = _pad(a_eq)
        a_ub_np = _pad(a_ub) if a_ub else np.zeros((0, nvar))
        a_all = np.vstack([a_eq_np, a_ub_np]) if len(a_ub_np) else a_eq_np
        lo = np.concatenate([np.asarray(b_eq), np.full(len(b_ub), -np.inf)])
        hi = np.concatenate([np.asarray(b_eq), np.asarray(b_ub)])
        integ = np.zeros(nvar)
        integ[i_z : i_z + n] = 1
        lo_b = np.array([(-np.inf if b[0] is None else b[0]) for b in bounds])
        hi_b = np.array([(np.inf if b[1] is None else b[1]) for b in bounds])
        result = milp(obj, constraints=LinearConstraint(a_all, lo, hi),
                      integrality=integ, bounds=Bounds(lo_b, hi_b))
        if not result.success:
            raise RuntimeError(f"单阶段 MILP 失败（S={s_count}, n={n}）：{result.message}")
        return result.x[i_x : i_x + n]

    result = linprog(
        obj, A_eq=np.asarray(a_eq), b_eq=np.asarray(b_eq),
        A_ub=np.asarray(a_ub) if a_ub else None,
        b_ub=np.asarray(b_ub) if b_ub else None,
        bounds=bounds, method="highs",
    )
    if not result.success:
        raise RuntimeError(f"单阶段 LP 失败（S={s_count}, n={n}）：{result.message}")
    return result.x[i_x : i_x + n]


def causal_cumulative_reserve(
    errors_kw: np.ndarray,
    quantile: float = 0.80,
    min_history: int = 20,
) -> np.ndarray:
    """因果的**窗口累计**净负荷残差分位数（kWh），逐日返回。

    §三.1：不逐时段留保险，只问"这一整段窗口总共会偏多少"——

        $$ R_d = Q_{q}\\Big(\\big\\{\\textstyle\\sum_t \\xi_{j,t}\\Delta t : j<d\\big\\}\\Big). $$

    第 $d$ 行只用 $j<d$ 的残差，是因果量；历史不足 ``min_history`` 天时取 0。
    """
    n_days = errors_kw.shape[0]
    out = np.zeros(n_days)
    for d in range(n_days):
        hist = errors_kw[:d]
        if len(hist) < min_history:
            continue
        out[d] = float(np.quantile(hist.sum(axis=1) * DT, quantile))
    return out


# ==========================================================================
# 二、短时域滚动执行器（MPC）
# ==========================================================================
def execute_day_mpc(
    commitment: np.ndarray,
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    price: np.ndarray,
    start_energy: float,
    *,
    load_fc_kw: np.ndarray | None = None,
    pv_fc_kw: np.ndarray | None = None,
    horizon: int = 36,
    force_end_soc: float | None = None,
    greedy: bool = False,
) -> dict:
    """给定承诺 $x$，对当日实际负载/光伏执行储能充放电与紧急购电。

    ``greedy=True`` 复刻上游 ``execute_actual``：有剩余立即充电、有缺口立即放电、
    电池不足再紧急购电。
    ``greedy=False`` 启用滚动 MPC：**充电仍是贪心**（余量不充就只能弃掉，充电
    永远弱占优），但**缺口怎么补**交给 H 步前瞻 LP——用 5c 买紧急电还是动电池，
    由窗口内的价格路径决定。MPC 在每个出现缺口的时段重解一次，只用**预报**
    看未来、用**实际**执行当前，因此不含完美信息。

    目标（$x$ 已固定，$cx$ 为常数，从目标中略去）：

        $$ \\min\\;\\sum_{\\tau=t}^{t+H} 5c_\\tau e_\\tau \\;-\\; \\lambda\\,E_{t+H},
           \\qquad \\lambda=\\eta_d\\,\\overline{c}_{\\text{当日}}. $$

    ``load_fc_kw`` / ``pv_fc_kw`` 缺省时退化为用实际值看未来（仅用于对照，
    不是可实现的策略）。
    """
    n = len(commitment)
    load_act = load_kw * DT
    pv_act = pv_kw * DT
    if load_fc_kw is None:
        load_fc_kw = load_kw
    if pv_fc_kw is None:
        pv_fc_kw = pv_kw
    load_fc = np.nan_to_num(load_fc_kw, nan=0.0) * DT
    pv_fc = np.nan_to_num(pv_fc_kw, nan=0.0) * DT

    lam = end_water_value(price)          # 当日口径，与计划 LP 的水值一致

    energy = float(np.clip(start_energy, E_MIN, E_MAX))
    out = {k: np.zeros(n) for k in
           ("charge", "discharge", "emergency", "unused_plan",
            "curtailment", "pv_used", "plan_used", "soc")}

    t = 0
    while t < n:
        avail = commitment[t] + pv_act[t]
        if avail >= load_act[t]:                       # 有余量：贪心充电
            surplus = avail - load_act[t]
            charge = min(surplus, STEP_MAX, max(0.0, (E_MAX - energy) / ETA_C))
            energy += ETA_C * charge
            consumption = load_act[t] + charge
            pv_used = min(pv_act[t], consumption)
            plan_used = min(commitment[t], max(0.0, consumption - pv_used))
            out["charge"][t] = charge
            out["pv_used"][t] = pv_used
            out["plan_used"][t] = plan_used
            out["curtailment"][t] = max(0.0, pv_act[t] - pv_used)
            out["unused_plan"][t] = max(0.0, commitment[t] - plan_used)
            out["soc"][t] = energy
            t += 1
            continue

        deficit = load_act[t] - avail
        vmax = min(deficit, STEP_MAX, max(0.0, (energy - E_MIN) * ETA_D))
        if greedy or vmax <= 1e-12:
            discharge = vmax
            emergency = max(0.0, deficit - discharge)
        else:
            discharge, emergency = _deficit_choice(
                t, commitment, load_act, pv_act, load_fc, pv_fc, price,
                energy, deficit, vmax, lam, horizon, force_end_soc,
            )
        energy -= discharge / ETA_D
        out["discharge"][t] = discharge
        out["emergency"][t] = emergency
        out["pv_used"][t] = pv_act[t]
        out["plan_used"][t] = commitment[t]
        out["soc"][t] = energy
        t += 1

    out["end_energy"] = float(energy)
    return out


def _deficit_choice(
    t: int,
    commitment: np.ndarray,
    load_act: np.ndarray,
    pv_act: np.ndarray,
    load_fc: np.ndarray,
    pv_fc: np.ndarray,
    price: np.ndarray,
    energy: float,
    deficit: float,
    vmax: float,
    lam: float,
    horizon: int,
    force_end_soc: float | None,
) -> tuple[float, float]:
    """缺口时段的 H 步前瞻 LP：返回 (放电量, 紧急购电量)。"""
    n = len(commitment)
    end = min(t + horizon, n)
    m = end - t
    i_u, i_v, i_e, i_s, i_w = 0, m, 2 * m, 3 * m, 4 * m
    nvar = 5 * m

    obj = np.zeros(nvar)
    obj[i_u : i_u + m] = 1e-7
    obj[i_v : i_v + m] = 1e-7
    obj[i_e : i_e + m] = EMERGENCY_MULTIPLIER * price[t:end]
    if force_end_soc is None:
        obj[i_s + m - 1] = -lam
    else:
        obj[i_s + m - 1] = 0.0

    # 充电与弃光的上限都只能是"余量"，因此不可能凭空造电、也不可能用弃光
    # 冒充放电把缺口糊过去（否则 LP 会选 w 而不选 v，白拿 1e-7 的便宜）。
    u_max = np.zeros(m)
    w_max = np.zeros(m)
    for j in range(m):
        tau = t + j
        if j == 0:
            continue                             # 当前时段是缺口，充电弃光均为 0
        surplus = max(0.0, commitment[tau] + pv_fc[tau] - load_fc[tau])
        u_max[j] = min(STEP_MAX, surplus)
        w_max[j] = surplus

    a_eq, b_eq = [], []
    for j in range(m):
        row = np.zeros(nvar)
        row[i_u + j], row[i_v + j], row[i_s + j] = -ETA_C, 1.0 / ETA_D, 1.0
        if j:
            row[i_s + j - 1] = -1.0
            b_eq.append(0.0)
        else:
            b_eq.append(energy)
        a_eq.append(row)

    if force_end_soc is not None:
        end_bound = (float(np.clip(force_end_soc, E_MIN, E_MAX)),) * 2
    else:
        end_bound = (E_MIN, E_MAX)
    soc_bounds = [(E_MIN, E_MAX)] * (m - 1) + [end_bound]

    a_ub, b_ub = [], []
    for j in range(m):
        tau = t + j
        if j == 0:
            supply = pv_act[tau] + commitment[tau] - load_act[tau]
        else:
            supply = pv_fc[tau] + commitment[tau] - load_fc[tau]
        # 供 ≥ 需：-(v+e) + u + w ≤ pv + x - load
        row = np.zeros(nvar)
        row[i_u + j], row[i_v + j], row[i_e + j], row[i_w + j] = 1.0, -1.0, -1.0, 1.0
        a_ub.append(row)
        b_ub.append(float(supply))

    upper_v = min(STEP_MAX, vmax) if m else 0.0
    bounds = (
        [(0.0, float(u_max[j])) for j in range(m)]
        + [(0.0, upper_v)] + [(0.0, STEP_MAX)] * (m - 1)
        + [(0.0, None)] * m
        + soc_bounds
        + [(0.0, float(w_max[j])) for j in range(m)]
    )

    res = linprog(
        obj, A_eq=np.asarray(a_eq), b_eq=np.asarray(b_eq),
        A_ub=np.asarray(a_ub), b_ub=np.asarray(b_ub),
        bounds=bounds, method="highs",
    )
    if not res.success:
        return vmax, max(0.0, deficit - vmax)       # 兜底回退到贪心
    v = float(res.x[i_v])
    v = min(max(v, 0.0), vmax)
    return v, max(0.0, deficit - v)


# ==========================================================================
# 三、全年调度循环（两种信息结构）
# ==========================================================================
_DAY_KEYS = ("charge", "discharge", "emergency", "unused_plan",
             "curtailment", "pv_used", "plan_used", "soc")


def run_day_single(
    commitment: np.ndarray,
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    price: np.ndarray,
    start_energy: float,
    *,
    load_fc_kw: np.ndarray | None = None,
    pv_fc_kw: np.ndarray | None = None,
    horizon: int = 36,
    force_end_soc: float | None = None,
    mpc: bool = True,
) -> dict:
    """单决策时刻（问题2/4-2）的一日执行。返回 executed 字典。"""
    if not mpc:
        return _greedy_execute(commitment, load_kw, pv_kw, start_energy)
    return execute_day_mpc(
        commitment, load_kw, pv_kw, price, start_energy,
        load_fc_kw=load_fc_kw, pv_fc_kw=pv_fc_kw,
        horizon=horizon, force_end_soc=force_end_soc,
    )


_AUDIT_KEYS = ("max_balance_err", "max_purchase_identity_err", "max_pv_identity_err",
               "max_soc_recursion_err", "soc_violations", "power_violations",
               "simultaneous_charge_discharge", "negative_flow", "slots")


def audit_execution(act: dict, commitment: np.ndarray, load_kw: np.ndarray,
                    pv_kw: np.ndarray, start_energy: float) -> dict:
    """对一次执行的逐时段输出做物理核对，返回各误差的最大值与越界计数。

    与上游冻结核对（`verify/out/表_物理可行性.csv`）同口径：
    能量平衡、购电恒等式、光伏恒等式、SOC 递推，以及 SOC/功率越界、
    同时充放电、负流量。调用方只需把结果按项取 max / 求和，无需保存逐时段输出。
    """
    load = load_kw * DT
    pv = pv_kw * DT
    n = len(commitment)
    charge, discharge = act["charge"][:n], act["discharge"][:n]
    emer, unused = act["emergency"][:n], act["unused_plan"][:n]
    pv_used, plan_used = act["pv_used"][:n], act["plan_used"][:n]
    curt, soc = act["curtailment"][:n], act["soc"][:n]

    balance = (plan_used + emer + discharge + pv_used) - (load + charge)
    purchase_id = (plan_used + unused) - commitment[:n]
    pv_id = (pv_used + curt) - pv

    e = np.empty(n)
    prev = float(np.clip(start_energy, E_MIN, E_MAX))
    for t in range(n):
        prev = prev + ETA_C * charge[t] - discharge[t] / ETA_D
        e[t] = prev
    soc_err = soc - e

    return {
        "max_balance_err": float(np.abs(balance).max()),
        "max_purchase_identity_err": float(np.abs(purchase_id).max()),
        "max_pv_identity_err": float(np.abs(pv_id).max()),
        "max_soc_recursion_err": float(np.abs(soc_err).max()),
        "soc_violations": int(((soc < E_MIN - 1e-6) | (soc > E_MAX + 1e-6)).sum()),
        "power_violations": int(((charge < -1e-9) | (charge > STEP_MAX + 1e-6)
                                 | (discharge < -1e-9) | (discharge > STEP_MAX + 1e-6)).sum()),
        "simultaneous_charge_discharge": int(((charge > 1e-9) & (discharge > 1e-9)).sum()),
        "negative_flow": int(sum(int((a < -1e-9).sum()) for a in
                                 (charge, discharge, emer, unused, pv_used, plan_used, curt))),
        "slots": int(n),
    }


def merge_audit(total: dict, one: dict) -> dict:
    """把单日审计结果并入累计器：误差取 max，计数求和，时段数累加。"""
    for k in _AUDIT_KEYS:
        v = one[k]
        if k == "slots":
            total[k] = total.get(k, 0) + v
        elif k.endswith("violations") or k in ("simultaneous_charge_discharge",
                                               "negative_flow"):
            total[k] = total.get(k, 0) + v
        else:
            total[k] = max(total.get(k, 0.0), v)
    return total


def _greedy_execute(commitment, load_kw, pv_kw, start_energy) -> dict:
    """复刻上游 ``execute_actual``（逐位一致），用于基线对照。"""
    load = load_kw * DT
    pv = pv_kw * DT
    energy = float(np.clip(start_energy, E_MIN, E_MAX))
    n = len(commitment)
    out = {k: np.zeros(n) for k in _DAY_KEYS}
    for t in range(n):
        available = commitment[t] + pv[t]
        if available >= load[t]:
            surplus = available - load[t]
            charge = min(surplus, STEP_MAX, max(0.0, (E_MAX - energy) / ETA_C))
            energy += ETA_C * charge
            consumption = load[t] + charge
            pv_used = min(pv[t], consumption)
            plan_used = min(commitment[t], max(0.0, consumption - pv_used))
            out["charge"][t] = charge
            out["pv_used"][t] = pv_used
            out["plan_used"][t] = plan_used
            out["curtailment"][t] = max(0.0, pv[t] - pv_used)
            out["unused_plan"][t] = max(0.0, commitment[t] - plan_used)
        else:
            deficit = load[t] - available
            discharge = min(deficit, STEP_MAX, max(0.0, (energy - E_MIN) * ETA_D))
            energy -= discharge / ETA_D
            out["discharge"][t] = discharge
            out["emergency"][t] = max(0.0, deficit - discharge)
            out["pv_used"][t] = pv[t]
            out["plan_used"][t] = commitment[t]
        out["soc"][t] = energy
    out["end_energy"] = float(energy)
    return out


def settle_period(commitment_path: np.ndarray, price: np.ndarray) -> dict:
    """链式结算（题面 #15）：基准量取路径最小值，相邻变动分别计价。

    ``commitment_path`` 形状 ``(n_stage, n_slot)``。
    """
    base = breach = overbuy = down_tot = up_tot = 0.0
    for t in range(commitment_path.shape[1]):
        seq = [float(commitment_path[j, t]) for j in range(commitment_path.shape[0])]
        c = float(price[t])
        base += c * min(seq)
        down = up = 0.0
        prev = seq[0]
        for cur in seq[1:]:
            if cur < prev:
                down += prev - cur
            else:
                up += cur - prev
            prev = cur
        breach += BREACH_RATE * c * down
        overbuy += OVERBUY_RATE * c * up
        down_tot += down
        up_tot += up
    return {
        "base_yuan": base,
        "breach_yuan": breach,
        "overbuy_yuan": overbuy,
        "total_yuan": base + breach + overbuy,
        "adjusted_kwh": down_tot + up_tot,
        "down_kwh": down_tot,
        "up_kwh": up_tot,
    }
