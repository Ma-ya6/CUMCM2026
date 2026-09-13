# -*- coding: utf-8 -*-
"""附录 · 四问核心求解代码

本文件汇集 CUMCM 2026 C 题**四个问题的求解主干代码**，按问题分节，节内按
"预测 → 计划 → 执行 → 结算"的决策链排列：

    §0  公共物理与经济常数
    §1  问题1：单日线性规划（约束 A 固定端点 / 约束 B 首尾自由）
    §2  问题2：因果预测器 + 分位数安全余量 + 储备型 LP + 滚动 MPC 执行
    §3  问题3：官方预报融合 + 链式承诺增量 MILP + 贪心执行 + 链式结算
    §4  问题4：波动电价的因果预测，与问题2/3 的模型装配
    §5  全年因果调度主循环（四问共用，靠开关切换）
    §6  附件读取与一键复现入口（main()）

对应源码：代码/问题1/solve_q1_ab.py、代码/共享/opt_core.py、代码/共享/run_year.py、
        代码/问题2/deps/{deterministic_baseline.py, seasonal/*}、
        代码/问题3/deps/forecasts.py、代码/问题4/问题4-2/deps/price_forecast.py

与原源码的对照（仅列改名与合并，其余同名同签名）：
    solve_q1(...)                    ← solve_a() / solve_b()，合并为 fixed_endpoint 开关
    causal_cumulative_reserve(...)   ← opt_core 同名函数（主链实际在 run_year 内联，同口径）
    greedy_execute(...)              ← opt_core._greedy_execute()
    execute_year_end(...)            ← run_year.execute()
    simulate_year(...)               ← run_year.simulate()，去掉落盘、审计与明细整理
    official_points(a, raw)          ← run_year.official_points(a)，raw 省略时自动读附件 3
    build_pv_fusion(official, own, actual)
                                     ← forecasts.build_pv_fusion(data, own_pv)，输入改为数组
    generate_causal_price_forecast(price, initial_price, dates, stress_by_day, ...)
                                     ← price_forecast.generate_causal_price_forecast(data, diagnostics)
    load_inputs_fixed()              ← 问题2/3 deterministic_baseline.load_inputs()
    load_inputs_variable()           ← 问题4-2 deterministic_baseline.load_inputs()
    prepare_all()                    ← run_year.prepare()，去掉 npz 缓存与指纹
    main()                           ← run_year.main() + solve_q1_ab.main()，
                                       落盘到 结果/附录复现/（见 §6）

为保持可读，只保留**进入决策路径**的计算代码，省略下列非建模内容：
  · 提交工作簿与图表的落盘（Excel 整表回填、npz 缓存、出图）
  · 仅用于事后评价的诊断函数：season_label、summarize_by_season、
    weekly_effect_significance、WeeklyEffectModel.diagnostics、forecast_metrics 等
  · 缓存指纹、命令行入口、物理审计、出图、敏感性/消融补跑
  · 未进入主链的分支：CVaR 情景与随机规划、settle="legacy" 复刻口径、
    逐步贪心基线的对照、年终 force_end_soc 参数、CausalEnsemble.window_fn

省略的都是 I/O、诊断与对照实验。**模型结构、约束、目标函数与参数取值均与原
代码逐位一致**，本文件中的函数已逐函数与原实现做过数值比对（最大差 0.0）。
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


# Windows 控制台默认 GBK，无法表示正文中的 U+2212（减号）等字符，直接 print 会抛
# UnicodeEncodeError 中断复现；统一把标准流切到 UTF-8，同时消除中文乱码。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")


# ===========================================================================
# §0  公共物理与经济常数
#     源码：代码/共享/opt_core.py 第 30—45 行、代码/问题2/deps/deterministic_baseline.py
# ===========================================================================
DT = 1.0 / 6.0            # 时段时长 Δt = 10 min，单位 h
N_SLOT = 144              # 日内时段数 T
SLOTS_PER_HOUR = 6
SEED = 20260910           # 固定随机种子，保证逐位可复现
ETA_C = 0.90              # 充电效率 η_c
ETA_D = 0.90              # 放电效率 η_d
E_MIN = 1200.0            # 储电量下限 E_min，kWh
E_MAX = 10800.0           # 储电量上限 E_max，kWh
E_INITIAL = 6000.0        # 初始与年终储电量 E_0，kWh
P_MAX_KW = 5000.0         # 最大充放电功率 P_st^max，kW
STEP_MAX = P_MAX_KW * DT  # 单时段最大充放电量，kWh
EMERGENCY_MULTIPLIER = 5.0    # 紧急购电电价倍数（问题2/4-2）
BREACH_RATE = 0.5             # 链式结算：下调承诺的违约系数
OVERBUY_RATE = 1.5            # 链式结算：上调承诺的超购系数
BIG_M = 1.0e5                 # 精确链式结算 MILP 的松弛常数
ISSUE_HOURS = (0, 6, 12, 18)  # 问题3/4-3 的四个决策发布时刻
PURCHASE_TOL = 1.0e-6         # 购电承诺的下界舍入容差，kWh
FORMAL_START = pd.Timestamp("2025-02-01")   # 正式输出期起点（334 天）
NIGHT_KW = 20.0           # 历史均值低于此值的小时视为夜间，光伏融合不做回归
MIN_STACK_SAMPLES = 40    # 逐（发布时刻, 小时）回归所需的最少样本数


@dataclass
class DataBundle:
    """问题1—4 的输入数据包，全部来自附件 1、2、4。"""
    dates: pd.DatetimeIndex
    load_kw: np.ndarray        # (365, 144) 实际负载功率
    pv_kw: np.ndarray          # (365, 144) 实际光伏功率
    price: np.ndarray          # 问题1—3 为 (144,) 分时电价；问题4 为 (365,144) 逐日波动电价
    initial_load: np.ndarray   # (144,) 冷启动用的首日负载曲线
    initial_pv: np.ndarray     # (144,) 冷启动用的首日光伏曲线
    initial_price: np.ndarray | None = None   # (144,) 附件1 单条电价曲线，仅供问题4 第 0 天冷启动


def end_water_value(price: np.ndarray) -> float:
    """日末储电的水值 λ = η_d × 当日平均电价，作为计划 LP 与执行器的终值口径。"""
    return float(ETA_D * np.mean(price))


def clip_purchase(x: np.ndarray) -> np.ndarray:
    """求解器出口：容差内的负购电量归零，明显负值仍报错。

    LP/MILP 会返回量级 1e-7 kWh 的"负零"，直接传给执行器会被物理审计判为
    负流量。此处只对 |x| < PURCHASE_TOL 的下界舍入归零。
    """
    if x.size and float(x.min()) < -PURCHASE_TOL:
        raise RuntimeError(f"购电承诺出现明显负值 {float(x.min()):.6e} kWh")
    return np.maximum(x, 0.0)


# ===========================================================================
# §1  问题1：单日线性规划（约束 A 固定端点 / 约束 B 首尾自由）
#     源码：代码/问题1/solve_q1_ab.py 的 solve_a() 与 solve_b()
#
#   两者除第 (4) 条日末回归约束外逐行相同，此处合并为一个函数，用
#   fixed_endpoint 开关切换——这正是问题1 唯一需要讨论的建模差别。
#
#     min  Σ_t c_t · x_t · Δt
#     s.t. (1) x_t + P_t^fc + v_t = L_t + u_t + w_t         电量平衡
#          (2) E_t = E_{t-1} + η_c·u_t·Δt − v_t·Δt/η_d      储能动态
#          (3) E_min ≤ E_t ≤ E_max                          储电量范围
#          (4) A: E_T = E_0 = 6000   B: E_T − E_0 = 0        日末回归
#          (5) x_t ≥ 0, 0 ≤ u_t,v_t ≤ P_st^max, w_t ≥ 0      变量范围
#
#   因 η_c·η_d = 0.81 < 1，最优解自动满足 min(u_t, v_t) = 0，无需 0-1 变量，
#   模型保持纯线性，用 HiGHS 一次求出全局最优。
#   约束 A 要求 E_T = E_0 = 6000，取 E_0 = 6000 后 A 的任意可行解在 B 中均可行，
#   故 B 的可行域更大，必有 C(B) ≤ C(A)。
# ===========================================================================
def solve_q1(
    c: np.ndarray,
    L: np.ndarray,
    P_fc: np.ndarray,
    *,
    fixed_endpoint: bool = True,
) -> dict:
    """求解问题1 单日模型。

    c / L / P_fc：长度 T = 144 的电价（元/kWh）、负载功率（kW）、光伏功率（kW）。
    fixed_endpoint=True  → 约束 A：E_0 = E_T = 6000 kWh；
    fixed_endpoint=False → 约束 B：E_T − E_0 = 0，共同取值由优化决定。
    """
    T = N_SLOT
    EPS_W = 1e-6        # 仅为消除弃光量不唯一的数值项，不影响费用

    # 解向量布局：x(T) | u(T) | v(T) | E(T+1) | w(T)
    iX, iU, iV, iE, iW = 0, T, 2 * T, 3 * T, 4 * T + 1
    nv = 5 * T + 1

    # ---- 目标函数：min Σ c_t x_t Δt
    c_obj = np.zeros(nv)
    c_obj[iX:iX + T] = c * DT
    c_obj[iW:iW + T] = EPS_W

    # ---- 等式约束 (1)(2)(4)
    n_eq = 2 * T + (2 if fixed_endpoint else 1)
    A_eq = np.zeros((n_eq, nv))
    b_eq = np.zeros(n_eq)
    for t in range(T):
        # (1) x_t − u_t + v_t − w_t = L_t − P_t^fc
        A_eq[t, iX + t] = 1.0
        A_eq[t, iU + t] = -1.0
        A_eq[t, iV + t] = 1.0
        A_eq[t, iW + t] = -1.0
        b_eq[t] = L[t] - P_fc[t]
        # (2) E_t − E_{t−1} − η_c Δt u_t + (Δt/η_d) v_t = 0
        A_eq[T + t, iE + t] = -1.0
        A_eq[T + t, iE + t + 1] = 1.0
        A_eq[T + t, iU + t] = -ETA_C * DT
        A_eq[T + t, iV + t] = DT / ETA_D
    # (4) 日末回归
    if fixed_endpoint:
        A_eq[2 * T, iE] = 1.0                       # E_0 = 6000
        b_eq[2 * T] = E_INITIAL
        A_eq[2 * T + 1, iE + T] = 1.0               # E_T = 6000
        b_eq[2 * T + 1] = E_INITIAL
    else:
        A_eq[2 * T, iE] = 1.0                       # E_T − E_0 = 0
        A_eq[2 * T, iE + T] = -1.0
        b_eq[2 * T] = 0.0

    # ---- 变量边界 (3)(5)
    bounds = ([(0.0, None)] * T                 # x_t ≥ 0
              + [(0.0, P_MAX_KW)] * T           # 0 ≤ u_t ≤ P_st^max
              + [(0.0, P_MAX_KW)] * T           # 0 ≤ v_t ≤ P_st^max
              + [(E_MIN, E_MAX)] * (T + 1)      # E_min ≤ E_t ≤ E_max
              + [(0.0, None)] * T)              # w_t ≥ 0

    res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("问题1 求解失败：" + str(res.message))

    z = res.x
    return {
        "x": z[iX:iX + T], "u": z[iU:iU + T], "v": z[iV:iV + T],
        "E": z[iE:iE + T + 1], "w": z[iW:iW + T],
        "C": float(res.fun),
    }


# ===========================================================================
# §2  问题2：因果预测 + 分位数安全余量 + 储备型 LP + 滚动 MPC 执行
#
#   问题2 的输入全部不可预知：第 d 天 0:00 制定当日计划时，当天 144 个时段的
#   负载与光伏都尚未发生。因此整条决策链只允许使用 j < d 的已实现值：
#
#     2.1 基础预测器：近期加权基线 / 岭回归 / 浅层 LightGBM 的误差加权集成
#     2.2 季节感知：只用已执行残差度量季节漂移，并用光谱窗口代理昼长
#     2.3 在线校正：光伏有效窗口 → 季节水平 → 周内水平 → 周型形状，四级级联
#     2.4 全年因果预测生成 generate_causal_forecasts
#     2.5 安全余量：历史窗口累计残差的 0.80 分位数（报童临界比）
#     2.6 计划：储备型 LP stage_lp(settle="none", reserve=R)
#     2.7 执行：短时域滚动 MPC
# ===========================================================================

# ---------------------------------------------------------------------------
# 2.1  基础预测器（源码：代码/问题2/deps/deterministic_baseline.py 第 113—284 行）
# ---------------------------------------------------------------------------
def recent_weighted(data: np.ndarray, day: int, initial: np.ndarray) -> np.ndarray:
    """预热期（第 7 天前）的 1/2/3 日滞后加权基线。"""
    if day == 0:
        return initial.copy()
    lags = [(1, 0.60), (2, 0.30), (3, 0.10)]
    valid = [(lag, weight) for lag, weight in lags if day - lag >= 0]
    total = sum(weight for _, weight in valid)
    return sum(weight * data[day - lag] for lag, weight in valid) / total


def weekly_load_baseline(data: np.ndarray, day: int, initial: np.ndarray) -> np.ndarray:
    """同星期几基线：取 7 日前曲线，并按前两周能量比做水平校正。

    小区用电有明显的周内结构（工作日/周末作息），同星期几参照比单纯的时间
    滞后更贴合；比值裁剪到 [0.85, 1.15] 防止单周异常被放大。
    """
    if day < 7:
        return recent_weighted(data, day, initial)
    if day >= 14:
        previous_week = data[day - 7 : day].sum()
        earlier_week = data[day - 14 : day - 7].sum()
        ratio = np.clip(previous_week / max(earlier_week, 1e-9), 0.85, 1.15)
    else:
        ratio = 1.0
    return np.maximum(data[day - 7] * ratio, 0.0)


def recent_pv_baseline(data: np.ndarray, day: int, initial: np.ndarray) -> np.ndarray:
    """光伏基线：1/2/7 日滞后加权。光伏由天气主导，无周内结构。"""
    if day == 0:
        return initial.copy()
    candidates = [(1, 0.50), (2, 0.30), (7, 0.20)]
    valid = [(lag, weight) for lag, weight in candidates if day - lag >= 0]
    total = sum(weight for _, weight in valid)
    pred = sum(weight * data[day - lag] for lag, weight in valid) / total
    return np.maximum(pred, 0.0)


def day_features(data: np.ndarray, dates: pd.DatetimeIndex, day: int) -> np.ndarray:
    """构造第 day 天的回归特征矩阵 (N_SLOT, 15)：滞后、滑动均值、日内/周内/年内周期。"""
    slot = np.arange(N_SLOT)

    def lag(k: int) -> np.ndarray:
        index = max(day - k, 0)
        return data[index]

    def rolling(days: int) -> np.ndarray:
        start = max(0, day - days)
        if start == day:
            return data[0]
        return data[start:day].mean(axis=0)

    recent7 = data[max(0, day - 7) : day]
    prior7 = data[max(0, day - 14) : max(0, day - 7)]
    recent_level = float(recent7.mean()) if recent7.size else float(data[0].mean())
    prior_level = float(prior7.mean()) if prior7.size else recent_level
    trend = np.clip(recent_level / max(prior_level, 1e-9), 0.8, 1.2)

    date = dates[day]
    features = np.column_stack(
        [
            lag(1),
            lag(2),
            lag(7),
            lag(14),
            rolling(3),
            rolling(7),
            rolling(14),
            np.sin(2 * np.pi * slot / N_SLOT),
            np.cos(2 * np.pi * slot / N_SLOT),
            np.full(N_SLOT, np.sin(2 * np.pi * date.dayofweek / 7)),
            np.full(N_SLOT, np.cos(2 * np.pi * date.dayofweek / 7)),
            np.full(N_SLOT, np.sin(2 * np.pi * date.dayofyear / 365)),
            np.full(N_SLOT, np.cos(2 * np.pi * date.dayofyear / 365)),
            np.full(N_SLOT, trend),
            np.full(N_SLOT, float(date.dayofweek in (4, 5))),
        ]
    )
    return features


def training_matrix_between(
    data: np.ndarray,
    dates: pd.DatetimeIndex,
    start_day: int,
    end_day: int,
) -> tuple[np.ndarray, np.ndarray]:
    """取 [start_day, end_day) 区间的训练样本，前 14 天留给特征构造。"""
    days = range(max(14, start_day), end_day)
    x = np.vstack([day_features(data, dates, day) for day in days])
    y = np.concatenate([data[day] for day in days])
    return x, y


def training_matrix(
    data: np.ndarray,
    dates: pd.DatetimeIndex,
    end_day: int,
    window_days: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    return training_matrix_between(data, dates, end_day - window_days, end_day)


def make_lightgbm(n_estimators: int) -> lgb.LGBMRegressor:
    """构造受约束的浅层模型，避免十分钟强相关样本诱导过细叶节点。"""
    return lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=7,
        max_depth=3,
        min_child_samples=N_SLOT,
        min_split_gain=0.01,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=5.0,
        random_state=SEED,
        # 单线程并启用确定性模式，保证相同历史前缀得到逐位一致的预测。
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def fit_lightgbm_causal(
    data: np.ndarray,
    dates: pd.DatetimeIndex,
    end_day: int,
    window_days: int = 180,
    validation_days: int = 21,
) -> tuple[lgb.LGBMRegressor, int]:
    """只用预测日以前的数据早停，并按最佳轮数在完整历史窗上重拟合。"""
    start_day = max(14, end_day - window_days)
    split_day = max(start_day + 21, end_day - validation_days)
    x_train, y_train = training_matrix_between(data, dates, start_day, split_day)
    x_valid, y_valid = training_matrix_between(data, dates, split_day, end_day)
    probe = make_lightgbm(600)
    probe.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )
    best_iteration = int(probe.best_iteration_ or 200)
    best_iteration = int(np.clip(best_iteration, 30, 400))
    x_full, y_full = training_matrix_between(data, dates, start_day, end_day)
    model = make_lightgbm(best_iteration)
    model.fit(x_full, y_full)
    return model, best_iteration


def adaptive_error_weights(
    available, history: dict[str, list[float]], lookback: int = 28
) -> dict[str, float]:
    """由近期误差动态组合三个候选模型。

    指数加权近期误差给最近一天最高权重；同时掺入上四分位误差，使偶发失稳的
    复杂模型被实质降权，而不是靠平均误差掩盖。
    """
    names = list(available)
    if len(names) == 1:
        return {names[0]: 1.0}
    scores = []
    for name in names:
        errors = np.asarray(history.get(name, [])[-lookback:], float)
        if errors.size:
            recency = np.exp(np.linspace(-2.5, 0.0, errors.size))
            mean_error = float(np.average(errors, weights=recency))
            score = 0.85 * mean_error + 0.15 * float(np.quantile(errors, 0.75))
        else:
            score = np.nan
        scores.append(score)
    scores = np.asarray(scores, float)
    if not np.isfinite(scores).all():
        return dict.fromkeys(names, 1.0 / len(names))
    relative_loss = scores / max(float(scores.min()), 1e-9) - 1.0
    raw = np.exp(-6.0 * relative_loss)
    weights = raw / raw.sum()
    weights = np.clip(weights, 0.03, 0.80)
    weights /= weights.sum()
    return dict(zip(names, weights))


# ---------------------------------------------------------------------------
# 2.2  季节感知（源码：代码/问题2/deps/seasonal/season_state.py）
#
#   题目只给出电价、负载、光伏三类序列，没有给出纬度、日出日落或任何气象量。
#   因此季节必须由**已发生序列自身**的统计性质感知，共三条通道：
#     ① 水平漂移 Δ_lev：最近短窗与更早参考窗的残差均值之差，归一化；
#     ② 尺度漂移 Δ_sca：两窗残差均方根之比的对数，刻画不确定性放大；
#     ③ 光伏有效发电窗口：窗口长度是昼长的数据代理，窗口移动方向即季节推进方向。
#   三者合成 [0,1] 的季节应力 σ_d；σ_d 越大表示"当前已偏离模型依赖的旧季节"，
#   模型应更快遗忘、更保守。
# ---------------------------------------------------------------------------
PV_ACTIVE_ABS_THRESHOLD_KW = 20.0
PV_ACTIVE_REL_THRESHOLD = 0.01


@dataclass
class DriftConfig:
    """漂移检测窗口与映射参数，均为结构性先验，不从全年回测结果调参。"""

    short_window: int = 7
    reference_window: int = 21
    level_weight: float = 0.60
    scale_weight: float = 0.40
    level_sensitivity: float = 1.00
    scale_sensitivity: float = 2.00
    smoothing: float = 0.50
    # 漂移→记忆长度映射：窗口 = clip(基窗 / (1 + shrink_gain * stress), 下限, 基窗)
    shrink_gain: float = 2.50
    min_window_ratio: float = 0.25


@dataclass(frozen=True)
class SeasonState:
    """某一日由历史数据推断出的季节状态，0:00 可确定，不含未来信息。"""

    date: pd.Timestamp
    doy: int
    pv_window_start: int          # 由历史光伏曲线推断的有效发电窗口（时段序号，0 基）
    pv_window_end: int
    pv_window_length: int
    daylight_proxy_hours: float          # 昼长的数据代理
    daylight_proxy_trend_hours: float    # 相对 14 天前的变化，正为变长
    drift_level: float
    drift_scale: float
    stress: float
    drift_direction: float


class PVActiveWindowTracker:
    """由历史光伏曲线因果推断每日有效发电窗口。

    对每个已执行日，用绝对与相对双阈值找出光伏功率超过阈值的首末时段；当日
    无有效发电时沿用前一天窗口。预测下一日窗口时，用最近若干天窗口起止的稳健
    水平加上被裁剪的趋势外推，并保留一个时段缓冲，避免阈值噪声截断有效光伏。
    """

    def __init__(
        self,
        n_slot: int,
        abs_threshold_kw: float = PV_ACTIVE_ABS_THRESHOLD_KW,
        rel_threshold: float = PV_ACTIVE_REL_THRESHOLD,
        trend_window: int = 14,
        level_window: int = 7,
        buffer_slots: int = 1,
    ) -> None:
        self.n_slot = int(n_slot)
        self.abs_threshold_kw = float(abs_threshold_kw)
        self.rel_threshold = float(rel_threshold)
        self.trend_window = int(trend_window)
        self.level_window = int(level_window)
        self.buffer_slots = int(buffer_slots)
        self.starts: list[int] = []
        self.ends: list[int] = []

    @staticmethod
    def _fallback_window(n_slot: int) -> tuple[int, int]:
        return int(round(0.29 * n_slot)), int(round(0.76 * n_slot))

    def observe(self, pv_kw: np.ndarray) -> None:
        """登记一个已执行日的实际光伏曲线，更新窗口历史。"""
        curve = np.asarray(pv_kw, float)
        threshold = max(self.abs_threshold_kw, self.rel_threshold * float(curve.max()))
        active = np.flatnonzero(curve > threshold)
        if active.size:
            self.starts.append(int(active[0]))
            self.ends.append(int(active[-1]))
        elif self.starts:
            self.starts.append(self.starts[-1])
            self.ends.append(self.ends[-1])
        else:
            start, end = self._fallback_window(self.n_slot)
            self.starts.append(start)
            self.ends.append(end)

    def predict_window(self) -> tuple[int, int]:
        """预测下一日的有效发电窗口。只用已登记的窗口历史。"""
        if not self.starts:
            return self._fallback_window(self.n_slot)
        recent_start = np.asarray(self.starts[-self.trend_window :], float)
        recent_end = np.asarray(self.ends[-self.trend_window :], float)
        start_level = float(np.median(recent_start[-self.level_window :]))
        end_level = float(np.median(recent_end[-self.level_window :]))
        start_trend = float(np.median(np.diff(recent_start))) if recent_start.size > 1 else 0.0
        end_trend = float(np.median(np.diff(recent_end))) if recent_end.size > 1 else 0.0
        start = int(np.clip(round(start_level + np.clip(start_trend, -1.0, 1.0)), 0, self.n_slot - 1))
        end = int(np.clip(round(end_level + np.clip(end_trend, -1.0, 1.0)), 0, self.n_slot - 1))
        if end < start:
            start, end = self._fallback_window(self.n_slot)
        return start, end

    def apply_window(self, prediction: np.ndarray) -> np.ndarray:
        """把预测窗口之外的光伏置零，窗口两端各留缓冲时段。"""
        start, end = self.predict_window()
        corrected = np.asarray(prediction, float).copy()
        corrected[: max(start - self.buffer_slots, 0)] = 0.0
        corrected[min(end + self.buffer_slots + 1, self.n_slot) :] = 0.0
        return np.maximum(corrected, 0.0)

    def window_length_series(self, lookback: int = 14) -> tuple[float, float]:
        """返回最近窗口长度（小时）及其相对更早窗口的变化。"""
        if not self.starts:
            start, end = self._fallback_window(self.n_slot)
            return (end - start + 1) / 6.0, 0.0
        lengths = (np.asarray(self.ends, float) - np.asarray(self.starts, float) + 1.0) / 6.0
        recent = float(np.median(lengths[-min(lookback, lengths.size) :]))
        if lengths.size > lookback:
            earlier = float(np.median(lengths[-min(2 * lookback, lengths.size) : -lookback]))
        else:
            earlier = recent
        return recent, recent - earlier


class SeasonalDriftDetector:
    """只用已执行日残差度量季节/工作点漂移，输出 [0,1] 的季节应力。

    第 d 天的输出只依赖 j<d 的残差。水平漂移回答"预测是否系统性偏离了新季节的
    平均水平"，尺度漂移回答"残差幅度是否整体变大"。
    """

    def __init__(self, config: DriftConfig | None = None) -> None:
        self.config = config or DriftConfig()
        self._residuals: list[np.ndarray] = []
        self._smoothed: float = 0.0
        self._level: float = 0.0
        self._scale: float = 0.0

    def update(self, residual: np.ndarray) -> None:
        """登记一个已执行日的逐时段残差（实际 − 预测）。"""
        self._residuals.append(np.asarray(residual, float).copy())

    def _windows(self):
        cfg = self.config
        need = cfg.short_window + cfg.reference_window
        if len(self._residuals) < need:
            return None
        recent = np.asarray(self._residuals[-cfg.short_window :], float)
        older = np.asarray(self._residuals[-need : -cfg.short_window], float)
        return recent, older

    def stats(self) -> tuple[float, float, float]:
        """返回 (水平漂移, 尺度漂移, 应力)。"""
        windows = self._windows()
        if windows is None:
            return 0.0, 0.0, self._smoothed
        recent, older = windows
        older_rms = float(np.sqrt(np.mean(np.square(older)))) + 1e-9
        recent_rms = float(np.sqrt(np.mean(np.square(recent)))) + 1e-9
        self._level = float(recent.mean() - older.mean())
        self._scale = float(np.log(recent_rms / older_rms))
        level_shift = abs(self._level) / older_rms
        scale_shift = abs(self._scale)
        return level_shift, scale_shift, self._smoothed

    def stress(self) -> float:
        """[0,1] 季节应力；历史不足参考窗时返回已有平滑值（不臆造变化）。"""
        windows = self._windows()
        if windows is None:
            return self._smoothed
        recent, older = windows
        cfg = self.config
        older_rms = float(np.sqrt(np.mean(np.square(older)))) + 1e-9
        recent_rms = float(np.sqrt(np.mean(np.square(recent)))) + 1e-9
        self._level = float(recent.mean() - older.mean())
        self._scale = float(np.log(recent_rms / older_rms))
        level_shift = abs(self._level) / older_rms
        scale_shift = abs(self._scale)
        raw = cfg.level_weight * (1.0 - np.exp(-cfg.level_sensitivity * level_shift)) + cfg.scale_weight * (
            1.0 - np.exp(-cfg.scale_sensitivity * scale_shift)
        )
        raw = float(np.clip(raw, 0.0, 1.0))
        # 单向平滑：突变立即反映，恢复缓慢，避免旧季节长期占据记忆。
        if raw >= self._smoothed:
            self._smoothed = raw
        else:
            self._smoothed = cfg.smoothing * self._smoothed + (1.0 - cfg.smoothing) * raw
        return self._smoothed

    def drift_direction(self) -> float:
        """水平漂移方向：正表示实际正在高于预测（水平上行）。"""
        windows = self._windows()
        if windows is None:
            return 0.0
        recent, older = windows
        scale = float(np.sqrt(np.mean(np.square(older)))) + 1e-9
        return float(np.clip(self._level / scale, -1.0, 1.0))

    def effective_window(self, base_window: int, minimum: int = 5) -> int:
        """把季节应力映射为缩短后的记忆长度：σ 越大，窗口越短。"""
        cfg = self.config
        factor = 1.0 / (1.0 + cfg.shrink_gain * self.stress())
        floor = max(minimum, int(round(base_window * cfg.min_window_ratio)))
        return int(np.clip(round(base_window * factor), floor, base_window))


class SeasonSensor:
    """把漂移检测器与光伏窗口跟踪器合成统一的季节状态输出。"""

    def __init__(self, n_slot: int, drift_config: DriftConfig | None = None) -> None:
        self.detector = SeasonalDriftDetector(drift_config)
        self.pv_window = PVActiveWindowTracker(n_slot)

    def observe(
        self,
        *,
        date: pd.Timestamp,
        load_residual: np.ndarray,
        pv_actual_kw: np.ndarray,
    ) -> None:
        """登记一个已执行日；只应在该日结束后调用。"""
        self.detector.update(load_residual)
        self.pv_window.observe(pv_actual_kw)

    def state(self, date: pd.Timestamp) -> SeasonState:
        """给出某日的季节状态。只使用此前登记的历史。"""
        start, end = self.pv_window.predict_window()
        length_hours, trend_hours = self.pv_window.window_length_series()
        level_shift, scale_shift, _ = self.detector.stats()
        return SeasonState(
            date=pd.Timestamp(date),
            doy=int(pd.Timestamp(date).dayofyear),
            pv_window_start=int(start),
            pv_window_end=int(end),
            pv_window_length=int(end - start + 1),
            daylight_proxy_hours=float(length_hours),
            daylight_proxy_trend_hours=float(trend_hours),
            drift_level=float(level_shift),
            drift_scale=float(scale_shift),
            stress=float(self.detector.stress()),
            drift_direction=float(self.detector.drift_direction()),
        )


# ---------------------------------------------------------------------------
# 2.3  在线校正管线（源码：代码/问题2/deps/seasonal/{online_correction, weekly_effect}.py）
#
#   基础集成只负责水平与日内形状的骨架，其输出 p̂⁽⁰⁾ 依次经过四级校正，每一级
#   只修正上一级残留的、属于自己那一类的误差，因而各通道可正交识别：
#
#     p̂⁽¹⁾ = p̂⁽⁰⁾ · 1{t ∈ W_d}          第 1 级：光伏有效发电窗口（仅光伏）
#     p̂⁽²⁾ = p̂⁽¹⁾ · λ_d                  第 2 级：季节水平因子（记忆窗由 σ_d 决定）
#     p̂⁽³⁾ = p̂⁽²⁾ · φ_{k(d)}             第 3 级：周内水平因子（全周均值为 1）
#     p̂⁽⁴⁾ = [p̂⁽³⁾ + ψ_{g(d),t}·E⁽³⁾]_+  第 4 级：周型形状轮廓
#
#   λ_d 由全部星期几混合估计，φ_k 以"相对全周均值的偏离"形式给出，二者在期望
#   意义上互不重叠，不会争夺同一份误差。调用顺序严格为"先 apply、后 update"。
# ---------------------------------------------------------------------------
@dataclass
class WeeklyEffectConfig:
    """周效应模块的结构超参数（先验值，非由全年回测挑选）。"""

    # 建立 1 月先验所用的日索引上界（不含）：0..30 即 1 月 1 日—1 月 31 日。
    prior_end_day: int = 31
    # 先验起始日索引。必须 ≥7：基础预测器在第 7 天之前只能靠 1/2/3 日滞后，
    # 尚不具备同星期几参照，其周内偏差是**预热期伪影**而非真实周效应。
    prior_start_day: int = 7
    prior_pseudo_count: float = 5.0
    daily_decay: float = 0.985       # 累计量的逐日衰减，有效记忆约 10 周
    factor_shrink_kappa: float = 3.0
    factor_clip: float = 0.03        # 基础预测器已由 7 日滞后捕获大部分周内效应
    energy_strength: float = 1.0
    shape_prior_pseudo_count: float = 5.0
    # 形状轮廓每类有 144 个自由参数，而"工作日/周末"每类每周只新增 5/2 个样本，
    # 参数数与样本量之比极高，必须强收缩才能避免把天气噪声固化成"周效应"。
    shape_shrink_kappa: float = 20.0
    shape_smoothing: float = 0.40
    shape_cap_fraction: float = 0.02    # 单时段形状修正幅度上限（占预测日能量的比例）
    shape_strength: float = 1.0


class WeeklyEffectModel:
    """周内水平因子与周型形状轮廓的因果在线估计器。"""

    N_WEEKDAY = 7
    N_TYPE = 2  # 0 = 工作日（周一至周五），1 = 周末（周六、周日）

    def __init__(self, n_slot: int, config: WeeklyEffectConfig | None = None) -> None:
        self.n_slot = int(n_slot)
        self.config = config or WeeklyEffectConfig()
        self.energy_sum = np.zeros(self.N_WEEKDAY, float)
        self.energy_weight = np.zeros(self.N_WEEKDAY, float)
        self.shape_sum = np.zeros((self.N_TYPE, self.n_slot), float)
        self.shape_weight = np.zeros(self.N_TYPE, float)
        self.n_updates = 0
        self._fitted = False
        # 先验建立前逐日缓存观测；攒满 prior_end_day 天后一次性转为 1 月先验。
        self._prior_buffer: list[tuple[pd.Timestamp, float, np.ndarray]] = []

    @staticmethod
    def weekday_type(date: pd.Timestamp) -> int:
        """周六、周日为 1（周末），其余为 0（工作日）。"""
        return 1 if pd.Timestamp(date).dayofweek >= 5 else 0

    def _build_prior(self, observations: list[tuple[pd.Timestamp, float, np.ndarray]]) -> None:
        """把一批"已全部发生"的历史观测折算为等效样本数为先验强度的初值。"""
        cfg = self.config
        observed = np.zeros(self.N_WEEKDAY, float)
        ratio_sum = np.zeros(self.N_WEEKDAY, float)
        type_count = np.zeros(self.N_TYPE, float)
        shape_total = np.zeros((self.N_TYPE, self.n_slot), float)

        for date, ratio, shape in observations:
            k = int(pd.Timestamp(date).dayofweek)
            ratio_sum[k] += float(ratio)
            observed[k] += 1.0
            g = self.weekday_type(date)
            shape_total[g] += np.asarray(shape, float)
            type_count[g] += 1.0

        with np.errstate(invalid="ignore", divide="ignore"):
            mean_ratio = np.where(observed > 0, ratio_sum / np.maximum(observed, 1e-9), 1.0)
        self.energy_sum = mean_ratio * cfg.prior_pseudo_count
        self.energy_weight = np.where(observed > 0, cfg.prior_pseudo_count, 0.0)

        for g in range(self.N_TYPE):
            if type_count[g] > 0:
                self.shape_sum[g] = (shape_total[g] / type_count[g]) * cfg.shape_prior_pseudo_count
                self.shape_weight[g] = cfg.shape_prior_pseudo_count
        self._fitted = True

    def _weekday_means(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(
                self.energy_weight > 1e-9,
                self.energy_sum / np.maximum(self.energy_weight, 1e-9),
                1.0,
            )
        return means

    def energy_factor(self, date: pd.Timestamp) -> float:
        """第 date 天的周内水平因子（全周均值为 1，因果）。"""
        cfg = self.config
        if not self._fitted:
            return 1.0
        means = self._weekday_means()
        pooled = float(np.mean(means))
        if pooled <= 1e-9:
            return 1.0
        k = int(pd.Timestamp(date).dayofweek)
        deviation = means[k] / pooled
        weight = float(self.energy_weight[k])
        shrink = weight / (weight + cfg.factor_shrink_kappa) if weight > 0 else 0.0
        factor = 1.0 + cfg.energy_strength * shrink * (deviation - 1.0)
        return float(np.clip(factor, 1.0 - cfg.factor_clip, 1.0 + cfg.factor_clip))

    def shape_profile(self, date: pd.Timestamp) -> np.ndarray:
        """第 date 天所属周型的归一化形状轮廓（因果）。"""
        cfg = self.config
        if not self._fitted:
            return np.zeros(self.n_slot, float)
        g = self.weekday_type(date)
        weight = float(self.shape_weight[g])
        if weight <= 1e-9:
            return np.zeros(self.n_slot, float)
        profile = self.shape_sum[g] / weight
        shrink = weight / (weight + cfg.shape_shrink_kappa)
        profile = shrink * profile
        a = cfg.shape_smoothing
        profile = a * np.roll(profile, 1) + (1.0 - 2.0 * a) * profile + a * np.roll(profile, -1)
        return profile

    def shape_correction(self, date: pd.Timestamp, predicted_energy: float) -> np.ndarray:
        """把形状轮廓换算为功率量纲的加性修正（受日能量比例上限约束）。"""
        profile = self.shape_profile(date)
        scale = max(float(predicted_energy), 1e-9)
        cap = self.config.shape_cap_fraction * scale
        return np.clip(self.config.shape_strength * scale * profile, -cap, cap)

    def update(
        self,
        day_index: int,
        date: pd.Timestamp,
        ratio: float,
        shape_residual_normalized: np.ndarray,
    ) -> None:
        """第 day_index 天结束后并入当日观测。只应传入已实现的实际值。

        先验尚未建立时只缓存 [prior_start_day, prior_end_day) 内的观测；到达
        prior_end_day 后一次性转为先验。在此之前 energy_factor 恒为 1、
        shape_profile 恒为 0，即"无周效应"，不会用不足一周的样本臆造结构。
        """
        if not self._fitted:
            cfg = self.config
            if cfg.prior_start_day <= day_index < cfg.prior_end_day:
                self._prior_buffer.append(
                    (pd.Timestamp(date), float(ratio), np.asarray(shape_residual_normalized, float))
                )
            if day_index >= cfg.prior_end_day - 1:
                if self._prior_buffer:
                    self._build_prior(self._prior_buffer)
                else:
                    self._fitted = True
                self._prior_buffer = []
            return
        cfg = self.config
        decay = cfg.daily_decay
        self.energy_sum *= decay
        self.energy_weight *= decay
        self.shape_sum *= decay
        self.shape_weight *= decay

        k = int(pd.Timestamp(date).dayofweek)
        self.energy_sum[k] += float(ratio)
        self.energy_weight[k] += 1.0

        g = self.weekday_type(date)
        self.shape_sum[g] += np.asarray(shape_residual_normalized, float)
        self.shape_weight[g] += 1.0
        self.n_updates += 1


@dataclass
class CorrectionConfig:
    """在线校正管线的超参数（结构性先验，非由全年回测挑选）。"""

    use_level: bool = True
    level_base_window: int = 21
    level_min_window: int = 5
    level_shrink_kappa: float = 3.0
    level_clip: float = 0.15
    level_strength: float = 1.0
    ew_divisor: float = 3.0          # 指数加权的相对时间常数：tau = window / ew_divisor
    use_pv_window: bool = True
    use_weekly_effect: bool = True
    weekly: WeeklyEffectConfig = field(default_factory=WeeklyEffectConfig)


@dataclass(frozen=True)
class CorrectionOutcome:
    """一日校正的可追溯输出。"""

    prediction: np.ndarray
    base_prediction: np.ndarray
    masked_prediction: np.ndarray
    season_state: SeasonState
    level_factor: float
    weekly_factor: float
    level_window_days: int
    window_masked: bool


class SeasonalLevelTracker:
    """日能量比值的因果指数加权估计器，记忆长度由季节应力自适应决定。"""

    def __init__(self, config: CorrectionConfig) -> None:
        self.config = config
        self._ratios: list[float] = []

    def update(self, ratio: float) -> None:
        self._ratios.append(float(ratio))

    def factor(self, window_days: int) -> float:
        """只用最近 window_days 个已实现比值估计水平因子。"""
        cfg = self.config
        if not self._ratios:
            return 1.0
        window = max(1, int(window_days))
        history = np.asarray(self._ratios[-window:], float)
        history = history[np.isfinite(history)]
        if history.size == 0:
            return 1.0
        age = np.arange(history.size - 1, -1, -1, dtype=float)
        tau = max(window / cfg.ew_divisor, 1.0)
        weights = np.exp(-age / tau)
        weights /= weights.sum()
        raw = float(weights @ history)
        # 样本不足时向 1 收缩，避免最初几天用极少样本做大幅修正。
        shrink = history.size / (history.size + cfg.level_shrink_kappa)
        factor = 1.0 + cfg.level_strength * shrink * (raw - 1.0)
        return float(np.clip(factor, 1.0 - cfg.level_clip, 1.0 + cfg.level_clip))


class CausalCorrectionPipeline:
    """单目标（负载或光伏）的因果在线校正管线。"""

    def __init__(
        self,
        *,
        target: str,
        n_slot: int,
        dt_hours: float,
        sensor: SeasonSensor,
        config: CorrectionConfig | None = None,
    ) -> None:
        if target not in ("load", "pv"):
            raise ValueError("target 只能是 'load' 或 'pv'")
        self.target = target
        self.n_slot = int(n_slot)
        self.dt_hours = float(dt_hours)
        self.sensor = sensor
        self.config = config or CorrectionConfig()
        self.level = SeasonalLevelTracker(self.config)
        # 周效应默认只作用于负载：小区周末作息影响用电，光伏由天气主导，无周内结构。
        self.weekly: WeeklyEffectModel | None = (
            WeeklyEffectModel(self.n_slot, self.config.weekly)
            if (self.config.use_weekly_effect and target == "load")
            else None
        )

    def _masked(self, base_prediction: np.ndarray, date: pd.Timestamp) -> np.ndarray:
        if self.target == "pv" and self.config.use_pv_window:
            return self.sensor.pv_window.apply_window(base_prediction)
        return np.asarray(base_prediction, float)

    def _stage_predictions(self, base_prediction: np.ndarray, date: pd.Timestamp):
        """按级联顺序重算各阶段输出，供 apply 与 update 共用，保证口径一致。"""
        state = self.sensor.state(date)
        masked = self._masked(base_prediction, date)
        window_days = self.sensor.detector.effective_window(
            self.config.level_base_window, self.config.level_min_window
        )
        level_factor = self.level.factor(window_days) if self.config.use_level else 1.0
        after_level = masked * level_factor
        weekly_factor = 1.0
        if self.weekly is not None:
            weekly_factor = self.weekly.energy_factor(date)
        after_weekly = after_level * weekly_factor
        return masked, after_level, after_weekly, level_factor, weekly_factor, window_days

    def apply(self, base_prediction: np.ndarray, date: pd.Timestamp) -> CorrectionOutcome:
        """第 date 天 0:00 调用：只用历史状态生成本日校正预测。"""
        base = np.asarray(base_prediction, float)
        masked, after_level, after_weekly, level_factor, weekly_factor, window_days = (
            self._stage_predictions(base, date)
        )
        prediction = after_weekly
        if self.weekly is not None:
            energy_after_weekly = float(after_weekly.sum() * self.dt_hours)
            prediction = after_weekly + self.weekly.shape_correction(date, energy_after_weekly)
        prediction = np.maximum(prediction, 0.0)
        return CorrectionOutcome(
            prediction=prediction,
            base_prediction=base,
            masked_prediction=masked,
            season_state=self.sensor.state(date),
            level_factor=float(level_factor),
            weekly_factor=float(weekly_factor),
            level_window_days=int(window_days),
            window_masked=bool(self.target == "pv" and self.config.use_pv_window),
        )

    def update(
        self,
        base_prediction: np.ndarray,
        actual: np.ndarray,
        date: pd.Timestamp,
        day_index: int,
    ) -> None:
        """第 day_index 天执行完毕后调用：并入当日已实现观测。

        必须严格在 apply 之后调用。第 1→2 级由季节水平通道吸收"整日水平"偏差，
        第 2→3 级由周内水平因子吸收"该星期几"的残余水平偏差。
        """
        base = np.asarray(base_prediction, float)
        actual = np.asarray(actual, float)
        masked, after_level, after_weekly, _, _, _ = self._stage_predictions(base, date)

        energy_actual = float(actual.sum() * self.dt_hours)
        energy_masked = float(masked.sum() * self.dt_hours)
        energy_after_level = float(after_level.sum() * self.dt_hours)

        if self.config.use_level and energy_masked > 1e-9:
            self.level.update(energy_actual / energy_masked)

        if self.weekly is not None:
            ratio_weekly = energy_actual / max(energy_after_level, 1e-9)
            energy_after_weekly = float(after_weekly.sum() * self.dt_hours)
            shape_residual = (actual - after_weekly) / max(energy_after_weekly, 1e-9)
            self.weekly.update(day_index, date, ratio_weekly, shape_residual)


def build_pipelines(
    *,
    n_slot: int,
    dt_hours: float,
    sensor: SeasonSensor,
    config: CorrectionConfig | None = None,
) -> dict[str, CausalCorrectionPipeline]:
    """构造负载与光伏两条校正管线，共享同一个季节感知器。"""
    cfg = config or CorrectionConfig()
    return {
        target: CausalCorrectionPipeline(
            target=target, n_slot=n_slot, dt_hours=dt_hours, sensor=sensor, config=cfg
        )
        for target in ("load", "pv")
    }


# ---------------------------------------------------------------------------
# 2.4  因果自适应预测器（源码：代码/问题2/deps/seasonal/adaptive_forecast.py）
#
#   逐日推进的因果闭环：
#     第 d 天 0:00  季节感知器给出 σ_d 与自适应记忆长度 W_d^mem
#                   基础集成按 W_d^train = f(σ_d) 取训练窗，输出 p̂⁽⁰⁾
#                   级联校正 → p̂⁽⁴⁾ → 进入当日计划
#     第 d 天 24:00 用当日实际值更新集成误差权重、校正管线、季节感知器
#
#   训练窗随 σ_d 收缩的理由：若季节已切换而训练窗仍是固定的 180 天，模型会被
#   上一个季节的样本主导，表现为"慢半拍"。σ_d 上升时把训练窗与水平记忆窗一并
#   缩短，使模型在数日内完成向新季节的迁移；σ_d 回落时窗口自动恢复，避免长期
#   使用过短窗口而放大噪声。
# ---------------------------------------------------------------------------
@dataclass
class ForecastConfig:
    """因果自适应预测器的结构与超参数（先验值，非由全年回测挑选）。"""

    ridge_min_day: int = 28
    lgb_min_day: int = 60
    ridge_refit_days: int = 7
    lgb_refit_days: int = 14
    base_window_days: int = 180
    window_min_days: int = 45
    use_adaptive_window: bool = True
    # σ 上升超过该幅度时立即重拟合，使季节切换不被固定重拟合周期拖延。
    stress_refit_jump: float = 0.30
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)


class CausalEnsemble:
    """baseline / ridge / lightgbm 的因果加权集成，训练窗由季节应力决定。"""

    def __init__(
        self,
        *,
        values: np.ndarray,
        dates: pd.DatetimeIndex,
        initial: np.ndarray,
        baseline_fn,
        config: ForecastConfig,
    ) -> None:
        self.values = values
        self.dates = dates
        self.initial = initial
        self.baseline_fn = baseline_fn
        self.config = config
        self.weights_by_day: list[dict[str, float]] = []
        self._error_history: dict[str, list[float]] = {"baseline": [], "ridge": [], "lightgbm": []}
        self._ridge = None
        self._lgb = None
        self._last_fit_day: dict[str, int] = {"ridge": -10**9, "lightgbm": -10**9}
        self._stress_at_fit: dict[str, float] = {"ridge": 0.0, "lightgbm": 0.0}
        self.refit_events: list[dict] = []

    def _compute_window(self, day: int, current_stress: float) -> int:
        """自适应训练窗：应力越大窗越短；应力首次明显上升时立即重拟合。"""
        cfg = self.config
        if day < cfg.ridge_min_day or not cfg.use_adaptive_window:
            return cfg.base_window_days
        window = int(
            np.clip(
                round(cfg.base_window_days / (1.0 + 2.5 * current_stress)),
                cfg.window_min_days,
                cfg.base_window_days,
            )
        )
        for label, period in (("ridge", cfg.ridge_refit_days), ("lightgbm", cfg.lgb_refit_days)):
            jump = current_stress - self._stress_at_fit[label]
            if jump >= cfg.stress_refit_jump:
                self._last_fit_day[label] = -10**9  # 强制重拟合
                self.refit_events.append(
                    {"day": int(day), "model": label, "reason": "season_stress_jump", "jump": float(jump)}
                )
        return window

    def predict(self, day: int, current_stress: float) -> np.ndarray:
        """只用 day 以前的数据生成第 day 天的集成预测。"""
        cfg = self.config
        window = self._compute_window(day, current_stress)
        candidates = {"baseline": self.baseline_fn(self.values, day, self.initial)}

        if day >= cfg.ridge_min_day:
            if self._ridge is None or day - self._last_fit_day["ridge"] >= cfg.ridge_refit_days:
                x_train, y_train = training_matrix(self.values, self.dates, day, window)
                self._ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                self._ridge.fit(x_train, y_train)
                self._last_fit_day["ridge"] = day
                self._stress_at_fit["ridge"] = current_stress
            candidates["ridge"] = self._ridge.predict(day_features(self.values, self.dates, day))

        if day >= cfg.lgb_min_day:
            if self._lgb is None or day - self._last_fit_day["lightgbm"] >= cfg.lgb_refit_days:
                self._lgb, _ = fit_lightgbm_causal(
                    self.values, self.dates, day, window_days=window
                )
                self._last_fit_day["lightgbm"] = day
                self._stress_at_fit["lightgbm"] = current_stress
            candidates["lightgbm"] = self._lgb.predict(day_features(self.values, self.dates, day))

        for name in candidates:
            candidates[name] = np.maximum(np.asarray(candidates[name], float), 0.0)

        weights = adaptive_error_weights(candidates.keys(), self._error_history)
        prediction = np.zeros(N_SLOT, float)
        for name, candidate in candidates.items():
            prediction = prediction + weights[name] * candidate
        self.weights_by_day.append(dict(weights))
        self._pending = candidates
        return np.maximum(prediction, 0.0)

    def observe(self, day: int) -> None:
        """第 day 天结束后登记各候选的当日误差，供次日加权。"""
        for name, candidate in self._pending.items():
            self._error_history[name].append(float(np.mean(np.abs(candidate - self.values[day]))))
        self._pending = {}


def generate_causal_forecasts(
    data: DataBundle,
    config: ForecastConfig | None = None,
    *,
    verbose: bool = False,
) -> dict:
    """逐日推进生成全年的因果负载/光伏预测。

    返回 load、pv 两条预测序列与被校正前的 load_base、pv_base，以及逐日诊断表。
    """
    cfg = config or ForecastConfig()
    # 防御性拷贝：绝不就地改写调用方传入的实际值数组，否则一次运行的中间写入会
    # 改变下一次运行的输入，使因果性检验失去意义。
    data = replace(
        data,
        load_kw=np.array(data.load_kw, dtype=float, copy=True),
        pv_kw=np.array(data.pv_kw, dtype=float, copy=True),
        price=np.array(data.price, dtype=float, copy=True),
    )
    n_days = len(data.dates)
    sensor = SeasonSensor(N_SLOT, cfg.drift)
    pipelines = build_pipelines(
        n_slot=N_SLOT, dt_hours=DT, sensor=sensor, config=cfg.correction
    )

    load_ensemble = CausalEnsemble(
        values=data.load_kw,
        dates=data.dates,
        initial=data.initial_load,
        baseline_fn=weekly_load_baseline,
        config=cfg,
    )
    pv_ensemble = CausalEnsemble(
        values=data.pv_kw,
        dates=data.dates,
        initial=data.initial_pv,
        baseline_fn=recent_pv_baseline,
        config=cfg,
    )

    forecasts = {
        "load": np.zeros_like(data.load_kw),
        "pv": np.zeros_like(data.pv_kw),
        "load_base": np.zeros_like(data.load_kw),
        "pv_base": np.zeros_like(data.pv_kw),
    }
    diagnostic_rows: list[dict] = []

    for day in range(n_days):
        date = data.dates[day]
        # 季节应力只用 j<day 的已执行残差，因此可在当日决策前读出。
        stress = sensor.detector.stress()

        base_load = load_ensemble.predict(day, stress)
        base_pv = pv_ensemble.predict(day, stress)

        outcome_load = pipelines["load"].apply(base_load, date)
        outcome_pv = pipelines["pv"].apply(base_pv, date)

        forecasts["load_base"][day] = base_load
        forecasts["pv_base"][day] = base_pv
        forecasts["load"][day] = outcome_load.prediction
        forecasts["pv"][day] = outcome_pv.prediction

        diagnostic_rows.append(
            {
                "date": date,
                "day_index": day,
                "season_stress": float(stress),
                "level_window_days": int(outcome_load.level_window_days),
                "load_level_factor": float(outcome_load.level_factor),
                "pv_level_factor": float(outcome_pv.level_factor),
                "load_weekly_factor": float(outcome_load.weekly_factor),
            }
        )

        # ---- 决策日结束：只用当日实际值更新所有在线状态 ----
        load_ensemble.observe(day)
        pv_ensemble.observe(day)
        pipelines["load"].update(base_load, data.load_kw[day], date, day)
        pipelines["pv"].update(base_pv, data.pv_kw[day], date, day)
        sensor.observe(
            date=date,
            load_residual=data.load_kw[day] - outcome_load.prediction,
            pv_actual_kw=data.pv_kw[day],
        )

        if verbose and (day + 1) % 60 == 0:
            print(f"  因果预测进度：{day + 1}/{n_days} 天", flush=True)

    forecasts["weights"] = {
        "load": load_ensemble.weights_by_day,
        "pv": pv_ensemble.weights_by_day,
    }
    forecasts["net"] = forecasts["load"] - forecasts["pv"]
    forecasts["net_error"] = (data.load_kw - data.pv_kw) - forecasts["net"]
    forecasts["diagnostics"] = pd.DataFrame(diagnostic_rows)
    forecasts["sensor"] = sensor
    return forecasts


# ---------------------------------------------------------------------------
# 2.5  因果分位数安全余量
#
#   问题2 只发布一次计划，实际负载/光伏与预测的偏差只能靠储能与紧急购电吸收。
#   紧急购电价为 5c，计划购电价为 c：多买 1 kWh 的期望代价是 c·P(过剩)，
#   少买 1 kWh 的期望代价是 4c·P(缺口)，报童临界比 4c/(4c+c) = 0.80，
#   故风险分位数 Q = 0.80 是**事前给定的理论值**，不由评价期回测挑选。
#
#   余量不是逐时段留保险，而是问"这一整段窗口总共会偏多少"：
#       R_d = Q_q( { Σ_t ξ_{j,t}·Δt : j < d } )
#   第 d 行只用 j<d 的窗口累计残差，是因果量；历史不足 20 天时取 0。
#   （源码：代码/共享/run_year.py simulate() 内的残差更新，与
#     代码/问题2/deps/deterministic_baseline.py 的 causal_residual_quantiles 同口径。）
# ---------------------------------------------------------------------------
RESERVE_QUANTILE = 0.80
RESERVE_MIN_HISTORY = 20


def causal_cumulative_reserve(
    errors_kw: np.ndarray,
    quantile: float = RESERVE_QUANTILE,
    min_history: int = RESERVE_MIN_HISTORY,
) -> np.ndarray:
    """因果的窗口累计净负荷残差分位数（kWh），逐日返回。"""
    n_days = errors_kw.shape[0]
    out = np.zeros(n_days)
    for d in range(n_days):
        hist = errors_kw[:d]
        if len(hist) < min_history:
            continue
        out[d] = float(np.quantile(hist.sum(axis=1) * DT, quantile))
    return out


# ---------------------------------------------------------------------------
# 2.6  单阶段计划 LP（源码：代码/共享/opt_core.py 的 stage_lp()
#
#   问题2/4-2 在每日 0:00 调用一次（settle="none"，无上一版承诺）；
#   问题3/4-3 在 06/12/18 各调用一次做增量修订（settle="correct"，带上一版
#   承诺 ref 与历史最小承诺 hist_min，见 §3.2）。
#
#   决策变量：x 计划购电、u 充电、v 放电、s 储电量、e_b/e_o 承诺上下调量、
#   w 弃光量；历史最小承诺项 −c·(m−x)^+ 是凹项，需引入 0-1 变量由 MILP 精确求解。
# ---------------------------------------------------------------------------
def stage_lp(
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    price: np.ndarray,
    start_energy: float,
    *,
    ref: np.ndarray | None = None,
    settle: str = "correct",
    reserve: float = 0.0,
    hist_min: np.ndarray | None = None,
) -> np.ndarray:
    """单阶段购电计划，返回承诺购电量向量 x（kWh/时段）。

    settle 口径：
      · "none"    —— 无上一版承诺（当日首次计划），目标就是 c·x；
      · "correct" —— 题面口径 φ = 0.5c·x + 1.0c·(x−r)^+（去掉常数项 0.5c·r），
                     即 1.0c 的附加费记在**上调量** e_o = (x−r)^+ 上。

    reserve 是储能储备 R（kWh），施加为 E_t ≥ E_min + R，承担预测误差风险。
    hist_min 给出链式结算中"历史最小承诺" m_t，非空时改用 MILP 精确求解。
    """
    n = len(price)
    load = np.maximum(load_kw, 0.0) * DT
    pv = np.maximum(pv_kw, 0.0) * DT

    i_x, i_u, i_v, i_s, i_eb, i_eo = 0, n, 2 * n, 3 * n, 4 * n, 5 * n
    i_w = 6 * n
    nvar = 7 * n

    obj = np.zeros(nvar)
    if settle == "none":
        obj[i_x : i_x + n] = price
    elif settle == "correct":
        obj[i_x : i_x + n] = BREACH_RATE * price
        obj[i_eo : i_eo + n] = (OVERBUY_RATE - BREACH_RATE) * price   # 1.0c 记在上调量
    else:
        raise ValueError(f"settle 只能是 none/correct，收到 {settle!r}")

    obj[i_u : i_u + n] = 1e-7          # 仅为消除充放电量不唯一的数值项
    obj[i_v : i_v + n] = 1e-7
    obj[i_w : i_w + n] = 1e-8
    obj[i_s + n - 1] = -end_water_value(price)   # 日末储电按水值 λ 计入终值

    a_eq, b_eq = [], []
    for t in range(n):                 # 电量平衡 x_t + P_t + v_t = L_t + u_t + w_t
        row = np.zeros(nvar)
        row[i_x + t], row[i_u + t], row[i_v + t] = 1.0, -1.0, 1.0
        row[i_w + t] = -1.0
        a_eq.append(row)
        b_eq.append(load[t] - pv[t])
    for t in range(n):                 # 储能动态 E_t = E_{t−1} + η_c u_t − v_t/η_d
        row = np.zeros(nvar)
        row[i_u + t], row[i_v + t], row[i_s + t] = -ETA_C, 1.0 / ETA_D, 1.0
        if t:
            row[i_s + t - 1] = -1.0
            b_eq.append(0.0)
        else:
            b_eq.append(start_energy)
        a_eq.append(row)
    if settle != "none":               # 上下调量定义：e_b − e_o = ref − x
        if ref is None:
            raise ValueError("settle 非 none 时必须给出上一版承诺 ref")
        for t in range(n):
            row = np.zeros(nvar)
            row[i_eb + t], row[i_eo + t], row[i_x + t] = 1.0, -1.0, 1.0
            a_eq.append(row)
            b_eq.append(float(ref[t]))

    # 储备边界必须"爬坡可达"，否则直接不可行：下界从起点按最大充电功率抬升。
    band = (E_MAX - E_MIN) / 2.0
    r_lo = min(max(reserve, 0.0), band)
    t_lo = (0 if start_energy >= E_MIN + r_lo
            else int(np.ceil((E_MIN + r_lo - start_energy) / (ETA_C * STEP_MAX))))
    soc_bounds = [(E_MIN + r_lo if t >= t_lo else E_MIN, E_MAX) for t in range(n)]

    bounds = ([(0.0, None)] * n + [(0.0, STEP_MAX)] * n + [(0.0, STEP_MAX)] * n
              + soc_bounds + [(0.0, None)] * n + [(0.0, None)] * n
              + [(0.0, None)] * n)

    # --- 精确链式结算：补上历史最小承诺项 −c·(m−x)^+（凹项，需 0-1 变量）---
    if hist_min is not None:
        if settle == "none":
            raise ValueError("hist_min 只在有上一版承诺时才有意义")
        # 真实阶段增量（去掉与 x 无关的常数 −c·m）：
        #   ΔF = c·min(m,x) + 0.5c·(m−x)^+ + 1.5c·(x−m)^+
        obj[i_x : i_x + n] = 0.0
        obj[i_eb : i_eb + n] = BREACH_RATE * price      # 0.5c
        obj[i_eo : i_eo + n] = OVERBUY_RATE * price     # 1.5c
        m_hist = np.asarray(hist_min, dtype=float)
        i_d, i_z = nvar, nvar + n          # d_t = min(m_t, x_t) 的线性化，z_t ∈ {0,1}
        nvar += 2 * n
        obj = np.concatenate([obj, np.zeros(2 * n)])
        obj[i_d : i_d + n] = -price
        bounds = tuple(bounds) + ((0.0, None),) * n + ((0.0, 1.0),) * n
        a_ub, b_ub = [], []
        for t in range(n):
            mt = max(float(m_hist[t]), 1e-9)   # d_t ≤ m_t，故可用 m_t 作紧的 big-M
            r1 = np.zeros(nvar); r1[i_d + t], r1[i_z + t] = 1.0, -mt
            a_ub.append(r1); b_ub.append(0.0)
            r2 = np.zeros(nvar); r2[i_d + t], r2[i_x + t], r2[i_z + t] = 1.0, 1.0, BIG_M
            a_ub.append(r2); b_ub.append(float(m_hist[t]) + BIG_M)

        a_eq_np = np.zeros((len(a_eq), nvar))
        for i, r in enumerate(a_eq):
            a_eq_np[i, : len(r)] = r
        a_ub_np = np.zeros((len(a_ub), nvar))
        for i, r in enumerate(a_ub):
            a_ub_np[i, : len(r)] = r
        lo = np.concatenate([np.asarray(b_eq), np.full(len(b_ub), -np.inf)])
        hi = np.concatenate([np.asarray(b_eq), np.asarray(b_ub)])
        integ = np.zeros(nvar); integ[i_z : i_z + n] = 1
        lo_b = np.array([(-np.inf if b[0] is None else b[0]) for b in bounds])
        hi_b = np.array([(np.inf if b[1] is None else b[1]) for b in bounds])
        result = milp(obj, constraints=LinearConstraint(np.vstack([a_eq_np, a_ub_np]), lo, hi),
                      integrality=integ, bounds=Bounds(lo_b, hi_b))
        if not result.success:
            raise RuntimeError(f"单阶段 MILP 失败（n={n}）：{result.message}")
        return clip_purchase(result.x[i_x : i_x + n])

    result = linprog(obj, A_eq=np.asarray(a_eq), b_eq=np.asarray(b_eq),
                     bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"单阶段 LP 失败（n={n}）：{result.message}")
    return clip_purchase(result.x[i_x : i_x + n])


# ---------------------------------------------------------------------------
# 2.7  短时域滚动执行器 MPC（源码：代码/共享/opt_core.py 的
#      execute_day_mpc() 与 _deficit_choice()）
#
#   承诺 x 已下达，执行时仍需决定：出现缺口是用储能放电，还是按 5c 紧急购电？
#   放电省下 5c 的紧急购电，但消耗储电、抬高后续缺口。滚动 MPC 在**每个出现
#   缺口的时段**重解一次 H 步前瞻 LP，只用**预报**看未来、用**实际**执行当前，
#   因此不含完美信息。充电侧仍为贪心——余量不充就只能弃掉，充电恒为弱占优。
# ---------------------------------------------------------------------------
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
) -> dict:
    """给定承诺 x，对当日实际负载/光伏执行储能充放电与紧急购电。

    前瞻目标（x 已固定，c·x 为常数，从目标中略去）：

        min  Σ_{τ=t}^{t+H} 5c_τ·e_τ − λ·E_{t+H},   λ = η_d·当日平均电价
    """
    n = len(commitment)
    load_act, pv_act = load_kw * DT, pv_kw * DT
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
        if vmax <= 1e-12:      # 电池已到下限，无需解 LP
            out["discharge"][t] = vmax
            out["emergency"][t] = max(0.0, deficit - vmax)
            out["pv_used"][t] = pv_act[t]
            out["plan_used"][t] = commitment[t]
            out["soc"][t] = energy
            t += 1
            continue
        discharge, emergency = _deficit_choice(
            t, commitment, load_act, pv_act, load_fc, pv_fc, price,
            energy, deficit, vmax, lam, horizon,
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
) -> tuple[float, float]:
    """缺口时段的 H 步前瞻 LP：返回（放电量, 紧急购电量）。"""
    n = len(commitment)
    end = min(t + horizon, n)
    m = end - t
    i_u, i_v, i_e, i_s, i_w = 0, m, 2 * m, 3 * m, 4 * m
    nvar = 5 * m

    obj = np.zeros(nvar)
    obj[i_u : i_u + m] = 1e-7
    obj[i_v : i_v + m] = 1e-7
    obj[i_e : i_e + m] = EMERGENCY_MULTIPLIER * price[t:end]
    obj[i_s + m - 1] = -lam                # 窗口末端储电按水值计入

    # 充电与弃光的上限都只能是"余量"：既不能凭空造电，也不能用弃光冒充放电
    # 把缺口糊过去（否则 LP 会选 w 而不选 v，白拿 1e-7 的便宜）。
    u_max = np.zeros(m)
    w_max = np.zeros(m)
    for j in range(m):
        tau = t + j
        if j == 0:
            continue                       # 当前时段是缺口，充电弃光均为 0
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

    a_ub, b_ub = [], []
    for j in range(m):
        tau = t + j
        if j == 0:
            supply = pv_act[tau] + commitment[tau] - load_act[tau]
        else:
            supply = pv_fc[tau] + commitment[tau] - load_fc[tau]
        # 供 ≥ 需：−(v+e) + u + w ≤ pv + x − load
        row = np.zeros(nvar)
        row[i_u + j], row[i_v + j], row[i_e + j], row[i_w + j] = 1.0, -1.0, -1.0, 1.0
        a_ub.append(row)
        b_ub.append(float(supply))

    upper_v = min(STEP_MAX, vmax) if m else 0.0
    bounds = ([(0.0, float(u_max[j])) for j in range(m)]
              + [(0.0, upper_v)] + [(0.0, STEP_MAX)] * (m - 1)
              + [(0.0, None)] * m
              + [(E_MIN, E_MAX)] * (m - 1) + [(E_MIN, E_MAX)]
              + [(0.0, float(w_max[j])) for j in range(m)])

    res = linprog(obj, A_eq=np.asarray(a_eq), b_eq=np.asarray(b_eq),
                  A_ub=np.asarray(a_ub), b_ub=np.asarray(b_ub),
                  bounds=bounds, method="highs")
    if not res.success:
        return vmax, max(0.0, deficit - vmax)          # 兜底回退到贪心
    v = min(max(float(res.x[i_v]), 0.0), vmax)
    return v, max(0.0, deficit - v)


# ===========================================================================
# §3  问题3：官方预报融合 + 链式承诺增量 MILP + 贪心执行 + 链式结算
#
#   问题3 与问题2 的差别有两点：
#     ① 决策发布四次（00/06/12/18），日内可修订；
#     ② 结算按题面第 15 条链式进行 —— 基准量取该时段整条承诺路径的最小值，
#        下调量按 0.5c 违约、上调量按 1.5c 超购。
#   因此修订是**有代价**的：宁可少调，也不要在价格低时调低、价格高时调高。
#   此外，问题3 可用附件 3 的四次官方光伏预报，与自有预测做逐小时线性融合。
# ===========================================================================

# ---------------------------------------------------------------------------
# 3.1  附件 3 官方预报：在物理区间的**右端点**取值并线性插值
#      （源码：代码/共享/run_year.py 的 official_points()）
# ---------------------------------------------------------------------------
def official_points(a: dict, raw: pd.DataFrame | None = None) -> np.ndarray:
    """把附件 3 的逐小时预报插值到 144 个十分钟时段，形状 (365, 4, 144)。

    第 k 次发布（k = 0/6/12/18 时）只覆盖 k 时之后的时段；锚点取发布时刻的
    实际光伏功率。夜间时段只能由**此前观测**或题目给定的初始光伏判定，不得
    使用当日尚未发生的实际值。raw 省略时自动读入附件 3。
    """
    raw = load_official_pv_table() if raw is None else raw.copy()
    raw.columns = ["date", "issue"] + list(range(1, 25))
    raw["date"] = pd.to_datetime(raw["date"].ffill())
    out = np.full((len(a["load"]), 4, N_SLOT), np.nan)
    for d, date in enumerate(pd.DatetimeIndex(a["dates"])):
        for kpos, k in enumerate(ISSUE_HOURS):
            rows = raw[(raw["date"] == date) & (raw["issue"].astype(str).str.strip() == f"{k}:00")]
            if len(rows) != 1:
                raise ValueError(f"forecast row missing {date} {k}")
            hourly = rows.iloc[0, 2:].to_numpy(float)
            anchor = a["pv"][d, k * 6 - 1] if k else (a["pv"][d - 1, -1] if d else 0.)
            points = np.arange(N_SLOT) / 6 + 1 / 6
            sl = slice(k * 6, N_SLOT)
            out[d, kpos, sl] = np.interp(points[sl], k + np.arange(25), np.r_[anchor, hourly])
            # 物理夜间掩码只能来自此前观测或初始给定光伏。
            prior = a["pv"][max(0, d - 30) : d]
            daylight = prior.max(axis=0) > 1e-8 if d else a["initial_pv"] > 1e-8
            out[d, kpos, (np.arange(N_SLOT) >= k * 6) & ~daylight] = 0.
    return out


# ---------------------------------------------------------------------------
# 3.2  官方预报与自有预报的因果融合
#      （源码：代码/问题3/deps/forecasts.py 的 fit_hourly_stack / _pooled_coef /
#        build_pv_fusion）
#
#   两次预报都有信息：官方预报给出未来的辐照趋势，自有预报给出本地历史水平。
#   对每个（发布时刻 k，绝对小时 h）拟合 actual ~ a·official + b·own + c 并回代。
#   第 d 天的权重只用 j<d 的样本，故是因果的；样本不足时依次退化为合并小时
#   回归、等权平均。
# ---------------------------------------------------------------------------
EVEN_BLEND = np.array([0.5, 0.5, 0.0])


def fit_hourly_stack(
    official: np.ndarray,
    own: np.ndarray,
    actual: np.ndarray,
    *,
    issue_hour: int,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, str]]:
    """对给定的发布时刻，逐小时拟合融合权重并回代。

    参数均为该发布时刻**覆盖范围内**的 (365, n_covered) 数组。
    """
    n_days, n_slot = official.shape
    fused = np.full((n_days, n_slot), np.nan)
    coef_by_hour: dict[int, np.ndarray] = {}
    source_by_hour: dict[int, str] = {}

    hours = np.arange(n_slot) // SLOTS_PER_HOUR + issue_hour   # 该列对应的绝对小时
    for h in np.unique(hours):
        cols = np.where(hours == h)[0]
        coef_by_hour[h] = EVEN_BLEND
        source_by_hour[h] = "等权(冷启动)"

        for d in range(n_days):
            enough = d * cols.size >= MIN_STACK_SAMPLES
            if enough:
                x_off = official[:d][:, cols].ravel()
                x_own = own[:d][:, cols].ravel()
                y = actual[:d][:, cols].ravel()
                design = np.stack([x_off, x_own, np.ones(x_off.size)], axis=1)
                coef, *_ = np.linalg.lstsq(design, y, rcond=None)
                if np.isfinite(coef).all():
                    coef_by_hour[h] = coef
                    source_by_hour[h] = "逐小时回归"
                else:
                    coef_by_hour[h] = _pooled_coef(official, own, actual, d)
                    source_by_hour[h] = "合并小时回归"
            else:
                coef_by_hour[h] = _pooled_coef(official, own, actual, d)
                source_by_hour[h] = "合并小时回归"

            c = coef_by_hour[h]
            fused[d, cols] = c[0] * official[d, cols] + c[1] * own[d, cols] + c[2]

    return np.clip(fused, 0.0, None), coef_by_hour, source_by_hour


def _pooled_coef(
    official: np.ndarray,
    own: np.ndarray,
    actual: np.ndarray,
    n_days: int,
) -> np.ndarray:
    """样本不足时退化为"该发布时刻全部白天时段"的合并回归。"""
    if n_days < 5:
        return EVEN_BLEND
    # 白天列只能由当前日之前的实测值识别，不能查看 n_days 之后的数据。
    day_cols = np.where(np.mean(actual[:n_days], axis=0) > NIGHT_KW)[0]
    if day_cols.size * n_days < MIN_STACK_SAMPLES:
        return EVEN_BLEND
    sub = np.ix_(np.arange(n_days), day_cols)
    x_off = official[sub].ravel()
    x_own = own[sub].ravel()
    y = actual[sub].ravel()
    design = np.stack([x_off, x_own, np.ones(x_off.size)], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef if np.isfinite(coef).all() else EVEN_BLEND


def build_pv_fusion(
    official_pv: np.ndarray,
    own_pv: np.ndarray,
    actual_pv: np.ndarray,
) -> np.ndarray:
    """对四个发布时刻分别构造融合光伏预报，返回 pv_fused，形状 (4, 365, 144)。

    未覆盖时段为 NaN。
    """
    n_days = official_pv.shape[0]
    pv_fused = np.full((len(ISSUE_HOURS), n_days, N_SLOT), np.nan)

    for k_pos, k in enumerate(ISSUE_HOURS):
        start = k * SLOTS_PER_HOUR           # 覆盖的起始列（0-based）
        fused, _, _ = fit_hourly_stack(
            official_pv[:, k_pos, start:], own_pv[:, start:], actual_pv[:, start:],
            issue_hour=k,
        )
        pv_fused[k_pos, :, start:] = fused
    return pv_fused


def forecast_at(a: dict, d: int, k: int, staged: bool) -> tuple[np.ndarray, np.ndarray]:
    """第 d 天第 k 个发布时刻可得的 (load, pv) 预报。

    分阶段问题用融合后的光伏预报与自有负载预报；已完成时段用实际负载校准
    当日剩余时段的水平（限幅 ±15%），夜间光伏强制置零。
    """
    load = a["load_stage_fc" if staged else "load_fc"][d].copy()
    pv = a["pv_fused"][ISSUE_HOURS.index(k), d].copy() if staged else a["pv_fc"][d].copy()
    pv = np.where(np.isfinite(pv), pv, a["pv_stage_fc" if staged else "pv_fc"][d])
    done = k * 6
    if done:
        ratio = a["load"][d, :done].sum() / max(load[:done].sum(), 1e-9)
        load[done:] *= np.clip(1 + .5 * (ratio - 1), .85, 1.15)
    prior = a["pv"][max(0, d - 30) : d]
    daylight = prior.max(axis=0) > 1e-8 if d else a["initial_pv"] > 1e-8
    pv[~daylight] = 0.
    return load, np.maximum(pv, 0)


# ---------------------------------------------------------------------------
# 3.3  贪心执行器（源码：代码/共享/opt_core.py 的 _greedy_execute()）
#
#   问题3 的执行不带前瞻：承诺已按四个时刻定好，执行时只在缺口处立即放电，
#   电池不足再紧急购电。前瞻能力体现在 06/12/18 的 MILP 修订里，不在执行侧。
# ---------------------------------------------------------------------------
def greedy_execute(
    commitment: np.ndarray,
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    start_energy: float,
) -> dict:
    """逐时段执行：有剩余立即充电、有缺口立即放电、电池不足再紧急购电。"""
    load = load_kw * DT
    pv = pv_kw * DT
    energy = float(np.clip(start_energy, E_MIN, E_MAX))
    n = len(commitment)
    out = {k: np.zeros(n) for k in
           ("charge", "discharge", "emergency", "unused_plan",
            "curtailment", "pv_used", "plan_used", "soc")}
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


# ---------------------------------------------------------------------------
# 3.4  链式承诺结算（源码：代码/共享/opt_core.py 的 settle_period()）
#
#   题面第 15 条：基准量取该时段**整条承诺路径的最小值**，其后每一次变动
#   分别计价——下调量按 0.5c 违约、上调量按 1.5c 超购。
# ---------------------------------------------------------------------------
def settle_period(commitment_path: np.ndarray, price: np.ndarray) -> dict:
    """链式结算。commitment_path 形状 (n_stage, n_slot)。"""
    base = breach = overbuy = down_tot = up_tot = 0.0
    for t in range(commitment_path.shape[1]):
        seq = [float(commitment_path[j, t]) for j in range(commitment_path.shape[0])]
        c = float(price[t])
        base += c * min(seq)                     # 基准量：路径最小值
        down = up = 0.0
        prev = seq[0]
        for cur in seq[1:]:
            if cur < prev:
                down += prev - cur
            else:
                up += cur - prev
            prev = cur
        breach += BREACH_RATE * c * down         # 下调 0.5c
        overbuy += OVERBUY_RATE * c * up         # 上调 1.5c
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


# ===========================================================================
# §4  问题4：波动电价下的重算
#
#   问题4 把附件 1 的分时电价换成附件 4 的波动电价，并规定电价本身也不可预知：
#   第 d 天 0:00 制定计划时，当天 144 个时段的电价尚未发生。因此问题4-2 与
#   问题4-3 分别复用问题2 与问题3 的模型骨架，只把
#       · 用于**决策**的电价   c_t  →  因果预测电价 ĉ_t
#       · 用于**结算**的电价   c_t  →  当日实际电价 c_t
#   两条通道分开。储能物理参数、约束、目标函数与结算规则完全不变。
# ===========================================================================

# ---------------------------------------------------------------------------
# 4.1  波动电价的因果预测（源码：代码/问题4/问题4-2/deps/price_forecast.py）
#
#   与负载/光伏同结构的三模型动态加权集成：近期加权基线 + 岭回归 + 浅层
#   LightGBM，按近期指数加权误差组合。季节应力直接取负载/光伏预测已算出的
#   逐日取值（只由 j<day 的已实现残差构成，故决策前可知）。
#   与负载/光伏的唯一差别：电价不接季节校正管线（光伏有效窗口与日电量水平
#   因子是光伏/负载专用的物理量），仅使用基础集成。
# ---------------------------------------------------------------------------
def generate_causal_price_forecast(
    price: np.ndarray,
    initial_price: np.ndarray,
    dates: pd.DatetimeIndex,
    stress_by_day: np.ndarray,
    config: ForecastConfig | None = None,
    *,
    verbose: bool = False,
) -> dict:
    """逐日推进生成全年因果电价预测。"""
    cfg = config or ForecastConfig()
    n_days = len(dates)
    if len(stress_by_day) != n_days:
        raise ValueError("诊断表长度与日期数不一致")

    ensemble = CausalEnsemble(
        values=np.array(price, dtype=float, copy=True),
        dates=dates,
        initial=np.array(initial_price, dtype=float, copy=True),
        # 电价具有明显的周内形状与近期水平漂移，与负载基线同构。
        baseline_fn=weekly_load_baseline,
        config=cfg,
    )

    forecast = np.zeros_like(price)
    for day in range(n_days):
        forecast[day] = ensemble.predict(day, float(stress_by_day[day]))
        ensemble.observe(day)
        if verbose and (day + 1) % 60 == 0:
            print(f"  电价预测进度：{day + 1}/{n_days} 天", flush=True)

    return {"price": forecast, "weights": ensemble.weights_by_day,
            "refit_events": ensemble.refit_events}


# ---------------------------------------------------------------------------
# 4.2  问题4-2 的模型装配
#
#   与 §2 完全相同，只把计划与执行所用的价格换成预测电价：
#     计划   x = stage_lp(load_fc, pv_fc, price_fc, E_start, settle="none", reserve=R)
#     执行   execute_day_mpc(x, load, pv, price_fc, E_start, load_fc, pv_fc, horizon=36)
#     结算   c_t 用当日**实际**波动电价，紧急购电按 5c_t
#   储备 R 与问题2 同口径：历史窗口累计净负荷残差的 0.80 分位数。
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 4.3  问题4-3 的模型装配
#
#   与 §3 完全相同，只把四个发布时刻的价格通道换成预测电价：
#     x⁰ = stage_lp(·, price_fc, E_start, settle="none")
#     xᵏ = stage_lp(·, price_fc, E_start, ref=xᵏ⁻¹, settle="correct", hist_min=m)
#   结算时基准量与实际波动电价相乘，链式上下调分别按 0.5c_t 与 1.5c_t 计价。
# ---------------------------------------------------------------------------


# ===========================================================================
# §5  全年因果调度主循环（四问共用）
#     源码：代码/共享/run_year.py 的 simulate()（已去掉落盘、审计与明细整理）
#
#   开关：
#     staged   —— 问题3/4-3 为 True：四个时刻发布，用链式承诺增量 MILP 修订，
#                 执行走贪心（无前瞻）；问题2/4-2 为 False：只在 0:00 发布一次，
#                 执行走滚动 MPC。
#     variable —— 问题4 为 True：决策用预测电价、结算用实际电价。
#
#   两条时间轴必须分开：
#     物理轴 i（1..144，I_{d,1} 为 00:00—00:10）用于平衡与结算；
#     附件 5 计划轴 j（1..144，J_{d,1} 为 00:10—00:20）用于提交表格，
#     映射 J_{d,j} = I_{d,j+1}、J_{d,144} = I_{d+1,1}。
#   日内第一个物理时段（00:00—00:10）执行的是**前一日**末时段的承诺，因此
#   跨日承诺链用前一日最后一条修订路径的末值衔接：carry / carry_path。
#
#   因果性：第 d 天 0:00 的计划只使用 j<d 的历史实际残差构造储备 R_d，
#   预测只使用当前时刻可得的信息，不含任何未来实际值。
# ===========================================================================
def predicted_energy(e: float, x: float, load: np.ndarray, pv: np.ndarray) -> float:
    """由当前时段净流量把储电量前推一步（用于计划 LP 的起点）。"""
    net = x + (pv - load) * DT
    if net >= 0:
        return min(E_MAX, e + ETA_C * min(net, STEP_MAX))
    return max(E_MIN, e - min(-net, STEP_MAX) / ETA_D)


def execute_year_end(commit, load, pv, price, energy, lfc, pfc, *,
                     mpc: bool, last_day: bool, physical_start: int) -> dict:
    """当日执行。非年终日用问题2 的 MPC 或问题3 的贪心；年终日改用可达性保护。

    年终保护：最后一天必须让储电量精确回到 E_0 = 6000 kWh，而这一天又是因果的
    ——它不能预知未来，只能在当日逐步把储电放/充到目标。做法是给每个时段算出
    "为在年内回到目标所允许的储电下界"，并禁止充到目标之上。
    """
    if not last_day:
        if mpc:
            return execute_day_mpc(commit, load, pv, price, energy,
                                   load_fc_kw=lfc, pv_fc_kw=pfc, horizon=36)
        return greedy_execute(commit, load, pv, energy)

    out = {key: np.zeros(len(commit)) for key in
           ("charge", "discharge", "emergency", "unused_plan",
            "curtailment", "pv_used", "plan_used", "soc")}
    for j, x in enumerate(commit):
        l, p = load[j] * DT, pv[j] * DT
        t = physical_start + j
        remaining = 143 - t
        # 不得充到年终目标之上；多余储电对着真实负载放掉。
        hi = max(E_INITIAL, energy)
        lo = max(E_MIN, E_INITIAL - remaining * ETA_C * STEP_MAX)
        v = min(STEP_MAX, l, max(0., (energy - E_INITIAL) * ETA_D))
        if v <= 1e-9:
            deficit = max(0., l - x - p)
            v = min(deficit, STEP_MAX, max(0., (energy - lo) * ETA_D))
        e_after = energy - v / ETA_D
        surplus = max(0., x + p + v - l)
        u = 0. if v > 1e-9 else min(STEP_MAX, surplus, max(0., (hi - e_after) / ETA_C))
        if e_after + ETA_C * u < lo - 1e-8:
            v = 0.
            e_after = energy
            u = max(u, (lo - energy) / ETA_C)
        if u > STEP_MAX + 1e-6:
            raise RuntimeError("year-end target no longer physically reachable")
        consumption = l + u - v
        pv_used = min(p, consumption)
        plan_used = min(x, max(0., consumption - pv_used))
        emergency = max(0., consumption - pv_used - plan_used)
        energy = e_after + ETA_C * u
        values = dict(charge=u, discharge=v, emergency=emergency, unused_plan=x - plan_used,
                      curtailment=p - pv_used, pv_used=pv_used, plan_used=plan_used, soc=energy)
        for key, value in values.items():
            out[key][j] = value
    out["end_energy"] = float(energy)
    return out


def simulate_year(a: dict, problem: str, *, n_days: int | None = None) -> list[dict]:
    """全年逐日因果调度，返回逐日结算明细。

    a 为预测层产出的数据字典：实际与预测的负载/光伏/电价序列，形状 (365, 144)
    或 (365, 4, 144)。问题1 是独立单日模型，不经过本循环。
    """
    staged = problem in ("Q3", "Q4-3")
    mpc = not staged
    variable = problem.startswith("Q4")
    stage_hours = ISSUE_HOURS if staged else (0,)
    n_stage = len(ISSUE_HOURS)          # 计划数组恒定保留 4 条修订路径

    n = min(n_days or len(a["load"]), len(a["load"]))
    dates = pd.DatetimeIndex(a["dates"])
    e = E_INITIAL                       # 全年唯一的状态递推量：日末储电量
    carry = 0.0                         # 前一日 24:00 时段的承诺量（跨日连续）
    carry_path = np.zeros(n_stage)
    error_history = {k: [] for k in ISSUE_HOURS}
    rows = []

    for d in range(n):
        e_start = e
        last_day = n == 365 and d == 364
        pset = a["price_actual"][d] if variable else a["price_fixed"]
        pdec = a["price_fc"][d] if variable else a["price_fixed"]
        lf, pf = forecast_at(a, d, 0, staged)

        # 计划 LP 的时段按"决策时点可用的预测"左移一格：两个次日 0:00 端点
        # 都用当前可得的信息预测，不偷看未来实际值。
        pl = np.r_[lf[1:], lf[0]]
        pp = np.r_[pf[1:], 0.]
        pc = np.r_[pdec[1:], pdec[0]]
        pred_e = predicted_energy(e, carry, lf[0], pf[0])

        # 因果累计储备：历史 j<d 的窗口累计净负荷残差的 0.80 分位数
        hist = error_history[0]
        reserve = (max(0., float(np.quantile(hist, RESERVE_QUANTILE)))
                   if len(hist) >= RESERVE_MIN_HISTORY else 0.)
        x = stage_lp(pl, pp, pc, pred_e, settle="none", reserve=reserve)

        paths = np.tile(x, (n_stage, 1))
        min_path = x.copy()
        ex = {key: np.zeros(N_SLOT) for key in
              ("charge", "discharge", "emergency", "unused_plan",
               "curtailment", "pv_used", "plan_used", "soc")}

        # 当日第一个物理时段在 0:00 新计划算出之后才执行，用的是前一日承诺
        first = execute_year_end(
            np.array([carry]), a["load"][d, :1], a["pv"][d, :1], pdec[:1], e,
            lf[:1], pf[:1], mpc=mpc, last_day=last_day, physical_start=0)
        e = first["end_energy"]
        for key in ex:
            ex[key][0] = first[key][0]

        current_pred = {0: (lf.copy(), pf.copy())}
        effective = np.r_[carry, x[:143]]     # 物理轴上的实际承诺（下标 0 为 00:00）

        for pos, k in enumerate(stage_hours):
            begin = 1 if k == 0 else k * 6
            if k:
                # 06/12/18 的增量修订：带上一版承诺与历史最小承诺，MILP 精确结算
                lf, pf = forecast_at(a, d, k, staged)
                current_pred[k] = (lf.copy(), pf.copy())
                slot = begin - 1
                lp = np.r_[lf[begin:], lf[0]]
                ppv = np.r_[pf[begin:], 0.]
                cp = np.r_[pdec[begin:], pdec[0]]
                hist = error_history[k]
                reserve = (max(0., float(np.quantile(hist, RESERVE_QUANTILE)))
                           if len(hist) >= RESERVE_MIN_HISTORY else 0.)
                revised = stage_lp(lp, ppv, cp, e, ref=paths[pos - 1, slot:],
                                   settle="correct", reserve=reserve,
                                   hist_min=min_path[slot:])
                x[slot:] = revised
                paths[pos:, :] = x
                min_path[slot:] = np.minimum(min_path[slot:], revised)
                effective[begin:] = x[slot:143]

            end = stage_hours[pos + 1] * 6 if pos + 1 < len(stage_hours) else N_SLOT
            c = effective[begin:end]
            act = execute_year_end(
                c, a["load"][d, begin:end], a["pv"][d, begin:end],
                pdec[begin:end], e, lf[begin:end], pf[begin:end],
                mpc=mpc, last_day=last_day, physical_start=begin)
            e = act["end_energy"]
            for key in ex:
                ex[key][begin:end] = act[key]

        # 结算：把前一日 24:00 时段的承诺作为链的第 0 条，拼成完整承诺路径
        physical_paths = np.column_stack([carry_path, paths[:, :143]])
        settled = settle_period(physical_paths, pset)
        emergency_cost = float(EMERGENCY_MULTIPLIER * pset @ ex["emergency"])
        rows.append(dict(date=str(dates[d].date()), grid_cost_yuan=settled["total_yuan"],
                         base_cost_yuan=settled["base_yuan"],
                         breach_cost_yuan=settled["breach_yuan"],
                         overbuy_cost_yuan=settled["overbuy_yuan"],
                         emergency_cost_yuan=emergency_cost,
                         total_cost_yuan=settled["total_yuan"] + emergency_cost,
                         emergency_kwh=float(ex["emergency"].sum()),
                         start_soc_kwh=e_start, end_soc_kwh=e))

        # 残差入库：只用"已发生"的时段，下一日 0:00 才能看到
        for k, (l, p) in current_pred.items():
            b = max(1, k * 6)
            error_history[k].append(
                float(((a["load"][d, b:] - a["pv"][d, b:])
                       - (l[b:] - p[b:])).sum() * DT))

        carry = float(x[-1])                  # 跨日连续：末时段承诺递到次日的 00:00
        carry_path = paths[:, -1].copy()

    return rows


# ===========================================================================
# §6  附件读取与一键复现入口
#     源码：代码/问题1/solve_q1_ab.py 的 read_attachment1()、
#           代码/问题2/deps/deterministic_baseline.py 的 load_inputs()、
#           代码/问题4/问题4-2/deps/deterministic_baseline.py 的 load_inputs()、
#           代码/共享/run_year.py 的 _find_attachments() / prepare() / official_points()
#
#   §1—§5 的模型代码不含任何文件读写；本节是唯一的 I/O 依赖，用于让本文件
#   脱离原工程目录、仅凭 附件1—4 复现全部结果。
# ===========================================================================
MODEL_DIR = Path(__file__).resolve().parent
#  复现结果落盘目录，相对本文件定位：整个 支撑材料/ 拷到任何位置（含桌面）都能写。
#  刻意避开 结果/result/ 与 结果/问题X/，不覆盖交付包中的正式成果。
OUTPUT_DIR = MODEL_DIR.parent / "结果" / "附录复现"
_ATTACHMENT_DIR: Path | None = None


def locate_attachments() -> Path:
    """附件目录固定为 支撑材料/C题/附件，不向支撑材料之外搜索。"""
    candidate = MODEL_DIR.parent / "C题" / "附件"
    if (candidate / "附件1.xlsx").exists() and (candidate / "附件2.xlsx").exists():
        return candidate
    raise FileNotFoundError(
        "未找到 附件1.xlsx / 附件2.xlsx，请保持 支撑材料/C题/附件/ 目录完整")


def attachment_dir() -> Path:
    """附件目录（首次调用时定位并缓存）。"""
    global _ATTACHMENT_DIR
    if _ATTACHMENT_DIR is None:
        _ATTACHMENT_DIR = locate_attachments()
    return _ATTACHMENT_DIR


def load_inputs_fixed() -> DataBundle:
    """读附件 1、2，构造问题1—3 的输入：电价取附件 1 的单条分时电价。"""
    attach = attachment_dir()
    load_raw = pd.read_excel(attach / "附件2.xlsx", sheet_name=0)
    pv_raw = pd.read_excel(attach / "附件2.xlsx", sheet_name=1)
    base_raw = pd.read_excel(attach / "附件1.xlsx")

    dates = pd.DatetimeIndex(pd.to_datetime(load_raw.iloc[:, 0]))
    load_kw = load_raw.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(float)
    pv_kw = pv_raw.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(float)
    price = pd.to_numeric(base_raw.iloc[:, 1], errors="raise").to_numpy(float)
    initial_load = pd.to_numeric(base_raw.iloc[:, 2], errors="raise").to_numpy(float)
    initial_pv = pd.to_numeric(base_raw.iloc[:, 3], errors="raise").to_numpy(float)

    if load_kw.shape != (365, N_SLOT) or pv_kw.shape != (365, N_SLOT):
        raise ValueError(f"附件2维度异常：load={load_kw.shape}, pv={pv_kw.shape}")
    if len(price) != N_SLOT:
        raise ValueError(f"附件1电价长度异常：{len(price)}")
    if not np.isfinite(load_kw).all() or not np.isfinite(pv_kw).all():
        raise ValueError("附件2存在缺失或非数值数据")
    if (load_kw < 0).any() or (pv_kw < 0).any():
        raise ValueError("负载或光伏存在负值")

    return DataBundle(dates, load_kw, pv_kw, price, initial_load, initial_pv)


def load_inputs_variable() -> DataBundle:
    """读附件 1、2、4，构造问题4 的输入：电价换成逐日波动的 (365, 144) 序列。"""
    attach = attachment_dir()
    data = load_inputs_fixed()
    price_raw = pd.read_excel(attach / "附件4.xlsx")

    price_dates = pd.DatetimeIndex(pd.to_datetime(price_raw.iloc[:, 0]))
    price = price_raw.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(float)

    if price.shape != (365, N_SLOT):
        raise ValueError(f"附件4电价维度异常：{price.shape}")
    if not price_dates.equals(data.dates):
        raise ValueError("附件4与附件2的日期序列不一致")
    if not np.isfinite(price).all():
        raise ValueError("附件4存在缺失或非数值数据")
    if (price <= 0).any():
        raise ValueError("附件4电价存在非正值")

    return replace(data, price=price, initial_price=data.price.copy())


def load_official_pv_table() -> pd.DataFrame:
    """读附件 3 的官方光伏预报表，列名规整为 date / issue / 1..24。"""
    raw = pd.read_excel(attachment_dir() / "附件3.xlsx")
    raw.columns = ["date", "issue"] + list(range(1, 25))
    raw["date"] = pd.to_datetime(raw["date"].ffill())
    return raw


def prepare_all(*, verbose: bool = True) -> dict:
    """从附件重建全年预测层，返回 simulate_year() 所需的 a 字典。

    与 run_year.prepare() 同口径，只是不落 npz 缓存：
      ① 基础因果预测——光伏有效窗口 + 日电量水平校正，不含周内效应；
      ② 分阶段模型的自有预测——默认配置（含周内效应校正）；
      ③ 电价预测——复用 ① 的逐日季节应力，仅使用基础集成；
      ④ 附件 3 官方预报插值，再做官方/自有光伏的因果融合。
    """
    data = load_inputs_fixed()
    cfg_base = ForecastConfig(correction=CorrectionConfig(
        use_pv_window=True, use_level=True, use_weekly_effect=False))

    if verbose:
        print("重建负载/光伏因果预测（基础配置）……", flush=True)
    fc = generate_causal_forecasts(data, cfg_base, verbose=verbose)
    a = dict(load=data.load_kw, pv=data.pv_kw, price_fixed=data.price,
             load_fc=fc["load"], pv_fc=fc["pv"],
             initial_load=data.initial_load, initial_pv=data.initial_pv,
             dates=data.dates)

    if verbose:
        print("重建分阶段模型的自有预测（默认配置）……", flush=True)
    staged_fc = generate_causal_forecasts(data, ForecastConfig(), verbose=verbose)
    a["load_stage_fc"] = staged_fc["load"]
    a["pv_stage_fc"] = staged_fc["pv"]

    if verbose:
        print("重建波动电价的因果预测……", flush=True)
    data4 = load_inputs_variable()
    stress = fc["diagnostics"].sort_values("day_index")["season_stress"].to_numpy(float)
    pf = generate_causal_price_forecast(
        data4.price, data4.initial_price, data4.dates, stress, cfg_base, verbose=verbose)
    a["price_actual"] = data4.price
    a["price_fc"] = pf["price"]

    if verbose:
        print("拟合官方/自有光伏的因果融合……", flush=True)
    a["pv_fused"] = build_pv_fusion(official_points(a), a["pv_stage_fc"], a["pv"])
    return a


def q1_result_table(r: dict, data: DataBundle) -> pd.DataFrame:
    """问题1 的 LP 解整理为逐时段表，列与 结果/问题1/ 的既有表同口径。"""
    lab = pd.date_range("2000-01-01", periods=N_SLOT + 1,
                        freq="10min").strftime("%H:%M")
    return pd.DataFrame({
        "时间": np.arange(N_SLOT),
        "时间标签": lab[:N_SLOT],
        "时段 t": np.arange(1, N_SLOT + 1),
        "时段区间": [f"{a}-{b}" for a, b in zip(lab[:N_SLOT], lab[1:])],
        "c_t (元/kWh)": data.price,
        "L_t (kW)": data.initial_load,
        "P_t^fc (kW)": data.initial_pv,
        "净负荷 (kW)": data.initial_load - data.initial_pv,
        "x_t (kW)": r["x"],
        "u_t (kW)": r["u"],
        "v_t (kW)": r["v"],
        "w_t (kW)": r["w"],
        "x_t·Δt (kWh)": r["x"] * DT,
        "E_t (kWh)": r["E"][:N_SLOT],
    })


def main() -> None:
    """一键复现：问题1 的单日 LP，以及问题2—4 的全年 365 天因果调度。

    结果写入 结果/附录复现/；该路径相对本文件定位，故整个 支撑材料/ 目录拷到
    任何位置（含桌面）都能直接运行。刻意不写 结果/result/ 与 结果/问题X/，
    以免覆盖交付包中的正式成果。
    """
    data = load_inputs_fixed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("问题1：单日线性规划（附件1 的分时电价与首日负载/光伏曲线）")
    print("=" * 72)
    for label, tag, fixed in (("约束A：E_0 = E_T = 6000 kWh", "约束A（固定）", True),
                              ("约束B：E_T − E_0 = 0（自由）", "约束B（自由）", False)):
        r = solve_q1(data.price, data.initial_load, data.initial_pv,
                     fixed_endpoint=fixed)
        print(f"  {label}")
        print(f"    日购电费用 {r['C']:.2f} 元 | 末储电量 {r['E'][-1]:.2f} kWh "
              f"| 充 {r['u'].sum() * DT:.2f} / 放 {r['v'].sum() * DT:.2f} kWh "
              f"| 弃光 {r['w'].sum():.4f} kWh")
        q1_result_table(r, data).to_csv(OUTPUT_DIR / f"问题1_{tag}_完整结果.csv",
                                        index=False, encoding="utf-8-sig")

    a = prepare_all()

    print("=" * 72)
    print("问题2—4：全年 365 天逐日因果调度（决策只用当日之前的信息）")
    print("=" * 72)
    summary = []
    for prob in ("Q2", "Q3", "Q4-2", "Q4-3"):
        tic = time.perf_counter()
        rows = simulate_year(a, prob)
        cost = sum(r["total_cost_yuan"] for r in rows)
        emergency = sum(r["emergency_kwh"] for r in rows)
        formal_rows = [r for r in rows
                       if pd.Timestamp(r["date"]) >= FORMAL_START]
        formal = sum(r["total_cost_yuan"] for r in formal_rows)
        print(f"  {prob}：全年总费用 {cost:.2f} 元"
              f"（正式期 {len(formal_rows)} 天 {formal:.2f} 元）"
              f"| 紧急购电 {emergency:.2f} kWh | 年末储电量 {rows[-1]['end_soc_kwh']:.2f} kWh")
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"{prob}_daily.csv",
                                  index=False, encoding="utf-8-sig")
        record = {
            "problem": prob,
            "simulation_days": len(rows),
            "formal_days": len(formal_rows),
            "calendar_year_cost_yuan": cost,
            "formal_period_cost_yuan": formal,
            "final_soc_kwh": rows[-1]["end_soc_kwh"],
            "seconds": time.perf_counter() - tic,
            "source": "附录_核心求解代码.py 由 附件1-4 原样重算",
        }
        (OUTPUT_DIR / f"{prob}_summary.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append(record)

    pd.DataFrame([{k: v for k, v in s.items() if k != "source"}
                  for s in summary]).to_csv(OUTPUT_DIR / "全年汇总.csv",
                                            index=False, encoding="utf-8-sig")
    print(f"\n结果已写入 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
