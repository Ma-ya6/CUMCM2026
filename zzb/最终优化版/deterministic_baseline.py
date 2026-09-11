from __future__ import annotations

import json
import platform
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from scipy.optimize import linprog
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(
    "ignore", message="X does not have valid feature names.*", category=UserWarning
)


SEED = 20260910
DT = 1.0 / 6.0
N_SLOT = 144
ETA_C = 0.90
ETA_D = 0.90
E_MIN = 1200.0
E_MAX = 10800.0
E_INITIAL = 6000.0
P_MAX_KW = 5000.0
FORMAL_START = pd.Timestamp("2025-02-01")
SPECIFIED_DATES = pd.to_datetime(
    ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
)

MODEL_DIR = Path(__file__).resolve().parent
C_DIR = MODEL_DIR.parents[1]
ATTACHMENT_DIR = C_DIR / "附件"
RESULT_DIR = MODEL_DIR / "results"
FIGURE_DIR = MODEL_DIR / "figures"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DataBundle:
    dates: pd.DatetimeIndex
    load_kw: np.ndarray
    pv_kw: np.ndarray
    price: np.ndarray
    initial_load: np.ndarray
    initial_pv: np.ndarray


@dataclass
class StorageParameters:
    eta_c: float = ETA_C
    eta_d: float = ETA_D
    e_min: float = E_MIN
    e_max: float = E_MAX
    p_max_kw: float = P_MAX_KW


def load_inputs() -> DataBundle:
    attachment2 = ATTACHMENT_DIR / "附件2.xlsx"
    attachment1 = ATTACHMENT_DIR / "附件1.xlsx"
    load_raw = pd.read_excel(attachment2, sheet_name=0)
    pv_raw = pd.read_excel(attachment2, sheet_name=1)
    base_raw = pd.read_excel(attachment1)

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


def recent_weighted(data: np.ndarray, day: int, initial: np.ndarray) -> np.ndarray:
    if day == 0:
        return initial.copy()
    lags = [(1, 0.60), (2, 0.30), (3, 0.10)]
    valid = [(lag, weight) for lag, weight in lags if day - lag >= 0]
    total = sum(weight for _, weight in valid)
    return sum(weight * data[day - lag] for lag, weight in valid) / total


def weekly_load_baseline(data: np.ndarray, day: int, initial: np.ndarray) -> np.ndarray:
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
    if day == 0:
        return initial.copy()
    candidates = [(1, 0.50), (2, 0.30), (7, 0.20)]
    valid = [(lag, weight) for lag, weight in candidates if day - lag >= 0]
    total = sum(weight for _, weight in valid)
    pred = sum(weight * data[day - lag] for lag, weight in valid) / total
    return np.maximum(pred, 0.0)


def day_features(data: np.ndarray, dates: pd.DatetimeIndex, day: int) -> np.ndarray:
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
    available: Iterable[str], history: Dict[str, list[float]], lookback: int = 28
) -> Dict[str, float]:
    """指数加权近期误差；指数损失使持续劣化的复杂模型能被实质降权。"""
    names = list(available)
    if len(names) == 1:
        return {names[0]: 1.0}
    scores = []
    for name in names:
        errors = np.asarray(history.get(name, [])[-lookback:], float)
        if errors.size:
            # 最近一天权重最高；同时加入上四分位误差，抑制偶发失稳的模型。
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


def generate_walk_forward_forecasts(data: DataBundle) -> dict:
    n_days = len(data.dates)
    output = {}
    candidate_store = {}

    for label, values, initial, baseline_fn in [
        ("load", data.load_kw, data.initial_load, weekly_load_baseline),
        ("pv", data.pv_kw, data.initial_pv, recent_pv_baseline),
    ]:
        combined = np.zeros_like(values)
        candidates_by_name = {
            "baseline": np.full_like(values, np.nan),
            "ridge": np.full_like(values, np.nan),
            "lightgbm": np.full_like(values, np.nan),
        }
        weights_by_day = []
        error_history: Dict[str, list[float]] = {
            "baseline": [],
            "ridge": [],
            "lightgbm": [],
        }
        ridge_model = None
        lgb_model = None

        for day in range(n_days):
            candidates = {"baseline": baseline_fn(values, day, initial)}

            if day >= 28:
                if ridge_model is None or day % 7 == 0:
                    x_train, y_train = training_matrix(values, data.dates, day)
                    ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                    ridge_model.fit(x_train, y_train)
                candidates["ridge"] = ridge_model.predict(day_features(values, data.dates, day))

            if day >= 60:
                if lgb_model is None or day % 14 == 0:
                    lgb_model, _ = fit_lightgbm_causal(values, data.dates, day)
                candidates["lightgbm"] = lgb_model.predict(day_features(values, data.dates, day))

            for name in candidates:
                candidates[name] = np.maximum(candidates[name], 0.0)
                candidates_by_name[name][day] = candidates[name]

            weights = adaptive_error_weights(candidates.keys(), error_history)
            pred = sum(weights[name] * candidates[name] for name in candidates)
            combined[day] = np.maximum(pred, 0.0)
            weights_by_day.append(weights)

            for name, pred_candidate in candidates.items():
                error_history[name].append(float(np.mean(np.abs(pred_candidate - values[day]))))

        output[label] = combined
        candidate_store[label] = candidates_by_name
        output[f"{label}_weights"] = weights_by_day

    output["candidates"] = candidate_store
    output["net_error"] = (data.load_kw - data.pv_kw) - (output["load"] - output["pv"])
    return output


def causal_residual_quantiles(
    errors: np.ndarray,
    quantile: float,
    lookback_days: int = 90,
    short_window_days: int = 30,
    update_rate: float = 0.35,
) -> np.ndarray:
    """因果分位数安全余量：长短窗收缩、相邻时段平滑和逐日阻尼。"""
    margins = np.zeros_like(errors)
    for day in range(len(errors)):
        if day == 0:
            margins[day] = 0.0
            continue
        history = errors[max(0, day - lookback_days) : day]
        if len(history) < 7:
            value = float(np.quantile(history, quantile))
            raw = np.full(errors.shape[1], value)
        else:
            long_quantile = np.quantile(history, quantile, axis=0)
            short_history = history[-min(short_window_days, len(history)) :]
            short_quantile = np.quantile(short_history, quantile, axis=0)
            raw = 0.70 * long_quantile + 0.30 * short_quantile
            # 十分钟数据相邻性强，轻度循环平滑可减少逐时段样本分位数抖动。
            raw = 0.25 * np.roll(raw, 1) + 0.50 * raw + 0.25 * np.roll(raw, -1)
        if day == 1:
            margins[day] = raw
        else:
            margins[day] = update_rate * raw + (1.0 - update_rate) * margins[day - 1]
    return margins


def end_water_value(price: np.ndarray) -> float:
    """日末储电的残余价值（水值）。

    留在储能中的每 1 kWh，次日可按 η_d 折算放电折抵购电，故取
    λ = η_d × 平均电价。用它替代“强制日末回归”，既抑制逐日耗尽，
    又让计划在“价低时充、价高时放”之间做跨时段套利。
    """
    return float(ETA_D * np.mean(price))


def solve_plan(
    load_forecast_kw: np.ndarray,
    pv_forecast_kw: np.ndarray,
    price: np.ndarray,
    start_energy: float,
    params: StorageParameters,
    force_end_soc: float | None = None,
) -> dict:
    load = np.maximum(load_forecast_kw, 0.0) * DT
    pv = np.maximum(pv_forecast_kw, 0.0) * DT
    step_max = params.p_max_kw * DT
    n = N_SLOT
    q0, c0, d0, w0, s0 = 0, n, 2 * n, 3 * n, 4 * n
    # 求解变量统一用“每十分钟电量”(kWh/时段 = kW × Δt)，与论文功率(kW)口径
    # 经 Δt=1/6 线性等价；s0 为时段末储电量状态(kWh)。
    objective = np.zeros(5 * n)
    objective[q0 : q0 + n] = price
    objective[c0 : c0 + n] = 1e-7
    objective[d0 : d0 + n] = 1e-7
    objective[w0 : w0 + n] = 1e-8
    # 问题2.md #20：不强制日末回归，改给日末储电残余价值；年末强制 E0。
    if force_end_soc is None:
        objective[s0 + n - 1] = -end_water_value(price)
    a_eq, b_eq = [], []

    for t in range(n):
        row = np.zeros(5 * n)
        row[q0 + t] = 1.0
        row[c0 + t] = -1.0
        row[d0 + t] = 1.0
        row[w0 + t] = -1.0
        a_eq.append(row)
        b_eq.append(load[t] - pv[t])

        row = np.zeros(5 * n)
        row[c0 + t] = -params.eta_c
        row[d0 + t] = 1.0 / params.eta_d
        row[s0 + t] = 1.0
        if t:
            row[s0 + t - 1] = -1.0
            rhs = 0.0
        else:
            rhs = start_energy
        a_eq.append(row)
        b_eq.append(rhs)

    if force_end_soc is not None:
        end_bound = (float(np.clip(force_end_soc, params.e_min, params.e_max)),) * 2
    else:
        end_bound = (params.e_min, params.e_max)
    bounds = (
        [(0.0, None)] * n
        + [(0.0, step_max)] * n
        + [(0.0, step_max)] * n
        + [(0.0, None)] * n
        + [(params.e_min, params.e_max)] * (n - 1)
        + [end_bound]
    )
    result = linprog(
        objective,
        A_eq=np.asarray(a_eq),
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"计划LP失败：{result.message}")
    x = result.x
    return {
        "purchase": x[q0 : q0 + n],
        "planned_charge": x[c0 : c0 + n],
        "planned_discharge": x[d0 : d0 + n],
        "planned_curtailment": x[w0 : w0 + n],
        "planned_soc": x[s0 : s0 + n],
    }


def solve_full_year_oracle(
    data: DataBundle,
    params: StorageParameters,
    force_terminal_soc: bool = True,
) -> dict:
    """完美信息下全年联合优化的理论最低费用（真正下界）。

    将全年 365×144 个时段作为一个整体 LP，已知全部负载/光伏/电价，
    联合优化计划购电与储能充放电，不存在逐日分割、日末水值近似或
    预测误差，因此是任何日前策略（包括完美日前预测）都不可超越的
    费用下界。完美信息下紧急购电恒为 0（可在 1 倍电价下提前计划，
    不必用 5 倍电价），故目标只含计划购电费。

    force_terminal_soc=True 时强制年末储电回归 E0=6000，与其它方案
    相同期末口径（问题2.md #20）；False 时放任期末储电自由，给出
    绝对费用地板（会借耗尽储能进一步压低费用）。
    """
    load = data.load_kw.reshape(-1) * DT
    pv = data.pv_kw.reshape(-1) * DT
    price_all = np.tile(data.price, len(data.dates))
    step_max = params.p_max_kw * DT
    n = len(load)
    q0, c0, d0, w0, s0 = 0, n, 2 * n, 3 * n, 4 * n
    nvar = 5 * n
    objective = np.zeros(nvar)
    objective[q0 : q0 + n] = price_all
    objective[c0 : c0 + n] = 1e-7
    objective[d0 : d0 + n] = 1e-7
    objective[w0 : w0 + n] = 1e-8

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs: list[float] = []
    # 平衡约束：x - u + v - w = load - pv（每时段）。
    for t in range(n):
        rows += [t, t, t, t]
        cols += [q0 + t, c0 + t, d0 + t, w0 + t]
        vals += [1.0, -1.0, 1.0, -1.0]
        rhs.append(load[t] - pv[t])
    # 储能动态：-η_c u + v/η_d + E_t - E_{t-1} = 0（t=0 时 E_{-1}=E0）。
    off = n
    for t in range(n):
        rows += [off + t, off + t, off + t]
        cols += [c0 + t, d0 + t, s0 + t]
        vals += [-params.eta_c, 1.0 / params.eta_d, 1.0]
        if t:
            rows.append(off + t)
            cols.append(s0 + t - 1)
            vals.append(-1.0)
            rhs.append(0.0)
        else:
            rhs.append(E_INITIAL)

    a_eq = sparse.coo_matrix((vals, (rows, cols)), shape=(2 * n, nvar)).tocsr()
    b_eq = np.asarray(rhs)
    lb = np.zeros(nvar)
    ub = np.full(nvar, np.inf)
    lb[c0 : c0 + n] = 0.0
    ub[c0 : c0 + n] = step_max
    lb[d0 : d0 + n] = 0.0
    ub[d0 : d0 + n] = step_max
    lb[s0 : s0 + n] = params.e_min
    ub[s0 : s0 + n] = params.e_max
    if force_terminal_soc:
        lb[s0 + n - 1] = E_INITIAL
        ub[s0 + n - 1] = E_INITIAL
    bounds = list(zip(lb, ub))

    result = linprog(
        objective,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"全年oracle LP失败：{result.message}")
    x = result.x
    purchase = x[q0 : q0 + n]
    # 与模型1、逐日oracle一致：费用只统计正式输出期（2.1—12.31）。
    # 1 月仍参与 LP（保证进入 2 月的 SOC 由递推得到），但不计入费用。
    formal_day0 = int(np.where(data.dates == FORMAL_START)[0][0])
    formal_slot0 = formal_day0 * N_SLOT
    formal_mask = np.arange(n) >= formal_slot0
    return {
        "total_cost_yuan": float(price_all @ purchase),
        "formal_cost_yuan": float(price_all[formal_mask] @ purchase[formal_mask]),
        "planned_purchase_kwh": float(purchase[formal_mask].sum()),
        "final_soc_kwh": float(x[s0 + n - 1]),
        "status": int(result.status),
    }


def execute_actual(
    purchase: np.ndarray,
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    start_energy: float,
    params: StorageParameters,
) -> dict:
    load = load_kw * DT
    pv = pv_kw * DT
    step_max = params.p_max_kw * DT
    energy = float(np.clip(start_energy, params.e_min, params.e_max))
    charge = np.zeros(N_SLOT)
    discharge = np.zeros(N_SLOT)
    emergency = np.zeros(N_SLOT)
    unused_plan = np.zeros(N_SLOT)
    curtailment = np.zeros(N_SLOT)
    pv_used = np.zeros(N_SLOT)
    plan_used = np.zeros(N_SLOT)
    soc = np.zeros(N_SLOT)

    for t in range(N_SLOT):
        available = purchase[t] + pv[t]
        if available >= load[t]:
            surplus = available - load[t]
            charge[t] = min(
                surplus,
                step_max,
                max(0.0, (params.e_max - energy) / params.eta_c),
            )
            energy += params.eta_c * charge[t]
            consumption = load[t] + charge[t]
            pv_used[t] = min(pv[t], consumption)
            plan_used[t] = min(purchase[t], max(0.0, consumption - pv_used[t]))
            curtailment[t] = max(0.0, pv[t] - pv_used[t])
            unused_plan[t] = max(0.0, purchase[t] - plan_used[t])
        else:
            deficit = load[t] - available
            discharge[t] = min(
                deficit,
                step_max,
                max(0.0, (energy - params.e_min) * params.eta_d),
            )
            energy -= discharge[t] / params.eta_d
            emergency[t] = max(0.0, deficit - discharge[t])
            pv_used[t] = pv[t]
            plan_used[t] = purchase[t]
        soc[t] = energy

    return {
        "charge": charge,
        "discharge": discharge,
        "emergency": emergency,
        "unused_plan": unused_plan,
        "curtailment": curtailment,
        "pv_used": pv_used,
        "plan_used": plan_used,
        "soc": soc,
        "end_energy": float(energy),
    }


def run_dispatch(
    data: DataBundle,
    load_forecast: np.ndarray,
    pv_forecast: np.ndarray,
    margins_kw: np.ndarray,
    params: StorageParameters = StorageParameters(),
    keep_intervals: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_rows = []
    interval_rows = []
    energy = E_INITIAL

    for day, date in enumerate(data.dates):
        start_energy = energy
        effective_load = np.maximum(load_forecast[day] + margins_kw[day], 0.0)
        plan = solve_plan(
            effective_load,
            pv_forecast[day],
            data.price,
            start_energy,
            params,
            force_end_soc=E_INITIAL if day == len(data.dates) - 1 else None,
        )
        actual = execute_actual(
            plan["purchase"], data.load_kw[day], data.pv_kw[day], start_energy, params
        )
        energy = actual["end_energy"]
        plan_cost = float(data.price @ plan["purchase"])
        emergency_cost = float(5.0 * data.price @ actual["emergency"])
        if date >= FORMAL_START:
            daily_rows.append(
                {
                    "date": date,
                    "planned_purchase_kwh": float(plan["purchase"].sum()),
                    "planned_cost_yuan": plan_cost,
                    "emergency_kwh": float(actual["emergency"].sum()),
                    "emergency_cost_yuan": emergency_cost,
                    "total_cost_yuan": plan_cost + emergency_cost,
                    "unused_plan_kwh": float(actual["unused_plan"].sum()),
                    "curtailed_pv_kwh": float(actual["curtailment"].sum()),
                    "charge_kwh": float(actual["charge"].sum()),
                    "discharge_kwh": float(actual["discharge"].sum()),
                    "start_soc_kwh": float(start_energy),
                    "end_soc_kwh": energy,
                    "load_kwh": float(data.load_kw[day].sum() * DT),
                    "pv_kwh": float(data.pv_kw[day].sum() * DT),
                }
            )
            if keep_intervals:
                for t in range(N_SLOT):
                    interval_rows.append(
                        {
                            "date": date,
                            "slot": t + 1,
                            "time_end": f"{((t + 1) * 10) // 60 % 24:02d}:{((t + 1) * 10) % 60:02d}",
                            "load_actual_kw": data.load_kw[day, t],
                            "pv_actual_kw": data.pv_kw[day, t],
                            "load_forecast_kw": load_forecast[day, t],
                            "pv_forecast_kw": pv_forecast[day, t],
                            "net_safety_margin_kw": margins_kw[day, t],
                            "planned_purchase_kwh": plan["purchase"][t],
                            "plan_used_kwh": actual["plan_used"][t],
                            "unused_plan_kwh": actual["unused_plan"][t],
                            "charge_kwh": actual["charge"][t],
                            "discharge_kwh": actual["discharge"][t],
                            "emergency_kwh": actual["emergency"][t],
                            "curtailed_pv_kwh": actual["curtailment"][t],
                            "pv_used_kwh": actual["pv_used"][t],
                            "soc_kwh": actual["soc"][t],
                        }
                    )

    return pd.DataFrame(daily_rows), pd.DataFrame(interval_rows)


def metric_row(name: str, actual: np.ndarray, predicted: np.ndarray) -> dict:
    mask = np.isfinite(actual) & np.isfinite(predicted)
    y = actual[mask]
    p = predicted[mask]
    mae = float(mean_absolute_error(y, p))
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    mean_abs_level = float(np.mean(np.abs(y)))
    nmae = mae / max(mean_abs_level, 1e-9)
    return {
        "model": name,
        "n": int(mask.sum()),
        "mae_kw": mae,
        "rmse_kw": rmse,
        "r2": float(r2_score(y, p)),
        "nmae": nmae,
        "accuracy_1_minus_nmae": float(1.0 - nmae),
        "bias_kw": float(np.mean(p - y)),
    }


def evaluate_forecasts(data: DataBundle, forecasts: dict) -> pd.DataFrame:
    formal = data.dates >= FORMAL_START
    rows = []
    for target, actual in [
        ("load", data.load_kw),
        ("pv", data.pv_kw),
        ("net", data.load_kw - data.pv_kw),
    ]:
        pred = forecasts["load"] if target == "load" else forecasts["pv"]
        if target == "net":
            pred = forecasts["load"] - forecasts["pv"]
        row = metric_row(f"ensemble_{target}", actual[formal], pred[formal])
        row["target"] = target
        row["evaluation_period"] = "2025-02-01_to_2025-12-31"
        rows.append(row)

    common = np.where((data.dates >= pd.Timestamp("2025-03-02")))[0]
    for target, actual in [("load", data.load_kw), ("pv", data.pv_kw)]:
        for model_name, values in forecasts["candidates"][target].items():
            row = metric_row(f"{model_name}_{target}", actual[common], values[common])
            row["target"] = target
            row["evaluation_period"] = "common_period_from_2025-03-02"
            rows.append(row)
    return pd.DataFrame(rows)


def confidence_diagnostics(data: DataBundle, forecasts: dict) -> dict:
    formal_indices = np.where(data.dates >= FORMAL_START)[0]
    net_actual = data.load_kw - data.pv_kw
    net_pred = forecasts["load"] - forecasts["pv"]
    covered, widths = [], []
    for day in formal_indices:
        history = forecasts["net_error"][max(0, day - 60) : day]
        lower_res = np.quantile(history, 0.10, axis=0)
        upper_res = np.quantile(history, 0.90, axis=0)
        lower = net_pred[day] + lower_res
        upper = net_pred[day] + upper_res
        covered.extend(((net_actual[day] >= lower) & (net_actual[day] <= upper)).tolist())
        widths.extend((upper - lower).tolist())
    return {
        "nominal_interval": 0.80,
        "empirical_coverage": float(np.mean(covered)),
        "mean_interval_width_kw": float(np.mean(widths)),
        "coverage_error": float(np.mean(covered) - 0.80),
    }


def forecast_stability_diagnostics(data: DataBundle, forecasts: dict) -> dict:
    """量化逐日误差波动、跨月漂移和集成权重换手。"""
    formal_indices = np.where(data.dates >= FORMAL_START)[0]
    net_actual = data.load_kw - data.pv_kw
    net_pred = forecasts["load"] - forecasts["pv"]
    daily_mae = np.mean(np.abs(net_actual[formal_indices] - net_pred[formal_indices]), axis=1)
    months = data.dates[formal_indices].to_period("M")
    monthly_nmae = []
    for month in months.unique():
        chosen = formal_indices[months == month]
        monthly_nmae.append(
            float(
                np.mean(np.abs(net_actual[chosen] - net_pred[chosen]))
                / max(np.mean(np.abs(net_actual[chosen])), 1e-9)
            )
        )

    weight_turnover = {}
    for target in ["load", "pv"]:
        matrix = np.asarray(
            [
                [weights.get(name, 0.0) for name in ["baseline", "ridge", "lightgbm"]]
                for day, weights in enumerate(forecasts[f"{target}_weights"])
                if data.dates[day] >= FORMAL_START
            ]
        )
        weight_turnover[target] = float(np.mean(np.abs(np.diff(matrix, axis=0)).sum(axis=1)) / 2)

    return {
        "daily_net_mae_cv": float(np.std(daily_mae, ddof=1) / np.mean(daily_mae)),
        "monthly_net_nmae_std": float(np.std(monthly_nmae, ddof=1)),
        "mean_daily_weight_turnover": weight_turnover,
    }


def moving_block_bootstrap_ci(values: np.ndarray, block: int = 7, reps: int = 1000) -> list[float]:
    rng = np.random.default_rng(SEED)
    values = np.asarray(values, float)
    n = len(values)
    starts = np.arange(max(1, n - block + 1))
    estimates = []
    for _ in range(reps):
        sample = []
        while len(sample) < n:
            start = int(rng.choice(starts))
            sample.extend(values[start : start + block])
        estimates.append(np.mean(sample[:n]))
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def overfitting_diagnostic(data: DataBundle) -> pd.DataFrame:
    train_end = int(np.where(data.dates == pd.Timestamp("2025-09-01"))[0][0])
    valid_end = int(np.where(data.dates == pd.Timestamp("2025-11-01"))[0][0])
    rows = []
    for target, values in [("load", data.load_kw), ("pv", data.pv_kw)]:
        x_train, y_train = training_matrix(values, data.dates, train_end, window_days=365)
        models = {
            "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            "lightgbm": None,
        }
        ranges = {
            "train": range(14, train_end),
            "validation": range(train_end, valid_end),
            "test": range(valid_end, len(data.dates)),
        }
        for name, model in models.items():
            best_iteration = None
            if name == "lightgbm":
                model, best_iteration = fit_lightgbm_causal(
                    values,
                    data.dates,
                    train_end,
                    window_days=365,
                    validation_days=28,
                )
            else:
                model.fit(x_train, y_train)
            scores = {}
            normalized_scores = {}
            for split, days in ranges.items():
                x = np.vstack([day_features(values, data.dates, day) for day in days])
                y = np.concatenate([values[day] for day in days])
                scores[split] = float(mean_absolute_error(y, np.maximum(model.predict(x), 0.0)))
                normalized_scores[split] = scores[split] / max(float(np.mean(np.abs(y))), 1e-9)
            gap = (normalized_scores["test"] - normalized_scores["train"]) / max(
                normalized_scores["train"], 1e-9
            )
            validation_gap = (
                normalized_scores["test"] - normalized_scores["validation"]
            ) / max(normalized_scores["validation"], 1e-9)
            # 同时参考训练差距与真正的未来验证差距，避免单一任意阈值判断。
            overfit_flag = bool(
                (gap > 0.25 and validation_gap > 0.10) or validation_gap > 0.25
            )
            generalization_warning = bool(gap > 0.10 or validation_gap > 0.10)
            rows.append(
                {
                    "target": target,
                    "model": name,
                    "train_mae_kw": scores["train"],
                    "validation_mae_kw": scores["validation"],
                    "test_mae_kw": scores["test"],
                    "train_nmae": normalized_scores["train"],
                    "validation_nmae": normalized_scores["validation"],
                    "test_nmae": normalized_scores["test"],
                    "test_train_nmae_gap_ratio": gap,
                    "test_validation_nmae_gap_ratio": validation_gap,
                    "best_iteration": best_iteration,
                    "overfit_flag": overfit_flag,
                    "generalization_warning": generalization_warning,
                    "note": "该差距同时包含季节分布漂移，不能单独视为纯过拟合",
                }
            )
    return pd.DataFrame(rows)


def sensitivity_analysis(data: DataBundle, forecasts: dict) -> pd.DataFrame:
    rows = []
    cases = [
        ("base", 0.80, ETA_C, ETA_D, E_MIN, E_MAX, P_MAX_KW),
        ("quantile_0.70", 0.70, ETA_C, ETA_D, E_MIN, E_MAX, P_MAX_KW),
        ("quantile_0.90", 0.90, ETA_C, ETA_D, E_MIN, E_MAX, P_MAX_KW),
        ("efficiency_0.85", 0.80, 0.85, 0.85, E_MIN, E_MAX, P_MAX_KW),
        ("efficiency_0.95", 0.80, 0.95, 0.95, E_MIN, E_MAX, P_MAX_KW),
        ("power_4000", 0.80, ETA_C, ETA_D, E_MIN, E_MAX, 4000.0),
        ("power_6000", 0.80, ETA_C, ETA_D, E_MIN, E_MAX, 6000.0),
        ("usable_capacity_80pct", 0.80, ETA_C, ETA_D, 2160.0, 9840.0, P_MAX_KW),
        ("usable_capacity_120pct", 0.80, ETA_C, ETA_D, 240.0, 11760.0, P_MAX_KW),
    ]
    for case, q, eta_c, eta_d, e_min, e_max, p_max in cases:
        margins = causal_residual_quantiles(forecasts["net_error"], q)
        params = StorageParameters(eta_c, eta_d, e_min, e_max, p_max)
        daily, _ = run_dispatch(data, forecasts["load"], forecasts["pv"], margins, params)
        rows.append(
            {
                "case": case,
                "safety_quantile": q,
                "eta_c": eta_c,
                "eta_d": eta_d,
                "e_min_kwh": e_min,
                "e_max_kwh": e_max,
                "p_max_kw": p_max,
                "total_cost_yuan": float(daily.total_cost_yuan.sum()),
                "emergency_kwh": float(daily.emergency_kwh.sum()),
                "unused_plan_kwh": float(daily.unused_plan_kwh.sum()),
                "curtailed_pv_kwh": float(daily.curtailed_pv_kwh.sum()),
            }
        )
    result = pd.DataFrame(rows)
    base_cost = float(result.loc[result.case == "base", "total_cost_yuan"].iloc[0])
    result["cost_change_pct"] = (result.total_cost_yuan / base_cost - 1.0) * 100.0
    return result


def robustness_analysis(data: DataBundle, forecasts: dict) -> pd.DataFrame:
    rows = []
    base_margin = causal_residual_quantiles(forecasts["net_error"], 0.80)
    for level in [0.05, 0.10]:
        for seed_offset in range(10):
            rng = np.random.default_rng(SEED + seed_offset + int(level * 1000))
            day_load_bias = rng.normal(0.0, level, size=(len(data.dates), 1))
            day_pv_bias = rng.normal(0.0, level, size=(len(data.dates), 1))
            slot_load_noise = rng.normal(0.0, level / 3, size=data.load_kw.shape)
            slot_pv_noise = rng.normal(0.0, level / 3, size=data.pv_kw.shape)
            load_perturbed = np.maximum(forecasts["load"] * (1 + day_load_bias + slot_load_noise), 0.0)
            pv_perturbed = np.maximum(forecasts["pv"] * (1 + day_pv_bias + slot_pv_noise), 0.0)
            daily, _ = run_dispatch(data, load_perturbed, pv_perturbed, base_margin)
            rows.append(
                {
                    "perturbation_level": level,
                    "replicate": seed_offset + 1,
                    "total_cost_yuan": float(daily.total_cost_yuan.sum()),
                    "emergency_kwh": float(daily.emergency_kwh.sum()),
                    "unused_plan_kwh": float(daily.unused_plan_kwh.sum()),
                }
            )
    result = pd.DataFrame(rows)
    return result


def plot_results(
    data: DataBundle,
    metrics: pd.DataFrame,
    daily: pd.DataFrame,
    interval: pd.DataFrame,
    sensitivity: pd.DataFrame,
    robustness: pd.DataFrame,
    overfit: pd.DataFrame,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    ensemble = metrics[metrics.model.str.startswith("ensemble_")]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(ensemble.target, ensemble.accuracy_1_minus_nmae * 100, color=["#2878B5", "#F8AC8C", "#6F4E7C"])
    ax.set_ylabel("准确率 1-NMAE（%）")
    ax.set_title("模型1滚动预测准确率（2—12月样本外）")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_预测准确率.png", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(daily.date, daily.total_cost_yuan, lw=1.0, color="#2878B5")
    axes[0].plot(daily.date, daily.total_cost_yuan.rolling(14, min_periods=1).mean(), lw=2.0, color="#C82423", label="14日均值")
    axes[0].set_ylabel("每日总费用（元）")
    axes[0].legend(frameon=False)
    axes[1].plot(daily.date, daily.emergency_kwh, lw=1.0, color="#F8AC8C")
    axes[1].set_ylabel("紧急购电量（kWh）")
    axes[1].set_xlabel("日期")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
    fig.suptitle("模型1全年滚动调度结果")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_每日费用与紧急购电.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, date in zip(axes.flat, SPECIFIED_DATES):
        subset = interval[interval.date == date]
        hour = np.arange(1, N_SLOT + 1) / 6
        ax.plot(hour, subset.load_actual_kw, label="实际负载", color="#2878B5")
        ax.plot(hour, subset.pv_actual_kw, label="实际光伏", color="#F8AC8C")
        ax.plot(hour, subset.planned_purchase_kwh / DT, label="计划购电功率", color="#6F4E7C", lw=1.2)
        ax.plot(hour, subset.emergency_kwh / DT, label="紧急购电功率", color="#C82423", lw=1.0)
        ax.set_title(str(date.date()))
        ax.set_xlim(0, 24)
        ax.set_xlabel("时刻（h）")
        ax.set_ylabel("功率（kW）")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("指定日期负载、光伏与购电策略", y=0.99)
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.955), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGURE_DIR / "03_指定日期调度.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    s = sensitivity.sort_values("cost_change_pct")
    axes[0].barh(s.case, s.cost_change_pct, color="#9AC9DB")
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_xlabel("相对基准费用变化（%）")
    axes[0].set_title("参数灵敏度")
    r = robustness.groupby("perturbation_level").total_cost_yuan.agg(["mean", "std"])
    axes[1].errorbar(r.index * 100, r["mean"], yerr=r["std"], marker="o", capsize=5, color="#C82423")
    axes[1].set_xlabel("预测扰动标准差（%）")
    axes[1].set_ylabel("全年总费用（元）")
    axes[1].set_title("随机扰动鲁棒性（均值±标准差）")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_灵敏度与鲁棒性.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, target in zip(axes, ["load", "pv"]):
        part = overfit[overfit.target == target]
        x = np.arange(len(part))
        width = 0.25
        ax.bar(x - width, part.train_mae_kw, width, label="训练")
        ax.bar(x, part.validation_mae_kw, width, label="验证")
        ax.bar(x + width, part.test_mae_kw, width, label="测试")
        ax.set_xticks(x, part.model)
        ax.set_ylabel("MAE（kW）")
        ax.set_title("负载" if target == "load" else "光伏")
    axes[0].legend(frameon=False)
    fig.suptitle("固定模型时间外推与过拟合诊断")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_过拟合诊断.png", dpi=220)
    plt.close(fig)


def validate_physical_results(interval: pd.DataFrame, params: StorageParameters) -> dict:
    balance = (
        interval.plan_used_kwh
        + interval.pv_used_kwh
        + interval.discharge_kwh
        + interval.emergency_kwh
        - interval.load_actual_kw * DT
        - interval.charge_kwh
    )
    return {
        "max_abs_energy_balance_error_kwh": float(balance.abs().max()),
        "soc_below_min_count": int((interval.soc_kwh < params.e_min - 1e-7).sum()),
        "soc_above_max_count": int((interval.soc_kwh > params.e_max + 1e-7).sum()),
        "charge_power_violation_count": int(
            (interval.charge_kwh > params.p_max_kw * DT + 1e-7).sum()
        ),
        "discharge_power_violation_count": int(
            (interval.discharge_kwh > params.p_max_kw * DT + 1e-7).sum()
        ),
        "simultaneous_charge_discharge_count": int(
            ((interval.charge_kwh > 1e-8) & (interval.discharge_kwh > 1e-8)).sum()
        ),
        "negative_flow_count": int(
            (
                interval[
                    [
                        "planned_purchase_kwh",
                        "plan_used_kwh",
                        "unused_plan_kwh",
                        "charge_kwh",
                        "discharge_kwh",
                        "emergency_kwh",
                        "curtailed_pv_kwh",
                        "pv_used_kwh",
                    ]
                ]
                < -1e-8
            )
            .sum()
            .sum()
        ),
    }


def main() -> None:
    started = time.perf_counter()
    np.random.seed(SEED)
    data = load_inputs()
    forecasts = generate_walk_forward_forecasts(data)
    margins = causal_residual_quantiles(forecasts["net_error"], 0.80)

    daily, interval = run_dispatch(
        data, forecasts["load"], forecasts["pv"], margins, keep_intervals=True
    )
    metrics = evaluate_forecasts(data, forecasts)
    confidence = confidence_diagnostics(data, forecasts)
    stability = forecast_stability_diagnostics(data, forecasts)
    overfit = overfitting_diagnostic(data)
    sensitivity = sensitivity_analysis(data, forecasts)
    robustness = robustness_analysis(data, forecasts)
    physical_validation = validate_physical_results(interval, StorageParameters())

    baseline_load = np.vstack(
        [weekly_load_baseline(data.load_kw, d, data.initial_load) for d in range(len(data.dates))]
    )
    baseline_pv = np.vstack(
        [recent_pv_baseline(data.pv_kw, d, data.initial_pv) for d in range(len(data.dates))]
    )
    baseline_error = (data.load_kw - data.pv_kw) - (baseline_load - baseline_pv)
    baseline_margin = causal_residual_quantiles(baseline_error, 0.80)
    baseline_daily, _ = run_dispatch(data, baseline_load, baseline_pv, baseline_margin)

    oracle_daily, _ = run_dispatch(
        data,
        data.load_kw,
        data.pv_kw,
        np.zeros_like(data.load_kw),
    )
    oracle_full_year = solve_full_year_oracle(data, StorageParameters())
    oracle_full_year_free = solve_full_year_oracle(
        data, StorageParameters(), force_terminal_soc=False
    )

    formal = data.dates >= FORMAL_START
    net_abs_day_mae = np.mean(
        np.abs(
            (data.load_kw - data.pv_kw)[formal]
            - (forecasts["load"] - forecasts["pv"])[formal]
        ),
        axis=1,
    )
    total_cost_ci = moving_block_bootstrap_ci(daily.total_cost_yuan.to_numpy())
    net_mae_ci = moving_block_bootstrap_ci(net_abs_day_mae)

    specified_daily = daily[daily.date.isin(SPECIFIED_DATES)].copy()
    comparison = pd.DataFrame(
        [
            {
                "model": "weekly_recent_baseline",
                "total_cost_yuan": baseline_daily.total_cost_yuan.sum(),
                "emergency_kwh": baseline_daily.emergency_kwh.sum(),
                "unused_plan_kwh": baseline_daily.unused_plan_kwh.sum(),
            },
            {
                "model": "model1_dynamic_ensemble",
                "total_cost_yuan": daily.total_cost_yuan.sum(),
                "emergency_kwh": daily.emergency_kwh.sum(),
                "unused_plan_kwh": daily.unused_plan_kwh.sum(),
            },
            {
                "model": "oracle_perfect_information_lower_bound",
                "total_cost_yuan": oracle_daily.total_cost_yuan.sum(),
                "emergency_kwh": oracle_daily.emergency_kwh.sum(),
                "unused_plan_kwh": oracle_daily.unused_plan_kwh.sum(),
            },
            {
                "model": "oracle_full_year_joint_lower_bound",
                "total_cost_yuan": oracle_full_year["formal_cost_yuan"],
                "emergency_kwh": 0.0,
                "unused_plan_kwh": 0.0,
            },
        ]
    )

    base_cost = float(baseline_daily.total_cost_yuan.sum())
    model_cost = float(daily.total_cost_yuan.sum())
    oracle_cost = float(oracle_daily.total_cost_yuan.sum())
    summary = {
        "run": {
            "seed": SEED,
            "formal_start": str(FORMAL_START.date()),
            "formal_days": int(formal.sum()),
            "runtime_seconds": None,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "lightgbm": lgb.__version__,
        },
        "prediction": {
            row["target"]: {
                key: float(row[key])
                for key in ["mae_kw", "rmse_kw", "r2", "nmae", "accuracy_1_minus_nmae", "bias_kw"]
            }
            for _, row in metrics[metrics.model.str.startswith("ensemble_")].iterrows()
        },
        "average_dynamic_weights": {
            target: {
                name: float(
                    np.mean(
                        [
                            weights.get(name, 0.0)
                            for day, weights in enumerate(forecasts[f"{target}_weights"])
                            if data.dates[day] >= FORMAL_START
                        ]
                    )
                )
                for name in ["baseline", "ridge", "lightgbm"]
            }
            for target in ["load", "pv"]
        },
        "dispatch": {
            "planned_purchase_kwh": float(daily.planned_purchase_kwh.sum()),
            "planned_cost_yuan": float(daily.planned_cost_yuan.sum()),
            "emergency_kwh": float(daily.emergency_kwh.sum()),
            "emergency_cost_yuan": float(daily.emergency_cost_yuan.sum()),
            "total_cost_yuan": model_cost,
            "unused_plan_kwh": float(daily.unused_plan_kwh.sum()),
            "curtailed_pv_kwh": float(daily.curtailed_pv_kwh.sum()),
            "min_soc_kwh": float(interval.soc_kwh.min()),
            "max_soc_kwh": float(interval.soc_kwh.max()),
            "days_with_emergency": int((daily.emergency_kwh > 1e-8).sum()),
        },
        "comparison": {
            "baseline_total_cost_yuan": base_cost,
            "model1_cost_saving_yuan": base_cost - model_cost,
            "model1_cost_saving_pct": (base_cost - model_cost) / base_cost,
            "oracle_lower_bound_yuan": oracle_cost,
            "gap_to_oracle_pct": (model_cost - oracle_cost) / oracle_cost,
            "oracle_full_year_joint_yuan": oracle_full_year["formal_cost_yuan"],
            "oracle_full_year_free_yuan": oracle_full_year_free["formal_cost_yuan"],
            "gap_to_full_year_oracle_pct": (model_cost - oracle_full_year["formal_cost_yuan"])
            / oracle_full_year["formal_cost_yuan"],
            "forecast_error_component_yuan": model_cost - oracle_cost,
            "myopia_component_yuan": oracle_cost - oracle_full_year["formal_cost_yuan"],
            "final_actual_soc_kwh": float(daily.end_soc_kwh.iloc[-1]),
        },
        "confidence": {
            **confidence,
            "mean_daily_cost_95pct_block_bootstrap_ci_yuan": total_cost_ci,
            "mean_daily_net_mae_95pct_block_bootstrap_ci_kw": net_mae_ci,
        },
        "overfitting": {
            "flagged_models": int(overfit.overfit_flag.sum()),
            "warning_models": int(overfit.generalization_warning.sum()),
            "rows": overfit.to_dict("records"),
        },
        "stability": {
            **stability,
            "safety_margin_mean_day_change_std_kw": float(
                pd.DataFrame(margins[formal]).mean(axis=1).diff().std()
            ),
        },
        "sensitivity": {
            "max_abs_cost_change_pct": float(sensitivity.cost_change_pct.abs().max()),
            "most_sensitive_case": str(
                sensitivity.loc[sensitivity.cost_change_pct.abs().idxmax(), "case"]
            ),
        },
        "robustness": {
            str(level): {
                "mean_total_cost_yuan": float(part.total_cost_yuan.mean()),
                "std_total_cost_yuan": float(part.total_cost_yuan.std(ddof=1)),
                "mean_emergency_kwh": float(part.emergency_kwh.mean()),
            }
            for level, part in robustness.groupby("perturbation_level")
        },
        "physical_validation": physical_validation,
    }
    summary["run"]["runtime_seconds"] = float(time.perf_counter() - started)

    daily.to_csv(RESULT_DIR / "daily_dispatch.csv", index=False, encoding="utf-8-sig")
    interval.to_csv(RESULT_DIR / "interval_dispatch.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(RESULT_DIR / "prediction_metrics.csv", index=False, encoding="utf-8-sig")
    overfit.to_csv(RESULT_DIR / "overfitting_diagnostic.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(RESULT_DIR / "sensitivity_analysis.csv", index=False, encoding="utf-8-sig")
    robustness.to_csv(RESULT_DIR / "robustness_analysis.csv", index=False, encoding="utf-8-sig")
    weight_rows = []
    for target in ["load", "pv"]:
        for day, weights in enumerate(forecasts[f"{target}_weights"]):
            if data.dates[day] >= FORMAL_START:
                weight_rows.append({"date": data.dates[day], "target": target, **weights})
    pd.DataFrame(weight_rows).fillna(0.0).to_csv(
        RESULT_DIR / "dynamic_weights.csv", index=False, encoding="utf-8-sig"
    )
    specified_daily.to_csv(RESULT_DIR / "specified_dates_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(RESULT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    with (RESULT_DIR / "frozen_numbers.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    plot_results(data, metrics, daily, interval, sensitivity, robustness, overfit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
