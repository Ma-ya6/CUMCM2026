"""季节感知的联合残差场景生成器（不使用天文信息）。

场景的作用是把"预测点值"扩展成"预测分布"，供两阶段 SAA-CVaR 优化使用。生成
过程中有三处体现周效应与季节效应，且全部因果：

1. **季节相似度用数据代理而非天文量**。历史日与目标日的季节相近程度由
   **光伏有效发电窗口长度**（昼长的数据代理）之差的核函数度量，而不是由
   赤纬、昼长公式或地点纬度度量。夏季与冬季的窗口长度差异大，因而互相
   抽到对方残差的概率低；相邻季节自然过渡。

2. **周型相似度**。残差的分布形态与星期几有关（周末的用电水平与形状都不同），
   因此同星期几、同周型（工作日/周末）的历史日获得更高抽样概率。这样抽出的
   残差与当日已施加的周内水平因子 $\\phi_{k(d)}$ 口径一致，不会把"周内结构"
   误当成"随机不确定性"。

3. **季节应力控制记忆长度与场景宽度**。季节应力 $\\sigma_d$ 高时缩短回看窗
   （旧季节样本退场更快），并按 $\\sigma_d$ 放大残差尺度，使模型在季节切换期
   对不确定性给出更宽的刻画，而不是沿用旧季节的窄分布。

因果性：候选日集合恒为 $\\{j: j<d\\}$；第 $d$ 天使用的窗口长度只取 $j<d$ 的
已观测值，目标日窗口取因果预测值。诊断字段只在事后计算，不反馈进决策。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScenarioSet:
    """一天的负载—光伏联合场景。数组单位为 kW。"""

    load_kw: np.ndarray
    pv_kw: np.ndarray
    probabilities: np.ndarray
    source_days: np.ndarray
    residual_scale: float
    effective_history_days: int
    season_stress: float


@dataclass
class SeasonAwareScenarioConfig:
    """场景生成的结构超参数（先验值，非由全年回测挑选）。"""

    base_lookback_days: int = 90
    min_lookback_days: int = 21
    recency_tau: float = 45.0
    # 同周型（工作日/周末）与同星期几的抽样加权。
    weekday_type_boost: float = 0.75
    same_weekday_boost: float = 0.35
    # 昼长代理相似度的核带宽（小时）与地板值。
    daylight_tau_hours: float = 3.0
    daylight_floor: float = 0.35
    # 季节应力把回看窗再压缩的强度。
    stress_lookback_gain: float = 2.0
    # 残差中心的近期因果偏差窗口。
    bias_window_days: int = 28
    bias_tau: float = 10.0
    # 残差尺度比值的裁剪与应力的放大强度。
    scale_clip: tuple[float, float] = (0.85, 1.25)
    stress_scale_gain: float = 0.15


def sampling_probabilities(
    dates: pd.DatetimeIndex,
    day: int,
    candidate_days: np.ndarray,
    daylight_proxy_hours: np.ndarray,
    target_daylight_hours: float,
    config: SeasonAwareScenarioConfig,
) -> np.ndarray:
    """近期、同星期属性与相近季节的历史残差获得更高抽样概率。"""
    age = day - candidate_days
    recency = np.exp(-age / config.recency_tau)

    target_weekday = dates[day].dayofweek
    source_weekday = dates[candidate_days].dayofweek.to_numpy()
    target_is_weekend = target_weekday >= 5
    source_is_weekend = source_weekday >= 5
    weekday_factor = np.where(source_is_weekend == target_is_weekend, 1.0 + config.weekday_type_boost, 1.0)
    weekday_factor = weekday_factor + config.same_weekday_boost * (source_weekday == target_weekday)

    source_hours = daylight_proxy_hours[candidate_days]
    season_factor = config.daylight_floor + (1.0 - config.daylight_floor) * np.exp(
        -np.abs(source_hours - target_daylight_hours) / config.daylight_tau_hours
    )

    weights = recency * weekday_factor * season_factor
    return weights / weights.sum()


def generate_season_aware_scenarios(
    *,
    dates: pd.DatetimeIndex,
    day: int,
    load_forecast_kw: np.ndarray,
    pv_forecast_kw: np.ndarray,
    load_errors_kw: np.ndarray,
    pv_errors_kw: np.ndarray,
    n_scenarios: int,
    rng: np.random.Generator,
    daylight_proxy_hours: np.ndarray,
    target_daylight_hours: float,
    season_stress: float,
    config: SeasonAwareScenarioConfig | None = None,
) -> ScenarioSet:
    """只用目标日以前的历史生成联合残差场景，并按季节应力在线校准。"""
    cfg = config or SeasonAwareScenarioConfig()
    if n_scenarios < 1:
        raise ValueError("n_scenarios 必须为正整数")

    if day == 0:
        load = np.repeat(load_forecast_kw[None, :], n_scenarios, axis=0)
        pv = np.repeat(pv_forecast_kw[None, :], n_scenarios, axis=0)
        return ScenarioSet(
            load, pv, np.full(n_scenarios, 1.0 / n_scenarios), np.zeros(n_scenarios, int), 1.0, 0, 0.0
        )

    net_errors = load_errors_kw[:day] - pv_errors_kw[:day]
    daily_scale = np.sqrt(np.mean(np.square(net_errors), axis=1))

    # 应力越大，旧季节样本越快退场。
    lookback = int(
        np.clip(
            round(cfg.base_lookback_days / (1.0 + cfg.stress_lookback_gain * float(season_stress))),
            cfg.min_lookback_days,
            cfg.base_lookback_days,
        )
    )
    start = max(0, day - lookback)
    candidate_days = np.arange(start, day, dtype=int)
    probabilities = sampling_probabilities(
        dates, day, candidate_days, daylight_proxy_hours, target_daylight_hours, cfg
    )

    # 系统重采样：在保持概率排序的前提下降低蒙特卡洛波动。
    positions = (rng.random() + np.arange(n_scenarios)) / n_scenarios
    chosen = candidate_days[np.searchsorted(np.cumsum(probabilities), positions, side="right")]
    chosen = np.clip(chosen, candidate_days[0], candidate_days[-1])

    # 场景中心跟随最近 bias_window_days 的因果偏差估计。
    recent_start = max(0, day - cfg.bias_window_days)
    age = day - np.arange(recent_start, day)
    recent_weights = np.exp(-age / cfg.bias_tau)
    recent_weights /= recent_weights.sum()
    target_load_bias = recent_weights @ load_errors_kw[recent_start:day]
    target_pv_bias = recent_weights @ pv_errors_kw[recent_start:day]
    sampled_load_error = load_errors_kw[chosen].copy()
    sampled_pv_error = pv_errors_kw[chosen].copy()
    sampled_load_error += target_load_bias - sampled_load_error.mean(axis=0)
    sampled_pv_error += target_pv_bias - sampled_pv_error.mean(axis=0)

    # 尺度校准：近期尺度 / 候选加权尺度，再按季节应力放大，反映切换期不确定性。
    recent_window = min(7, day)
    recent_scale = float(np.median(daily_scale[day - recent_window : day]))
    candidate_scale = float(np.average(daily_scale[candidate_days], weights=probabilities))
    maturity = float(np.clip(np.sqrt(len(candidate_days) / 60.0), 0.75, 1.0))
    scale_ratio = float(
        np.clip(recent_scale / max(candidate_scale, 1e-6), cfg.scale_clip[0], cfg.scale_clip[1])
    )
    residual_scale = float(
        np.clip(
            maturity * scale_ratio * (1.0 + cfg.stress_scale_gain * float(season_stress)),
            cfg.scale_clip[0],
            cfg.scale_clip[1] * (1.0 + cfg.stress_scale_gain),
        )
    )

    load = np.maximum(load_forecast_kw[None, :] + residual_scale * sampled_load_error, 0.0)
    pv = np.maximum(pv_forecast_kw[None, :] + residual_scale * sampled_pv_error, 0.0)
    return ScenarioSet(
        load_kw=load,
        pv_kw=pv,
        probabilities=np.full(n_scenarios, 1.0 / n_scenarios),
        source_days=chosen,
        residual_scale=residual_scale,
        effective_history_days=len(candidate_days),
        season_stress=float(season_stress),
    )


def scenario_diagnostic_row(
    *,
    date: pd.Timestamp,
    scenarios: ScenarioSet,
    actual_load_kw: np.ndarray,
    actual_pv_kw: np.ndarray,
) -> dict:
    """事后诊断只用于评价，不反馈给当日场景生成或优化。"""
    net = scenarios.load_kw - scenarios.pv_kw
    actual_net = actual_load_kw - actual_pv_kw
    lower = np.quantile(net, 0.05, axis=0)
    upper = np.quantile(net, 0.95, axis=0)
    return {
        "date": date,
        "n_scenarios": len(scenarios.probabilities),
        "unique_source_days": int(np.unique(scenarios.source_days).size),
        "net_mean_kw": float(net.mean()),
        "net_std_kw": float(net.std()),
        "net_90pct_coverage": float(np.mean((actual_net >= lower) & (actual_net <= upper))),
        "actual_net_mae_from_scenario_mean_kw": float(np.mean(np.abs(actual_net - net.mean(axis=0)))),
        "residual_scale": float(scenarios.residual_scale),
        "effective_history_days": int(scenarios.effective_history_days),
        "season_stress": float(scenarios.season_stress),
    }
