"""问题4的电价因果预测：与负载/光伏同结构的三模型动态加权集成。

问题4中电价与负载同为**不可预知量**：第 d 天 0:00 制定当天计划时，当天 144 个
时段的电价尚未发生，只能用 j<d 的实际电价预测。本模块复用问题2既有预测器
（近期加权基线 + 岭回归 + 浅层 LightGBM，按近期指数加权误差动态组合），
不新增任何待标定参数；季节应力直接取负载/光伏预测已算出的逐日取值。

与负载/光伏的唯一差别：电价不接季节校正管线（光伏有效窗口与日电量水平因子
是光伏/负载专用的物理量），仅使用基础集成。

第 0 天用附件1的单条电价曲线冷启动，与 ``initial_load``/``initial_pv`` 同构。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import deterministic_baseline as db
from seasonal import CausalEnsemble, ForecastConfig


def forecast_metrics(actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> dict:
    """正式评价期的电价预测精度。"""
    y = actual[mask]
    p = predicted[mask]
    err = p - y
    mae = float(np.abs(err).mean())
    return {
        "mae_yuan_per_kwh": mae,
        "rmse_yuan_per_kwh": float(np.sqrt(np.mean(err**2))),
        "nmae": mae / float(np.mean(np.abs(y))),
        "bias_yuan_per_kwh": float(err.mean()),
        "mean_actual_yuan_per_kwh": float(y.mean()),
    }


def generate_causal_price_forecast(
    data: db.DataBundle,
    diagnostics: pd.DataFrame,
    config: ForecastConfig | None = None,
    *,
    verbose: bool = False,
) -> dict:
    """逐日推进生成全年因果电价预测。

    ``diagnostics`` 是 ``generate_causal_forecasts`` 返回的逐日诊断表，其中的
    ``season_stress`` 只由 j<day 的已实现残差构成，因此在第 day 天决策前可知。
    """
    cfg = config or ForecastConfig()
    stress_by_day = (
        diagnostics.sort_values("day_index")["season_stress"].to_numpy(float)
    )
    n_days = len(data.dates)
    if len(stress_by_day) != n_days:
        raise ValueError("诊断表长度与日期数不一致")

    ensemble = CausalEnsemble(
        values=np.array(data.price, dtype=float, copy=True),
        dates=data.dates,
        initial=np.array(data.initial_price, dtype=float, copy=True),
        # 电价具有明显的周内形状与近期水平漂移，与负载基线同构。
        baseline_fn=db.weekly_load_baseline,
        config=cfg,
        window_fn=None,
    )

    forecast = np.zeros_like(data.price)
    for day in range(n_days):
        forecast[day] = ensemble.predict(day, float(stress_by_day[day]))
        ensemble.observe(day)
        if verbose and (day + 1) % 60 == 0:
            print(f"  电价预测进度：{day + 1}/{n_days} 天", flush=True)

    mask = np.asarray(data.dates >= db.FORMAL_START)
    return {
        "price": forecast,
        "weights": ensemble.weights_by_day,
        "refit_events": ensemble.refit_events,
        "metrics": forecast_metrics(data.price, forecast, mask),
    }
