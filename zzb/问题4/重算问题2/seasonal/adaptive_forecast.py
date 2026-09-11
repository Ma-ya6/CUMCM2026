"""因果自适应预测器：基础集成 + 自适应训练窗 + 周/季节级联校正。

本模块把问题2模型2的预测器改造成一个**逐日推进的因果闭环**：

```
第 d 天 0:00
  ├─ 季节感知器给出 σ_d 与自适应记忆长度 W_d^mem
  ├─ 基础集成按 W_d^train = f(σ_d) 取训练窗，输出 p̂^(0)
  ├─ 级联校正（光伏窗口 / 季节水平 / 周内水平 / 周型形状）→ p̂^(4)
  └─ 用 p̂^(4) 进入随机优化制定当日计划
第 d 天 24:00
  └─ 用当日实际值更新：集成误差权重、校正管线、季节感知器
```

**为什么训练窗要跟着 σ_d 走**。若季节已切换而训练窗仍是固定的 180 天，模型会
被上一个季节的样本主导，预测器表现为"慢半拍"。σ_d 上升时把训练窗、水平记忆窗与
场景回看窗一并缩短，使模型在数日内完成向新季节的迁移；σ_d 回落时窗口自动恢复，
避免长期使用过短窗口而放大噪声。这是"感知季节变化"与"适应新季节"的同一套机制
在两个尺度上的体现。

**无天文信息**。本模块不调用 ``deterministic_baseline.solar_geometry``，也不使用
纬度、昼长、太阳高度角等题目未给出的量；季节完全由历史序列的统计漂移与光伏
有效发电窗口的移动来刻画。``tests/test_no_leakage.py`` 中的天文信息审计会断言这一点。

**因果性**。第 d 天的每一项预测只依赖 $j<d$ 的实际值与已生成的预测值；没有任何
跨日的前瞻、插值或全序列平滑。``tests/test_no_leakage.py`` 用"未来扰动不变性"
（改动 $j\\ge k$ 的实际值后，$j<k$ 的预测必须逐位不变）逐日机械验证。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import deterministic_baseline as base
from .online_correction import CorrectionConfig, CausalCorrectionPipeline, build_pipelines
from .season_state import DriftConfig, SeasonSensor


@dataclass
class ForecastConfig:
    """因果自适应预测器的结构与超参数（先验值，非由全年回测挑选）。"""

    # ---- 基础集成 ----
    ridge_min_day: int = 28
    lgb_min_day: int = 60
    ridge_refit_days: int = 7
    lgb_refit_days: int = 14
    base_window_days: int = 180
    window_min_days: int = 45
    # 关闭后训练窗恒为 base_window_days（用于消融：验证自适应窗的贡献）。
    use_adaptive_window: bool = True
    # σ 上升超过该幅度时立即重拟合，使季节切换不被固定重拟合周期拖延。
    stress_refit_jump: float = 0.30
    # ---- 级联校正 ----
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
        window_fn,
    ) -> None:
        self.values = values
        self.dates = dates
        self.initial = initial
        self.baseline_fn = baseline_fn
        self.config = config
        self.window_fn = window_fn  # 输入 day，返回当日的训练窗天数（因果）
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
                x_train, y_train = base.training_matrix(self.values, self.dates, day, window)
                self._ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                self._ridge.fit(x_train, y_train)
                self._last_fit_day["ridge"] = day
                self._stress_at_fit["ridge"] = current_stress
            candidates["ridge"] = self._ridge.predict(base.day_features(self.values, self.dates, day))

        if day >= cfg.lgb_min_day:
            if self._lgb is None or day - self._last_fit_day["lightgbm"] >= cfg.lgb_refit_days:
                self._lgb, _ = base.fit_lightgbm_causal(
                    self.values, self.dates, day, window_days=window
                )
                self._last_fit_day["lightgbm"] = day
                self._stress_at_fit["lightgbm"] = current_stress
            candidates["lightgbm"] = self._lgb.predict(base.day_features(self.values, self.dates, day))

        for name in candidates:
            candidates[name] = np.maximum(np.asarray(candidates[name], float), 0.0)

        weights = base.adaptive_error_weights(candidates.keys(), self._error_history)
        prediction = np.zeros(base.N_SLOT, float)
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
    data,
    config: ForecastConfig | None = None,
    *,
    verbose: bool = False,
) -> dict:
    """逐日推进生成全年的因果负载/光伏预测，并返回逐日诊断。

    返回值包含 ``load``、``pv`` 两条预测序列、``candidates``、``weights``，
    以及一张逐日诊断表 ``diagnostics``（季节状态与各通道因子）。
    """
    cfg = config or ForecastConfig()
    # 防御性拷贝：本函数绝不就地改写调用方传入的实际值数组。
    # 这既保证"同一次运行的结果只由输入决定"，也保证多次调用之间互不污染——
    # 否则一次运行的中间写入会改变下一次运行的输入，使因果性检验失去意义。
    data = replace(
        data,
        load_kw=np.array(data.load_kw, dtype=float, copy=True),
        pv_kw=np.array(data.pv_kw, dtype=float, copy=True),
        price=np.array(data.price, dtype=float, copy=True),
    )
    n_days = len(data.dates)
    sensor = SeasonSensor(base.N_SLOT, cfg.drift)
    pipelines = build_pipelines(
        n_slot=base.N_SLOT, dt_hours=base.DT, sensor=sensor, config=cfg.correction
    )

    load_ensemble = CausalEnsemble(
        values=data.load_kw,
        dates=data.dates,
        initial=data.initial_load,
        baseline_fn=base.weekly_load_baseline,
        config=cfg,
        window_fn=None,
    )
    pv_ensemble = CausalEnsemble(
        values=data.pv_kw,
        dates=data.dates,
        initial=data.initial_pv,
        baseline_fn=base.recent_pv_baseline,
        config=cfg,
        window_fn=None,
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
                "drift_level": float(outcome_load.season_state.drift_level),
                "drift_scale": float(outcome_load.season_state.drift_scale),
                "drift_direction": float(outcome_load.season_state.drift_direction),
                "pv_window_start": int(outcome_pv.season_state.pv_window_start),
                "pv_window_end": int(outcome_pv.season_state.pv_window_end),
                "daylight_proxy_hours": float(outcome_pv.season_state.daylight_proxy_hours),
                "daylight_proxy_trend_hours": float(outcome_pv.season_state.daylight_proxy_trend_hours),
                "load_weight_baseline": float(load_ensemble.weights_by_day[-1].get("baseline", np.nan)),
                "load_weight_ridge": float(load_ensemble.weights_by_day[-1].get("ridge", np.nan)),
                "load_weight_lightgbm": float(load_ensemble.weights_by_day[-1].get("lightgbm", np.nan)),
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

    forecasts["candidates"] = {"load": load_ensemble.weights_by_day}
    forecasts["weights"] = {
        "load": load_ensemble.weights_by_day,
        "pv": pv_ensemble.weights_by_day,
    }
    forecasts["net"] = forecasts["load"] - forecasts["pv"]
    forecasts["net_error"] = (data.load_kw - data.pv_kw) - forecasts["net"]
    forecasts["diagnostics"] = pd.DataFrame(diagnostic_rows)
    forecasts["sensor"] = sensor
    forecasts["pipelines"] = pipelines
    return forecasts
