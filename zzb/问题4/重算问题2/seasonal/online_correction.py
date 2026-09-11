"""在线校正管线：把基础预测器输出依次通过季节通道与周效应通道。

**级联结构**。基础集成预测器（``deterministic_baseline`` 的 baseline/ridge/lightgbm
加权组合）只负责水平与日内形状的骨架。其输出 $\\hat p^{(0)}_{d,t}$ 依次经过四级
校正，每一级只修正**上一级残留的、属于自己那一类的误差**，因而各通道可正交识别：

$$
\\hat p^{(1)}_{d,t}=\\hat p^{(0)}_{d,t}\\cdot \\mathbb 1\\{t\\in W_d\\},
\\qquad
\\hat p^{(2)}_{d,t}=\\hat p^{(1)}_{d,t}\\, \\lambda_d,
\\qquad
\\hat p^{(3)}_{d,t}=\\hat p^{(2)}_{d,t}\\, \\phi_{k(d)},
\\qquad
\\hat p^{(4)}_{d,t}=\\bigl[\\hat p^{(3)}_{d,t}+\\psi_{g(d),t}E^{(3)}_d\\bigr]_+ .
$$

- 第 1 级 $W_d$ 是**光伏有效发电窗口**（仅光伏），由历史光伏曲线推断，是昼长季节
  变化的纯数据代理；
- 第 2 级 $\\lambda_d$ 是**季节水平因子**，记忆长度 $W_d^{\\mathrm{mem}}$ 由季节应力
  $\\sigma_d$ 决定：$\\sigma_d$ 越大（越可能已进入新季节），窗口越短，水平因子越快
  跟上新水平。这是"感知季节变化并适应新季节"的执行机构；
- 第 3 级 $\\phi_{k}$ 是**周内水平因子**，全周均值为 1，只承担周内哪几天偏高；
- 第 4 级 $\\psi_{g,t}$ 是**周型形状轮廓**，承担工作日/周末的日内曲线形状差异。

**正交性**。$\\lambda_d$ 由全部星期几混合估计（日能量比值的指数加权均值），
$\\phi_k$ 则以"相对全周均值的偏离"形式给出，二者在期望意义上互不重叠：前者吸收
季节与整体水平漂移，后者吸收周内结构。因此不会出现两个通道争夺同一份误差、
互相放大。

**因果性**。$\\lambda_d,\\phi_k,\\psi_{g,\\cdot}$ 在第 $d$ 天 0:00 只由 $j<d$ 的
已实现比值累计得到；``update`` 只应在第 $d$ 天执行完毕、实际值已知后调用，
且调用顺序严格为"先 ``apply``、后 ``update``"。日内在同一日内不做任何再调整。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .season_state import SeasonSensor, SeasonState
from .weekly_effect import WeeklyEffectConfig, WeeklyEffectModel


@dataclass
class CorrectionConfig:
    """在线校正管线的超参数（结构性先验，非由全年回测挑选）。"""

    # ---- 季节水平通道 ----
    use_level: bool = True
    level_base_window: int = 21
    level_min_window: int = 5
    level_shrink_kappa: float = 3.0
    level_clip: float = 0.15
    level_strength: float = 1.0
    # 指数加权的相对时间常数：tau = window / ew_divisor。
    ew_divisor: float = 3.0
    # ---- 光伏有效发电窗口 ----
    use_pv_window: bool = True
    # ---- 周效应通道 ----
    use_weekly_effect: bool = True
    weekly: WeeklyEffectConfig = field(default_factory=WeeklyEffectConfig)


@dataclass(frozen=True)
class CorrectionOutcome:
    """一日校正的可追溯输出，用于冻结结果与事后归因。"""

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

    @property
    def n_observed(self) -> int:
        return len(self._ratios)

    def factor(self, window_days: int) -> float:
        """只用最近 ``window_days`` 个已实现比值估计水平因子。"""
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

    # ------------------------------------------------------------------ 内部
    def _masked(self, base_prediction: np.ndarray, date: pd.Timestamp) -> np.ndarray:
        if self.target == "pv" and self.config.use_pv_window:
            return self.sensor.pv_window.apply_window(base_prediction)
        return np.asarray(base_prediction, float)

    def _stage_predictions(
        self, base_prediction: np.ndarray, date: pd.Timestamp
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int]:
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

    # ------------------------------------------------------------------ 对外
    def apply(self, base_prediction: np.ndarray, date: pd.Timestamp) -> CorrectionOutcome:
        """第 ``date`` 天 0:00 调用：只用历史状态生成本日校正预测。"""
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
        """第 ``day_index`` 天执行完毕后调用：并入当日已实现观测。

        必须严格在 ``apply`` 之后调用；``day_index`` 用于周效应先验的窗口判定。
        """
        base = np.asarray(base_prediction, float)
        actual = np.asarray(actual, float)
        masked, after_level, after_weekly, _, _, _ = self._stage_predictions(base, date)

        energy_actual = float(actual.sum() * self.dt_hours)
        energy_masked = float(masked.sum() * self.dt_hours)
        energy_after_level = float(after_level.sum() * self.dt_hours)

        # 第 1→2 级：季节水平通道吸收"整日水平"偏差。
        if self.config.use_level and energy_masked > 1e-9:
            self.level.update(energy_actual / energy_masked)

        # 第 2→3 级：周内水平因子吸收"该星期几"的残余水平偏差。
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
