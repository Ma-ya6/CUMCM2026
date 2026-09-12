"""季节感知模块（纯数据驱动，不使用任何题目未给出的天文或气象信息）。

题目只给出电价、负载、光伏三类序列，**没有**给出地点、纬度、经度、日出日落
时刻、太阳高度角、温度或任何气象变量。因此本模块不引入任何天文假设，季节变化
只能通过**已发生序列自身**的统计性质被感知。共三条通道，全部在每天 0:00 仅依赖
$j<d$ 的历史数据：

1. **水平漂移** $\Delta_{\mathrm{lev}}$：最近短窗与更早参考窗的逐时段预测残差均值之差，
   归一化后反映负荷/光伏整体水平随季节迁移产生的系统性偏移（温度、辐照水平的
   季节变化都体现为这一项）。

2. **尺度漂移** $\Delta_{\mathrm{sca}}$：两窗残差均方根之比的对数。季节切换期
   （升温/降温、辐照快速变化）不确定性放大，该指标随之上升。

3. **光伏有效发电窗口**：由历史光伏曲线推断当日有效发电的起止时段。窗口的
   起止位置与长度随季节前移/后移、伸长/缩短，因此窗口长度是**昼长的数据代理**，
   窗口的移动方向就是季节推进方向。该量完全由历史光伏功率反推，不含天文假设。

三条通道合成 $[0,1]$ 的"季节应力" $\\sigma_d$，用于缩短记忆长度（让新季节更快
进入模型）、放大场景宽度、提高风险权重。$\\sigma_d$ 越大表示"当前正处于模型所
依赖的旧季节之外的工况"，模型应更快遗忘、更保守。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

# 光伏有效发电窗口推断的默认阈值（与功率单位为 kW 的口径一致）。
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
    # 由历史光伏曲线推断的有效发电窗口（时段序号，0 基）。
    pv_window_start: int
    pv_window_end: int
    pv_window_length: int
    # 昼长的数据代理：有效发电窗口长度换算为小时。
    daylight_proxy_hours: float
    # 窗口长度相对 14 天前的变化（小时），正为变长（趋向夏季），负为变短（趋向冬季）。
    daylight_proxy_trend_hours: float
    drift_level: float
    drift_scale: float
    stress: float
    drift_direction: float

    @property
    def season_label(self) -> str:
        """仅用于事后分组展示，不进入任何决策路径。"""
        month = self.date.month
        if month in (3, 4, 5):
            return "春季"
        if month in (6, 7, 8):
            return "夏季"
        if month in (9, 10, 11):
            return "秋季"
        return "冬季"


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

    def lengths_hours(self) -> np.ndarray:
        """已登记各日的窗口长度（小时），索引与已执行日一一对应。

        仅可对 $j<d$ 的取值使用，第 $d$ 天自身的窗口在 0:00 尚未观测到。
        """
        if not self.starts:
            return np.zeros(0, float)
        return (
            np.asarray(self.ends, float) - np.asarray(self.starts, float) + 1.0
        ) / 6.0

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

    第 $d$ 天的输出只依赖 $j<d$ 的残差，因此不泄露未来信息。水平漂移回答
    "预测是否系统性偏离了新季节的平均水平"，尺度漂移回答"残差幅度是否整体变大"。
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

    @property
    def n_observed(self) -> int:
        return len(self._residuals)

    def _windows(self) -> tuple[np.ndarray, np.ndarray] | None:
        cfg = self.config
        need = cfg.short_window + cfg.reference_window
        if len(self._residuals) < need:
            return None
        recent = np.asarray(self._residuals[-cfg.short_window :], float)
        older = np.asarray(self._residuals[-need : -cfg.short_window], float)
        return recent, older

    def stats(self) -> tuple[float, float, float]:
        """返回 (水平漂移, 尺度漂移, 应力)，raw 分量未平滑。"""
        windows = self._windows()
        if windows is None:
            return 0.0, 0.0, self._smoothed
        recent, older = windows
        cfg = self.config
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
        """把季节应力映射为缩短后的记忆长度。"""
        cfg = self.config
        factor = 1.0 / (1.0 + cfg.shrink_gain * self.stress())
        floor = max(minimum, int(round(base_window * cfg.min_window_ratio)))
        return int(np.clip(round(base_window * factor), floor, base_window))


class SeasonSensor:
    """把漂移检测器与光伏窗口跟踪器合成统一的季节状态输出。"""

    def __init__(
        self,
        n_slot: int,
        drift_config: DriftConfig | None = None,
    ) -> None:
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


def summarize_by_season(
    dates: pd.DatetimeIndex, frame: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    """按日历季节分组汇总，仅用于事后评价，不反馈进任何决策路径。"""
    tags = pd.Series([_season_label(pd.Timestamp(d)) for d in dates], index=frame.index)
    return frame.groupby(tags)[list(columns)].agg(["sum", "mean", "count"])


def _season_label(date: pd.Timestamp) -> str:
    month = pd.Timestamp(date).month
    if month in (3, 4, 5):
        return "春季"
    if month in (6, 7, 8):
        return "夏季"
    if month in (9, 10, 11):
        return "秋季"
    return "冬季"
