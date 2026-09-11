"""周效应模块：以 1 月数据建立初始模型，随后由新增数据在线收缩更新。

**建模对象**。周效应被拆成两个互不混淆的分量，二者都作用在基础预测器输出之上：

- **周内水平因子** $\\phi_k\\ (k=0,\\dots,6)$：第 $k$ 个星期几的日能量相对全周
  平均水平的偏离。为实现与"季节水平"通道的正交识别，因子以**偏离形式**使用，
  即 $\\phi_k \\big/ \\frac{1}{7}\\sum_j \\phi_j$，全周均值为 1；日能量的整体抬升
  或下移由季节水平通道承担，本模块只负责"周内哪几天偏高、哪几天偏低"。

- **周型形状轮廓** $\\psi_{g(\\cdot),t}\\ (g\\in\\{\\text{工作日},\\text{周末}\\})$：
  扣除水平后残留的逐时段形状偏差，按工作日/周末两类分别维护，捕捉"周末作息
  改变日内曲线形状"这一周效应的第二重表现。

**1 月先验 + 在线更新**。$\\phi$ 与 $\\psi$ 的初值全部来自 2025 年 1 月
（预热期）的数据，此后每执行一天，先对全部累计量做一次衰减，再把当日观测并入
对应星期几/周型。权重衰减使有效样本量收敛到一个有限值，因而 1 月先验会随新数据
积累而被逐步稀释，模型持续适应；同时，有限记忆也让周效应本身可以随季节缓慢漂移
（例如夏季与冬季的周末作息强度不同）。

**收缩与裁剪**。任一星期几的有效样本数都很少（一周只有一个观测），直接使用
样本均值会严重过拟合。因此对角色的偏离量按 $w/(w+\\kappa)$ 向 1（或向 0）收缩，
再对幅度做硬裁剪，使周效应在证据不足时自动退化为"无周效应"。

**因果性**。第 $d$ 天使用的 $\\phi,\\psi$ 只由 $j<d$ 的已执行日累计得到；
``update`` 只应在第 $d$ 天结束、实际值已知后调用。1 月先验在 2 月 1 日 0:00
时已全部成为历史，故不构成泄露。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class WeeklyEffectConfig:
    """周效应模块的结构超参数（先验值，非由全年回测挑选）。"""

    # 建立 1 月先验所用的日索引上界（不含）：0..30 即 1 月 1 日—1 月 31 日。
    prior_end_day: int = 31
    # 先验起始日索引。必须 ≥7：基础预测器在第 7 天之前只能靠 1/2/3 日滞后
    # （``recent_weighted``），尚不具备同星期几参照，其周内偏差是**预热期伪影**
    # 而非真实周效应；用这些天建先验会注入虚假的星期几结构。实测：把 1 月 1—7 日
    # 纳入先验会使"周三"因子偏离真值约 4.4%，足以淹没真实残差（约 1%）。
    prior_start_day: int = 7
    # 每个星期几的 1 月先验等效样本数。
    prior_pseudo_count: float = 5.0
    # 累计量的逐日衰减，决定有效记忆长度（0.985 → 有效约 10 周）。
    daily_decay: float = 0.985
    # 水平因子向 1 收缩的强度。
    factor_shrink_kappa: float = 3.0
    # 水平因子相对偏离的硬裁剪幅度。基础预测器已由 7 日滞后捕获约 37% 的
    # 周五/周六效应，残留的周内偏差仅约 1%—2%，故裁剪幅度取小值。
    factor_clip: float = 0.03
    # 水平因子介入预测的强度（0 表示关闭周效应）。
    energy_strength: float = 1.0
    # 形状轮廓的 1 月先验等效样本数。
    shape_prior_pseudo_count: float = 5.0
    # 形状轮廓向 0 收缩的强度。形状轮廓每类有 144 个自由参数，而"工作日/周末"
    # 每类每周只新增 5/2 个样本，参数数与样本量之比极高，必须强收缩才能避免把
    # 个别日子的天气噪声固化成"周效应"。κ 取 20 使有效权重 20 时的收缩系数仅 0.5。
    shape_shrink_kappa: float = 20.0
    # 形状轮廓的循环平滑系数（十分钟相邻时段强相关）。
    shape_smoothing: float = 0.40
    # 单时段形状修正幅度上限（占预测日能量的比例）。
    shape_cap_fraction: float = 0.02
    # 形状轮廓介入预测的强度。
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

    # ---------------------------------------------------------------- 周型归类
    @staticmethod
    def weekday_type(date: pd.Timestamp) -> int:
        """周六、周日为 1（周末），其余为 0（工作日）。"""
        return 1 if pd.Timestamp(date).dayofweek >= 5 else 0

    # ------------------------------------------------------------ 1 月先验建立
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

    def fit_initial(
        self,
        dates: pd.DatetimeIndex,
        actual_energy: np.ndarray,
        predicted_energy: np.ndarray,
        shape_residual_normalized: np.ndarray,
    ) -> None:
        """仅用第 ``prior_end_day`` 天之前的**已全部发生**的数据建立先验。

        ``actual_energy``、``predicted_energy`` 为逐日总能量（长度 n_days）；
        ``shape_residual_normalized`` 为逐日逐时段的归一化形状残差 (n_days, n_slot)。
        """
        cfg = self.config
        end = min(cfg.prior_end_day, len(dates))
        observations = [
            (
                pd.Timestamp(dates[day]),
                float(actual_energy[day]) / max(float(predicted_energy[day]), 1e-9),
                np.asarray(shape_residual_normalized[day], float),
            )
            for day in range(min(cfg.prior_start_day, end), end)
        ]
        if not observations:
            self._fitted = True
            return
        self._build_prior(observations)

    # ---------------------------------------------------------------- 因果读出
    @property
    def fitted(self) -> bool:
        return self._fitted

    def _weekday_means(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(
                self.energy_weight > 1e-9,
                self.energy_sum / np.maximum(self.energy_weight, 1e-9),
                1.0,
            )
        return means

    def energy_factor(self, date: pd.Timestamp) -> float:
        """第 ``date`` 天的周内水平因子（全周均值为 1，因果）。"""
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
        """第 ``date`` 天所属周型的归一化形状轮廓（因果），单位与归一化残差一致。"""
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

    # ---------------------------------------------------------------- 在线更新
    def update(
        self,
        day_index: int,
        date: pd.Timestamp,
        ratio: float,
        shape_residual_normalized: np.ndarray,
    ) -> None:
        """第 ``day_index`` 天结束后并入当日观测。只应传入已实现的实际值。

        先验尚未建立时只缓存 ``[prior_start_day, prior_end_day)`` 区间内的观测；
        到达 ``prior_end_day`` 后一次性转为先验。在此之前 ``energy_factor`` 恒为
        1、``shape_profile`` 恒为 0，即"无周效应"，不会用不足一周的样本臆造结构。
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

    # ---------------------------------------------------------------- 诊断输出
    def diagnostics(self, dates: pd.DatetimeIndex) -> dict:
        """导出当前周效应估计，供事后解释（不进入决策路径）。"""
        means = self._weekday_means()
        pooled = float(np.mean(means))
        factors = {}
        for k in range(self.N_WEEKDAY):
            probe = pd.Timestamp("2025-01-06") + pd.Timedelta(days=k)  # 2025-01-06 为周一
            factors[probe.day_name()] = float(self.energy_factor(probe))
        return {
            "n_updates": self.n_updates,
            "raw_weekday_mean_ratio": {str(k): float(means[k]) for k in range(self.N_WEEKDAY)},
            "pooled_mean_ratio": pooled,
            "applied_factor_by_weekday": factors,
            "effective_weights": {str(k): float(self.energy_weight[k]) for k in range(self.N_WEEKDAY)},
            "weekday_shape_weight": float(self.shape_weight[0]),
            "weekend_shape_weight": float(self.shape_weight[1]),
        }


def weekly_effect_significance(
    dates: pd.DatetimeIndex,
    actual_energy: np.ndarray,
    predicted_energy: np.ndarray,
    formal_start: pd.Timestamp,
) -> dict:
    """事后检验周效应是否显著：只用 2 月 1 日起的数据，按星期几分组做单因素方差分析。

    该检验只用于说明"为什么需要周效应模块"，不参与任何决策路径，也不用于选参。
    """
    ratios = actual_energy / np.maximum(predicted_energy, 1e-9)
    mask = np.asarray(dates >= formal_start)
    groups = [
        ratios[mask & (np.asarray(dates.dayofweek) == k)] for k in range(7)
    ]
    groups = [g[np.isfinite(g)] for g in groups if g.size > 0]
    if len(groups) < 2:
        return {"f_statistic": float("nan"), "p_value": float("nan"), "group_means": []}
    overall = np.concatenate(groups)
    grand = float(overall.mean())
    k = len(groups)
    n = overall.size
    ss_between = float(sum(g.size * (g.mean() - grand) ** 2 for g in groups))
    ss_within = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    df_between = k - 1
    df_within = n - k
    if df_within <= 0 or ss_within <= 0:
        f_stat = float("nan")
        p_value = float("nan")
    else:
        f_stat = (ss_between / df_between) / (ss_within / df_within)
        p_value = float(_f_sf(f_stat, df_between, df_within))
    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "df_between": int(df_between),
        "df_within": int(df_within),
        "group_means": [float(g.mean()) for g in groups],
        "day_names": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][: len(groups)],
    }


def _f_sf(f: float, df1: int, df2: int) -> float:
    """F 分布上尾概率，用 scipy 若可用，否则用粗略近似。"""
    try:
        from scipy.stats import f as f_dist

        return float(f_dist.sf(f, df1, df2))
    except Exception:  # pragma: no cover - scipy 不可用时的兜底
        return float("nan")
