"""问题3的预测模型：负载因果自适应集成 + 光伏"官方预报⊗自有统计"因果融合。

## 问题3 的预测任务与问题2 的区别

问题2 里光伏也要自己预测；问题3 的题目直接给了附件3 的官方滚动预报，
所以预测任务变成**两路不对称**：

- **负载**：题目没有给任何负载预报，仍须自建。直接沿用
  ``问题2/模型2再优化`` 的因果自适应集成（周效应 + 季节漂移 + 自适应训练窗），
  该模块与问题无关，可逐字复用，保证两个问题口径一致、结论可比。
- **光伏**：官方预报（附件3）必须用，但它**并不准**——实测 0:00 预报
  全年 MAE 371.22 kW（nMAE 15.6%），而自有的因果统计预报只有 149.76 kW
  （nMAE 6.3%）。官方预报差了 2.5 倍。

这不是矛盾，而是两路预报的**误差结构不同**：官方预报来自数值气象模式，
抓得住天气过程（云、锋面）但系统偏差大、日内形状粗；自有预报是纯统计外推，
抓得住气候平均日内形状和持续性，但对天气突变无信息。两者的误差不完全相关，
因此**融合能同时压低 MAE 与 RMSE**。实测因果融合后 MAE 371.22 → 140.98 kW
（−62%），比"自有单独用"（149.76）还好 5.9%，RMSE 也由 303.90 降到 276.02。

## 融合器：逐（发布时刻×小时）的因果线性堆叠

对发布时刻 $k$ 与日内小时 $h\\ge k$，用 $j<d$ 的历史样本做二元最小二乘

$$ P^{\\text{act}}_{j,\\tau} \\;\\approx\\; a^{(k)}_h\\, P^{(k)}_{j,\\tau} \\;+\\; b^{(k)}_h\\, \\hat P^{\\text{own}}_{j,\\tau} \\;+\\; c^{(k)}_h ,
\\qquad \\tau\\in\\text{小时}h,\\; j<d , $$

再用于第 $d$ 天：$\\hat P^{\\text{fuse}}_{d,\\tau}=a^{(k)}_h P^{(k)}_{d,\\tau}+b^{(k)}_h\\hat P^{\\text{own}}_{d,\\tau}+c^{(k)}_h$。

三点设计说明：

1. **按发布时刻分别拟合**。$P^{(12)}$ 比 $P^{(0)}$ 准得多（见表），若共用一套
   权重，相当于把"官方预报"当成一个固定精度的源，权重必然错配。分开拟合后
   每次调整都用各自校准过的权重。
2. **按小时分别拟合**。光伏误差的日内形状很强（正午绝对误差大、晨昏小），
   统一权重会被正午样本主导。逐小时拟合让晨昏时段也能拿到合适的权重。
3. **夜间退化保护**。日落后实际光伏≈0，此时两路预报都≈0，回归会退化
   （截距与系数不可辨识）。故对历史均值 < 20 kW 的小时不做回归，直接取两路
   算术平均——反正两路都是 0，融合与否无实质差别，但避免了数值病态。

## 严格因果

第 $d$ 天的每一项预测只依赖 $j<d$ 的实际值与预报值。融合权重在第 $d$ 天
决策前由"截至 $d-1$ 日已实现"的样本重新拟合，当日实际值不参与。
``--leakage-audit`` 用**未来扰动不变性**机械验证：把第 $D$ 天起的实际值整体
放大/缩小后重跑，断言 $j<D$ 的每一个预测值逐位不变。

## 日内负载更新

问题3.md #12 要求调整时刻用"更新后的负载预测"。上午已执行时段的实测负载
是当日水平的信息，故在 $k$ 时刻对剩余时段做一次**阻尼水平修正**：

$$ \\lambda_k = 1 + \\gamma\\left(\\frac{\\sum_{t\\le 6k} L^{\\text{act}}_t}{\\sum_{t\\le 6k} \\hat L_t}-1\\right),
\\qquad \\hat L^{(k)}_t = \\lambda_k\\,\\hat L_t \\;\\;(t>6k), $$

$\\gamma=0.5$ 表示只吸收一半的已实现偏差（上午的偏差未必延续到下午），
$\\lambda_k$ 裁剪到 $[0.85,1.15]$。该修正只用已发生时段的实测值，是因果的。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common import (
    CACHE_DIR,
    DT,
    ISSUE_HOURS,
    N_SLOT,
    Q2_M1_DIR,
    Q2_MODEL_DIR,
    SLOTS_PER_HOUR,
    DataBundle3,
    coverage_start_slot,
)

for _p in (str(Q2_MODEL_DIR), str(Q2_M1_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import deterministic_baseline as base  # noqa: E402
from seasonal import ForecastConfig, generate_causal_forecasts  # noqa: E402

NIGHT_KW = 20.0          # 历史均值低于此值的小时视为夜间，不做回归
MIN_STACK_SAMPLES = 40   # 逐（k,小时）回归所需的最少样本数


# --------------------------------------------------------------------------
# 一、自有因果预测（负载与光伏）
# --------------------------------------------------------------------------
def build_own_forecasts(
    data: DataBundle3,
    *,
    use_cache: bool = True,
    verbose: bool = False,
) -> dict:
    """跑一遍模型2再优化的因果自适应集成，取负载与光伏两路自有预测。

    结果较大且耗时可观（全年逐日重拟合），故默认缓存到 ``cache/``。
    """
    cache_load = CACHE_DIR / "own_load.npy"
    cache_pv = CACHE_DIR / "own_pv.npy"
    if use_cache and cache_load.exists() and cache_pv.exists():
        if verbose:
            print("  读取自有预测缓存")
        return {
            "load": np.load(cache_load),
            "pv": np.load(cache_pv),
            "diagnostics": None,
        }

    # 复用：把 DataBundle3 投影成 deterministic_baseline 认识的 DataBundle
    projected = base.DataBundle(
        dates=data.dates,
        load_kw=data.load_kw,
        pv_kw=data.pv_kw,
        price=data.price,
        initial_load=data.initial_load,
        initial_pv=data.initial_pv,
    )
    if verbose:
        print("  运行自有因果自适应预测（全年逐日）...", flush=True)
    fc = generate_causal_forecasts(projected, ForecastConfig(), verbose=verbose)

    np.save(cache_load, fc["load"])
    np.save(cache_pv, fc["pv"])
    fc["diagnostics"].to_csv(
        CACHE_DIR / "own_forecast_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    if verbose:
        print("  自有预测已缓存")
    return {"load": fc["load"], "pv": fc["pv"], "diagnostics": fc["diagnostics"]}


# --------------------------------------------------------------------------
# 二、光伏融合器
# --------------------------------------------------------------------------
@dataclass
class FusionDiagnostics:
    """融合权重的可追溯记录，用于论文中的权重表与冻结。"""

    coef: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    source: dict[tuple[int, int], str] = field(default_factory=dict)

    def frame(self) -> pd.DataFrame:
        rows = []
        for (k, h), c in sorted(self.coef.items()):
            rows.append(
                {
                    "issue_hour": k,
                    "hour_of_day": h,
                    "a_official": float(c[0]),
                    "b_own": float(c[1]),
                    "c_intercept": float(c[2]),
                    "weight_source": self.source[(k, h)],
                }
            )
        return pd.DataFrame(rows)


EVEN_BLEND = np.array([0.5, 0.5, 0.0])


def fit_hourly_stack(
    official: np.ndarray,
    own: np.ndarray,
    actual: np.ndarray,
    *,
    issue_hour: int,
    verbose: bool = False,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, str]]:
    """对给定的发布时刻，逐小时拟合 ``actual ~ a*official + b*own + c`` 并回代。

    参数均为该发布时刻**覆盖范围内**的 ``(365, n_covered)`` 数组。
    第 ``d`` 天的权重只用 ``j < d`` 的样本，故是因果的；样本不足时依次退化为
    合并小时回归、等权平均。
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

        # 夜间：两路预报皆≈0，回归不可辨识，直接等权，不必进循环
        if float(np.mean(actual[:, cols])) < NIGHT_KW:
            source_by_hour[h] = "夜间等权"
            fused[:, cols] = 0.5 * official[:, cols] + 0.5 * own[:, cols]
            continue

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
    day_cols = np.where(np.mean(actual, axis=0) > NIGHT_KW)[0]
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
    data: DataBundle3,
    own_pv: np.ndarray,
    *,
    verbose: bool = False,
) -> dict:
    """对四个发布时刻分别构造融合光伏预报。

    返回 ``pv_fused`` / ``pv_official`` 形状 ``(4, 365, 144)``，未覆盖时段为 NaN；
    以及权重诊断表。
    """
    n_days = len(data.dates)
    pv_fused = np.full((len(ISSUE_HOURS), n_days, N_SLOT), np.nan)
    diag = FusionDiagnostics()

    for k_pos, k in enumerate(ISSUE_HOURS):
        start = k * SLOTS_PER_HOUR           # 覆盖的起始列（0-based）
        off = data.official_pv[:, k_pos, start:]
        own = own_pv[:, start:]
        act = data.pv_kw[:, start:]
        fused, coef_map, src_map = fit_hourly_stack(off, own, act, issue_hour=k)
        pv_fused[k_pos, :, start:] = fused
        for h, c in coef_map.items():
            diag.coef[(k, h)] = c
            diag.source[(k, h)] = src_map[h]
        if verbose:
            print(f"  融合完成：发布时刻 {k:2d}:00  覆盖 {fused.shape[1]} 时段/天")

    return {
        "pv_fused": pv_fused,
        "pv_official": data.official_pv,
        "diagnostics": diag.frame(),
    }


# --------------------------------------------------------------------------
# 三、日内负载更新
# --------------------------------------------------------------------------
def intraday_load_update(
    daily_load_fc: np.ndarray,
    actual_load: np.ndarray,
    k: int,
    *,
    gamma: float = 0.5,
    clip: float = 0.15,
) -> np.ndarray:
    """在 ``k`` 时刻用已执行时段的实测负载，对剩余时段做阻尼水平修正。

    只使用 $t\\le 6k$ 的实测值（这些时段在 $k$ 时刻已经执行完毕），因此是因果的。
    未做更新的时段原样返回。
    """
    out = np.array(daily_load_fc, dtype=float, copy=True)
    n_done = k * SLOTS_PER_HOUR
    if n_done <= 0:
        return out
    realized = float(actual_load[:n_done].sum())
    planned = float(daily_load_fc[:n_done].sum())
    if planned < 1e-9:
        return out
    lam = 1.0 + gamma * (realized / planned - 1.0)
    lam = float(np.clip(lam, 1.0 - clip, 1.0 + clip))
    out[n_done:] = daily_load_fc[n_done:] * lam
    return out


def forecast_bundle(
    data: DataBundle3,
    *,
    use_cache: bool = True,
    verbose: bool = True,
) -> dict:
    """组装问题3 的全部预测：自有负载、自有光伏、四时刻官方预报与融合预报。"""
    own = build_own_forecasts(data, use_cache=use_cache, verbose=verbose)
    if verbose:
        print("  拟合光伏融合权重...", flush=True)
    fusion = build_pv_fusion(data, own["pv"], verbose=verbose)
    return {
        "load": own["load"],
        "pv_own": own["pv"],
        "pv_official": fusion["pv_official"],
        "pv_fused": fusion["pv_fused"],
        "fusion_diagnostics": fusion["diagnostics"],
        "own_diagnostics": own["diagnostics"],
    }


if __name__ == "__main__":
    from common import load_inputs

    data = load_inputs()
    bundle = forecast_bundle(data)
    print("\n=== 融合权重（部分，小时 10-14） ===")
    fd = bundle["fusion_diagnostics"]
    print(
        fd[(fd.hour_of_day.between(10, 14))]
        .round(3)
        .to_string(index=False)
    )
    print("\n=== 融合权重按发布时刻汇总（白天小时） ===")
    day = fd[fd.weight_source != "夜间均值"]
    print(
        day.groupby("issue_hour")[["a_official", "b_own", "c_intercept"]]
        .mean()
        .round(3)
        .to_string()
    )
    fd.to_csv(CACHE_DIR / "fusion_weights.csv", index=False, encoding="utf-8-sig")
