"""问题3 各预测源的准确度度量：百分比准确度、误差方向分解与可比的头部对比。

## 指标口径

对实际值 $y$ 与预测值 $\\hat y$：

- **MAE** $=\\frac1n\\sum|\\hat y-y|$（kW）
- **RMSE** $=\\sqrt{\\frac1n\\sum(\\hat y-y)^2}$（kW）
- **nMAE** $=\\mathrm{MAE}/\\overline{|y|}$（无量纲）
- **百分比准确度** $=1-\\mathrm{nMAE}$，本文以百分数报告
- **偏差** $=\\overline{\\hat y-y}$（kW，带符号：负值代表系统性"少估"）
- **R²** $=1-\\frac{\\sum(\\hat y-y)^2}{\\sum(y-\\bar y)^2}$

**净负荷不裁剪**。中午光伏可能超过负荷，净负荷为负（实测最低约 −6602 kW），
故净负荷一律不做非负裁剪；只有负荷与光伏本身裁剪到 $\\ge 0$。

## 为什么必须区分"覆盖"与"精度"

附件3 的四个发布时刻覆盖的时段数不同（100% / 75% / 50% / 25%），且覆盖窗口
的系统性质不同：18:00 的预报只覆盖 18:00–24:00，那是光伏接近零的夜间，
绝对误差自然小（MAE 19 kW），但相对误差极大（nMAE≈1.0，准确度≈0）。
若把它们放在一张表里比大小，会得出"18:00 预报最准"的荒谬结论。

因此本模块输出两张表：

1. ``accuracy_by_source.csv``——各源在**各自覆盖范围内**的精度，用于回答
   "每一路预报本身有多准"；
2. ``accuracy_common_window.csv``——限制到**共同窗口**，用于回答
   "新发布的预报是否真的比旧的更准"，这是问题3 调整机制成立的前提。

共同窗口取所有被比较源都覆盖的时段：比较 0:00 与 6:00 用 $t\\ge 37$，
三者用 $t\\ge 73$，四者用 $t\\ge 109$。

## 误差的经济方向

对调度而言，误差的**符号**比大小更重要：少估 1 kW·h 要用 5 倍电价紧急购电，
多估 1 kW·h 则计划电量作废。故同时报告少估占比与 4:1 加权的非对称损失，
为后面"调整策略影响"的分析提供传导链条。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from common import (
    DT,
    ISSUE_HOURS,
    N_SLOT,
    SLOTS_PER_HOUR,
    RESULT_DIR,
    DataBundle3,
    formal_split,
)

# 共同窗口：n 个源共同覆盖所需的最晚起始时段（1-based）
COMMON_WINDOW = {1: 1, 2: 37, 3: 73, 4: 109}


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y, p))
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    denom = float(np.mean(np.abs(y)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae_kw": mae,
        "rmse_kw": rmse,
        "nmae": mae / max(denom, 1e-9),
        "accuracy_pct": 100.0 * (1.0 - mae / max(denom, 1e-9)),
        "bias_kw": float(np.mean(p - y)),
        "r2": 1.0 - float(np.sum((p - y) ** 2)) / max(ss_tot, 1e-9),
        "actual_mean_kw": denom,
        "n_obs": int(y.size),
    }


def _error_direction(y: np.ndarray, p: np.ndarray, price: np.ndarray) -> dict:
    """少估/多估的电量与加权的非对称经济损失。

    少估 1 kW·h 须以 5 倍价紧急购电，相对计划价多付 $4p$；多估 1 kW·h 则计划
    电量作废、损失 $p$。故单位权重为 4:1。
    """
    under = np.maximum(y - p, 0.0) * DT           # 少估（须紧急购电）
    over = np.maximum(p - y, 0.0) * DT            # 多估（计划作废）
    tot = float(under.sum() + over.sum())
    return {
        "underforecast_kwh": float(under.sum()),
        "overforecast_kwh": float(over.sum()),
        "under_share": float(under.sum() / max(tot, 1e-9)),
        "asymmetric_loss_yuan": float(price @ (4.0 * under + over)),
    }


def _tile_price(price: np.ndarray, n_days: int) -> np.ndarray:
    return np.tile(price, n_days)


def build_source_table(
    data: DataBundle3,
    bundle: dict,
    splits: dict[str, np.ndarray],
) -> pd.DataFrame:
    """各预测源 × 目标 × 分期的精度表（诚实标注各自的覆盖范围）。"""
    n_days = len(data.dates)
    actual_net = data.load_kw - data.pv_kw
    rows: list[dict] = []

    # ---- 负载：自有因果预测（唯一来源，题目未给负载预报） ----
    for split, dmask in splits.items():
        y = data.load_kw[dmask].ravel()
        p = np.maximum(bundle["load"][dmask], 0.0).ravel()
        rows.append({"target": "load", "source": "自有因果集成", "coverage": "100%",
                     "split": split, **_metrics(y, p)})

    # ---- 光伏：官方预报 / 自有统计 / 因果融合，逐发布时刻 ----
    for k_pos, k in enumerate(ISSUE_HOURS):
        cov_start = k * SLOTS_PER_HOUR
        cov_pct = f"{100 * (N_SLOT - cov_start) / N_SLOT:.0f}%"
        sources = (
            ("官方预报", data.official_pv[:, k_pos]),
            ("因果融合", bundle["pv_fused"][k_pos]),
            ("自有统计", bundle["pv_own"]),
        )
        for src_name, arr in sources:
            for split, dmask in splits.items():
                sub = np.zeros_like(dmask)
                sub[dmask] = True
                m = np.zeros((n_days, N_SLOT), bool)
                m[:, cov_start:] = True
                m &= sub[:, None]
                if not m.any():
                    continue
                finite = np.isfinite(arr) & m
                if not finite.any():
                    continue
                y = data.pv_kw[finite]
                p = np.maximum(arr[finite], 0.0)
                rows.append({
                    "target": "pv", "source": f"{src_name}@{k:02d}:00",
                    "coverage": cov_pct, "split": split, **_metrics(y, p),
                })

    # ---- 净负荷：调度实际使用的口径（融合光伏 @ 各发布时刻） ----
    for k_pos, k in enumerate(ISSUE_HOURS):
        cov_start = k * SLOTS_PER_HOUR
        for src_name, arr in (
            ("官方预报", data.official_pv[:, k_pos]),
            ("因果融合", bundle["pv_fused"][k_pos]),
            ("自有统计", bundle["pv_own"]),
        ):
            for split, dmask in splits.items():
                m = np.zeros((n_days, N_SLOT), bool)
                m[:, cov_start:] = True
                m &= dmask[:, None]
                if not m.any():
                    continue
                finite = np.isfinite(arr) & m
                y = actual_net[finite]
                p = bundle["load"][finite] - np.maximum(arr[finite], 0.0)
                row = {"target": "net", "source": f"{src_name}@{k:02d}:00",
                       "coverage": f"{100 * (N_SLOT - cov_start) / N_SLOT:.0f}%",
                       "split": split, **_metrics(y, p)}
                if split == "全年正式期":
                    prices = np.tile(data.price, (n_days, 1))[finite]
                    row.update(_error_direction(y, p, prices))
                rows.append(row)

    return pd.DataFrame(rows)


def build_common_window_table(
    data: DataBundle3,
    bundle: dict,
    splits: dict[str, np.ndarray],
) -> pd.DataFrame:
    """相邻发布时刻在共同窗口上的配对比较。

    回答的是**决策相关**的问题：第 $k$ 时刻拿到新预报时，它相对于我手上
    已经在用的 $k^-$ 预报是否真的更准？故窗口取两者共同覆盖的部分，
    即 $(k^-,k]$ 之后的剩余全天——6:00 vs 0:00 用 $t\\ge 37$，
    12:00 vs 6:00 用 $t\\ge 73$，18:00 vs 12:00 用 $t\\ge 109$。

    **必须注意**：18:00 vs 12:00 的窗口 18:00–24:00 是**夜间**，光伏已接近零，
    两路预报都≈0，该窗口上的光伏精度对比是退化的（四路 MAE 都在 2.9 kW 左右，
    差异不到 1%）。这一格不能用来论证"18:00 的预报没用"——18:00 的调整价值
    主要来自**负载**侧的更新与储能的再优化，而非光伏。故本表对退化格给出显式
    标记 ``degenerate``，并且"是否需要 18:00 调整"的结论以
    ``adjustment_impact.py`` 的实际费用分解为准，不以本表为准。
    """
    n_days = len(data.dates)
    rows: list[dict] = []
    # (上一发布时刻, 本发布时刻)；k_prev=None 表示 0:00 相对"无预报"的绝对水平
    for k_prev, k_pos in ((0, 1), (6, 2), (12, 3)):
        k = ISSUE_HOURS[k_pos]
        # 共同窗口 = 较晚发布时刻的覆盖范围。两侧都必须限制到同一窗口，
        # 否则会拿"旧预报的整天精度"和"新预报的剩余时段精度"相比，结论无效。
        cov_start = k * SLOTS_PER_HOUR
        window = f"{k:02d}:00-24:00"
        for src_name, arr_pair in (
            ("官方预报", (data.official_pv[:, ISSUE_HOURS.index(k_prev)],
                          data.official_pv[:, k_pos])),
            ("因果融合", (bundle["pv_fused"][ISSUE_HOURS.index(k_prev)],
                          bundle["pv_fused"][k_pos])),
        ):
            for split, dmask in splits.items():
                m = np.zeros((n_days, N_SLOT), bool)
                m[:, cov_start:] = True
                m &= dmask[:, None]
                # 同一窗口上同时评估"旧预报"与"新预报"，构成配对比较
                for role, arr in (("旧", arr_pair[0]), ("新", arr_pair[1])):
                    finite = np.isfinite(arr) & m
                    if not finite.any():
                        continue
                    y = data.pv_kw[finite]
                    p = np.maximum(arr[finite], 0.0)
                    row = {
                        "prev_issue_hour": k_prev,
                        "issue_hour": k,
                        "common_window": window,
                        "source": src_name,
                        "role": role,
                        "split": split,
                        **_metrics(y, p),
                    }
                    # 夜间窗口标记：该窗口实际光伏均值过低时，光伏精度对比不可用
                    row["degenerate"] = bool(row["actual_mean_kw"] < 50.0)
                    rows.append(row)
    return pd.DataFrame(rows)


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """给论文用的紧凑汇总：全年正式期，负载/光伏/净负荷的百分比准确度。"""
    formal = frame[frame.split == "全年正式期"]
    keep_src = [
        "自有因果集成",
        "自有统计@00:00",
        "官方预报@00:00", "因果融合@00:00",
        "官方预报@06:00", "因果融合@06:00",
        "官方预报@12:00", "因果融合@12:00",
        "官方预报@18:00", "因果融合@18:00",
    ]
    out = formal[formal.source.isin(keep_src)][
        ["target", "source", "coverage", "mae_kw", "rmse_kw", "nmae",
         "accuracy_pct", "bias_kw", "r2"]
    ].copy()
    return out.sort_values(["target", "source"]).reset_index(drop=True)


def main(verbose: bool = True) -> dict[str, pd.DataFrame]:
    from common import load_inputs
    from forecasts import forecast_bundle

    data = load_inputs()
    bundle = forecast_bundle(data, verbose=verbose)
    splits = formal_split(data.dates)

    src = build_source_table(data, bundle, splits)
    src.to_csv(RESULT_DIR / "accuracy_by_source.csv", index=False, encoding="utf-8-sig")

    common = build_common_window_table(data, bundle, splits)
    common.to_csv(
        RESULT_DIR / "accuracy_common_window.csv", index=False, encoding="utf-8-sig"
    )

    summary = build_summary(src)
    summary.to_csv(RESULT_DIR / "accuracy_summary.csv", index=False, encoding="utf-8-sig")

    if verbose:
        pd.set_option("display.width", 250)
        print("\n=== 全年正式期：各预测源准确度 ===")
        print(summary.round(3).to_string(index=False))
        formal = common[common.split == "全年正式期"].copy()
        # 配对比较：同一窗口、同一源上，"新预报相对旧预报"的 MAE 变化
        pair = formal.pivot_table(
            index=["prev_issue_hour", "issue_hour", "common_window", "source", "degenerate"],
            columns="role",
            values=["mae_kw", "nmae", "bias_kw"],
        )
        pair.columns = [f"{a}_{b}" for a, b in pair.columns]
        pair = pair.reset_index()
        pair["mae_gain"] = pair["mae_kw_旧"] - pair["mae_kw_新"]
        pair["mae_gain_pct"] = 100.0 * pair["mae_gain"] / pair["mae_kw_旧"]
        pair.to_csv(
            RESULT_DIR / "accuracy_issue_pairs.csv", index=False, encoding="utf-8-sig"
        )
        print("\n=== 光伏：同一共同窗口上 新预报 vs 旧预报（配对） ===")
        print(
            pair[["prev_issue_hour", "issue_hour", "common_window", "source",
                  "mae_kw_旧", "mae_kw_新", "mae_gain", "mae_gain_pct", "degenerate"]]
            .round(3).to_string(index=False)
        )

    return {"by_source": src, "common_window": common, "summary": summary}


if __name__ == "__main__":
    main()
