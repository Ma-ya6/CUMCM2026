"""完美电价口径下，负载/光伏预测误差与安全余量的成本分解（诊断用）。

四个口径的电价全部为附件4实际值（LP 已知），只切换负载/光伏侧的两件事：

  W1  实际负载/光伏 + 余量0      = 完美预测、逐日决策
  W2  实际负载/光伏 + q=0.80余量  = 预测完美但仍按历史残差留安全余量
  W3  因果预测     + 余量0        = 有预测误差但不留安全余量
  W4  因果预测     + q=0.80余量   = 问题4"完美电价"对照口径

W2−W1 = 安全余量的净成本；W3−W1 = 预测误差的净成本；
W4−W1 = 两者合计（含交互项）。运行：python forecast_gap_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import deterministic_baseline as db
from seasonal import CorrectionConfig, ForecastConfig, generate_causal_forecasts


HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "results"
Q = 0.80


def run(data, load_fc, pv_fc, margins, params) -> dict:
    daily, _ = db.run_dispatch(
        data, load_fc, pv_fc, margins, params, price_forecast=data.price
    )
    formal = daily[daily.date >= db.FORMAL_START]
    cost = float(formal.total_cost_yuan.sum())
    purchase = float(formal.planned_purchase_kwh.sum())
    return {
        "total_cost_yuan": cost,
        "planned_purchase_kwh": purchase,
        "planned_cost_yuan": float(formal.planned_cost_yuan.sum()),
        "planned_weighted_price": float(formal.planned_cost_yuan.sum() / purchase),
        "emergency_kwh": float(formal.emergency_kwh.sum()),
        "emergency_cost_yuan": float(formal.emergency_cost_yuan.sum()),
        "unused_plan_kwh": float(formal.unused_plan_kwh.sum()),
        "curtailment_kwh": float(formal.curtailed_pv_kwh.sum()),
        "days_with_emergency": int((formal.emergency_kwh > 1e-8).sum()),
    }


def main() -> None:
    data = db.load_inputs()
    params = db.StorageParameters()
    n_days = len(data.dates)

    correction = CorrectionConfig(
        use_pv_window=True, use_level=True, use_weekly_effect=False
    )
    forecasts = generate_causal_forecasts(data, ForecastConfig(correction=correction))
    load_fc, pv_fc = forecasts["load"], forecasts["pv"]
    margins = db.causal_residual_quantiles(
        (data.load_kw - data.pv_kw) - (load_fc - pv_fc), Q
    )
    zero = np.zeros((n_days, db.N_SLOT))

    scenarios = [
        ("W1 实际值+余量0", data.load_kw, data.pv_kw, zero),
        ("W2 实际值+q0.80余量", data.load_kw, data.pv_kw, margins),
        ("W3 因果预测+余量0", load_fc, pv_fc, zero),
        ("W4 因果预测+q0.80余量", load_fc, pv_fc, margins),
    ]
    rows = [(name, run(data, lf, pf, mg, params)) for name, lf, pf, mg in scenarios]
    table = pd.DataFrame([{"口径": name, **numbers} for name, numbers in rows])
    table.to_csv(RESULT_DIR / "预测误差与余量分解诊断.csv", index=False, encoding="utf-8-sig")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    base = rows[0][1]["total_cost_yuan"]
    w2 = rows[1][1]["total_cost_yuan"]
    w3 = rows[2][1]["total_cost_yuan"]
    w4 = rows[3][1]["total_cost_yuan"]
    total = w4 - base
    print("\n分解（元 / 占 W4−W1 %）：")
    for label, part in (
        ("安全余量 W2−W1", w2 - base),
        ("预测误差 W3−W1", w3 - base),
        ("交互项（合计−两者）", total - (w2 - base) - (w3 - base)),
    ):
        print(f"  {label}: {part:,.2f} 元  ({100.0 * part / total:.1f}%)")
    print(f"  合计 W4−W1: {total:,.2f} 元")


if __name__ == "__main__":
    main()
