"""问题2（附件1固定电价）与问题4（附件4波动电价）的 gap 对照（诊断用）。

对两个电价口径各算三层：
  下界      全知 + 全年统一优化
  完美逐日  实际负载/光伏 + 余量0 + 逐日决策
  可实现    因果预测 + q=0.80 安全余量 + 逐日决策（问题2/问题4的正式结果）

问题2的可实现值直接复用最终优化版的正式结果，同时在本脚本下重算校验。
运行：python q2_vs_q4_gap.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import deterministic_baseline as db
from oracle_lower_bound import annual_numbers, solve_full_year_oracle
from seasonal import CorrectionConfig, ForecastConfig, generate_causal_forecasts


HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "results"
PROJECT = HERE.parents[3]
RESULT_DIR = PROJECT / "结果" / "问题4" / "问题4-2"
Q2_RESULT_DIR = PROJECT / "结果" / "问题2"


def main() -> None:
    data = db.load_inputs()
    params = db.StorageParameters()
    n_days = len(data.dates)
    zero = np.zeros((n_days, db.N_SLOT))

    fixed_price = np.tile(data.initial_price, (n_days, 1))
    scenarios = {
        "问题2 附件1固定电价": replace(data, price=fixed_price),
        "问题4 附件4波动电价": data,
    }

    correction = CorrectionConfig(
        use_pv_window=True, use_level=True, use_weekly_effect=False
    )
    forecasts = generate_causal_forecasts(data, ForecastConfig(correction=correction))
    margins = db.causal_residual_quantiles(
        (data.load_kw - data.pv_kw) - (forecasts["load"] - forecasts["pv"]), 0.80
    )

    q2 = pd.read_csv(Q2_RESULT_DIR / "Q2_daily.csv", encoding="utf-8-sig")
    realized_q2 = float(q2[q2.date >= "2025-02-01"].C_total_yuan.sum())
    frozen = json.loads((RESULT_DIR / "Q4-2_summary.json").read_text(encoding="utf-8"))
    realized_q4 = frozen["dispatch"]["total_cost_yuan"]

    rows = []
    for name, scenario in scenarios.items():
        oracle = solve_full_year_oracle(scenario, params)
        lower = annual_numbers(
            oracle["purchase_kwh"],
            oracle["charge_kwh"],
            oracle["discharge_kwh"],
            oracle["curtailment_kwh"],
            scenario.price.reshape(-1),
            scenario.dates,
        )["total_cost_yuan"]

        perfect_daily, _ = db.run_dispatch(
            scenario, scenario.load_kw, scenario.pv_kw, zero, params,
            price_forecast=scenario.price,
        )
        realized_daily, _ = db.run_dispatch(
            scenario, forecasts["load"], forecasts["pv"], margins, params,
            price_forecast=scenario.price,
        )
        realized_daily = realized_daily[realized_daily.date >= db.FORMAL_START]

        rows.append(
            {
                "口径": name,
                "全知全年统一下界/元": lower,
                "完美预测逐日决策/元": float(perfect_daily.total_cost_yuan.sum()),
                "因果预测+余量q0.80/元": float(realized_daily.total_cost_yuan.sum()),
            }
        )

    table = pd.DataFrame(rows)
    table["正式结果/元"] = [realized_q2, realized_q4]
    table["可实现−下界/元"] = table["正式结果/元"] - table["全知全年统一下界/元"]
    table["可实现−下界/%"] = (
        100.0 * table["可实现−下界/元"] / table["全知全年统一下界/元"]
    )
    table.to_csv(RESULT_DIR / "问题2问题4下界对照.csv", index=False, encoding="utf-8-sig")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print("\n问题2正式结果 %.2f；问题4正式结果 %.2f；费用差 %.2f" % (realized_q2, realized_q4, realized_q4 - realized_q2))
    print(
        "下界差 %.2f；gap 差 %.2f"
        % (
            table.loc[1, "全知全年统一下界/元"] - table.loc[0, "全知全年统一下界/元"],
            table.loc[1, "可实现−下界/元"] - table.loc[0, "可实现−下界/元"],
        )
    )


if __name__ == "__main__":
    main()
