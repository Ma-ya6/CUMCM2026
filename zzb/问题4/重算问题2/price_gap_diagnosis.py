"""问题4 vs 问题2 费用差的口径分解（诊断用，不属于正式交付结果）。

固定同一套因果负载/光伏预测与同一 q=0.80 安全余量，只替换电价口径：

  C1  附件1单条曲线（365天相同，已知）       = 问题2口径，本脚本重算
  C2  附件4日内平均形状（365天相同，已知）   = 去掉逐日波动、保留附件4日内形状，本脚本重算
  C3  附件4实际电价（逐日波动，LP已知）      = 完美电价信息，取自冻结结果
  C4  附件4实际电价（逐日波动，因果预测）    = 问题4正式结果，取自冻结结果

C2−C1 = 日内形状差异；C3−C2 = 逐日波动性；C4−C3 = 电价不可预知性。
负载/光伏预测与余量直接从 results/全年逐10分钟策略.csv 复用，不重跑预测。
运行：python price_gap_diagnosis.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import deterministic_baseline as db
from seasonal import CorrectionConfig, ForecastConfig, generate_causal_forecasts


HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "results"
Q2_RESULT_DIR = Path(
    r"D:\Files\Work\Mathematical_Modeling\GuoSai\2026\C题\CUMCM2026\zzb\最终优化版\results"
)


def run_known_price(data, price_grid, load_fc, pv_fc, margins, params) -> dict:
    """给定已知电价曲线（LP 目标即该曲线，结算也用该曲线）的全年逐日调度。"""
    scenario = replace(data, price=price_grid)
    daily, _ = db.run_dispatch(
        scenario, load_fc, pv_fc, margins, params, price_forecast=price_grid
    )
    formal = daily[daily.date >= db.FORMAL_START]
    return {
        "total_cost_yuan": float(formal.total_cost_yuan.sum()),
        "planned_cost_yuan": float(formal.planned_cost_yuan.sum()),
        "planned_purchase_kwh": float(formal.planned_purchase_kwh.sum()),
        "planned_weighted_price": float(
            formal.planned_cost_yuan.sum() / formal.planned_purchase_kwh.sum()
        ),
        "emergency_kwh": float(formal.emergency_kwh.sum()),
        "emergency_cost_yuan": float(formal.emergency_cost_yuan.sum()),
        "charge_kwh": float(formal.charge_kwh.sum()),
        "discharge_kwh": float(formal.discharge_kwh.sum()),
        "curtailment_kwh": float(formal.curtailed_pv_kwh.sum()),
        "unused_plan_kwh": float(formal.unused_plan_kwh.sum()),
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
        (data.load_kw - data.pv_kw) - (load_fc - pv_fc), 0.80
    )

    frozen = json.loads(
        (RESULT_DIR / "最终冻结结果.json").read_text(encoding="utf-8")
    )
    c3 = {"total_cost_yuan": frozen["perfect_price_information"]["total_cost_yuan"]}
    c4 = {"total_cost_yuan": frozen["dispatch"]["total_cost_yuan"]}

    rows = [
        (
            "C1 附件1单条曲线（已知）",
            run_known_price(
                data, np.tile(data.initial_price, (n_days, 1)), load_fc, pv_fc, margins, params
            ),
        ),
        (
            "C2 附件4日内平均形状（已知）",
            run_known_price(
                data,
                np.tile(data.price.mean(axis=0), (n_days, 1)),
                load_fc,
                pv_fc,
                margins,
                params,
            ),
        ),
        ("C3 附件4实际电价（完美信息）", c3),
        ("C4 附件4实际电价（因果预测）", c4),
    ]
    table = pd.DataFrame([{"口径": name, **numbers} for name, numbers in rows])
    table.to_csv(RESULT_DIR / "电价口径分解诊断.csv", index=False, encoding="utf-8-sig")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    c1_cost = rows[0][1]["total_cost_yuan"]
    c2_cost = rows[1][1]["total_cost_yuan"]
    print("\n分解（元 / 占问题2→问题4总差 %）：")
    total_gap = c4["total_cost_yuan"] - c1_cost
    for label, part in (
        ("日内形状差异 C2−C1", c2_cost - c1_cost),
        ("逐日波动性 C3−C2", c3["total_cost_yuan"] - c2_cost),
        ("电价不可预知 C4−C3", c4["total_cost_yuan"] - c3["total_cost_yuan"]),
    ):
        print(f"  {label}: {part:,.2f} 元  ({100.0 * part / total_gap:.1f}%)")
    print(f"  合计 C4−C1: {total_gap:,.2f} 元")

    print("\n附件1曲线 vs 附件4日内平均形状：")
    for label, curve in (
        ("附件1单条曲线", data.initial_price),
        ("附件4日内平均(全期)", data.price.mean(axis=0)),
        ("附件4日内平均(正式期)", data.price[data.dates >= db.FORMAL_START].mean(axis=0)),
    ):
        print(
            f"  {label}: 均值{curve.mean():.4f} 最低{curve.min():.4f} 最高{curve.max():.4f} "
            f"峰谷差{curve.max() - curve.min():.4f}"
        )
    print(
        "  日内形状相关系数: %.4f"
        % np.corrcoef(data.initial_price, data.price.mean(axis=0))[0, 1]
    )

    q2 = pd.read_csv(Q2_RESULT_DIR / "全年逐日结果.csv", encoding="utf-8-sig")
    q2_formal = q2[q2.date >= "2025-02-01"]
    print(
        "\n问题2原版: 总费用 %.2f 计划费 %.2f 计划量 %.2f 均价 %.4f 紧急量 %.2f "
        "充电 %.2f 放电 %.2f 弃光 %.2f"
        % (
            q2_formal.C_total_yuan.sum(),
            q2_formal.C_grid_yuan.sum(),
            q2_formal.x_plan_kwh.sum(),
            q2_formal.C_grid_yuan.sum() / q2_formal.x_plan_kwh.sum(),
            q2_formal.e_emergency_kwh.sum(),
            q2_formal.u_charge_kwh.sum(),
            q2_formal.v_discharge_kwh.sum(),
            q2_formal.w_pv_waste_kwh.sum(),
        )
    )


if __name__ == "__main__":
    main()
