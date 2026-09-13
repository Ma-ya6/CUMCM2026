"""波动电价下问题3 的代价阶梯（四个口径）。

口径与 问题4/重算问题2 的 8.1 节逐条对应：

  1. 理论完美最低值 ：全年统一优化，已知全部实际负载/光伏/电价（``理论最优.json``）
  2. 完美预测但逐日决策：负载/光伏/电价全换成实际值、$R_d=0$，但保持四时点滚动结构
  3. 完美电价       ：负载/光伏仍因果预测，决策电价换成当天实际电价
  4. 可实现策略     ：负载/光伏/电价的因果预测 + $R_d$（正式结果）

口径 3 与 4 的差别只来自"电价不可预知"；口径 2 与 3 的差别只来自
"负载/光伏不可预知 + 储备"。四个口径的储能物理约束、初始 SOC=6000 kWh、
年末 SOC 回归 6000 kWh、费用统计区间（2025-02-01 起）完全一致。

运行：python cost_ladder.py    结果：results/代价阶梯.csv
"""

from __future__ import annotations

import json

import pandas as pd

from common import RESULT_DIR, load_inputs, load_price_channel
from dispatch_core import simulate_year, summarize
from forecasts import forecast_bundle
from models import StorageReserveLP

ORACLE_JSON = RESULT_DIR / "理论最优.json"


def main() -> None:
    if not ORACLE_JSON.exists():
        raise FileNotFoundError(f"缺少 {ORACLE_JSON}，请先运行 python oracle_lower_bound3.py")

    data = load_inputs()
    bundle = forecast_bundle(data, verbose=False)
    price_day, price_fc = load_price_channel()
    frozen = json.loads((RESULT_DIR / "最终冻结结果.json").read_text(encoding="utf-8"))
    model = StorageReserveLP()

    print("口径2：完美预测 + R_d=0（保持四时点滚动）...", flush=True)
    perfect = simulate_year(
        data, bundle, model,
        load_source="perfect", pv_source="perfect",
        price_day=price_day, price_fc=None,
    )
    agg2 = summarize(perfect, label="完美预测但逐日决策（余量0）")

    rows = [
        {
            "口径": "理论完美最低值（全知、全年统一优化）",
            "总费用_元": json.loads(ORACLE_JSON.read_text(encoding="utf-8"))[
                "oracle_full_year_joint_yuan"
            ],
            "计划购电量_kWh": float("nan"),
            "紧急购电量_kWh": 0.0,
        },
        {
            "口径": "完美预测但逐日决策（余量0）",
            "总费用_元": agg2["total_cost_yuan"],
            "计划购电量_kWh": agg2["planned_purchase_kwh"],
            "紧急购电量_kWh": agg2["emergency_kwh"],
        },
        {
            "口径": "完美电价、负载/光伏仍因果预测",
            "总费用_元": frozen["perfect_price_information"]["total_cost_yuan"],
            "计划购电量_kWh": frozen["perfect_price_information"]["planned_purchase_kwh"],
            "紧急购电量_kWh": frozen["perfect_price_information"]["emergency_kwh"],
        },
        {
            "口径": "可实现策略（本文提交）",
            "总费用_元": frozen["annual"]["total_cost_yuan"],
            "计划购电量_kWh": frozen["annual"]["planned_purchase_kwh"],
            "紧急购电量_kWh": frozen["annual"]["emergency_kwh"],
        },
    ]
    table = pd.DataFrame(rows)
    base = float(table.loc[0, "总费用_元"])
    table["相对全知下界_元"] = table["总费用_元"] - base
    table["相对全知下界_%"] = 100 * table["相对全知下界_元"] / base
    table.to_csv(RESULT_DIR / "代价阶梯.csv", index=False, encoding="utf-8-sig")

    print("\n===== 波动电价下问题3 的代价阶梯 =====")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print("\n口径2 紧急购电 kWh =", agg2["emergency_kwh"])


if __name__ == "__main__":
    main()
