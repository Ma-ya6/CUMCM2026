"""跑七个决策模型的全年仿真，输出最终花费对照。

预测层与 模型3 逐位相同（同一 ``common``／``forecasts``／``cache``），
物理执行与经济结算共用 ``dispatch_core``，因此各模型之间的费用差**只来自
决策模型本身**。

自检：M2 用模型3 的全套口径（分位余量 q=0.80、锚定凸化、同一执行器），
必须复现冻结值 13,751,025.02 元（本层最新一次全量重跑得到的模型3 答案）；
否则说明底座被改动过，结果不可比。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import RESULT_DIR, load_inputs
from dispatch_core import simulate_year, summarize, verify_against_baseline
from forecasts import forecast_bundle
from models import all_models

# 冻结值 = 本层模型3（M2 口径）最新一次全量重跑得到的正式期费用。
MODEL3_FROZEN_COST_YUAN = 13751025.022640
# 对比基准 = 主链（代码/共享/run_year.py）问题3 的正式期 334 天费用，
# 即 结果/汇总/ 冻结数字中 Q3 在 q=0.80 的值（与论文表 D 同一口径）。
MAINLINE_COST_YUAN = 13439044.36781069
DEV_END = pd.Timestamp("2025-08-31")


def split_costs(daily: pd.DataFrame) -> dict[str, float]:
    dev = daily[daily.date <= DEV_END]
    hold = daily[daily.date > DEV_END]
    return {
        "开发期(2-8月)_元": float(dev.total_cost_yuan.sum()),
        "留出期(9-12月)_元": float(hold.total_cost_yuan.sum()),
        "全年正式期_元": float(daily.total_cost_yuan.sum()),
    }


ARCHIVE = RESULT_DIR / "方案档案.csv"


def archive(table: pd.DataFrame) -> None:
    """把本次结果 upsert 进累积档案，重跑只更新同名方案，不丢历史方案。

    参数扫描（不同 q、不同储备形态、不同决策时点数）与主表共用这一份档案，
    避免每次实验都覆盖上一次的结果、也避免为了看旧数而重跑。
    """
    cols = ["方案", "决策模型", "建模范式", "风险处理", "开发期(2-8月)_元",
            "留出期(9-12月)_元", "全年正式期_元", "紧急购电_kWh", "紧急购电费_元",
            "电网结算费_元", "超购费_元", "未用计划电量_kWh", "调整量_kWh",
            "计划购电量_kWh", "更新时间"]
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    new = table.drop(columns=["相对主链_元", "相对主链_%"], errors="ignore").copy()
    new = new.rename(columns={"代号": "方案"})
    new["更新时间"] = now
    new = new[[c for c in cols if c in new.columns]]
    if ARCHIVE.exists():
        old = pd.read_csv(ARCHIVE, encoding="utf-8-sig")
        old = old[~old.方案.isin(new.方案)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(ARCHIVE, index=False, encoding="utf-8-sig")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    verify_against_baseline()
    data = load_inputs()
    bundle = forecast_bundle(data, verbose=False)
    target = MODEL3_FROZEN_COST_YUAN

    rows, daily_all = [], []
    for model in all_models():
        daily = simulate_year(data, bundle, model)
        daily_all.append(daily)
        agg = summarize(daily, label=model.code)
        row = {
            "代号": model.code,
            "决策模型": model.name,
            "建模范式": model.paradigm,
            "风险处理": model.risk,
            **split_costs(daily),
            "紧急购电_kWh": agg["emergency_kwh"],
            "紧急购电费_元": agg["emergency_cost_yuan"],
            "电网结算费_元": agg["grid_cost_yuan"],
            "超购费_元": agg["overbuy_cost_yuan"],
            "违约费_元": agg["breach_cost_yuan"],
            "未用计划电量_kWh": agg["unused_plan_kwh"],
            "调整量_kWh": agg["adjusted_kwh"],
            "链式减两元差额_元": agg["reversal_gap_yuan"],
            "计划购电量_kWh": agg["planned_purchase_kwh"],
        }
        rows.append(row)
        print(f"  {model.code} {model.name:16s} 全年 {row['全年正式期_元']:>15,.2f}  "
              f"紧急 {row['紧急购电_kWh']:>10,.1f} kWh", flush=True)

    table = pd.DataFrame(rows)
    archive(table)
    m2 = float(table.loc[table.代号 == "M2", "全年正式期_元"].iloc[0])
    anchor_ok = abs(m2 - target) < 1e-6
    table["相对主链_元"] = table["全年正式期_元"] - MAINLINE_COST_YUAN
    table["相对主链_%"] = 100 * table["相对主链_元"] / MAINLINE_COST_YUAN
    table = table.sort_values("全年正式期_元").reset_index(drop=True)
    table.to_csv(RESULT_DIR / "决策模型对比.csv", index=False, encoding="utf-8-sig")
    pd.concat(daily_all, ignore_index=True).to_csv(
        RESULT_DIR / "决策模型逐日结果.csv", index=False, encoding="utf-8-sig"
    )

    meta = {
        "model3_frozen_cost_yuan": target,
        "M2_reproduces_model3": bool(anchor_ok),
        "M2_cost_yuan": m2,
        "mainline_cost_yuan": MAINLINE_COST_YUAN,
        "argmin_dev_period": table.loc[table["开发期(2-8月)_元"].idxmin(), "代号"],
        "argmin_holdout_period": table.loc[table["留出期(9-12月)_元"].idxmin(), "代号"],
        "argmin_full_period": table.iloc[0]["代号"],
    }
    (RESULT_DIR / "决策模型元数据.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n自检：M2 复现模型3 冻结值 =", anchor_ok,
          f"（{m2:,.2f} vs {target:,.2f}）")
    show = table[["代号", "决策模型", "建模范式", "开发期(2-8月)_元",
                  "留出期(9-12月)_元", "全年正式期_元", "相对主链_%",
                  "紧急购电_kWh", "未用计划电量_kWh"]]
    print("\n===== 决策模型对照 =====")
    print(show.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))


if __name__ == "__main__":
    main()
