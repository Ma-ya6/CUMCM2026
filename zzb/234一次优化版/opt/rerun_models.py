# -*- coding: utf-8 -*-
"""二、用**修正后的预测层**重跑上游 M1–M10 决策模型对比表。

背景：上游 ``问题3/模型3决策优化版/results/决策模型对比.csv`` 是**旧预测层**
（``forecasts.py:_pooled_coef`` 查看全年实测光伏识别昼夜列，属跨日未来信息
泄露）跑出来的，而问题3 / 问题4-3 的最终结论建立在新预测层上。两张表不同口径，
不能同时进论文。

更要紧的是它带着一个**结论冲突**：

- 上游表里 M2（分位安全余量 LP）与 M6（精确链式结算 MILP）**逐位相同**
  （13,748,164.499748461 vs …463），据此断言"锚定凸化不是近似而是精确的"；
- 而 ``opt`` 侧修正后的实测是 exact − fixed = −2.71 万元、95% CI 不含 0，
  即同一对目标函数**不等价**。

差异只能来自预测层或执行器。本脚本把 9 个模型在新预测层上原样重跑一遍
（``models.py``、``dispatch_core.py``、结算规则一字不动，只换预测层），
用同一张表回答"M2≡M6 是否仍然成立"。

**只读上游代码，不覆盖上游 ``results/``**：结果写进 ``opt/out/``。

跑法：``python rerun_models.py --problem Q3``
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OPT_DIR = Path(__file__).resolve().parent
C_DIR = OPT_DIR.parent.parent
OUT_DIR = OPT_DIR / "out"
sys.path.insert(0, str(OPT_DIR))

DIRS = {"Q3": C_DIR / "问题3" / "模型3决策优化版",
        "Q4-3": C_DIR / "问题4" / "重算问题3"}
UPSTREAM_Q3 = C_DIR / "问题3" / "模型3决策优化版" / "results" / "决策模型对比.csv"


def run(problem: str) -> pd.DataFrame:
    for m in ("common", "dispatch_core", "forecasts", "models",
              "deterministic_baseline", "seasonal"):
        sys.modules.pop(m, None)
    root = DIRS[problem]
    sys.path.insert(0, str(root))
    import common                                        # noqa: E402
    import dispatch_core                                 # noqa: E402
    import forecasts                                     # noqa: E402
    from models import all_models                        # noqa: E402

    data = common.load_inputs()
    bundle = forecasts.forecast_bundle(data, use_cache=False, verbose=False)
    kw = {}
    if problem == "Q4-3":
        price_day, price_fc = common.load_price_channel()
        kw = {"price_day": price_day, "price_fc": price_fc}

    rows = []
    for model in all_models():
        t0 = time.time()
        daily = dispatch_core.simulate_year(data, bundle, model, **kw)
        agg = dispatch_core.summarize(daily, label=model.code)
        rows.append({
            "代号": model.code,
            "决策模型": model.name,
            "建模范式": model.paradigm,
            "风险处理": model.risk,
            "全年正式期_元": agg["total_cost_yuan"],
            "紧急购电_kWh": agg["emergency_kwh"],
            "紧急购电费_元": agg["emergency_cost_yuan"],
            "电网结算费_元": agg["grid_cost_yuan"],
            "超购费_元": agg["overbuy_cost_yuan"],
            "违约费_元": agg["breach_cost_yuan"],
            "未用计划电量_kWh": agg["unused_plan_kwh"],
            "调整量_kWh": agg["adjusted_kwh"],
            "链式减两元差额_元": agg["reversal_gap_yuan"],
            "计划购电量_kWh": agg["planned_purchase_kwh"],
            "秒": round(time.time() - t0, 1),
        })
        print(f"  {model.code:4s} {model.name:20s} 全年 "
              f"{rows[-1]['全年正式期_元']:>15,.2f}  "
              f"紧急 {rows[-1]['紧急购电_kWh']:>10,.1f} kWh  "
              f"({rows[-1]['秒']}s)", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=list(DIRS), default="Q3")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    table = run(args.problem)
    m2 = float(table.loc[table.代号 == "M2", "全年正式期_元"].iloc[0])
    table["相对M2_元"] = table["全年正式期_元"] - m2
    table["相对M2_%"] = 100 * table["相对M2_元"] / m2
    table = table.sort_values("全年正式期_元").reset_index(drop=True)

    stem = f"{args.problem}_模型对比_新预测层"
    table.to_csv(OUT_DIR / f"{stem}.csv", index=False, encoding="utf-8-sig")

    payload = {
        "problem": args.problem,
        "预测层": "修正后（forecasts._pooled_coef 只用 j<d 的实测光伏识别昼夜列）",
        "对照表": str(UPSTREAM_Q3),
        "M2_cost_yuan": m2,
        "行": table.to_dict(orient="records"),
    }

    # 与上游旧预测层表的逐行对照
    if args.problem == "Q3" and UPSTREAM_Q3.exists():
        old = pd.read_csv(UPSTREAM_Q3, encoding="utf-8-sig")
        key = old.set_index("代号")["全年正式期_元"]
        cmp_rows = []
        for _, r in table.iterrows():
            o = float(key.get(r["代号"], np.nan))
            cmp_rows.append({
                "代号": r["代号"], "决策模型": r["决策模型"],
                "旧预测层_元": o, "新预测层_元": r["全年正式期_元"],
                "变化_元": r["全年正式期_元"] - o if np.isfinite(o) else None,
                "变化_%": (100 * (r["全年正式期_元"] - o) / o) if np.isfinite(o) else None,
            })
        cmp = pd.DataFrame(cmp_rows)
        cmp.to_csv(OUT_DIR / f"{stem}_对照旧表.csv", index=False, encoding="utf-8-sig")
        payload["对照旧表"] = cmp.to_dict(orient="records")

        old_m2 = float(key["M2"]); old_m6 = float(key["M6"])
        new_m6 = float(table.loc[table.代号 == "M6", "全年正式期_元"].iloc[0])
        payload["M2与M6"] = {
            "旧预测层": {"M2": old_m2, "M6": old_m6, "差_元": old_m6 - old_m2,
                       "M6相对M2_%": 100 * (old_m6 - old_m2) / old_m2},
            "新预测层": {"M2": m2, "M6": new_m6, "差_元": new_m6 - m2,
                       "M6相对M2_%": 100 * (new_m6 - m2) / m2},
        }
        d = payload["M2与M6"]
        print(f"\nM2 与 M6：旧预测层差 {d['旧预测层']['差_元']:+.6f} 元 "
              f"({d['旧预测层']['M6相对M2_%']:+.3e}%)")
        print(f"           新预测层差 {d['新预测层']['差_元']:+,.2f} 元 "
              f"({d['新预测层']['M6相对M2_%']:+.4f}%)")

    (OUT_DIR / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print(f"\n已写出 {OUT_DIR / (stem + '.csv')}")


if __name__ == "__main__":
    main()
