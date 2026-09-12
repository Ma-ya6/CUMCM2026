"""发布时点消融：固定问题3/4-3 结构的其他全部条件，只改变启用的发布时刻集合。

仅在明确运行本文件时开始；产物落在 `图表/综合/补跑输出/hours_<变体>/`，不覆盖正式结果。
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_year as ry

OUT = HERE.parents[2] / "图表" / "综合" / "补跑输出"
TABLES = HERE.parents[2] / "图表" / "数据表"
FORMAL = "2025-02-01"

# 键为变体名，值为启用的发布时刻（小时）；问题3/4-3 的 0:00 计划恒开启。
VARIANTS = {
    "base_0": (0,),
    "only_6": (0, 6),
    "only_12": (0, 12),
    "only_18": (0, 18),
    "full": (0, 6, 12, 18),
    "cum_0612": (0, 6, 12),
    "drop_6": (0, 12, 18),
    "drop_12": (0, 6, 18),
}


def run_one(a, problem, name, hours):
    dest = OUT / f"hours_{name}" / problem
    dest.mkdir(parents=True, exist_ok=True)
    summary = dest / f"{problem}_summary.json"
    if summary.exists():
        rec = json.loads(summary.read_text(encoding="utf-8"))
        if rec.get("simulation_days") == 365:
            print(f"Resume: {name} {problem}", flush=True)
            rec.update(variant=name, hours=list(hours), passed=True)
            return rec
    ry.OUT_OVERRIDE = dest
    try:
        rec = ry.simulate(a, problem, issue_hours=hours)
    except RuntimeError as exc:
        shutil.rmtree(dest, ignore_errors=True)
        print(f"FAILED: {name} {problem}: {exc}", flush=True)
        return dict(problem=problem, variant=name, hours=list(hours),
                    passed=False, reason=str(exc))
    rec["variant"] = name
    rec["hours"] = list(hours)
    rec["passed"] = True
    return rec


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="发布时点消融；不改变正式结果")
    ap.add_argument("--problem", nargs="+", default=["Q3", "Q4-3"])
    ap.add_argument("--variant", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()
    a = ry.prepare()
    records = []
    for problem in args.problem:
        for name in args.variant:
            print(f"=== {problem} {name} hours={VARIANTS[name]}", flush=True)
            records.append(run_one(a, problem, name, VARIANTS[name]))
    tab = pd.DataFrame([{
        "problem": r["problem"], "variant": r["variant"],
        "issue_hours": "+".join(str(h) for h in r["hours"]),
        "passed": r["passed"],
        "formal_period_cost_yuan": r.get("formal_period_cost_yuan"),
        "calendar_year_cost_yuan": r.get("calendar_year_cost_yuan"),
        "final_soc_kwh": r.get("final_soc_kwh"),
        "max_balance_error_kwh": (r.get("audit") or {}).get("max_balance_err"),
        "negative_flow": (r.get("audit") or {}).get("negative_flow"),
        "seconds": r.get("seconds"),
        "note": r.get("reason") or "通过物理核验",
    } for r in records])
    TABLES.mkdir(parents=True, exist_ok=True)
    tab.to_csv(TABLES / "表H_发布时点消融_正式期334天.csv", index=False, encoding="utf-8-sig")
    print(tab.to_string(index=False), flush=True)
    print("Requested ablation runs completed. Original baseline results unchanged.")


if __name__ == "__main__":
    main()
