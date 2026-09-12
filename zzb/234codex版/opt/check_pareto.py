# -*- coding: utf-8 -*-
"""可靠性—经济 Pareto 汇总。

只读 ``opt/out/*_阶梯*.json``（已跑出的结果），不重跑仿真。

跑法：``python check_pareto.py``，输出 ``opt/可靠性Pareto.md``。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OPT_DIR = Path(__file__).resolve().parent
OUT_DIR = OPT_DIR / "out"

LAMBDA = 0.7662
E0 = 6000.0

# 各问题的最终策略档位名（取 q=0.80 的那一档）
FINAL_TAG = {"Q2": "reserve-mpc", "Q3": "exact-greedy",
             "Q4-2": "reserve-mpc", "Q4-3": "exact-greedy"}
UP_TAG = "legacy-greedy"


def load(problem: str, suffix: str) -> dict:
    f = OUT_DIR / f"{problem}_阶梯{suffix}.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text(encoding="utf-8"))


def adj(s: dict) -> float:
    """统一年末口径后的总费用。"""
    return s["total_cost_yuan"] - LAMBDA * (s["final_soc_kwh"] - E0)


def points(problem: str) -> list[tuple[str, dict]]:
    """(档位标签, 汇总) 列表，按 经济型→保守型 排序。"""
    out = []
    base = load(problem, "")
    if UP_TAG in base:
        out.append((f"上游口径 ({UP_TAG})", base[UP_TAG]))
    for q in ("0.60", "0.70", "0.75", "0.80", "0.85", "0.90", "0.95"):
        if q == "0.80":
            if FINAL_TAG[problem] in base:
                out.append(("q=0.80 (本次最终)", base[FINAL_TAG[problem]]))
            continue
        d = load(problem, f"_q{q}")
        if d:
            tag = next(iter(d))
            out.append((f"q={q}", d[tag]))
    return out


def pareto_front(rows: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """按 (费用, 紧急电量) 双目标取非支配子集。"""
    keep = []
    for name, s in rows:
        c, e = adj(s), s["emergency_kwh"]
        dominated = any(
            adj(s2) <= c and s2["emergency_kwh"] <= e
            and (adj(s2) < c or s2["emergency_kwh"] < e)
            for n2, s2 in rows if n2 != name
        )
        if not dominated:
            keep.append((name, s))
    return keep


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    L: list[str] = []
    A = L.append
    A("# 可靠性—经济 Pareto 汇总")
    A("")
    A(f"> 数据来源：`opt/out/*_阶梯*.json`（既有运行结果的后处理，未重跑）。"
      f"费用均为统一年末口径 $C-\\lambda(E_T-E_0)$，$\\lambda={LAMBDA}$。")
    A("")

    for p in ("Q2", "Q4-2", "Q3", "Q4-3"):
        rows = points(p)
        if not rows:
            continue
        up = dict(rows[0][1]) if rows and rows[0][0].startswith("上游") else None
        A(f"## {p}")
        A("")
        A("| 档位 | 校正后总费用/万元 | 紧急购电量/kWh | 紧急天数 | 单日 CVaR₉₅/万元 | "
          "未用计划电量/万kWh | 相对上游费用 | 相对上游紧急电量 |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|")
        for name, s in rows:
            c = adj(s)
            de = ""
            dc = ""
            if up is not None:
                dc = f"{(c-adj(up))/1e4:+,.2f} 万元"
                de = f"{s['emergency_kwh']-up['emergency_kwh']:+,.0f} kWh"
            A(f"| {name} | {c/1e4:,.2f} | {s['emergency_kwh']:,.0f} | "
              f"{s['days_with_emergency']} | {s['daily_cost_cvar95_yuan']/1e4:,.2f} | "
              f"{s['unused_plan_kwh']/1e4:,.2f} | {dc} | {de} |")
        A("")

        front = pareto_front(rows)
        A(f"**费用—紧急电量非支配点**（共 {len(rows)} 个档位，非支配 {len(front)} 个）："
          + "、".join(n for n, _ in front))
        A("")

        if up is not None:
            cap = up["emergency_kwh"]
            feas = [(n, s) for n, s in rows if s["emergency_kwh"] <= cap]
            if feas:
                best = min(feas, key=lambda kv: adj(kv[1]))
                A(f"**紧急购电量不超过上游（≤ {cap:,.0f} kWh）时最省的档位**：{best[0]}，"
                  f"校正后 {adj(best[1])/1e4:,.2f} 万元"
                  f"（相对上游 {(adj(best[1])-adj(up))/1e4:+,.2f} 万元）。")
            else:
                A(f"**紧急购电量不超过上游（≤ {cap:,.0f} kWh）的档位：无**——"
                  "所有已跑档位的紧急购电量都高于上游。")
            A("")

    out = OPT_DIR / "可靠性Pareto.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
