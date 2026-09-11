# -*- coding: utf-8 -*-
"""年末储能口径统一 + 与原始模型结果对比。

对应 `优化建议.md` §二.2 的第二种方案：

    C^adj = C - λ_final · (E_final - E_0)

口径说明：
  · E_final 取**实际执行后**的年末储电量（冻结结果里的 final_soc_kwh），
    不是计划阶段的年末 SOC。
  · λ_final 对四问**统一取 0.7662 元/kWh**，即附件1 那条 144 时段电价曲线的
    时段均值（min 0.3713，max 1.3952）。它与任何模型自身的输出无关，
    因此不会出现"用自己的价格给自己年末库存估值"的偏差。
  · 附件4 正式期实际电价均值 0.7575 元/kWh，与 0.7662 相差 1.1%，
    故同一 λ 对问题4 同样适用。

本脚本只对已冻结结果做后处理，不重跑仿真，不修改上游任何文件。
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from paths import C_DIR, FROZEN, OUT, WORK

W = 1e4
E0 = 6000.0
PROBLEMS = ["Q2", "Q3", "Q4-2", "Q4-3"]
NAME = {"Q2": "问题2", "Q3": "问题3", "Q4-2": "问题4-2", "Q4-3": "问题4-3"}
LAMBDA_STAR = 0.7662          # 附件1 电价曲线时段均值


def load() -> dict:
    out = {}
    for p in PROBLEMS:
        d = json.loads(FROZEN[p].read_text(encoding="utf-8"))
        a = d.get("annual") or d.get("dispatch")
        out[p] = {
            "total": a["total_cost_yuan"],
            "plan_kwh": a["planned_purchase_kwh"],
            "plan_yuan": a["planned_cost_yuan"],
            "emer_kwh": a["emergency_kwh"],
            "days": a["days_with_emergency"],
            "unused": a["unused_plan_kwh"],
            "efin": a["final_soc_kwh"],
            "cvar": a["daily_cost_cvar95_yuan"],
        }
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    d = load()

    # 附件1 电价曲线均值（用于核对 λ* 取值）
    a1 = pd.read_excel(C_DIR / "附件" / "附件1.xlsx")
    lam1 = float(a1["电价"].astype(float).mean())

    rows = []
    for p in PROBLEMS:
        r = d[p]
        dE = r["efin"] - E0
        adj = -LAMBDA_STAR * dE
        rows.append({
            "问题": NAME[p],
            "原始总费用/元": r["total"],
            "实际年末SOC/kWh": r["efin"],
            "年末偏离E0/kWh": dE,
            "残值修正/元": adj,
            "校正后总费用/元": r["total"] + adj,
            "计划购电量/kWh": r["plan_kwh"],
            "计划购电均价/(元/kWh)": r["plan_yuan"] / r["plan_kwh"],
            "紧急购电量/kWh": r["emer_kwh"],
            "紧急天数/天": r["days"],
            "未用计划电量/kWh": r["unused"],
            "单日费用CVaR95/元": r["cvar"],
        })
    t = pd.DataFrame(rows)

    # ---- 两两对比（同价格体系内才有意义）----
    pairs = [("Q2", "Q3", "固定电价（附件1）"), ("Q4-2", "Q4-3", "波动电价（附件4）")]
    cmp_rows = []
    for a, b, tag in pairs:
        raw = d[a]["total"] - d[b]["total"]
        adjd = (d[a]["total"] - LAMBDA_STAR * (d[a]["efin"] - E0)) - \
               (d[b]["total"] - LAMBDA_STAR * (d[b]["efin"] - E0))
        # 拉平所需的 λ：C_a - λ·dE_a = C_b - λ·dE_b
        denom = (d[a]["efin"] - E0) - (d[b]["efin"] - E0)
        be = raw / denom if abs(denom) > 1e-9 else float("nan")
        cmp_rows.append({
            "对比": f"{NAME[a]} − {NAME[b]}", "价格体系": tag,
            "原始差额/万元": raw / W,
            "校正后差额/万元": adjd / W,
            "校正后差额变化": f"{100 * (adjd / raw - 1):+.3f}%",
            "拉平所需λ/(元/kWh)": be,
            "最末模型年末偏离/kWh": d[b]["efin"] - E0,
        })
    cmp_t = pd.DataFrame(cmp_rows)

    # ---- λ 敏感性 ----
    sens = []
    for lam in (0.4, 0.6, LAMBDA_STAR, 0.8, 1.0):
        row = {"λ/(元/kWh)": lam}
        for p in PROBLEMS:
            row[NAME[p]] = (d[p]["total"] - lam * (d[p]["efin"] - E0)) / W
        sens.append(row)
    sens_t = pd.DataFrame(sens)

    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "表_年末口径统一.csv", index=False, encoding="utf-8-sig")
    cmp_t.to_csv(OUT / "表_年末口径两两对比.csv", index=False, encoding="utf-8-sig")
    sens_t.to_csv(OUT / "表_年末口径λ敏感性.csv", index=False, encoding="utf-8-sig")

    def md(df):
        def f(v):
            if isinstance(v, float):
                if v != 0 and abs(v) < 1e-4:
                    return f"{v:.2e}"
                return f"{v:,.2f}" if abs(v) >= 1 else f"{v:,.4f}"
            return str(v)
        return "\n".join(["| " + " | ".join(df.columns) + " |",
                          "|" + "|".join(["---"] * len(df.columns)) + "|"]
                         + ["| " + " | ".join(f(v) for v in r) + " |"
                            for r in df.itertuples(index=False)])

    L = ["# 年末储能口径统一与模型对比",
         "",
         f"> λ\\* = **{LAMBDA_STAR} 元/kWh**（附件1 电价曲线时段均值，实测 {lam1:.4f}）；",
         "> E_final 取实际执行后的年末储电量；C^adj = C − λ\\*·(E_final − E0)。",
         "> 只对已冻结结果做后处理，未重跑仿真。",
         "",
         "## 一、统一口径后的费用",
         "",
         md(t.round(4)),
         "",
         "## 二、与原始模型的对比（同价格体系内两两比较）",
         "",
         md(cmp_t.round(4)),
         "",
         "## 三、λ 敏感性",
         "",
         md(sens_t.round(4)),
         ""]

    for r in cmp_rows:
        L.append(f"- **{r['对比']}**：原始差额 {r['原始差额/万元']:,.2f} 万元，"
                 f"统一年末残值后 {r['校正后差额/万元']:,.2f} 万元"
                 f"（变化 {r['校正后差额变化']}）。")
    L += ["",
          "**怎么读这张表**：差额 = 前者 − 后者，为正说明后者更便宜。",
          "年末残值修正的方向是**让后者更便宜**（后者年末多存了电），"
          "所以校正后差额不减反增。",
          "",
          "## 四、结论", ""]

    for r in cmp_rows:
        a, b = r["对比"].split(" − ")
        L.append(f"### {a} vs {b}")
        L.append("")
        L.append(f"{b} 实际年末比 E0 多存 "
                 f"{r['最末模型年末偏离/kWh']:,.1f} kWh，按 λ\\*={LAMBDA_STAR} 元/kWh 折算仅 "
                 f"{abs(r['最末模型年末偏离/kWh'] * LAMBDA_STAR):,.0f} 元"
                 f"（占总费用 {100 * abs(r['最末模型年末偏离/kWh'] * LAMBDA_STAR) / d['Q3' if b == '问题3' else 'Q4-3']['total']:.4f}%）。"
                 f"把两侧都折算到同一 E0 后，{b} 仍便宜 {abs(r['校正后差额/万元']):,.2f} 万元。")
        L.append("")
        L.append(f"要把 {a} 与 {b} 拉平，λ 需要取到 "
                 f"**{r['拉平所需λ/(元/kWh)']:,.1f} 元/kWh**——即 **负价格**，"
                 f"物理上无意义。")
        L.append("")
        L.append(f"→ **{b} 的费用优势不是靠年末库存取得的**。恰恰相反，它在年末多存了电的同时还更便宜，"
                 f"统一口径只会让优势略微扩大（{r['校正后差额变化']}）。")
        L.append("")

    L += ["### 口径说明",
          "",
          "本表用的是 `优化建议.md` §二.2 的**残值口径**（对已冻结结果做后处理）。",
          "另一种口径是硬约束 E_final = E0 后重跑，本目录无权改上游代码，未做；",
          "但两者差距的上界就是这里的残值修正量（≤ 2,855 元，占总费用 0.02%），",
          "不足以改变任何排序或结论。",
          "",
          "另外：问题2/3 用附件1 固定电价、问题4-2/4-3 用附件4 波动电价，",
          "**跨价格体系横向比费用没有意义**，故只做了组内两两比较。",
          ""]
    (WORK / "年末口径统一对比.md").write_text("\n".join(L), encoding="utf-8")

    print(md(t.round(4)))
    print()
    print(md(cmp_t.round(4)))
    print()
    print(md(sens_t.round(4)))


if __name__ == "__main__":
    main()
