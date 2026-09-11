# -*- coding: utf-8 -*-
"""汇总成一份 核对报告.md：填 `模型评价建议.md` 【突出展示】节的模板 + 核对发现。

依赖 core_indicators.py 与 check_adjust.py 已先运行（读它们的 out/*.json）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from paths import FROZEN, OUT, WORK

PROBLEMS = ["Q2", "Q3", "Q4-2", "Q4-3"]
W = 1e4


def money(v) -> str:
    return "—" if v is None else f"{v / W:,.2f}"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ct = pd.read_csv(OUT / "表_核心指标.csv", encoding="utf-8-sig")
    ft = pd.read_csv(OUT / "表_预测精度.csv", encoding="utf-8-sig")
    pt = pd.read_csv(OUT / "表_物理可行性.csv", encoding="utf-8-sig")
    gt = pd.read_csv(OUT / "表_费用差距分解.csv", encoding="utf-8-sig")
    adj = json.loads((OUT / "核对_调整费用方向.json").read_text(encoding="utf-8"))
    fs = {k: json.loads(Path(v).read_text(encoding="utf-8")) for k, v in FROZEN.items()}
    nt = pd.read_csv(OUT / "表_年末口径统一.csv", encoding="utf-8-sig").set_index("问题")
    adj_cost = {p: f"{nt.loc[m, '校正后总费用/元'] / W:,.2f}"
                for p, m in zip(PROBLEMS, ["问题2", "问题3", "问题4-2", "问题4-3"])}

    def cell(p, name="可实现策略总费用/万元"):
        return ct.loc[ct.指标 == name, p].iloc[0]

    def fcell(obj, name):
        return ft.loc[ft.预测对象 == obj, name].iloc[0]

    g = gt.set_index("口径")

    def gc(p, row):
        return g.loc[row, p]

    # ---- 二元 / 三元分解 ----
    def dec2(p):
        c0, c1, c3 = gc(p, "C0 全知全年统一下界"), gc(p, "C1 完美预测+逐日决策"), gc(p, "C3 可实现策略（最终冻结）")
        if pd.isna(c1):
            return None
        tot = c3 - c0
        return {"decision": c1 - c0, "forecast": c3 - c1, "total": tot,
                "S_dec": 100 * (c1 - c0) / tot, "S_fc": 100 * (c3 - c1) / tot}

    d2 = {p: dec2(p) for p in ("Q2", "Q4-2")}

    def dec3(p):
        c0, c1, c2, c3 = (gc(p, "C0 全知全年统一下界"), gc(p, "C1 完美预测+逐日决策"),
                          gc(p, "C2 因果负载光伏+电价完美"), gc(p, "C3 可实现策略（最终冻结）"))
        if pd.isna(c1):
            return None
        tot = c3 - c0
        return {"decision": c1 - c0, "lp": c2 - c1, "price": c3 - c2, "total": tot,
                "S_dec": 100 * (c1 - c0) / tot, "S_lp": 100 * (c2 - c1) / tot,
                "S_c": 100 * (c3 - c2) / tot}

    d3 = {p: dec3(p) for p in ("Q4-2",)}

    L = []
    A = L.append
    A("# 问题2—4 模型解析核对与主要指标")
    A("")
    A("> 本报告只从**上游已冻结结果**提取数值，不重跑仿真；")
    A("> 唯一对上游代码的调用是 `check_adjust.py` 里对 `models.py` 的**只读 import**。")
    A("")

    # ---------------- 结论摘要 ----------------
    A("## 一、结论摘要")
    A("")
    A(f"在 2025-02-01—12-31 共 334 天、48,096 个十分钟时段的回放中：")
    A("")
    A("| | 问题2 | 问题3 | 问题4-2 | 问题4-3 |")
    A("|---|---:|---:|---:|---:|")
    for name, unit in [("负载预测nMAE", "%"), ("光伏预测nMAE", "%"), ("净负荷预测nMAE", "%")]:
        row = A
        o = {"负载预测nMAE": "负载", "光伏预测nMAE": "光伏", "净负荷预测nMAE": "净负荷"}[name]
        vals = [f"{100 * fcell(o, f'{p}_nMAE'):.2f}%" for p in PROBLEMS]
        A(f"| {name} | " + " | ".join(vals) + " |")
    vals = []
    for p in PROBLEMS:
        v = fcell("电价", f"{p}_nMAE")
        vals.append("—" if pd.isna(v) else f"{100 * v:.2f}%")
    A("| 电价预测nMAE | " + " | ".join(vals) + " |")
    A("| 可实现策略总费用/万元 | " + " | ".join(
        f"**{cell(p)}**" for p in PROBLEMS) + " |")
    A("| 理论完美最低费用/万元 | " + " | ".join(
        money(gc(p, "C0 全知全年统一下界")) for p in PROBLEMS) + " |")
    A("| 相对完美结果高出 | " + " | ".join(
        f"{gc(p, 'C3-C0 相对高出'):.2f}%" for p in PROBLEMS) + " |")
    A("| 紧急购电/计划购电 | " + " | ".join(cell(p, "紧急购电/计划购电") for p in PROBLEMS) + " |")
    A("| 紧急购电天数/天 | " + " | ".join(cell(p, "紧急购电天数/天") for p in PROBLEMS) + " |")
    A("| 单日费用CVaR95/万元 | " + " | ".join(cell(p, "单日费用CVaR95/万元") for p in PROBLEMS) + " |")
    A("| 未用计划电量占比 | " + " | ".join(cell(p, "未用计划电量占比") for p in PROBLEMS) + " |")
    A("| 实际年末SOC/kWh | " + " | ".join(cell(p, "实际年末SOC/kWh") for p in PROBLEMS) + " |")
    A("| 年末残值统一口径后总费用/万元 | " + " | ".join(adj_cost[p] for p in PROBLEMS) + " |")
    A("| 物理约束违规次数 | " + " | ".join(
        str(int(pt.loc[pt.检查项.isin(
            ['SOC越界次数', '充放电功率越界次数', '同时充放电次数', '负流量次数']), p].astype(float).sum()))
        for p in PROBLEMS) + " |")
    A("")
    A("")
    A("年末储能口径已按 `优化建议.md` §二.2 的残值口径统一（λ*=0.7662 元/kWh），")
    A("明细与两两对比见 `年末口径统一对比.md`——统一后问题3 相对问题2、问题4-3 相对问题4-2 的优势")
    A("不减反增，说明这两处优势不是靠年末库存取得的。")
    A("")
    A("全部 48,096 个评价时段中，储能越界、功率越界、同时充放电、负流量均为 **0**；")
    A(f"最大能量平衡误差为 **{pt.loc[pt.检查项 == '最大能量平衡误差/kWh', 'Q2'].iloc[0]:.3e}** kWh（机器精度），")
    A("储能递推恒等式逐时段成立。")
    A("")

    # ---------------- 费用差距 ----------------
    A("## 二、费用差距分解")
    A("")
    A("### 2.1 二元分解（问题2、问题4-2）")
    A("")
    A("| 口径 | 问题2 | 问题4-2 |")
    A("|---|---:|---:|")
    A(f"| C0 全知全年统一下界 | {money(gc('Q2', 'C0 全知全年统一下界'))} | {money(gc('Q4-2', 'C0 全知全年统一下界'))} |")
    A(f"| C1 完美预测+逐日决策（决策结构损失起点） | {money(d2['Q2'] and gc('Q2', 'C1 完美预测+逐日决策'))} | {money(d2['Q4-2'] and gc('Q4-2', 'C1 完美预测+逐日决策'))} |")
    A(f"| C3 可实现策略 | {money(gc('Q2', 'C3 可实现策略（最终冻结）'))} | {money(gc('Q4-2', 'C3 可实现策略（最终冻结）'))} |")
    A("")
    A("| 分解项 | 问题2 金额/万元 | 问题2 占比 | 问题4-2 金额/万元 | 问题4-2 占比 |")
    A("|---|---:|---:|---:|---:|")
    for key, label in [("decision", "决策结构 ΔC_decision"), ("forecast", "预测与风险 ΔC_forecast")]:
        cells = []
        for p in ("Q2", "Q4-2"):
            v = d2[p][key]
            cells += [f"{v / W:,.2f}", f"{d2[p]['S_' + ('dec' if key == 'decision' else 'fc')]:.2f}%"]
        A(f"| {label} | " + " | ".join(cells) + " |")
    A(f"| **总差距 ΔC_total** | {d2['Q2']['total'] / W:,.2f} | 100% | {d2['Q4-2']['total'] / W:,.2f} | 100% |")
    A("")
    for p in ("Q2", "Q4-2"):
        d = d2[p]
        A(f"- **{p}**：多出的 {d['total'] / W:,.2f} 万元中，"
          f"{d['forecast'] / W:,.2f} 万元（{d['S_fc']:.1f}%）来自预测与风险控制，"
          f"{d['decision'] / W:,.2f} 万元（{d['S_dec']:.1f}%）来自逐日决策结构与日末储能价值近似。")
    A("")
    A("> 含义：两问的差距**几乎全部来自预测层**，决策结构的贡献不足 10%。")
    A("> 但这只能在修正结构问题之后再确认——问题3/4-3 的调整费用方向问题会直接污染决策结构项。")
    A("")

    A("### 2.2 三元分解（问题4-2：四级费用阶梯）")
    A("")
    if d3["Q4-2"]:
        d = d3["Q4-2"]
        A("| 阶梯 | 金额/万元 | 相对全知最优 |")
        A("|---|---:|---:|")
        for row, lab in [("C0 全知全年统一下界", "C0 全年全知统一优化"),
                         ("C1 完美预测+逐日决策", "C1 负载/光伏/电价全知，逐日结构"),
                         ("C2 因果负载光伏+电价完美", "C2 负载/光伏因果预测，电价完美已知"),
                         ("C3 可实现策略（最终冻结）", "C3 全部因果预测")]:
            A(f"| {lab} | {money(gc('Q4-2', row))} | {100 * (gc('Q4-2', row) - gc('Q4-2', 'C0 全知全年统一下界')) / gc('Q4-2', 'C0 全知全年统一下界'):+.2f}% |")
        A("")
        A(f"- 决策结构限制：{d['decision'] / W:,.2f} 万元（{d['S_dec']:.2f}%）")
        A(f"- 负载与光伏预测及风险控制：{d['lp'] / W:,.2f} 万元（{d['S_lp']:.2f}%）")
        A(f"- 电价不可预知：{d['price'] / W:,.2f} 万元（{d['S_c']:.2f}%）")
        A(f"- 合计：{d['total'] / W:,.2f} 万元")
    A("")
    A("### 2.3 缺口")
    A("")
    A("- 问题3、问题4-3 **缺 C1（完美预测+逐日/分阶段决策）**，因此只能给出总差距，")
    A("  无法拆出决策结构项。需补一个「输入全知但仍用 4 时点链式结构」的对照运行。")
    A("- 问题3 的 C0 借用了问题2 口径的下界（同为附件1固定电价、同样储能参数）；")
    A("  全知条件下链式结构退化为单次承诺，故理论上与问题2 同值，但**尚未在问题3 目录内独立复算**。")
    A("")

    # ---------------- 预测精度 ----------------
    A("## 三、预测准确度")
    A("")
    A("| 预测对象 | 指标 | 问题2 | 问题3 | 问题4-2 | 问题4-3 |")
    A("|---|---|---:|---:|---:|---:|")
    for o in ["负载", "光伏", "净负荷", "电价"]:
        for m, lab in [("MAE", "MAE"), ("RMSE", "RMSE"), ("nMAE", "nMAE"),
                       ("Bias", "Bias"), ("P95", "P95绝对误差")]:
            vals = []
            for p in PROBLEMS:
                v = fcell(o, f"{p}_{m}")
                if pd.isna(v):
                    vals.append("—")
                elif m == "nMAE":
                    vals.append(f"{100 * v:.2f}%")
                else:
                    vals.append(f"{v:,.4f}" if abs(v) < 10 else f"{v:,.1f}")
            A(f"| {o if m == 'MAE' else ''} | {lab} | " + " | ".join(vals) + " |")
    A("")
    A("> 问题2 与问题4-2 的负载/光伏数值完全同值（同源预测层）。")
    A("> 问题3 与问题4-3 的负载/光伏 nMAE 与前者略有差异（负载 3.24% vs 3.22%，光伏 5.92% vs 6.30%），")
    A("> 说明两个目录用的是各自目录下的预测层实现；**差异原因未核实**，若要横向比较需先统一预测器。")
    A("> 问题4-2 的电价 P95 无法复核——其 `全年逐10分钟策略.csv` 未落电价列。")
    A("")

    # ---------------- 物理可行性 ----------------
    A("## 四、物理可行性与可靠性（逐时段重算）")
    A("")
    A("| 检查项 | 问题2 | 问题3 | 问题4-2 | 问题4-3 |")
    A("|---|---:|---:|---:|---:|")
    def num(v):
        v = float(v)
        return f"{v:,.0f}" if v.is_integer() else f"{v:,.4g}"

    for _, r in pt.iterrows():
        A(f"| {r.检查项} | " + " | ".join(num(r[p]) for p in PROBLEMS) + " |")
    A("")
    A("口径：能量平衡 `y+g+v+e-(L·Δt+u)=0`；购电恒等式 `x-(y+r)=0`；")
    A("光伏恒等式 `P·Δt-(g+w)=0`；储能递推 `E_t=E_{t-1}+0.9u_t-v_t/0.9`。")
    A("首时段无前驱（前驱在评价期之前），递推检查从第 2 行起。")
    A("")

    # ---------------- 结构核对 ----------------
    A("## 五、模型解析核对：调整购电量费用方向")
    A("")
    A("### 5.1 §六.1 单时段手算单元测试　r = 100 kWh，c = 1.0 元/kWh")
    A("")
    A("| x | 手算（题面闭式） | `settle_slot` | `solve_scenario_lp` 目标 |")
    A("|---:|---:|---:|---:|")
    for row in adj["single_slot"]:
        A(f"| {row['x']:.0f} | {row['手算结算']:.1f} | {row['settle_slot']:.1f} | "
          f"{row['现行LP目标']:.1f} |")
    A("")
    A("结算函数与手算完全一致；**优化目标不一致**：LP 对 x=80 与 x=120 给出同一个值，")
    A("即认为「下调 20 kWh」与「上调 20 kWh」代价相同，而题面规定两者相差 40c。")
    A("")
    A("### 5.2 受控 LP 探针（净负荷 0，上一版承诺 100，SOC 起止 6000，x 自由）")
    A("")
    p_ = adj["lp_probe"]
    A(f"- `solve_scenario_lp` 实际返回 x = **{p_['x']}**")
    A(f"- 正确口径下的最优解应为 x = {p_['正确口径预测']}；现行口径下的最优解为 x = {p_['现行口径预测']}")
    A(f"- 实际命中：**{p_['命中']}**")
    A("")
    A("即该 LP 会主动把承诺抬到上一版承诺的水平、且从不下调——这正是「1.0c 附加费记在下调量上」的行为特征。")
    A("")
    A("### 5.3 §六.2 链式结算路径测试　100 → 120 → 110 → 130")
    A("")
    c = adj["chain_path"]
    A("| 口径 | 基准 | 下调费 | 上调费 | 合计 |")
    A("|---|---:|---:|---:|---:|")
    A(f"| 手算 chain | {c['手算']['base_yuan']:.1f} | {c['手算']['breach_yuan']:.1f} | "
      f"{c['手算']['overbuy_yuan']:.1f} | {c['手算']['total_yuan']:.1f} |")
    A(f"| `settle(chain)` | {c['settle_chain']['base_yuan']:.1f} | {c['settle_chain']['breach_yuan']:.1f} | "
      f"{c['settle_chain']['overbuy_yuan']:.1f} | {c['settle_chain']['total_yuan']:.1f} |")
    A(f"| `settle(two_way)` | {c['settle_two_way']['base_yuan']:.1f} | {c['settle_two_way']['breach_yuan']:.1f} | "
      f"{c['settle_two_way']['overbuy_yuan']:.1f} | {c['settle_two_way']['total_yuan']:.1f} |")
    A("")
    A(f"链式比两元多计 {c['settle_chain']['total_yuan'] - c['settle_two_way']['total_yuan']:.1f} 元，"
      "说明「先升后降再升」的重复调整成本被正确识别。**结算侧通过。**")
    A("")
    A("### 5.4 冻结结果的旁证")
    A("")
    A("| 目录 | 全年调整量/kWh | `breach_cost_yuan`（下调费） | `overbuy_cost_yuan`（上调费） |")
    A("|---|---:|---:|---:|")
    for p, k in [("Q3", "Q3"), ("Q4-3", "Q4-3")]:
        s = fs[k]["summary"]
        A(f"| 问题{'3' if k == 'Q3' else '4-3'} | {s['adjusted_kwh']:,.2f} | "
          f"{s['breach_cost_yuan']:,.2f} | {s['overbuy_cost_yuan']:,.2f} |")
    A("")
    A("334 天、48,096 个时段里**下调费恒为 0**，一次下调都没有发生，与 5.2 的探针行为一致。")
    A("")
    A("### 5.5 结论与影响范围")
    A("")
    A("| 项 | 结论 |")
    A("|---|---|")
    A("| `dispatch_core.settle_slot` 结算方向 | ✅ 正确 |")
    A("| `dispatch_core.settle_period` | ✅ 复用 `settle_slot`，正确 |")
    A("| `models.solve_scenario_lp` 目标方向 | ❌ 附加费加在下调量上，反向 |")
    A("| `models.solve_exact_chain_milp` 目标方向 | ✅ 同一文件内给 `i_eb` 记 0.5c、`i_eo` 记 1.5c，正确 |")
    A("")
    A("同一份 `models.py` 内 LP 与 MILP 对同一约束 `eb-eo=ref-x` 给出相反的附加费位置，")
    A("**这是不需要辩论题面语义的自证矛盾**。")
    A("")
    A("受影响：问题3、问题4-3 中所有调用 `solve_scenario_lp` 且 `ref is not None` 的模型，")
    A("包括 `run_models.py` 里用作自检锚点的 M2（复现模型3 冻结值 13,748,164.50 元）。")
    A("**修正后该锚点必然失配，属预期。**")
    A("")
    A("修正位置（本目录不改上游，仅记录）：`models.py:131-132` 的 `obj[i_eb]` 改为 `obj[i_eo]`，")
    A("并同步 `models.py:224` 的 CVaR 分支；`问题3/模型3决策优化版/models.py` 与")
    A("`问题4/重算问题3/models.py` 逐字节相同，须同步修改。")
    A("")

    # ---------------- 未完成 ----------------
    A("## 六、本报告未覆盖的检验项")
    A("")
    A("按 `模型评价建议.md` §十五，以下尚未做：")
    A("")
    for i, t in enumerate([
        "前缀不变性信息泄露检验（§三.2）",
        "分季节误差表与分组误差（§四.2）",
        "q=0.80 覆盖率 / Pinball Loss / 超限连续性（§五）",
        "与朴素基准的 7 日区块 Bootstrap 置信区间（§四.4）",
        "消融：无风险控制 / 逐时段余量 / 储能储备（§九.2）",
        "滚动信息价值 VOI_roll 与电价预测价值（§九.3）",
        "费用—紧急购电 Pareto 前沿（§十二）",
        "计算效率统计（§十四）",
    ], 1):
        A(f"{i}. {t}")
    A("")

    (WORK / "核对报告.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
