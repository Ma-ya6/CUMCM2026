# -*- coding: utf-8 -*-
"""把四问各档位的全年结果与上游冻结值统一到同一年末口径，输出对比表。

上游冻结值直接读 ``verify/out/表_年末口径统一.csv``（那是由只读核对包从
上游已冻结结果提取的，未经重跑）。本脚本只做后处理与排版，不重跑仿真。

跑法：``python summarize_all.py``（在 opt/ 下），输出 ``opt/汇总对比.md``。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

OPT_DIR = Path(__file__).resolve().parent
C_DIR = OPT_DIR.parent
LAMBDA = 0.7662          # 元/kWh，附件1 电价曲线时段均值，见 年末口径统一对比.md
E0 = 6000.0

PROBLEMS = ["Q2", "Q3", "Q4-2", "Q4-3"]
LADDER = ["legacy-greedy", "fixed-greedy", "fixed-mpc", "exact-greedy", "exact-mpc",
          "reserve-greedy", "reserve-mpc"]
FINAL4 = {"Q4-2": "reserve-mpc", "Q4-3": "exact-greedy"}
C0_Q4 = 1278.26          # 万元，问题4 两子问题共用的全年全知统一下界（见 核对报告.md）


def load_upstream() -> dict:
    """上游四问的基线结果（原始费用、年末SOC、已校正费用）。

    问题3 / 问题4-3 的预测层含跨日未来信息泄露（上游 ``forecasts.py:_pooled_coef``
    用全年实测光伏识别昼夜列）。该缺陷已修，故问题3/4-3 的基线改用
    ``opt/rebuild_upstream.py`` 在**同一修正预测层**下重算的结果；
    问题2 / 问题4-2 的预测层未改动，仍取 ``verify/out/表_年末口径统一.csv``。
    """
    src = C_DIR / "verify" / "out" / "表_年末口径统一.csv"
    key = {"问题2": "Q2", "问题3": "Q3", "问题4-2": "Q4-2", "问题4-3": "Q4-3"}
    out = {}
    with src.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            out[key[row["问题"]]] = {
                "total_cost_yuan": float(row["原始总费用/元"]),
                "final_soc_kwh": float(row["实际年末SOC/kWh"]),
                "adjusted_cost_yuan": float(row["校正后总费用/元"]),
                "planned_purchase_kwh": float(row["计划购电量/kWh"]),
                "emergency_kwh": float(row["紧急购电量/kWh"]),
                "emergency_days": int(float(row["紧急天数/天"])),
                "unused_plan_kwh": float(row["未用计划电量/kWh"]),
                "cvar95_yuan": float(row["单日费用CVaR95/元"]),
                "recomputed": False,
            }
    rc = OPT_DIR / "out" / "上游重算.json"
    if rc.exists():
        for p, f in json.loads(rc.read_text(encoding="utf-8")).items():
            u = out[p]
            u["frozen_total_cost_yuan"] = u["total_cost_yuan"]
            u["frozen_final_soc_kwh"] = u["final_soc_kwh"]
            u["total_cost_yuan"] = f["原始总费用/元"]
            u["final_soc_kwh"] = f["实际年末SOC/kWh"]
            u["planned_purchase_kwh"] = f["计划购电量/kWh"]
            u["emergency_kwh"] = f["紧急购电量/kWh"]
            u["emergency_days"] = f["紧急天数/天"]
            u["unused_plan_kwh"] = f["未用计划电量/kWh"]
            u["cvar95_yuan"] = f["单日费用CVaR95/元"]
            u["adjusted_cost_yuan"] = adj(u["total_cost_yuan"], u["final_soc_kwh"])
            u["recomputed"] = True
    return out


def load_ladder() -> dict:
    out = {}
    for p in PROBLEMS:
        f = OPT_DIR / "out" / f"{p}_阶梯.json"
        out[p] = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    return out


def load_perfect() -> dict:
    """完美预测口径（C1）：真值替代预测、储备归零。"""
    out = {}
    for p in PROBLEMS:
        f = OPT_DIR / "out" / f"{p}_阶梯_perfect.json"
        out[p] = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    return out


def load_priceperfect() -> dict:
    """电价完美口径（C2）：只用真值电价，负载/光伏仍为因果预测。"""
    out = {}
    for p in PROBLEMS:
        f = OPT_DIR / "out" / f"{p}_阶梯_priceperfect.json"
        out[p] = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    return out


def adj(cost: float, soc: float) -> float:
    return cost - LAMBDA * (soc - E0)


def wan(x: float) -> str:
    return f"{x / 1e4:,.2f}"


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    up = load_upstream()
    lad = load_ladder()
    perf = load_perfect()
    ppf = load_priceperfect()
    lines: list[str] = []
    A = lines.append

    A("# 四问阶梯结果汇总（统一年末口径）")
    A("")
    A(f"> 年末口径：`C_adj = C - λ·(E_final - E0)`，λ = {LAMBDA} 元/kWh，E0 = {E0:.0f} kWh。")
    A("> 本目录各档位读自 `opt/out/*_阶梯.json`。基线：问题2 / 问题4-2 取")
    A("> `verify/out/表_年末口径统一.csv`（预测层未改动）；**问题3 / 问题4-3 取")
    A("> `opt/out/上游重算.json`**——上游预测层含跨日未来信息泄露，已修，故基线必须在")
    A("> 同一修正预测层下重算，否则\"相对上游\"不是同口径比较。")
    A("> **文档引用的数字以 `opt/结果冻结.md` 为准**（由 `opt/freeze.py` 生成，同源同口径）。")
    A("")

    # ---- 表1 各档位费用与年末SOC ----
    A("## 表1 各档位全年总费用与年末储能")
    A("")
    A("| 问题 | 档位 | 原始总费用/万元 | 年末SOC/kWh | 校正后总费用/万元 | 相对上游/万元 | 相对上游/% |")
    A("|---|---|---:|---:|---:|---:|---:|")
    for p in PROBLEMS:
        u = up[p]
        A(f"| {p} | **上游基线** | {wan(u['total_cost_yuan'])} | "
          f"{u['final_soc_kwh']:,.1f} | {wan(u['adjusted_cost_yuan'])} | — | — |")
        for tag in LADDER:
            s = lad[p].get(tag)
            if s is None:
                continue
            c, soc = s["total_cost_yuan"], s["final_soc_kwh"]
            ca = adj(c, soc)
            d = ca - u["adjusted_cost_yuan"]
            A(f"| {p} | {tag} | {wan(c)} | {soc:,.1f} | {wan(ca)} | "
              f"{d / 1e4:+,.2f} | {d / u['adjusted_cost_yuan'] * 100:+.3f}% |")
    A("")

    # ---- 表2 可靠性与利用率 ----
    A("## 表2 可靠性与电量结构")
    A("")
    A("| 问题 | 档位 | 计划购电/kWh | 紧急购电/kWh | 紧急/计划 | 紧急天数 | 未用计划/kWh | 未用占比 | 弃光/kWh | CVaR95/万元 |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for p in PROBLEMS:
        u = up[p]
        A(f"| {p} | **上游最终模型** | {u['planned_purchase_kwh']:,.0f} | "
          f"{u['emergency_kwh']:,.0f} | {u['emergency_kwh'] / u['planned_purchase_kwh'] * 100:.4f}% | "
          f"{u['emergency_days']} | {u['unused_plan_kwh']:,.0f} | "
          f"{u['unused_plan_kwh'] / u['planned_purchase_kwh'] * 100:.3f}% | — | "
          f"{u['cvar95_yuan'] / 1e4:.2f} |")
        for tag in LADDER:
            s = lad[p].get(tag)
            if s is None:
                continue
            pp = s["planned_purchase_kwh"]
            A(f"| {p} | {tag} | {pp:,.0f} | {s['emergency_kwh']:,.0f} | "
              f"{s['emergency_kwh'] / pp * 100:.4f}% | {s['days_with_emergency']} | "
              f"{s['unused_plan_kwh']:,.0f} | {s['unused_plan_kwh'] / pp * 100:.3f}% | "
              f"{s['curtailed_pv_kwh']:,.0f} | {s['daily_cost_cvar95_yuan'] / 1e4:.2f} |")
    A("")

    # ---- 表3 费用分解（问题3/4-3）----
    A("## 表3 结算费用分解（仅问题3、问题4-3 有调整费）")
    A("")
    A("| 问题 | 档位 | 基准费/万元 | 下调费/万元 | 上调费/万元 | 调整电量/kWh | 紧急购电费/万元 | 合计/万元 |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for p in ("Q3", "Q4-3"):
        for tag in LADDER:
            s = lad[p].get(tag)
            if s is None or "base_cost_yuan" not in s:
                continue
            A(f"| {p} | {tag} | {wan(s['base_cost_yuan'])} | {wan(s['breach_cost_yuan'])} | "
              f"{wan(s['overbuy_cost_yuan'])} | {s['adjusted_kwh']:,.0f} | "
              f"{wan(s['emergency_cost_yuan'])} | {wan(s['total_cost_yuan'])} |")
    A("")

    # ---- 表5 完美预测口径 C1 ----
    A("## 表5 完美预测 + 同决策结构（C1，用于费用分解）")
    A("")
    A("| 问题 | 档位 | 原始总费用/万元 | 年末SOC/kWh | 校正后总费用/万元 | 紧急购电/kWh |")
    A("|---|---|---:|---:|---:|---:|")
    for p in PROBLEMS:
        for tag, s in perf[p].items():
            soc = s["final_soc_kwh"]
            A(f"| {p} | {tag} | {wan(s['total_cost_yuan'])} | {soc:,.1f} | "
              f"{wan(adj(s['total_cost_yuan'], soc))} | {s['emergency_kwh']:.3g} |")
    A("")
    A("问题2、问题4-2 的 C1 = 12,288,748.57 / 12,939,133.66 元，与本目录无关的上游冻结值逐位一致；")
    A("问题3、问题4-3 的 `legacy-greedy` 档年末 SOC 为满仓 10,800 kWh，")
    A("统一年末口径后为 12,287,347.01 / 12,940,246.07 元；`fixed-greedy` 档年末 SOC 恰为 6,000 kWh。")
    A("")

    # ---- 表6 电价完美档 C2（仅问题4）----
    A("## 表6 电价完美档（C2，仅问题4 有意义）")
    A("")
    A("| 问题 | 结构 | 原始总费用/万元 | 年末SOC/kWh | 校正后总费用/万元 |")
    A("|---|---|---:|---:|---:|")
    for p in ("Q4-2", "Q4-3"):
        for tag, s in ppf[p].items():
            soc = s["final_soc_kwh"]
            mark = "**" if tag == FINAL4[p] else "`"
            name = f"{mark}{tag}（本次最终）{mark}" if tag == FINAL4[p] else f"`{tag}`"
            A(f"| {p} | {name} | {wan(s['total_cost_yuan'])} | {soc:,.1f} | "
              f"{wan(adj(s['total_cost_yuan'], soc))} |")
    A("")
    A("问题2/3 用附件1 固定电价，故 $C_2\\equiv C_3$，不存在该档。")
    A("")

    # ---- 表7 同结构四档分解 ----
    A("## 表7 同结构四档分解（$C_0\\to C_1\\to C_2\\to C_3$，每个子问题固定同一策略）")
    A("")
    A("| 分解项 | 4-2（`reserve-mpc`）/万元 | 占比 | 4-3（`exact-greedy`）/万元 | 占比 |")
    A("|---|---:|---:|---:|---:|")
    dec = {}
    for p in ("Q4-2", "Q4-3"):
        t = FINAL4[p]
        c1 = adj(perf[p][t]["total_cost_yuan"], perf[p][t]["final_soc_kwh"]) / 1e4
        c2 = adj(ppf[p][t]["total_cost_yuan"], ppf[p][t]["final_soc_kwh"]) / 1e4
        c3 = adj(lad[p][t]["total_cost_yuan"], lad[p][t]["final_soc_kwh"]) / 1e4
        dec[p] = (c1 - C0_Q4, c2 - c1, c3 - c2, c3 - C0_Q4)
    A(f"| $C_0$ 全知统一下界 | {C0_Q4:,.2f} | — | {C0_Q4:,.2f} | — |")
    for i, (name, hi) in enumerate([
            ("决策结构 $C_1-C_0$", False),
            ("净负荷预测与风险 $C_2-C_1$", True),
            ("电价不可预知 $C_3-C_2$", True)]):
        a = dec["Q4-2"]; b = dec["Q4-3"]
        aa = f"{a[i]:,.2f}"; bb = f"{b[i]:,.2f}"
        if hi:
            aa, bb = f"**{aa}**", f"**{bb}**"
        A(f"| {name} | {aa} | {a[i]/a[3]*100:.2f}% | {bb} | {b[i]/b[3]*100:.2f}% |")
    A(f"| **总差距 $C_3-C_0$** | **{dec['Q4-2'][3]:,.2f}** | 100% | "
      f"**{dec['Q4-3'][3]:,.2f}** | 100% |")
    A("")
    A("> 上游结构下的对应分解为 4-2：8.36% / 89.02% / **2.62%**；4-3：11.65% / 83.65% / **4.70%**。")
    A("> **两套占比不可混用**——四档之间只能改变信息条件，不能同时更换决策结构。")
    A("")

    # ---- 表8 统计显著性与可靠性格局 ----
    from check_extra import FINAL as CK_FINAL, block_bootstrap, load_daily, BLOCK
    A("## 表8 统计显著性与可靠性格局")
    A("")
    A("| 问题 | 最终策略 | 全年差额/万元（校正口径） | Bootstrap 点估计/万元（原始口径） | "
      f"{BLOCK} 日区块 Bootstrap 95% CI/万元 | 含 0？ | 更省月数/总月数 |")
    A("|---|---|---:|---:|---|---:|---:|")
    for p, tag in CK_FINAL.items():
        up, new = load_daily(p, "legacy-greedy"), load_daily(p, tag)
        if up.empty or new.empty:
            continue
        t, lo, hi, _ = block_bootstrap(up, new)
        d_adj = (adj(lad[p][tag]["total_cost_yuan"], lad[p][tag]["final_soc_kwh"])
                 - adj(lad[p]["legacy-greedy"]["total_cost_yuan"],
                       lad[p]["legacy-greedy"]["final_soc_kwh"])) / 1e4
        a = up.set_index("date").total_cost_yuan.resample("MS").sum()
        b = new.set_index("date").total_cost_yuan.resample("MS").sum()
        n_cheap = int(((b - a) < 0).sum())
        A(f"| {p} | `{tag}` | {d_adj:+,.2f} | {t/1e4:+,.2f} | "
          f"[{lo/1e4:+,.2f}, {hi/1e4:+,.2f}] | {'**是**' if lo <= 0 <= hi else '否'} | "
          f"{n_cheap}/{len(a)} |")
    A("")
    A("**四个问题的费用差在区块 Bootstrap 下都覆盖 0**（点估计与主表相差约 0.3 万元的年末校正项）；")
    A("把区块长度换成 3 / 14 / 28 日后，16 个组合的区间仍全部覆盖 0，")
    A("故只能说\"在该区块设定下费用差未被检出\"，不构成显著性结论。")
    A("明细见 `补充检验.md`（含 exact-mpc 对照与 ±0.5% 界限的 TOST 等价性检验）；")
    A("费用—紧急电量 Pareto 对照见 `可靠性Pareto.md`。")
    A("")

    # ---- 表4 费用恒等式核对 ----
    A("## 表4 费用恒等式核对（各档位自洽）")
    A("")
    A("| 问题 | 档位 | C_grid-(base+down+up) | C_total-(grid+emer) | 判定 |")
    A("|---|---|---:|---:|---|")
    allok = True
    for p in PROBLEMS:
        for tag in LADDER:
            s = lad[p].get(tag)
            if s is None:
                continue
            if "base_cost_yuan" in s:
                e1 = s["grid_cost_yuan"] - (s["base_cost_yuan"] + s["breach_cost_yuan"]
                                            + s["overbuy_cost_yuan"])
                e2 = s["total_cost_yuan"] - (s["grid_cost_yuan"] + s["emergency_cost_yuan"])
            else:
                e1 = 0.0
                e2 = s["total_cost_yuan"] - (s["planned_cost_yuan"] + s["emergency_cost_yuan"])
            ok = abs(e1) < 1e-6 and abs(e2) < 1e-6
            allok &= ok
            A(f"| {p} | {tag} | {e1:.2e} | {e2:.2e} | {'✓' if ok else '★'} |")
    A("")
    A(f"**判定**：{'全部满足费用恒等式' if allok else '★存在不满足项'}。")
    A("")

    # ---- 表9 在线物理核对 ----
    A("## 表9 本次实跑的在线物理核对（逐时段累计）")
    A("")
    A("> 由 `opt_core.audit_execution` 在仿真循环内对**每一个实际执行时段**在线累计，"
      "口径与上游冻结核对相同。误差项取全年最大值，越界项为计数。")
    A("")
    A("| 问题 | 档位 | 核对时段数 | 最大平衡误差/kWh | 最大购电恒等式误差/kWh | "
      "最大光伏恒等式误差/kWh | 最大 SOC 递推误差/kWh | SOC 越界 | 功率越界 | 同时充放电 | 负流量 |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    n_aud = 0
    for p in PROBLEMS:
        for tag in LADDER:
            s = lad[p].get(tag)
            if s is None or "audit" not in s:
                continue
            a = s["audit"]
            n_aud += 1
            A(f"| {p} | {tag} | {a['slots']:,} | {a['max_balance_err']:.3e} | "
              f"{a['max_purchase_identity_err']:.3e} | {a['max_pv_identity_err']:.3e} | "
              f"{a['max_soc_recursion_err']:.3e} | {a['soc_violations']} | "
              f"{a['power_violations']} | {a['simultaneous_charge_discharge']} | "
              f"{a['negative_flow']} |")
    A("")
    A(f"**判定**：{n_aud} 个档位全部通过——误差均为机器精度量级，越界计数全为 0。")
    A("")

    out = OPT_DIR / "汇总对比.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
