# -*- coding: utf-8 -*-
"""主要指标提取：按 `模型评价建议.md` 【突出展示】节的表 2 / 表 6 / 表 10 口径。

全部数值来自上游**已冻结**的结果文件（只读），本脚本不重跑任何仿真。
产出：out/核心指标汇总.json 与 out/核心指标汇总.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from paths import (DT, E_INITIAL, E_MAX, E_MIN, FROZEN, ORACLE_Q2Q4, ORACLE_Q4_2,
                   ORACLE_Q4_3, OUT, P_MAX, SLOTS)

W = 1e4  # 元 -> 万元
PROBLEMS = ["Q2", "Q3", "Q4-2", "Q4-3"]


# --------------------------------------------------------------------------
def load_frozen() -> dict:
    return {k: json.loads(Path(v).read_text(encoding="utf-8")) for k, v in FROZEN.items()}


def load_slots() -> dict:
    return {k: pd.read_csv(v, encoding="utf-8-sig") for k, v in SLOTS.items()}


# --------------------------------------------------------------------------
def forecast_table(frozen: dict, slots: dict) -> pd.DataFrame:
    """表 6 预测准确度：MAE / RMSE / nMAE / Bias / P95 绝对误差。

    MAE..Bias 直接取自冻结 JSON；P95 由逐 10 分钟策略 CSV 重算。
    """
    # 冻结 JSON 里 Q2/Q4-2 有 forecast 块，Q3/Q4-3 没有；后者一律由 CSV 重算。
    # 有 JSON 时优先用 JSON 值，CSV 重算值仍照常输出以便对照。
    rows = []
    specs = [
        ("负载", "load", "L_actual_kw", "L_fc_kw"),
        ("光伏", "pv", "P_act_kw", "P_fc_kw"),
        ("净负荷", "net_load", None, None),
    ]
    for label, key, act, fc in specs:
        rec = {"预测对象": label}
        for p in PROBLEMS:
            d = slots[p]
            if key == "net_load":
                a = d.L_actual_kw - d.P_act_kw
                f = d.L_fc_kw - d.P_fc_kw
            else:
                a, f = d[act], d[fc]
            e = a - f
            j = frozen[p].get("forecast", {}).get(key, {})
            rec[f"{p}_MAE"] = j.get("mae_kw", float(e.abs().mean()))
            rec[f"{p}_RMSE"] = j.get("rmse_kw", float(np.sqrt((e ** 2).mean())))
            rec[f"{p}_nMAE"] = j.get("nmae", float(e.abs().mean() / a.abs().mean()))
            rec[f"{p}_Bias"] = j.get("bias_kw", float(e.mean()))
            rec[f"{p}_P95"] = float(e.abs().quantile(0.95))
        rows.append(rec)

    rec = {"预测对象": "电价"}
    for p in PROBLEMS:
        d = slots[p]
        j = frozen[p].get("price_forecast", {})
        if not j:
            for s in ("MAE", "RMSE", "nMAE", "Bias", "P95"):
                rec[f"{p}_{s}"] = None
            continue
        rec[f"{p}_MAE"] = j.get("mae_yuan_per_kwh")
        rec[f"{p}_RMSE"] = j.get("rmse_yuan_per_kwh")
        rec[f"{p}_nMAE"] = j.get("nmae")
        rec[f"{p}_Bias"] = j.get("bias_yuan_per_kwh")
        if "c_actual_yuan_per_kwh" in d.columns:
            e = d.c_actual_yuan_per_kwh - d.c_fc_yuan_per_kwh
            rec[f"{p}_P95"] = float(e.abs().quantile(0.95))
        else:
            rec[f"{p}_P95"] = None   # 该目录的策略 CSV 未落电价列，P95 无法复核
    rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def annual_block(frozen: dict, p: str) -> dict:
    """各问题冻结 JSON 里年块 / 调度块的键名不统一，这里做一次归一。"""
    d = frozen[p]
    return d.get("annual") or d.get("dispatch")


def cost_table(frozen: dict, slots: dict) -> pd.DataFrame:
    """表 2 核心指标。"""
    rows = []

    def add(name, vals, fmt="{:.4g}"):
        rows.append({"指标": name, **{p: fmt.format(v) if v is not None else "—"
                                      for p, v in zip(PROBLEMS, vals)}})

    a = {p: annual_block(frozen, p) for p in PROBLEMS}
    tot = [a[p]["total_cost_yuan"] for p in PROBLEMS]
    plan = [a[p]["planned_purchase_kwh"] for p in PROBLEMS]
    emer = [a[p]["emergency_kwh"] for p in PROBLEMS]
    days = [a[p]["days_with_emergency"] for p in PROBLEMS]
    cvar = [a[p]["daily_cost_cvar95_yuan"] for p in PROBLEMS]
    unused = [a[p]["unused_plan_kwh"] for p in PROBLEMS]
    fsoc = [a[p]["final_soc_kwh"] for p in PROBLEMS]

    add("可实现策略总费用/万元", [v / W for v in tot], "{:,.2f}")
    add("计划购电量/kWh", plan, "{:,.0f}")
    add("计划购电费/万元", [a[p]["planned_cost_yuan"] / W for p in PROBLEMS], "{:,.2f}")
    add("紧急购电量/kWh", emer, "{:,.0f}")
    add("紧急购电/计划购电", [100 * e / q for e, q in zip(emer, plan)], "{:.4f}%")
    add("紧急购电天数/天", days, "{:.0f}")
    add("单日费用CVaR95/万元", [v / W for v in cvar], "{:,.2f}")
    add("未用计划电量/kWh", unused, "{:,.0f}")
    add("未用计划电量占比", [100 * u / q for u, q in zip(unused, plan)], "{:.3f}%")
    add("实际年末SOC/kWh", fsoc, "{:,.1f}")
    add("年末SOC偏离E0/kWh", [v - E_INITIAL for v in fsoc], "{:+,.1f}")

    pv = [a[p].get("curtailed_pv_kwh") for p in PROBLEMS]
    add("弃光电量/kWh", pv, "{:,.0f}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def physical_table(slots: dict) -> pd.DataFrame:
    """表 10 可靠性与物理可行性——从逐 10 分钟策略 CSV 逐时段重算。"""
    cols = ["最大能量平衡误差/kWh", "最大购电恒等式误差/kWh", "最大光伏恒等式误差/kWh",
            "最大SOC递推误差/kWh", "首时段隐含起始SOC/kWh", "SOC越界次数",
            "充放电功率越界次数", "同时充放电次数", "负流量次数", "评价时段数"]
    out = {}
    for p in PROBLEMS:
        d = slots[p]
        n = len(d)
        L, P = d.L_actual_kw.to_numpy(), d.P_act_kw.to_numpy()
        u, v = d.u_charge_kwh.to_numpy(), d.v_discharge_kwh.to_numpy()
        y = d.y_used_kwh.to_numpy()
        e = d.e_emergency_kwh.to_numpy()
        w = d.w_pv_waste_kwh.to_numpy()
        g = d.g_pv_used_kwh.to_numpy()
        r = d.r_purchase_waste_kwh.to_numpy()
        s = d.E_end_kwh.to_numpy()
        xcol = "x_final_kwh" if "x_final_kwh" in d.columns else "x_plan_kwh"
        x = d[xcol].to_numpy()

        bal = y + g + v + e - (L * DT + u)                      # 应 = 0
        buy = x - (y + r)                                       # 应 = 0
        pvi = P * DT - (g + w)                                  # 应 = 0
        # 逐时段递推；首行没有前驱（前驱在评价期之前），故从第 2 行起检查，
        # 首行的前驱反解出来单列报告。
        soc = s[1:] - (s[:-1] + 0.9 * u[1:] - v[1:] / 0.9)      # 应 = 0
        implied_start = s[0] - 0.9 * u[0] + v[0] / 0.9

        out[p] = [
            float(np.abs(bal).max()),
            float(np.abs(buy).max()),
            float(np.abs(pvi).max()),
            float(np.abs(soc).max()),
            float(implied_start),
            int((s < E_MIN - 1e-6).sum() + (s > E_MAX + 1e-6).sum()),
            int((u > P_MAX * DT + 1e-6).sum() + (v > P_MAX * DT + 1e-6).sum()),
            int(((u > 1e-6) & (v > 1e-6)).sum()),
            int((np.minimum.reduce([x, u, v, y, e, w, g, r]) < -1e-9).sum()),
            n,
        ]
    return pd.DataFrame(out, index=cols).rename_axis("检查项").reset_index()


# --------------------------------------------------------------------------
def gap_table(frozen: dict) -> pd.DataFrame:
    """费用差距分解：全知下界 → 决策结构 → 负载/光伏预测 → 电价预测 → 可实现。"""
    o2 = pd.read_csv(ORACLE_Q2Q4, encoding="utf-8-sig")
    o42 = json.loads(Path(ORACLE_Q4_2).read_text(encoding="utf-8"))
    o43 = json.loads(Path(ORACLE_Q4_3).read_text(encoding="utf-8"))
    a = {p: annual_block(frozen, p) for p in PROBLEMS}
    q2o = o2[o2.口径.str.contains("问题2")].iloc[0]

    rows = [
        {"口径": "C0 全知全年统一下界",
         "Q2": q2o["全知全年统一下界/元"], "Q3": q2o["全知全年统一下界/元"],
         "Q4-2": o42["oracle_full_year_joint"]["total_cost_yuan"],
         "Q4-3": o43["oracle_full_year_joint_yuan"]},
        {"口径": "C1 完美预测+逐日决策",
         "Q2": q2o["完美预测逐日决策/元"], "Q3": None,
         "Q4-2": o42["perfect_forecast_daily_decision"]["total_cost_yuan"], "Q4-3": None},
        {"口径": "C2 因果负载光伏+电价完美",
         "Q2": None, "Q3": None,
         "Q4-2": o42["perfect_price"]["total_cost_yuan"],
         "Q4-3": frozen["Q4-3"]["perfect_price_information"]["total_cost_yuan"]},
        {"口径": "C3 可实现策略（最终冻结）",
         "Q2": a["Q2"]["total_cost_yuan"], "Q3": a["Q3"]["total_cost_yuan"],
         "Q4-2": a["Q4-2"]["total_cost_yuan"], "Q4-3": a["Q4-3"]["total_cost_yuan"]},
    ]
    t = pd.DataFrame(rows)
    c0 = t.iloc[0]
    gap = t.iloc[-1]
    t.loc[len(t)] = {"口径": "C3-C0 绝对差额/万元",
                     **{p: (gap[p] - c0[p]) / W for p in PROBLEMS}}
    t.loc[len(t)] = {"口径": "C3-C0 相对高出",
                     **{p: 100 * (gap[p] - c0[p]) / c0[p] for p in PROBLEMS}}
    return t


# --------------------------------------------------------------------------
def to_md(df: pd.DataFrame) -> str:
    def fmt(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        if isinstance(v, float):
            if v != 0 and abs(v) < 1e-4:        # 机器精度量级，用科学计数法
                return f"{v:.2e}"
            return f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:,.2f}"
        return str(v)
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):     # 控制台按 UTF-8 输出，避免 GBK 崩溃
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    frozen, slots = load_frozen(), load_slots()
    OUT.mkdir(parents=True, exist_ok=True)

    ft = forecast_table(frozen, slots)
    ct = cost_table(frozen, slots)
    pt = physical_table(slots)
    gt = gap_table(frozen)

    ft.to_csv(OUT / "表_预测精度.csv", index=False, encoding="utf-8-sig")
    ct.to_csv(OUT / "表_核心指标.csv", index=False, encoding="utf-8-sig")
    pt.to_csv(OUT / "表_物理可行性.csv", index=False, encoding="utf-8-sig")
    gt.to_csv(OUT / "表_费用差距分解.csv", index=False, encoding="utf-8-sig")

    md = ["# 问题2—4 主要指标（自冻结结果提取）", "",
          "> 数值全部来自上游已冻结结果文件，由 `verify/core_indicators.py` 提取，未重跑仿真。", "",
          "## 表 1　核心指标", "", to_md(ct), "",
          "## 表 2　预测准确度", "", to_md(ft), "",
          "## 表 3　物理可行性与可靠性（逐时段重算）", "", to_md(pt), "",
          "## 表 4　费用差距分解（元）", "", to_md(gt), ""]
    (OUT / "核心指标汇总.md").write_text("\n".join(md), encoding="utf-8")

    print(to_md(ct)); print()
    print(to_md(ft)); print()
    print(to_md(pt)); print()
    print(to_md(gt))


if __name__ == "__main__":
    main()
