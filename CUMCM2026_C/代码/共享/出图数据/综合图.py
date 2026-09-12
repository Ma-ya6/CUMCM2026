# -*- coding: utf-8 -*-
"""跨问题综合图：只读 图表/数据表/ 下已导出的 CSV，重绘 图表/综合/ 下的 8 张图。

不重算任何决策、不调用求解器，因此可在全年计算与出图数据导出之后随时重跑。

八张图各自的作用：
    Y-1 四问费用总览        一眼看出四问成本递进与「距全知下界还有多少空间」，
                            并把全知阶梯 C0/C1/C2/C3 画在同一张图上定位差距来源。
    Y-2 购电来源构成        把全年购电拆成「计划内实购 / 紧急购电 / 未用计划 / 弃光」，
                            用于判断两问费用差异究竟出在哪一类电量上。
    Y-3 承诺结算代价分解    只对分阶段发布的 Q3/Q4-3，逐月拆出基础购电费与
                            违约（下调 0.5 倍）、超购（上调 1.5 倍）费，量化多次改计划的代价。
    Y-4 储能 SOC 热力图      365 天 × 144 时段的 SOC 真值矩阵，检查储能是否存在
                            长期贴边、季节性闲置或过度循环。
    Y-5 逐日费用分布箱线    按月份给出四问逐日费用的箱线分布，看月度波动与季节风险，
                            而不是只看年度合计。
    Y-6 预测误差分布        负荷与光伏的预测误差直方图叠加 R 所用分位数，
                            说明安全余量 R 的取法依据。
    Y-7 SOC 越界裕度        逐日 SOC 距上下限的最近距离，说明储能约束是否真的紧，
                            以及年度末可达性保护的作用区间。
    Y-8 负荷光伏净负荷概览  全年负荷、光伏、净负荷的月度均值曲线与净负荷热力图，
                            给出四个问题共同的物理背景。

用法：python 综合图.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # 代码/共享/出图数据
ROOT = HERE.parent                              # 代码/共享
PROJECT = ROOT.parent.parent                    # 项目根
TABLES = PROJECT / "图表" / "数据表"
OUT = PROJECT / "图表" / "综合"
OUT.mkdir(parents=True, exist_ok=True)

TAGS = {"Q2": "reserve-mpc", "Q3": "exact-greedy",
        "Q4-2": "reserve-mpc", "Q4-3": "exact-greedy"}
LABEL = {"Q2": "问题2（固定电价·单次发布）",
         "Q3": "问题3（固定电价·四次发布）",
         "Q4-2": "问题4-2（波动电价·单次发布）",
         "Q4-3": "问题4-3（波动电价·四次发布）"}
SHORT = {"Q2": "Q2", "Q3": "Q3", "Q4-2": "Q4-2", "Q4-3": "Q4-3"}
COLOR = {"Q2": "#005792", "Q3": "#009473", "Q4-2": "#00A3FF", "Q4-3": "#E57373"}
FORMAL_START = "2025-02-01"
SOC_MIN, SOC_MAX = 1200.0, 10800.0

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def table(problem: str, name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{problem}_{TAGS[problem]}_{name}.csv")


def daily(problem: str, formal: bool = True) -> pd.DataFrame:
    # 主目录的表B 只覆盖正式期 334 天；自然年 365 天口径在 自然年365天/ 子目录。
    base = TABLES if formal else TABLES / "自然年365天"
    d = pd.read_csv(base / f"{problem}_{TAGS[problem]}_表B_逐日.csv")
    d["date"] = pd.to_datetime(d["date"])
    return d[d.date >= FORMAL_START].reset_index(drop=True) if formal else d


def slots(problem: str) -> pd.DataFrame:
    a = table(problem, "表A_逐10分钟")
    a["date"] = pd.to_datetime(a["date"])
    return a


def box(ax):
    for s in ax.spines.values():
        s.set_color("#333333")
        s.set_linewidth(0.8)
    ax.grid(True, color="#DDE3EA", lw=0.6, ls="-")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9.5, colors="#333333")


def save(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path.relative_to(PROJECT)}")


# ---------------------------------------------------------------- Y-1
def df_ready() -> bool:
    """表F 由 `出图数据/run_experiments.py` 按需生成（约 38 次全年仿真），不随 export 自动补算。"""
    return (TABLES / "表F_全知下界阶梯_正式期334天.csv").exists()


def y1_cost_overview():
    f = pd.read_csv(TABLES / "表F_全知下界阶梯_正式期334天.csv").set_index("问题")
    problems = ["Q2", "Q3", "Q4-2", "Q4-3"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))

    ax = axes[0]
    x = np.arange(len(problems))
    w = 0.36
    annual = [daily(p, formal=False).total_cost_yuan.sum() / 1e4 for p in problems]
    formal = [daily(p, formal=True).total_cost_yuan.sum() / 1e4 for p in problems]
    ax.bar(x - w / 2, annual, w, color="#B8C4D4", edgecolor="#333333", lw=0.7, label="2025 自然年 365 天")
    ax.bar(x + w / 2, formal, w, color=[COLOR[p] for p in problems], edgecolor="#333333", lw=0.7,
           label="正式期 334 天")
    for xi, (a, b) in enumerate(zip(annual, formal)):
        ax.text(xi - w / 2, a + 8, f"{a:.0f}", ha="center", fontsize=9, color="#333333")
        ax.text(xi + w / 2, b + 8, f"{b:.0f}", ha="center", fontsize=9, color="#333333")
    ax.set_ylim(0, max(annual + formal) * 1.30)   # 给顶部图例留位，避免压住柱顶标注
    ax.set_xticks(x); ax.set_xticklabels([SHORT[p] for p in problems], fontsize=10.5)
    ax.set_ylabel("总费用 / 万元", fontsize=11)
    ax.set_title("两套口径下的总费用", fontsize=12, pad=8)
    ax.legend(frameon=False, fontsize=10)
    box(ax)

    ax = axes[1]
    c0 = [f.loc[p, "C0"] for p in problems]
    c1 = [f.loc[p, "C1"] for p in problems]
    c2 = [f.loc[p, "C2"] for p in problems]
    c3 = [f.loc[p, "C3"] for p in problems]
    ax.plot(x, c0, "o--", color="#9AA4B2", lw=1.6, ms=6, label="C0 全知物理 LP 下界")
    ax.plot(x, c1, "s-", color="#00A3FF", lw=1.6, ms=6, label="C1 全知负荷光伏")
    ax.plot(x, c2, "^-", color="#009473", lw=1.6, ms=6, label="C2 全知负荷光伏与电价")
    ax.plot(x, c3, "D-", color="#005792", lw=2.0, ms=6, label="C3 实际因果执行")
    for xi in x:
        gap = c3[xi] - c0[xi]
        ax.annotate("", xy=(xi, c3[xi]), xytext=(xi, c0[xi]),
                    arrowprops=dict(arrowstyle="<->", color="#E57373", lw=1.2))
        ax.text(xi + 0.06, (c0[xi] + c3[xi]) / 2, f"差 {gap:.1f}", fontsize=9, color="#C0392B")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[p] for p in problems], fontsize=10.5)
    ax.set_ylabel("正式期 334 天费用 / 万元", fontsize=11)
    ax.set_title("因果执行与全知下界的阶梯差距", fontsize=12, pad=8)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    box(ax)

    fig.suptitle("图 Y-1  四问费用总览：成本递进与优化空间", fontsize=13.5, y=1.02)
    save(fig, "Y-1_四问费用总览.png")


# ---------------------------------------------------------------- Y-2
def y2_purchase_mix():
    problems = ["Q2", "Q3", "Q4-2", "Q4-3"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    x = np.arange(len(problems))
    big = [("计划内实购", "planned_purchase_kwh", "#005792")]
    small = [("紧急购电", "emergency_kwh", "#E57373"),
             ("未用计划", "unused_plan_kwh", "#9AA4B2"),
             ("弃光", "curtailed_pv_kwh", "#009473")]
    # 左：计划内实购量级 ~2000 万 kWh；右：其余三项量级 ~10–130 万 kWh。
    # 分轴画，否则小项在堆叠图里完全不可见。
    for ax, series, title in ((axes[0], big, "计划内实购（主项）"),
                              (axes[1], small, "紧急购电 / 未用计划 / 弃光（放大）")):
        w = 0.8 / len(series)
        for k, (name, col, color) in enumerate(series):
            vals = np.array([daily(p)[col].sum() / 1e4 for p in problems])
            ax.bar(x + (k - (len(series) - 1) / 2) * w, vals, w * 0.9,
                   color=color, edgecolor="#333333", lw=0.6, label=name)
            for xi, v in enumerate(vals):
                ax.text(xi + (k - (len(series) - 1) / 2) * w, v * 1.02, f"{v:.0f}",
                        ha="center", fontsize=9, color="#333333")
        ax.set_xticks(x); ax.set_xticklabels([SHORT[p] for p in problems], fontsize=10.5)
        ax.set_title(title, fontsize=12, pad=8)
        ax.legend(frameon=False, fontsize=10)
        box(ax)
    axes[0].set_ylabel("正式期 334 天电量 / 万 kWh", fontsize=11)
    axes[1].set_ylabel("正式期 334 天电量 / 万 kWh", fontsize=11)
    fig.suptitle("图 Y-2  购电来源构成：成本差异出在哪一类电量", fontsize=13.5, y=1.02)
    save(fig, "Y-2_购电来源构成.png")


# ---------------------------------------------------------------- Y-3
def y3_commitment_decomposition():
    problems = ["Q3", "Q4-3"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), sharey=True)
    for ax, p in zip(axes, problems):
        d = daily(p).copy()
        d["month"] = d.date.dt.strftime("%m月")
        g = d.groupby("month", sort=True)[["base_cost_yuan", "breach_cost_yuan",
                                           "overbuy_cost_yuan", "emergency_cost_yuan"]].sum() / 1e4
        months = g.index.tolist()
        bottom = np.zeros(len(g))
        for name, col, color in [("基础购电费", "base_cost_yuan", "#005792"),
                                 ("下调违约费", "breach_cost_yuan", "#E57373"),
                                 ("上调超购费", "overbuy_cost_yuan", "#E5A663"),
                                 ("紧急购电费", "emergency_cost_yuan", "#9AA4B2")]:
            vals = g[col].to_numpy()
            ax.bar(months, vals, 0.6, bottom=bottom, color=color,
                   edgecolor="white", lw=0.8, label=name)
            bottom += vals
        ax.set_title(LABEL[p], fontsize=11.5, pad=8)
        ax.set_ylabel("费用 / 万元", fontsize=11)
        ax.tick_params(axis="x", labelsize=9.5, rotation=45)
        box(ax)
    axes[1].legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.suptitle("图 Y-3  承诺结算代价分解：下调 0.5 倍违约与上调 1.5 倍超购", fontsize=13.5, y=1.02)
    save(fig, "Y-3_承诺结算代价分解.png")


# ---------------------------------------------------------------- Y-4
def y4_soc_heatmap():
    problems = ["Q2", "Q3", "Q4-2", "Q4-3"]
    fig, axes = plt.subplots(4, 1, figsize=(13.4, 10.4))
    for ax, p in zip(axes, problems):
        a = slots(p)
        a = a[a.date >= FORMAL_START]
        m = a.E_end_kwh.to_numpy().reshape(-1, 144) / 1000.0
        im = ax.imshow(m, aspect="auto", origin="lower", cmap="RdYlBu_r",
                       vmin=SOC_MIN / 1000, vmax=SOC_MAX / 1000,
                       extent=[0, 24, 0, m.shape[0]])
        ax.set_ylabel("正式期天数", fontsize=10)
        ax.set_title(f"{LABEL[p]}  最低 {m.min():.2f} MWh / 最高 {m.max():.2f} MWh",
                     fontsize=11, pad=6)
        ax.set_xticks(np.arange(0, 25, 3))
        ax.tick_params(labelsize=9)
        cb = fig.colorbar(im, ax=ax, pad=0.008, fraction=0.020)
        cb.set_label("储电量 / MWh", fontsize=9)
        cb.ax.tick_params(labelsize=8.5)
    axes[-1].set_xlabel("时刻", fontsize=11)
    fig.suptitle("图 Y-4  储能储电量热力图：全年是否存在贴边、闲置或过度循环", fontsize=13.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save(fig, "Y-4_储能SOC热力图.png")


# ---------------------------------------------------------------- Y-5
def y5_monthly_box():
    problems = ["Q2", "Q3", "Q4-2", "Q4-3"]
    fig, ax = plt.subplots(figsize=(13.4, 5.4))
    months = sorted(daily("Q2").date.dt.strftime("%m月").unique())
    step = 0.20
    for k, p in enumerate(problems):
        d = daily(p).copy()
        d["month"] = d.date.dt.strftime("%m月")
        data = [d.loc[d.month == m, "total_cost_yuan"].to_numpy() / 1e4 for m in months]
        pos = np.arange(len(months)) + (k - 1.5) * step
        bp = ax.boxplot(data, positions=pos, widths=step * 0.85, patch_artist=True,
                        medianprops=dict(color="#333333", lw=1.4),
                        whiskerprops=dict(color="#666666", lw=0.9),
                        capprops=dict(color="#666666", lw=0.9),
                        flierprops=dict(marker="o", ms=2.4, mfc="#E57373",
                                        mec="#E57373", alpha=0.55))
        for patch in bp["boxes"]:
            patch.set(facecolor=COLOR[p], alpha=0.62, edgecolor="#333333", lw=0.6)
        ax.plot([], [], "s", color=COLOR[p], ms=8, alpha=0.62, label=SHORT[p])
    ax.set_xticks(np.arange(len(months))); ax.set_xticklabels(months, fontsize=10)
    ax.set_ylabel("逐日总费用 / 万元", fontsize=11)
    ax.legend(frameon=False, fontsize=10, ncol=4)
    box(ax)
    fig.suptitle("图 Y-5  逐日费用月度分布：均值之外的波动与季节风险", fontsize=13.5, y=1.0)
    save(fig, "Y-5_逐日费用月度分布.png")


# ---------------------------------------------------------------- Y-6
def y6_forecast_error():
    problems = ["Q2", "Q3", "Q4-2", "Q4-3"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))

    ax = axes[0]
    # 只画净负荷误差：光伏误差夜间恒为 0，会把负荷误差的分布完全压掉。
    a = slots("Q2")
    a = a[a.date >= FORMAL_START]
    net = (a.L_fc_kw - a.L_actual_kw - (a.P_fc_kw - a.P_act_kw)).to_numpy()
    lo, hi = np.quantile(net, [0.002, 0.998])         # 截断极端离群点，否则分布不可见
    ax.hist(net, bins=np.linspace(lo, hi, 90), color="#005792", alpha=0.72,
            edgecolor="none", label=f"净负荷预测误差（截断至 99.6% 区间）")
    for qq, color, ls in ((0.50, "#9AA4B2", ":"), (0.80, "#E57373", "--"), (0.90, "#C0392B", "-.")):
        v = np.quantile(net, qq)
        ax.axvline(v, color=color, lw=1.7, ls=ls, label=f"{int(qq*100)}% 分位 = {v:.1f} kW")
    ax.axvline(0, color="#333333", lw=0.9)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("净负荷预测误差 / kW", fontsize=11)
    ax.set_ylabel("时段数", fontsize=11)
    ax.set_title("净负荷预测误差分布与安全余量分位数", fontsize=12, pad=8)
    ax.legend(frameon=False, fontsize=9.5)
    box(ax)

    ax = axes[1]
    x = np.arange(len(problems))
    rows = []
    for p in problems:
        e = table(p, "预测误差明细")
        rows.append((np.abs(e.load_error_kw).mean(), np.abs(e.pv_error_kw).mean(),
                     np.sqrt((e.net_error_kw ** 2).mean()),
                     np.abs(e.naive_load_error_kw).mean()))
    rows = np.array(rows)
    w = 0.20
    ax.bar(x - 1.5 * w, rows[:, 0], w, color="#005792", edgecolor="#333333", lw=0.6, label="负荷 MAE")
    ax.bar(x - 0.5 * w, rows[:, 1], w, color="#009473", edgecolor="#333333", lw=0.6, label="光伏 MAE")
    ax.bar(x + 0.5 * w, rows[:, 2], w, color="#00A3FF", edgecolor="#333333", lw=0.6, label="净负荷 RMSE")
    ax.bar(x + 1.5 * w, rows[:, 3], w, color="#B8C4D4", edgecolor="#333333", lw=0.6,
           label="负荷 MAE（朴素基准）")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[p] for p in problems], fontsize=10.5)
    ax.set_ylabel("误差 / kW", fontsize=11)
    ax.set_title("四问预测精度与朴素基准对比", fontsize=12, pad=8)
    ax.legend(frameon=False, fontsize=9.5)
    box(ax)

    fig.suptitle("图 Y-6  预测误差分布：安全余量 R 的取法依据", fontsize=13.5, y=1.02)
    save(fig, "Y-6_预测误差分布.png")


# ---------------------------------------------------------------- Y-7
def y7_soc_margin():
    problems = ["Q2", "Q3", "Q4-2", "Q4-3"]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.2), sharex=True)
    for ax, p in zip(axes.ravel(), problems):
        a = slots(p)
        a = a[a.date >= FORMAL_START]
        m = a.E_end_kwh.to_numpy().reshape(-1, 144)
        up = (SOC_MAX - m.max(axis=1)) / 1000.0
        dn = (m.min(axis=1) - SOC_MIN) / 1000.0
        days = np.arange(1, len(m) + 1)
        ax.plot(days, up, color="#005792", lw=1.0, label="距上限裕度")
        ax.plot(days, dn, color="#E57373", lw=1.0, label="距下限裕度")
        ax.fill_between(days, 0, up, color="#005792", alpha=0.12)
        ax.fill_between(days, 0, dn, color="#E57373", alpha=0.12)
        ax.set_title(f"{LABEL[p]}  贴边天数(裕度<0.2 MWh)：上限 {int((up < 0.2).sum())} / "
                     f"下限 {int((dn < 0.2).sum())}", fontsize=10.5, pad=6)
        ax.set_ylabel("裕度 / MWh", fontsize=10)
        box(ax)
    axes[0, 0].legend(frameon=False, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("正式期天数", fontsize=10)
    fig.suptitle("图 Y-7  SOC 越界裕度：储能约束是否真的紧", fontsize=13.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "Y-7_SOC越界裕度.png")


# ---------------------------------------------------------------- Y-8
def y8_physical_overview():
    a = slots("Q2")
    a = a[a.date >= FORMAL_START].reset_index(drop=True)
    n_day = len(a) // 144                 # 表A 只覆盖正式期，天数由数据本身决定
    load = a.L_actual_kw.to_numpy().reshape(n_day, 144)
    pv = a.P_act_kw.to_numpy().reshape(n_day, 144)
    net = load - pv

    fig, axes = plt.subplots(2, 1, figsize=(13.2, 8.0))

    ax = axes[0]
    month_mean = a.assign(month=a.date.dt.strftime("%m月")).groupby("month")[
        ["L_actual_kw", "P_act_kw"]].mean()
    ax.plot(month_mean.index, month_mean.L_actual_kw, "o-", color="#00A3FF", lw=1.8, ms=5,
            label="小区负载")
    ax.plot(month_mean.index, month_mean.P_act_kw, "s-", color="#009473", lw=1.8, ms=5,
            label="光伏出力")
    ax.plot(month_mean.index, (month_mean.L_actual_kw - month_mean.P_act_kw), "^--",
            color="#005792", lw=1.8, ms=5, label="净负荷")
    ax.axhline(0, color="#9AA4B2", lw=0.9)
    ax.set_ylabel("功率 / kW", fontsize=11)
    ax.set_title(f"正式期 {n_day} 天月度均值：负载、光伏与净负荷", fontsize=12, pad=8)
    ax.legend(frameon=False, fontsize=10, ncol=3)
    box(ax)

    ax = axes[1]
    im = ax.imshow(net, aspect="auto", origin="lower", cmap="coolwarm",
                   extent=[0, 24, 0, n_day],
                   vmin=-np.abs(net).max(), vmax=np.abs(net).max())
    ax.set_xlabel("时刻", fontsize=11)
    ax.set_ylabel("正式期天数", fontsize=11)
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_title("净负荷热力图（红=需购电，蓝=光伏富余）", fontsize=12, pad=8)
    ax.tick_params(labelsize=9.5)
    cb = fig.colorbar(im, ax=ax, pad=0.008, fraction=0.022)
    cb.set_label("净负荷 / kW", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    fig.suptitle("图 Y-8  物理背景：负载、光伏与净负荷（正式期 334 天）", fontsize=13.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "Y-8_负荷光伏净负荷概览.png")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("跨问题综合图 → 图表/综合/ ：")
    ready = df_ready()
    if ready:
        y1_cost_overview()
    else:
        print("  跳过 Y-1：缺 表F，先跑 代码/共享/出图数据/run_experiments.py all")
    y2_purchase_mix()
    y3_commitment_decomposition()
    y4_soc_heatmap()
    y5_monthly_box()
    y6_forecast_error()
    y7_soc_margin()
    y8_physical_overview()
    print(f"完成：{8 if ready else 7} 张。")


if __name__ == "__main__":
    main()
