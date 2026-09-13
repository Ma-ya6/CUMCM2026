# -*- coding: utf-8 -*-
"""问题 2 / 3 / 4 的结果图与跨问题图（X 组）。

只读 `图表/数据表/` 下已导出的 CSV，不重算任何决策、不调用求解器，
因此可在 `project.py run` 与 `project.py export` 之后随时重跑。

每次运行覆盖写出 17 张 png：

    图表/问题2/   Q2-1 … Q2-4     4 张
    图表/问题3/   Q3-1 … Q3-5     5 张
    图表/问题4/   Q4-1 … Q4-4     4 张
    图表/综合/    X-1 … X-4       4 张

每张图的作用见 `图表/图表说明.md`。用法：python plot_result_figures.py

几条固定口径（与 `README_出图数据使用说明.md` 一致）：
  1. `*_kw` 是功率，`*_kwh` 是十分钟电量；画在同一根轴上时电量乘 6（TO_KW）变成功率。
  2. 除 Q4-1/Q4-4 用 365 天电价矩阵外，全部按 334 天正式期（2/1—12/31）口径。
  3. 本版 `correction_applied_yuan = 0`，年终 SOC 恰好回到 6000，图上费用即实际费用。
  4. 图内不写图题，标题由文件名承担。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # 代码/共享/出图数据
ROOT = HERE.parent                              # 代码/共享
PROJECT = ROOT.parent.parent                    # 项目根
TABLES = PROJECT / "图表" / "数据表"
CHARTS = PROJECT / "图表"

# plot_style.py 放在 代码/问题1/ 下（问题一的三个画图脚本按该位置 import），此处复用同一份。
sys.path.insert(0, str(PROJECT / "代码" / "问题1"))
from plot_style import (C_MAIN, C_SECOND, C_FOURTH, C_RED_LIGHT,   # noqa: E402
                        C_GUIDE, C_PURPLE, C_TEXT, style_frame_grid,
                        apply_base_style)

apply_base_style()

TO_KW = 6.0                                     # 十分钟电量 → 平均功率
C_THIRD = "#DDCC77"                             # 沙黄 · 首次承诺 x0
C_FIFTH = "#9D2B2B"                             # 深红 · 第五档
TAGS = {"Q2": "reserve-mpc", "Q3": "exact-greedy",
        "Q4-2": "reserve-mpc", "Q4-3": "exact-greedy"}
LABEL = {"Q2": "问题2", "Q3": "问题3", "Q4-2": "问题4-2", "Q4-3": "问题4-3"}
COLOR = {"Q2": C_MAIN, "Q3": C_FOURTH, "Q4-2": C_SECOND, "Q4-3": C_PURPLE}
WD_NAME = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


# ---------------------------------------------------------------- 读取
def tab(problem: str, name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{problem}_{TAGS[problem]}_{name}.csv",
                       encoding="utf-8-sig")


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  已写出 {path.relative_to(PROJECT)}")


def hours(n: int, start: float = 0.0):
    """n 个十分钟段的左端时刻（小时）。"""
    return start + np.arange(n) / 6.0


def typical_week(a: pd.DataFrame, cols) -> tuple[pd.DataFrame, dict]:
    """按星期几对全年逐时段求平均，得到 7×144 的典型周。

    不是挑某七天，而是同一星期几的所有日期对 144 个时段逐一取算术平均。
    返回（平均表, {星期几: 天数}）。
    """
    a = a.copy()
    a["wd"] = pd.to_datetime(a["date"]).dt.weekday
    days = a.groupby("wd")["date"].nunique().to_dict()
    out = []
    for wd in range(7):
        g = a[a.wd == wd]
        m = g.groupby("t")[list(cols)].mean().reset_index()
        m["wd"] = wd
        out.append(m)
    return pd.concat(out, ignore_index=True), days


def week_axis(ax, days: dict) -> None:
    """典型周横轴：0—168 小时，每 24 小时一个刻度，标星期几与实际天数。"""
    ax.set_xlim(0, 168)
    ax.set_xticks(np.arange(0, 169, 24))
    ax.set_xticklabels([f"{WD_NAME[i]}\n({days.get(i, 0)}天)" for i in range(7)]
                       + [""], fontsize=9.5, color=C_TEXT)
    for x in range(24, 168, 24):
        ax.axvline(x, color=C_GUIDE, linewidth=0.6, linestyle=":", zorder=0)


def day_axis(ax) -> None:
    """单日横轴：0—24 小时，每 4 小时一个刻度。"""
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.set_xticklabels([f"{h}:00" for h in range(0, 25, 4)], color=C_TEXT)


def four_dates_grid(fig, four):
    """四个指定日期的 2×2 面板。"""
    axes = fig.subplots(2, 2)
    for ax, date in zip(axes.ravel(), four):
        ax.set_title(str(date), fontsize=11, color=C_TEXT)
        day_axis(ax)
        style_frame_grid(ax)
    return axes


def read_four() -> list[str]:
    """四个指定日期的字符串，取自调度表本身，避免与导出脚本的常量各写一份。"""
    d = tab("Q2", "四个指定日期_调度")
    return sorted(d["date"].astype(str).unique().tolist())


# ================================================================ 问题二
def q2_1(four):  # noqa: ARG001  四个指定日期不参与本图
    """典型周调度（星期平均）。看工作日与周末的负荷差是否被储能吸收。"""
    a = tab("Q2", "表A_逐10分钟")
    w, days = typical_week(a, ["L_actual_kw", "P_act_kw", "x_plan_kwh",
                               "e_emergency_kwh", "E_end_kwh"])
    x = w["wd"].values * 24 + (w["t"].values - 1) / 6.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1.6, 1]})
    ax1.plot(x, w.L_actual_kw, color=C_SECOND, linewidth=1.5, label="小区负载")
    ax1.plot(x, w.P_act_kw, color=C_FOURTH, linewidth=1.5, label="光伏出力")
    ax1.plot(x, w.x_plan_kwh * TO_KW, color=C_MAIN, linewidth=1.5,
             label="计划购电")
    ax1.set_ylabel("功率 / kW", color=C_TEXT)
    ax1.legend(loc="upper left", ncol=3, framealpha=0.9)
    week_axis(ax1, days)
    style_frame_grid(ax1)

    ax2.plot(x, w.E_end_kwh, color=C_MAIN, linewidth=1.5, label="日末储电量")
    ax2.axhline(1200, color=C_GUIDE, linewidth=0.9, linestyle="--")
    ax2.axhline(10800, color=C_GUIDE, linewidth=0.9, linestyle="--")
    ax2.text(1, 1350, "下限 1200", fontsize=9, color=C_GUIDE, va="bottom")
    ax2.text(1, 10650, "上限 10800", fontsize=9, color=C_GUIDE, va="top")
    ax2.set_ylabel("储电量 / kWh", color=C_TEXT)
    ax2.set_xlabel("时刻", color=C_TEXT)
    ax2.legend(loc="lower right", framealpha=0.9)
    week_axis(ax2, days)
    style_frame_grid(ax2)
    save(fig, CHARTS / "问题2" / "Q2-1_典型周调度_星期平均.png")


def q2_2(four):
    """四个指定日期调度。看四季代表日的调度差异。"""
    d = tab("Q2", "四个指定日期_调度")
    fig = plt.figure(figsize=(13, 7))
    axes = four_dates_grid(fig, four)
    for ax, date in zip(axes.ravel(), four):
        g = d[d.date == date]
        h = hours(len(g))
        ax.plot(h, g.L_actual_kw, color=C_SECOND, linewidth=1.3, label="小区负载")
        ax.plot(h, g.P_act_kw, color=C_FOURTH, linewidth=1.3, label="光伏出力")
        ax.plot(h, g.x_plan_kwh * TO_KW, color=C_MAIN, linewidth=1.5,
                label="计划购电")
        ax.set_ylabel("功率 / kW", color=C_TEXT, fontsize=10)
        axr = ax.twinx()
        axr.plot(h, g.E_end_kwh, color=C_GUIDE, linewidth=1.4, label="储电量")
        axr.set_ylim(0, 12000)
        axr.set_ylabel("储电量 / kWh", color=C_TEXT, fontsize=10)
        style_frame_grid(axr, grid=False, frame=False)
    axes[0, 0].legend(loc="upper left", fontsize=9, framealpha=0.9)
    axes[0, 1].legend(handles=[plt.Line2D([], [], color=C_GUIDE, linewidth=1.4,
                                          label="储电量（右轴）")],
                      loc="upper left", fontsize=9, framealpha=0.9)
    fig.supxlabel("时刻", color=C_TEXT)
    save(fig, CHARTS / "问题2" / "Q2-2_四个指定日期调度.png")


def q2_3(four):  # noqa: ARG001
    """逐日费用与紧急购电。看紧急购电集中在哪些天。"""
    b = tab("Q2", "表B_逐日")
    x = pd.to_datetime(b["date"])
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(x, b.grid_cost_yuan / 1e4, color=C_MAIN, linewidth=1.3,
            label="计划内购电费")
    ax.plot(x, b.emergency_cost_yuan / 1e4, color=C_PURPLE, linewidth=1.3,
            label="紧急购电费（含 5 倍罚）")
    ax.set_ylabel("费用 / 万元", color=C_TEXT)
    ax.set_xlabel("日期", color=C_TEXT)
    axr = ax.twinx()
    axr.plot(x, b.emergency_kwh, color=C_RED_LIGHT, linewidth=1.3,
             label="紧急购电量（右轴）")
    axr.set_ylabel("紧急购电量 / kWh", color=C_TEXT)
    style_frame_grid(ax)
    style_frame_grid(axr, grid=False, frame=False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=3, framealpha=0.9)
    save(fig, CHARTS / "问题2" / "Q2-3_逐日费用与紧急购电.png")


def q2_4(four):  # noqa: ARG001
    """SOC 范围与日储备。上栏看储能是否越界，下栏看储备 R 的水平。"""
    s = tab("Q2", "SOC范围与储备")
    x = pd.to_datetime(s["date"])
    e_min = float(s["E_min_kwh"].iloc[0])
    e_max = float(s["E_max_kwh"].iloc[0])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.6, 6.0), sharex=True,
                                   gridspec_kw={"height_ratios": [1.35, 1]})
    ax1.fill_between(x, s.soc_min_kwh, s.soc_max_kwh, color=C_SECOND, alpha=0.25,
                     label="日内储电量范围")
    ax1.plot(x, s.end_soc_kwh, color=C_PURPLE, linewidth=2.0, label="24:00 储电量")
    ax1.axhline(e_max, color=C_GUIDE, linewidth=1.4, linestyle="--")
    ax1.axhline(e_min, color=C_GUIDE, linewidth=1.4, linestyle="--")
    ax1.text(x.iloc[2], e_max + 350, f"上限 {e_max:.0f}", fontsize=9.5,
             color="#78859F", va="bottom")
    ax1.text(x.iloc[2], e_min + 120, f"下限 {e_min:.0f}", fontsize=9.5,
             color="#78859F", va="bottom")
    ax1.set_ylim(0, 11800)
    ax1.set_ylabel("储电量 / kWh", color=C_TEXT)
    ax1.legend(loc="lower left", framealpha=0.9, fontsize=10,
               bbox_to_anchor=(0.008, 0.20), borderaxespad=0)
    style_frame_grid(ax1)

    ax2.plot(x, s.reserve_kwh, color=C_MAIN, linewidth=1.6)
    ax2.set_ylabel("日储备 R / kWh", color=C_TEXT)
    ax2.set_xlabel("日期", color=C_TEXT)
    style_frame_grid(ax2)

    save(fig, CHARTS / "问题2" / "Q2-4_SOC范围与日储备.png")


# ================================================================ 问题三
def q3_1(four):  # noqa: ARG001
    """典型周调度含承诺修订。看一天之内计划被改了多少、改在哪些时段。"""
    a = tab("Q3", "表A_逐10分钟")
    w, days = typical_week(a, ["L_actual_kw", "P_act_kw", "x0_kwh", "x3_kwh",
                               "E_end_kwh"])
    x = w["wd"].values * 24 + (w["t"].values - 1) / 6.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1.6, 1]})
    ax1.plot(x, w.L_actual_kw, color=C_SECOND, linewidth=1.4, label="小区负载")
    ax1.plot(x, w.P_act_kw, color=C_FOURTH, linewidth=1.4, label="光伏出力")
    ax1.plot(x, w.x0_kwh * TO_KW, color=C_GUIDE, linewidth=1.6,
             label="首次承诺 x0")
    ax1.plot(x, w.x3_kwh * TO_KW, color=C_MAIN, linewidth=1.6,
             label="最终承诺 x3")
    ax1.set_ylabel("功率 / kW", color=C_TEXT)
    ax1.legend(loc="upper left", ncol=4, framealpha=0.9, fontsize=9.5)
    week_axis(ax1, days)
    style_frame_grid(ax1)

    ax2.plot(x, w.E_end_kwh, color=C_MAIN, linewidth=1.5, label="日末储电量")
    ax2.axhline(1200, color=C_GUIDE, linewidth=0.9, linestyle="--")
    ax2.axhline(10800, color=C_GUIDE, linewidth=0.9, linestyle="--")
    ax2.set_ylabel("储电量 / kWh", color=C_TEXT)
    ax2.set_xlabel("时刻", color=C_TEXT)
    ax2.legend(loc="lower right", framealpha=0.9)
    week_axis(ax2, days)
    style_frame_grid(ax2)
    save(fig, CHARTS / "问题3" / "Q3-1_典型周调度含承诺修订_星期平均.png")


def q3_2(four):
    """四个指定日期的 x0→x3 承诺修订过程。"""
    d = tab("Q3", "四个指定日期_调度")
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.0))
    for ax, date in zip(axes.ravel(), four):
        g = d[d.date == date].sort_values("t")
        x = hours(len(g)) + 1.0 / 12.0          # 十分钟时段的中间时刻
        ax.plot(x, g.L_actual_kw, color=C_SECOND, linewidth=1.8, label="负载")
        ax.plot(x, g.P_act_kw, color=C_FOURTH, linewidth=1.8, label="光伏")
        ax.step(x, g.x0_kwh * TO_KW, color=C_THIRD, linewidth=1.6, where="mid",
                label="首次承诺 x0")
        ax.step(x, g.x3_kwh * TO_KW, color=C_MAIN, linewidth=1.6, where="mid",
                label="最终承诺 x3")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))
        ax.set_title(str(date), fontsize=11, color=C_TEXT, loc="left")
        ax.set_xlabel("时刻 / h", color=C_TEXT, fontsize=10)
        ax.set_ylabel("功率 / kW", color=C_TEXT, fontsize=10)
        style_frame_grid(ax)

        axr = ax.twinx()
        axr.plot(x, g.E_end_kwh, color=C_PURPLE, linewidth=1.6, linestyle="--",
                 label="SOC（右轴）")
        axr.set_ylim(0, 12000)
        axr.set_ylabel("储电量 / kWh", color=C_TEXT, fontsize=10)
        style_frame_grid(axr, grid=False, frame=True)

    handles = [Line2D([], [], color=C_SECOND, linewidth=1.8, label="负载"),
               Line2D([], [], color=C_FOURTH, linewidth=1.8, label="光伏"),
               Line2D([], [], color=C_THIRD, linewidth=1.6, label="首次承诺 x0"),
               Line2D([], [], color=C_MAIN, linewidth=1.6, label="最终承诺 x3"),
               Line2D([], [], color=C_PURPLE, linewidth=1.6, linestyle="--",
                      label="SOC（右轴）")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(top=0.90, hspace=0.45, wspace=0.45)
    save(fig, CHARTS / "问题3" / "Q3-2_四个指定日期调度.png")


def q3_3(four):  # noqa: ARG001
    """承诺路径阶梯。x0/x1/x2/x3 四条阶梯，看每次发布把计划改到哪里。"""
    c = tab("Q3", "表C_承诺路径_自然日对齐")
    day = sorted(c["date"].astype(str).unique())[-1]
    # 取一个中期日子，避免落在年初年末的边界保护区间；有 2025-07-01 就用它。
    if "2025-07-01" in set(c["date"].astype(str)):
        day = "2025-07-01"
    g = c[c.date.astype(str) == day]
    h = hours(len(g))
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.step(h, g.x0_kwh * TO_KW, where="post", color=C_GUIDE, linewidth=1.6,
            label="x0 · 00:00 首次承诺")
    ax.step(h, g.x1_kwh * TO_KW, where="post", color=C_SECOND, linewidth=1.6,
            label="x1 · 06:00 调整后")
    ax.step(h, g.x2_kwh * TO_KW, where="post", color=C_FOURTH, linewidth=1.6,
            label="x2 · 12:00 调整后")
    ax.step(h, g.x3_kwh * TO_KW, where="post", color=C_MAIN, linewidth=1.8,
            label="x3 · 18:00 调整后（最终执行）")
    ax.set_ylabel("购电功率 / kW", color=C_TEXT)
    ax.set_xlabel(f"{day} 时刻", color=C_TEXT)
    ax.legend(loc="upper left", ncol=2, framealpha=0.9, fontsize=9.5)
    day_axis(ax)
    style_frame_grid(ax)
    save(fig, CHARTS / "问题3" / "Q3-3_承诺路径阶梯.png")


def q3_4(four):  # noqa: ARG001
    """承诺修订量与月度调整。量化「改计划」的规模与方向。"""
    b = tab("Q3", "表B_逐日")
    m = tab("Q3", "月度统计")
    x = pd.to_datetime(b["date"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7))

    ax1.bar(x, b.path_up_kwh, width=1.0, color=C_RED_LIGHT, label="上调量")
    ax1.bar(x, -b.path_down_kwh, width=1.0, color=C_SECOND, label="下调量")
    ax1.axhline(0, color=C_TEXT, linewidth=0.8)
    ax1.set_ylabel("承诺修订量 / kWh", color=C_TEXT)
    ax1.legend(loc="upper left", ncol=2, framealpha=0.9)
    style_frame_grid(ax1)

    mx = pd.to_datetime(m["month"] + "-01")
    ax2.bar(mx, m.adjusted_kwh, width=20, color=C_MAIN, label="月度净调整量")
    ax2.set_ylabel("净调整量 / kWh", color=C_TEXT)
    ax2.set_xlabel("月份", color=C_TEXT)
    axr = ax2.twinx()
    axr.plot(mx, m.reversal_ratio_all_slots * 100, color=C_FOURTH,
             marker="o", markersize=4, linewidth=1.5,
             label="逆行段占比（占全部物理段，右轴）")
    axr.set_ylabel("逆行段占比 / %", color=C_TEXT)
    style_frame_grid(ax2)
    style_frame_grid(axr, grid=False, frame=False)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.9, fontsize=9.5)
    save(fig, CHARTS / "问题3" / "Q3-4_承诺修订量与月度调整.png")


def q3_5(four):  # noqa: ARG001
    """逐日费用构成。四类费用各占一张，看谁在波动、谁在主导。"""
    b = tab("Q3", "表B_逐日")
    x = pd.to_datetime(b["date"])
    parts = [("基准购电费", "base_cost_yuan", C_MAIN),
             ("超购费", "overbuy_cost_yuan", C_SECOND),
             ("违约费", "breach_cost_yuan", C_THIRD),
             ("紧急购电费", "emergency_cost_yuan", C_FIFTH)]

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.0), sharex=True)
    for ax, (name, col, color) in zip(axes.ravel(), parts):
        v = b[col].to_numpy(float)
        ax.fill_between(x, 0, v, color=color, alpha=0.25)
        ax.plot(x, v, color=color, linewidth=1.3)
        ax.set_title(f"{name}（全年合计 {v.sum():.4g} 元）", fontsize=10,
                     color=C_TEXT, loc="left")
        ax.set_ylabel("日费用 / 元", color=C_TEXT, fontsize=10)
        style_frame_grid(ax)
    for ax in axes[1]:
        ax.set_xlabel("日期", color=C_TEXT, fontsize=10)
    fig.subplots_adjust(hspace=0.32, wspace=0.24)
    save(fig, CHARTS / "问题3" / "Q3-5_逐日费用构成.png")


# ================================================================ 问题四
def price_matrix() -> pd.DataFrame:
    p = pd.read_csv(TABLES / "Q4_电价热力图_365天_矩阵.csv", encoding="utf-8-sig")
    return p


def q4_1(four):  # noqa: ARG001
    """电价热力图。问题 4 相对问题 2/3 唯一新增的输入，先看清它的规律。"""
    p = price_matrix()
    y = pd.to_datetime(p["date"])
    m = p.drop(columns=["date"]).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(13, 5.6))
    mesh = ax.pcolormesh(np.arange(m.shape[1] + 1), np.arange(m.shape[0] + 1),
                         m, cmap="viridis", shading="flat")
    ax.set_xlim(0, 144)
    ax.set_xticks(range(0, 145, 12))
    ax.set_xticklabels([f"{h}:00" for h in range(0, 25, 2)], fontsize=9)
    ax.set_ylabel("日期", color=C_TEXT)
    ax.set_xlabel("时刻", color=C_TEXT)
    step = max(1, len(y) // 12)
    ax.set_yticks(range(0, len(y), step))
    ax.set_yticklabels([d.strftime("%m-%d") for d in y[::step]], fontsize=9)
    cb = fig.colorbar(mesh, ax=ax, pad=0.015)
    cb.set_label("电价 / (元/kWh)", color=C_TEXT)
    ax.invert_yaxis()                                # 1 月在上，与日历阅读习惯一致
    style_frame_grid(ax, grid=False)
    save(fig, CHARTS / "问题4" / "Q4-1_电价热力图.png")


def q4_2(four):
    """四个指定日期的调度与实际电价。看储能是否在低价段充、高价段放。"""
    d = tab("Q4-3", "四个指定日期_调度")
    fig = plt.figure(figsize=(13, 7))
    axes = four_dates_grid(fig, four)
    for ax, date in zip(axes.ravel(), four):
        g = d[d.date == date]
        h = hours(len(g))
        ax.plot(h, g.L_actual_kw, color=C_SECOND, linewidth=1.1,
                alpha=0.85, label="小区负载")
        ax.plot(h, g.x_plan_kwh * TO_KW, color=C_MAIN, linewidth=1.5,
                label="计划购电")
        ax.set_ylabel("功率 / kW", color=C_TEXT, fontsize=10)
        axr = ax.twinx()
        axr.plot(h, g.price_actual, color=C_RED_LIGHT, linewidth=1.4,
                 label="实际电价")
        axr.set_ylabel("电价 / (元/kWh)", color=C_TEXT, fontsize=10)
        style_frame_grid(axr, grid=False, frame=False)
    axes[0, 0].legend(loc="upper left", fontsize=9, framealpha=0.9)
    axes[0, 1].legend(handles=[plt.Line2D([], [], color=C_RED_LIGHT, linewidth=1.4,
                                          label="实际电价（右轴）")],
                      loc="upper left", fontsize=9, framealpha=0.9)
    fig.supxlabel("时刻", color=C_TEXT)
    save(fig, CHARTS / "问题4" / "Q4-2_四个指定日期调度.png")


def q4_3(four):  # noqa: ARG001
    """Q4-2 与 Q4-3 对比。回答「波动电价下多时刻发布还值不值」。"""
    g = pd.read_csv(TABLES / "表G_最终档位指标汇总_正式期334天.csv",
                    encoding="utf-8-sig")
    row = {"Q4-2": g[g["问题"] == "Q4-2"].iloc[0], "Q4-3": g[g["问题"] == "Q4-3"].iloc[0]}
    panels = [("total_cost_yuan", "正式期总费用 / 万元"),
              ("emergency_kwh", "紧急购电量 / 万 kWh"),
              ("unused_plan_kwh", "未用计划量 / 万 kWh"),
              ("curtailed_pv_kwh", "弃光量 / 万 kWh")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (col, ylab) in zip(axes.ravel(), panels):
        vals = [row["Q4-2"][col] / 1e4, row["Q4-3"][col] / 1e4]
        bars = ax.bar(["问题4-2\n（一次发布）", "问题4-3\n（四次发布）"], vals,
                      color=[C_SECOND, C_PURPLE], width=0.52)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.2f}",
                    ha="center", va="bottom", fontsize=10, color=C_TEXT)
        ax.set_ylabel(ylab, color=C_TEXT)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.ticklabel_format(axis="y", style="plain")
        style_frame_grid(ax, grid=False)
    save(fig, CHARTS / "问题4" / "Q4-3_Q4-2与Q4-3对比.png")


def q4_4(four):
    """指定日期电价与全年平均。判断四个指定日期是否真的有代表性。"""
    p = price_matrix()
    p = p.set_index("date")
    avg = p.mean(axis=0).to_numpy(dtype=float)
    h = hours(p.shape[1])
    fig, ax = plt.subplots(figsize=(13, 4.8))
    for date, c in zip(four, [C_SECOND, C_FOURTH, C_RED_LIGHT, C_PURPLE]):
        if date in p.index:
            ax.plot(h, p.loc[date].to_numpy(dtype=float), color=c, linewidth=1.4,
                    alpha=0.9, label=str(date))
    ax.plot(h, avg, color=C_MAIN, linewidth=2.4, label="全年平均")
    ax.set_ylabel("电价 / (元/kWh)", color=C_TEXT)
    ax.set_xlabel("时刻", color=C_TEXT)
    ax.legend(loc="upper left", ncol=5, framealpha=0.9, fontsize=9.5)
    day_axis(ax)
    style_frame_grid(ax)
    save(fig, CHARTS / "问题4" / "Q4-4_指定日期电价与全年平均.png")


# ================================================================ 跨问题 X 组
def df_ready() -> bool:
    """表D/表F 由 `出图数据/run_experiments.py` 按需生成（约 38 次全年仿真），不随 export 自动补算。"""
    return ((TABLES / "表F_全知下界阶梯_正式期334天.csv").exists()
            and len(list(TABLES.glob("*_表D_分位数扫描_正式期334天.csv"))) == 4)


def scan(problem: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{problem}_{TAGS[problem]}_表D_分位数扫描_正式期334天.csv",
                       encoding="utf-8-sig")


def x1(four):  # noqa: ARG001
    """覆盖率校准。检查名义分位数 q 是否兑现为实测覆盖率。"""
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.plot([0.45, 0.95], [0.45, 0.95], color=C_GUIDE, linewidth=1.2,
            linestyle="--", label="风险 = 实测")
    for p in ("Q2", "Q3", "Q4-2", "Q4-3"):
        s = scan(p)
        ax.plot(s.q, s.coverage, color=COLOR[p], marker="o", markersize=4.5,
                linewidth=1.6, label=LABEL[p])
    ax.set_xlabel("风险分位数 q", color=C_TEXT)
    ax.set_ylabel("实测覆盖率", color=C_TEXT)
    ax.legend(loc="upper left", framealpha=0.9)
    style_frame_grid(ax)
    save(fig, CHARTS / "综合" / "X-1_覆盖率校准.png")


def x2(four):  # noqa: ARG001
    """费用差距分解。全知阶梯 C0→C3，定位差距出在哪一层。"""
    f = pd.read_csv(TABLES / "表F_全知下界阶梯_正式期334天.csv",
                    encoding="utf-8-sig").set_index("问题")
    probs = ["Q2", "Q3", "Q4-2", "Q4-3"]
    names = ["C0 全知物理下界", "C1 完美负荷/光伏", "C2 完美电价", "C3 实际（本版）"]
    colors = [C_GUIDE, C_SECOND, C_FOURTH, C_MAIN]
    xpos = np.arange(len(probs))
    fig, ax = plt.subplots(figsize=(11, 5))
    w = 0.2
    for i, (col, name, c) in enumerate(zip(["C0", "C1", "C2", "C3"], names, colors)):
        vals = [f.loc[p, col] for p in probs]
        bars = ax.bar(xpos + (i - 1.5) * w, vals, width=w, color=c, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}",
                    ha="center", va="bottom", fontsize=8, color=C_TEXT, rotation=90)
    ax.set_xticks(xpos)
    ax.set_xticklabels([LABEL[p] for p in probs], color=C_TEXT)
    ax.set_ylabel("正式期总费用 / 万元", color=C_TEXT)
    ax.set_ylim(0, f[["C0", "C1", "C2", "C3"]].to_numpy().max() * 1.22)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9.5)
    style_frame_grid(ax, grid=False)
    save(fig, CHARTS / "综合" / "X-2_费用差距分解.png")


def x3(four):  # noqa: ARG001
    """q 对费用的影响。回答「安全余量取多大最省」。"""
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for p in ("Q2", "Q3", "Q4-2", "Q4-3"):
        s = scan(p)
        ax.plot(s.q, s.total_cost_yuan / 1e4, color=COLOR[p], marker="o",
                markersize=4.5, linewidth=1.6, label=LABEL[p])
    ax.axvline(0.80, color=C_GUIDE, linewidth=1.1, linestyle="--")
    ax.text(0.803, ax.get_ylim()[0], " 基线 q=0.80", fontsize=9, color=C_GUIDE,
            va="bottom")
    ax.set_xlabel("风险分位数 q", color=C_TEXT)
    ax.set_ylabel("正式期总费用 / 万元", color=C_TEXT)
    ax.legend(loc="upper right", framealpha=0.9)
    style_frame_grid(ax)
    save(fig, CHARTS / "综合" / "X-3_q对费用的影响.png")


def x4(four):  # noqa: ARG001
    """风险费用权衡。多留余量 → 紧急购电少但计划内购电多。"""
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for p in ("Q2", "Q3", "Q4-2", "Q4-3"):
        s = scan(p)
        ax.plot(s.q, s.emergency_kwh, color=COLOR[p], marker="o", markersize=4.5,
                linewidth=1.6, label=LABEL[p])
    ax.set_xlabel("风险分位数 q", color=C_TEXT)
    ax.set_ylabel("紧急购电量 / kWh", color=C_TEXT)
    axr = ax.twinx()
    for p in ("Q2", "Q3", "Q4-2", "Q4-3"):
        s = scan(p)
        axr.plot(s.q, s.total_cost_yuan / 1e4, color=COLOR[p], linewidth=1.1,
                 linestyle=":", alpha=0.75)
    axr.set_ylabel("总费用 / 万元（虚线）", color=C_TEXT)
    style_frame_grid(ax)
    style_frame_grid(axr, grid=False, frame=False)
    ax.legend(loc="upper right", framealpha=0.9)
    save(fig, CHARTS / "综合" / "X-4_风险费用权衡.png")


# ---------------------------------------------------------------- 入口
def main() -> None:
    four = read_four()
    print(f"四个指定日期：{four}")
    q2_1(four); q2_2(four); q2_3(four); q2_4(four)
    q3_1(four); q3_2(four); q3_3(four); q3_4(four); q3_5(four)
    q4_1(four); q4_2(four); q4_3(four); q4_4(four)
    if df_ready():
        x1(four); x2(four); x3(four); x4(four)
        print("完成：17 张图。")
    else:
        print("跳过 X-1—X-4：缺 表D/表F，先跑 代码/共享/出图数据/run_experiments.py all")
        print("完成：13 张图。")


if __name__ == "__main__":
    main()
