# -*- coding: utf-8 -*-
"""
问题一 · 出图脚本（三张图版）

只出三张图：
    图 1  负载、光伏、净负荷（三条功率曲线）
    图 2  储能储电量（SOC）轨迹
    图 3  计划购电量

样式说明：
    1. 图例放在图内空白处，不占顶部空间
    2. 背景不加网格线（把下面 GRID_ON 改成 True 即可恢复淡横向网格）
    3. 曲线加粗，保证打印 PDF 不发虚
    4. 图标题放在图片下方（论文规范）

配色（按最新要求）：
    负载 棕   #8B5E3C
    光伏 黄   #E8A800（偏琥珀的黄，纯黄在白底上会看不清，所以压深了一点）
    净负荷 深灰虚线
    储能 深蓝紫
    购电量 沿用原蓝 #2E6F9E

数据来源：问题一结果 csv（问题一_完整结果.csv）。
用法：python plot_q1_three.py <结果csv> <输出目录> <标题后缀>
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

DT = 1.0 / 6.0

# ---- 配色 ----
C_LOAD = "#8B5E3C"      # 负载：棕
C_PV = "#E8A800"        # 光伏：黄（偏琥珀，白底上看得清）
C_NET = "#5A5A5A"       # 净负荷：深灰
C_SOC = "#3E3A8C"       # 储能：深蓝紫
C_BUY = "#2E6F9E"       # 购电量：原蓝
C_GRID = "#E2E2E2"

LW_MAIN = 2.8
LW_SEC = 2.2

GRID_ON = False         # False = 背景不加横线；True = 恢复淡横向网格


def style_ax(ax, xlabel=None, ylabel=None, xstep=2):
    """统一样式：无网格（或仅淡横向网格），去掉上右边框"""
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, xstep))
    if GRID_ON:
        ax.grid(axis="y", color=C_GRID, lw=0.9)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9E9E9E")
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors="#333333", labelsize=10.5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11.5, color="#333333")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11.5, color="#333333")


def caption(fig, text):
    """图标题置于图片下方（论文规范）"""
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.5, 0.015, text, ha="center", va="bottom",
             fontsize=12, color="#1A1A1A")


def main():
    csv_path, outdir = sys.argv[1], sys.argv[2]
    tag = sys.argv[3] if len(sys.argv) > 3 else ""
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    seg = df[df["x_t (kW)"].notna()].reset_index(drop=True)
    E = df["E_t (kWh)"].values
    price = seg["c_t (元/kWh)"].values
    L = seg["L_t (kW)"].values
    P = seg["P_t^fc (kW)"].values
    net = seg["净负荷 (kW)"].values
    x = seg["x_t (kW)"].values
    u = seg["u_t (kW)"].values
    v = seg["v_t (kW)"].values
    w = seg["w_t (kW)"].values

    hours = np.arange(len(seg)) / 6.0
    h24 = np.append(hours, 24.0)
    buy_kwh = x * DT
    fee = float((price * buy_kwh).sum())
    avg_price = fee / buy_kwh.sum()
    q_ch, q_dis = float((u * DT).sum()), float((v * DT).sum())

    # ============ 图 1  负载 / 光伏 / 净负荷 ============
    fig, ax = plt.subplots(figsize=(10.5, 4.9))
    ax.axhline(0, color="#B0B0B0", lw=1.0, zorder=3)
    ax.plot(hours, L, color=C_LOAD, lw=LW_MAIN, zorder=5, label="小区负载")
    ax.plot(hours, P, color=C_PV, lw=LW_MAIN, zorder=5, label="光伏发电")
    ax.plot(hours, net, color=C_NET, lw=LW_SEC, ls="--", zorder=6,
            label="净负荷（负载−光伏）")
    y_top = max(L.max(), P.max(), net.max()) * 1.18
    y_bot = min(0.0, net.min()) * 1.25 - 200
    ax.set_ylim(y_bot, y_top)
    style_ax(ax, xlabel="时刻 / h", ylabel="功率 / kW")
    ax.legend(frameon=False, fontsize=11, loc="upper left",
              ncol=2, borderaxespad=0.8)
    caption(fig, "图 1  典型日负载、光伏与净负荷" + tag)
    fig.savefig(os.path.join(outdir, "图1_负载光伏净负荷.png"), dpi=200)
    plt.close(fig)

    # ============ 图 2  储能：储电量 SOC ============
    fig, ax = plt.subplots(figsize=(10.5, 4.9))
    ax.axhline(1200, color="#B3A9BF", ls="--", lw=1.4, zorder=2)
    ax.axhline(10800, color="#B3A9BF", ls="--", lw=1.4, zorder=2)
    ax.plot(h24, E, color=C_SOC, lw=3.0, zorder=4, label="储电量 SOC")
    ei, ai = int(np.argmax(E)), int(np.argmin(E))
    ax.scatter([h24[ei], h24[ai]], [E[ei], E[ai]],
               s=48, color=C_SOC, zorder=5)
    ax.annotate("峰值 %.0f kWh（%.1f 时）" % (E[ei], h24[ei]),
                (h24[ei], E[ei]), xytext=(8, 10), textcoords="offset points",
                fontsize=10, color=C_SOC)
    ax.annotate("谷值 %.0f kWh（%.1f 时）" % (E[ai], h24[ai]),
                (h24[ai], E[ai]), xytext=(6, -18), textcoords="offset points",
                fontsize=10, color=C_SOC)
    ax.text(23.7, 10980, "上限 10800", fontsize=9.5, color="#7A7286", ha="right",
            va="bottom")
    ax.text(0.3, 1320, "下限 1200", fontsize=9.5, color="#7A7286", ha="left",
            va="bottom")
    ax.set_ylim(0, 11800)      # 纵轴从 0 起，谷值标注落在 0 与 1200 之间，不再压轴
    style_ax(ax, xlabel="时刻 / h", ylabel="储电量 / kWh")
    ax.legend(frameon=False, fontsize=11, loc="lower left", borderaxespad=0.8)
    caption(fig, "图 2  储能储电量日内轨迹" + tag)
    fig.savefig(os.path.join(outdir, "图2_储能SOC轨迹.png"), dpi=200)
    plt.close(fig)

    # ============ 图 3  计划购电量 ============
    fig, ax = plt.subplots(figsize=(10.5, 4.9))
    ax.plot(hours, buy_kwh, color=C_BUY, lw=LW_MAIN, zorder=5,
            label="计划购电量")
    ax.set_ylim(0, buy_kwh.max() * 1.35)
    style_ax(ax, xlabel="时刻 / h", ylabel="购电量 / kWh（每 10 分钟）")
    ax.legend(frameon=False, fontsize=11, loc="upper left", borderaxespad=0.8)
    caption(fig, "图 3  典型日计划购电量" + tag)
    fig.savefig(os.path.join(outdir, "图3_计划购电量.png"), dpi=200)
    plt.close(fig)

    print("  已生成 3 张图到 %s" % outdir)
    print("  全天购电量 %.1f kWh，平均购电价 %.4f 元/kWh" % (buy_kwh.sum(), avg_price))
    print("  充电 %.1f kWh，放电 %.1f kWh，弃光 %.1f kWh"
          % (q_ch, q_dis, float((w * DT).sum())))


if __name__ == "__main__":
    main()
