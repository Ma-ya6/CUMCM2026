# -*- coding: utf-8 -*-
"""
问题一 · 出图脚本（三张图版）

只出三张图：
    图 1  负载、光伏、净负荷（三条功率曲线）
    图 2  储能储电量（SOC）轨迹 + 充放电功率（左轴 kWh、右轴 kW）
    图 3  计划购电量

样式说明：
    1. 图例放在图内空白处，不占顶部空间
    2. 四周细黑框 + 内部细灰网格（与问题二「指定日期调度」那张图同一套观感）
    3. 曲线加粗，保证打印 PDF 不发虚
    4. 图内不画图题，标题由文件名承担，正文里的图题在 Word/LaTeX 里加

配色（取自上一层目录的 plot_style.py）：
    小区负载   C_SECOND  #00A3FF 亮蓝
    光伏       C_FOURTH  #009473 青绿
    净负荷     C_MAIN    #005792 深蓝（虚线）
    储电量 SOC C_MAIN    #005792 深蓝（alpha 0.8，比标注小字浅一档）
    计划购电量 C_MAIN    #005792 深蓝
    充电功率   C_FOURTH  #009473 青绿（alpha 0.30）
    放电功率   C_RED_LIGHT #E57373 浅红（alpha 0.42）
    参考线     C_GUIDE   #9AA4B2 灰蓝（零线、SOC 上下限）

数据来源：
    用法里的第 1 个参数，通常是 results\\约束A（固定）_完整结果.csv
    或 results\\约束B（自由）_完整结果.csv。
用法：python plot_q1_three.py <结果csv> <输出目录> <标题后缀>

注：第 3 个参数（"（约束A）"这类口径标签）保留只是为了兼容旧命令行，
    图内不再写图题，标题由文件名承担，正文图题在 Word/LaTeX 里加。
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import (C_MAIN, C_SECOND, C_FOURTH, C_RED_LIGHT, C_GUIDE,
                        C_TEXT, C_FRAME, FRAME_LW, style_frame_grid,
                        apply_base_style)                              # noqa: E402

apply_base_style()

DT = 1.0 / 6.0

# ---- 配色 ----
C_LOAD = C_SECOND       # 负载：#00A3FF 亮蓝
C_PV = C_FOURTH         # 光伏：#009473 青绿
C_NET = C_MAIN          # 净负荷：#005792 深蓝
C_SOC_TXT = C_MAIN      # 储能标注小字：#005792，比曲线深
C_BUY = C_MAIN          # 购电量：#005792 深蓝
C_CH = C_FOURTH         # 充电功率：#009473 青绿
C_DIS = C_RED_LIGHT     # 放电功率：#E57373 浅红
C_REF = C_GUIDE         # 参考线（零线、上下限）：#9AA4B2

# 充放电柱的透明度：充电浅、放电稍深，两根柱不叠在一起也能分清
BAR_ALPHA_CH = 0.30
BAR_ALPHA_DIS = 0.42

# 图 2 图例锚点（轴内比例坐标，loc="center left"）：
# 该带（约 62%~78% 高）在整段曲线上都空着 —— 柱高最多占轴高一半，
# SOC 曲线在 1.8h 之后都在 82% 以上，上下限虚线在 10% 与 92%。
LEG_ANCHOR_2 = (0.075, 0.70)

LW_MAIN = 2.4
LW_SEC = 2.2

def style_ax(ax, xlabel=None, ylabel=None, xstep=2, grid=True):
    """统一样式：四周细黑框 + 内部细灰网格（框线、网格色写在 plot_style）"""
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, xstep))
    style_frame_grid(ax, grid=grid)
    ax.tick_params(colors="#333333", labelsize=10.5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11.5, color="#333333")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11.5, color="#333333")


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
    fig, ax = plt.subplots(figsize=(9.6, 6.0))       # 16:10
    ax.axhline(0, color=C_REF, lw=1.0, zorder=3)
    ax.plot(hours, L, color=C_LOAD, lw=LW_MAIN * 0.8, zorder=5, label="小区负载")
    ax.plot(hours, P, color=C_PV, lw=LW_MAIN * 0.8, zorder=5, label="光伏发电")
    ax.plot(hours, net, color=C_DIS, lw=LW_SEC * 0.8, ls="--", zorder=6,
            label="净负荷（负载−光伏）")
    y_top = max(L.max(), P.max(), net.max()) * 1.18
    y_bot = min(0.0, net.min()) * 1.25 - 200
    ax.set_ylim(y_bot, y_top)
    ax.set_yticks(np.arange(-3000, y_top, 800))
    style_ax(ax, xlabel="时刻 / h", ylabel="功率 / kW")
    ax.legend(frameon=False, fontsize=11, loc="upper left",
              ncol=2, borderaxespad=0.8)
    fig.tight_layout()          
    fig.savefig(os.path.join(outdir, "图1_负载光伏净负荷.png"), dpi=200)
    plt.close(fig)

    # ============ 图 2  储能：储电量 SOC + 充放电功率 ============
    fig, ax = plt.subplots(figsize=(9.6, 6.0))       # 16:10
    ax2 = ax.twinx()                                 # 右轴只给充放电功率

    # 先画柱：同一时段模型不允许同时充放电，两支柱不会叠在一起；
    # 右轴上限取 2 倍峰值，柱子最高只占轴高一半，SOC 曲线区域留得住。
    p_max = float(max(u.max(), v.max()))
    p_top = float(np.ceil(p_max * 2.0 / 1000.0) * 1000.0)
    ax2.bar(hours, u,  color=C_CH, width=DT * 0.8,
            alpha=BAR_ALPHA_CH, lw=0, zorder=1, label="充电功率")
    ax2.bar(hours, v,  color=C_DIS, width=DT * 0.9,
            alpha=BAR_ALPHA_DIS, lw=0, zorder=1, label="放电功率")
    ax2.set_ylim(0, p_top * 0.75)
    ax2.set_yticks(np.arange(0, p_top, p_top / 5.0))
    ax2.set_ylabel("充放电功率 / kW", fontsize=11.5, color="#333333")
    for s in ("top", "left", "bottom"):
        ax2.spines[s].set_visible(False)
    ax2.spines["right"].set_color(C_FRAME)
    ax2.spines["right"].set_linewidth(FRAME_LW)
    ax2.tick_params(colors="#333333", labelsize=10.5)
    # 让左轴的 SOC 曲线、标注和图例画在柱子之上（twinx 的轴默认后画）
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.axhline(1200, color=C_REF, ls="--", lw=1.4, zorder=2)
    ax.axhline(10800, color=C_REF, ls="--", lw=1.4, zorder=2)
    ax.plot(h24, E, color=C_MAIN, lw=LW_MAIN * 0.8, zorder=4, label="储电量 SOC", alpha=0.8)
    ei, ai = int(np.argmax(E)), int(np.argmin(E))
    ax.scatter([h24[ei], h24[ai]], [E[ei], E[ai]],
               s=48, color=C_SOC_TXT, zorder=5)
    ax.annotate("峰值 %.0f kWh（%.1f 时）" % (E[ei], h24[ei]),
                (h24[ei], E[ei]), xytext=(8, 10), textcoords="offset points",
                fontsize=10, color=C_SOC_TXT)
    ax.annotate("谷值 %.0f kWh（%.1f 时）" % (E[ai], h24[ai]),
                (h24[ai], E[ai]), xytext=(6, -18), textcoords="offset points",
                fontsize=10, color=C_SOC_TXT,
                bbox=dict(facecolor="white", alpha=0.72, edgecolor="none",
                          pad=1.2))
    ax.text(23.7, 10980, "上限 10800", fontsize=9.5, color=C_TEXT, ha="right",
            va="bottom")
    ax.text(0.3, 1320, "下限 1200", fontsize=9.5, color=C_TEXT, ha="left",
            va="bottom", bbox=dict(facecolor="white", alpha=0.72,
                                   edgecolor="none", pad=1.2))
    ax.set_ylim(0, 11800)      # 纵轴从 0 起，谷值标注落在 0 与 1200 之间，不再压轴
    # 网格画在右轴（下层）上：这样它就是柱子的底纹，而不是压在柱子上
    style_ax(ax, xlabel="时刻 / h", ylabel="储电量 / kWh", grid=False)
    style_frame_grid(ax2, grid=True, frame=False)
    # 图例：SOC 与两支柱合在一处，放在轴内那条空带里
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=11,
              loc="center left", bbox_to_anchor=LEG_ANCHOR_2,
              borderaxespad=0, ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "图2_储能SOC与充放电功率.png"), dpi=200)
    plt.close(fig)

    # ============ 图 3  计划购电量 ============
    fig, ax = plt.subplots(figsize=(9.6, 6.0))       # 16:10
    ax.plot(hours, buy_kwh, color=C_BUY, lw=LW_MAIN * 0.8, zorder=5,
            label="计划购电量")
    ax.fill_between(hours, buy_kwh, color=C_BUY, zorder=5, alpha=0.6)
    ax.set_ylim(0, buy_kwh.max() * 1.25)
    style_ax(ax, xlabel="时刻 / h", ylabel="购电量 / kWh（每 10 分钟）")
    ax.legend(frameon=False, fontsize=11, loc="upper left", borderaxespad=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "图3_计划购电量.png"), dpi=200)
    plt.close(fig)

    print("  已生成 3 张图到 %s" % outdir)
    print("  全天购电量 %.1f kWh，平均购电价 %.4f 元/kWh" % (buy_kwh.sum(), avg_price))
    print("  充电 %.1f kWh，放电 %.1f kWh，弃光 %.1f kWh"
          % (q_ch, q_dis, float((w * DT).sum())))


if __name__ == "__main__":
    main()
