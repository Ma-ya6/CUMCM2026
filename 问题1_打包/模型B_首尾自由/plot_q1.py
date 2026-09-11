# -*- coding: utf-8 -*-
"""
问题一 · 出图脚本

读入问题一的完整结果 csv，生成 5 张图：
    图1  典型日电价、负载、光伏三联曲线
    图2  净负荷、计划购电量与电价
    图3  储能储电量（SOC）日内轨迹
    图4  逐时段充电 / 放电功率
    图5  有储能与无储能的购电量对比

用法：
    python plot_q1.py <结果csv路径> <输出目录> <标题后缀>

依赖：pandas、numpy、matplotlib
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

DT = 1.0 / 6.0


def main():
    csv_path = sys.argv[1]
    outdir = sys.argv[2]
    tag = sys.argv[3] if len(sys.argv) > 3 else ""
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    # 时间 0..143 是 144 个时段，最后一行只有储电量
    seg = df[df["x_t (kW)"].notna()].reset_index(drop=True)
    E = df["E_t (kWh)"].values                     # 145 个
    c = seg["c_t (元/kWh)"].values
    L = seg["L_t (kW)"].values
    P = seg["P_t^fc (kW)"].values
    net = seg["净负荷 (kW)"].values
    x = seg["x_t (kW)"].values
    u = seg["u_t (kW)"].values
    v = seg["v_t (kW)"].values
    hours = np.arange(144) / 6.0
    print("  时段数 %d，储电量点数 %d" % (len(seg), len(E)))

    # ---- 图1 三联曲线
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(hours, c, color="#E45756", lw=1.8, label="电价")
    ax.set_xlabel("时刻 / h")
    ax.set_ylabel("电价 / (元/kWh)", color="#E45756")
    ax.tick_params(axis="y", labelcolor="#E45756")
    ax2 = ax.twinx()
    ax2.plot(hours, L, color="#4C78A8", lw=1.6, label="小区负载")
    ax2.plot(hours, P, color="#54A24B", lw=1.6, label="光伏发电")
    ax2.set_ylabel("功率 / kW")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 2))
    ax.grid(alpha=0.3)
    ax.set_title("图 1  典型日电价、负载与光伏 %s" % tag)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "图1_电价负载光伏.png"), dpi=160)
    plt.close(fig)

    # ---- 图2 净负荷与计划购电
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.fill_between(hours, 0, net, color="#4C78A8", alpha=0.18, label="净负荷（负载−光伏）")
    ax.plot(hours, x, color="#4C78A8", lw=1.8, label="计划购电功率")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("时刻 / h"); ax.set_ylabel("功率 / kW")
    axb = ax.twinx()
    axb.plot(hours, c, color="#E45756", lw=1.3, ls="--", label="电价")
    axb.set_ylabel("电价 / (元/kWh)", color="#E45756")
    axb.tick_params(axis="y", labelcolor="#E45756")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axb.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 2))
    ax.grid(alpha=0.3)
    ax.set_title("图 2  净负荷与计划购电量 %s" % tag)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "图2_净负荷与计划购电.png"), dpi=160)
    plt.close(fig)

    # ---- 图3 SOC 轨迹
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.plot(np.arange(len(E)) / 6.0, E, color="#B279A2", lw=2.2, marker="o", ms=2.5,
            label="储电量 SOC")
    ax.axhline(1200, color="gray", ls="--", lw=1)
    ax.axhline(10800, color="gray", ls="--", lw=1)
    ax.text(0.4, 1220, "SOC 下限 1200 kWh", fontsize=9, color="gray")
    ax.text(0.4, 10850, "SOC 上限 10800 kWh", fontsize=9, color="gray")
    ax.set_xlabel("时刻 / h"); ax.set_ylabel("储电量 / kWh")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 2))
    ax.set_ylim(1000, 11200)
    ax.grid(alpha=0.3); ax.legend(frameon=False, loc="upper left")
    ax.set_title("图 3  储能储电量日内轨迹 %s" % tag)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "图3_储能SOC轨迹.png"), dpi=160)
    plt.close(fig)

    # ---- 图4 充放电功率
    fig, ax = plt.subplots(figsize=(10, 4.3))
    w = 0.13
    ax.bar(hours - w / 2, u * DT, width=w, color="#54A24B", alpha=0.9, label="充电电量（+）")
    ax.bar(hours + w / 2, -v * DT, width=w, color="#E45756", alpha=0.9, label="放电电量（−）")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("时刻 / h"); ax.set_ylabel("每时段充（+）/ 放（−）电量 / kWh")
    ax.set_xlim(-0.5, 24); ax.set_xticks(range(0, 25, 2))
    ax.grid(axis="y", alpha=0.3); ax.legend(frameon=False, loc="upper left")
    ax.set_title("图 4  逐时段充放电电量 %s" % tag)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "图4_充放电功率.png"), dpi=160)
    plt.close(fig)

    # ---- 图5 有储能 vs 无储能
    need = np.maximum(net, 0.0)
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.plot(hours, x * DT, color="#4C78A8", lw=1.8, label="有储能：计划购电量")
    ax.plot(hours, need * DT, color="#F58518", lw=1.5, ls="--", label="无储能：直接按净负荷购电")
    ax.fill_between(hours, x * DT, need * DT, where=(need > x),
                    color="#54A24B", alpha=0.25, label="储能替代的购电")
    ax.fill_between(hours, x * DT, need * DT, where=(need < x),
                    color="#E45756", alpha=0.25, label="为充电而多购的电")
    ax.set_xlabel("时刻 / h"); ax.set_ylabel("购电量 / kWh（每 10 分钟）")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 2))
    ax.grid(alpha=0.3); ax.legend(frameon=False, loc="upper left")
    ax.set_title("图 5  有储能与无储能的购电量对比 %s" % tag)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "图5_有储能与无储能对比.png"), dpi=160)
    plt.close(fig)

    print("  已生成 5 张图到 %s" % outdir)


if __name__ == "__main__":
    main()
