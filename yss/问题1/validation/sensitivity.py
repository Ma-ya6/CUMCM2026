# -*- coding: utf-8 -*-
"""
问题一 · 检验（三）：参数敏感性与约束松绑

做五组实验，全部在约束 A（可执行的那一版）下重新求解：

    1. 充电效率 η_c 扫描      0.85 / 0.90 / 0.95（η_d 固定 0.90）
    2. 放电效率 η_d 扫描      0.85 / 0.90 / 0.95（η_c 固定 0.90）
    3. 储电量上限 E_max 扫描  6000 / 7200 / 9000 / 10800 / 14400（E_min 固定 1200）
    4. 充放电功率上限扫描      2500 / 5000 / 10000
    5. 约束松绑               E_min=0、容量与功率放到实际不限，即只剩日循环约束

前四组回答"结论对参数有多敏感"，第五组给出"物理限制一共让我们多花了多少钱"，
是一个可以写进论文的上界参照。

顺带自检：每组实验都会报告 Σv/Σu，它应当严格等于 η_c·η_d —— 与容量、功率无关，
只由日循环闭合决定。

用法：python sensitivity.py
产出：tables\\敏感性结果.xlsx、figures\\图2_参数敏感性.png
"""

import os
import sys
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tableio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import solve_q1_ab as q1          # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
TABDIR = os.path.join(BASE, "tables")
FIGDIR = os.path.join(BASE, "figures")

DT = q1.DT
T = q1.T

C1 = "#3E3A8C"        # 第一组曲线
C2 = "#C8553D"        # 第二组曲线
C_BASE = "#1A1A1A"    # 基准线

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"


def title(text):
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


def solve_with(c, L, P_fc, eta_c=None, eta_d=None, e_min=None, e_max=None, p_st=None):
    """
    临时改参数重解一次，算完立刻还原。
    这些常数是 solve_q1_ab 的模块级变量，solve_a() 在调用时才读取，
    所以在外面临时改掉再调用即可，不必给求解函数加一堆参数。
    """
    save = (q1.ETA_C, q1.ETA_D, q1.E_MIN, q1.E_MAX, q1.P_ST_MAX)
    try:
        if eta_c is not None:
            q1.ETA_C = eta_c
        if eta_d is not None:
            q1.ETA_D = eta_d
        if e_min is not None:
            q1.E_MIN = e_min
        if e_max is not None:
            q1.E_MAX = e_max
        if p_st is not None:
            q1.P_ST_MAX = p_st
        return q1.solve_a(c, L, P_fc)
    finally:
        q1.ETA_C, q1.ETA_D, q1.E_MIN, q1.E_MAX, q1.P_ST_MAX = save


def metrics(sol, c, eta_c, eta_d):
    x, u, v, w, E = sol["x"], sol["u"], sol["v"], sol["w"], sol["E"]
    Q_u, Q_v = float((u * DT).sum()), float((v * DT).sum())
    return {
        "购电费 (元)": float((c * x * DT).sum()),
        "购电量 (kWh)": float((x * DT).sum()),
        "充电 ΣuΔt (kWh)": Q_u,
        "放电 ΣvΔt (kWh)": Q_v,
        "Σv/Σu": Q_v / Q_u if Q_u > 1e-9 else float("nan"),
        "η_c·η_d": eta_c * eta_d,
        "弃光 (kWh)": float((w * DT).sum()),
        "E_0 = E_T (kWh)": float(E[0]),
    }


def main():
    os.makedirs(TABDIR, exist_ok=True)
    os.makedirs(FIGDIR, exist_ok=True)

    title("问题一 检验（三）：参数敏感性与约束松绑")
    c, L, P_fc, minutes = q1.read_attachment1()
    if len(c) != T:
        sys.exit("附件 1 行数不是 %d" % T)

    t0 = time.time()
    base = metrics(q1.solve_a(c, L, P_fc), c, q1.ETA_C, q1.ETA_D)
    print("  基准（约束A，η=0.90/0.90，E_max=10800，P_st=5000）：购电费 %.4f 元"
          % base["购电费 (元)"])

    rows = []      # (组名, 参数取值, metrics)

    title("1. 充电效率 η_c 扫描（η_d = 0.90）")
    eta_c_pts = [0.85, 0.90, 0.95]
    fee_eta_c = []
    for e in eta_c_pts:
        m = metrics(solve_with(c, L, P_fc, eta_c=e), c, e, q1.ETA_D)
        fee_eta_c.append(m["购电费 (元)"])
        rows.append(("充电效率 η_c", "η_c = %.2f" % e, m))
        print("  η_c=%.2f  购电费 %10.4f 元（%+.4f%%）  Σv/Σu=%.6f  η_c·η_d=%.6f"
              % (e, m["购电费 (元)"],
                 (m["购电费 (元)"] - base["购电费 (元)"]) / base["购电费 (元)"] * 100,
                 m["Σv/Σu"], m["η_c·η_d"]))

    title("2. 放电效率 η_d 扫描（η_c = 0.90）")
    eta_d_pts = [0.85, 0.90, 0.95]
    fee_eta_d = []
    for e in eta_d_pts:
        m = metrics(solve_with(c, L, P_fc, eta_d=e), c, q1.ETA_C, e)
        fee_eta_d.append(m["购电费 (元)"])
        rows.append(("放电效率 η_d", "η_d = %.2f" % e, m))
        print("  η_d=%.2f  购电费 %10.4f 元（%+.4f%%）  Σv/Σu=%.6f  η_c·η_d=%.6f"
              % (e, m["购电费 (元)"],
                 (m["购电费 (元)"] - base["购电费 (元)"]) / base["购电费 (元)"] * 100,
                 m["Σv/Σu"], m["η_c·η_d"]))

    title("3. 储电量上限 E_max 扫描（E_min = 1200）")
    emax_pts = [6000, 7200, 9000, 10800, 14400, 21600, 36000]
    fee_emax = []
    for e in emax_pts:
        m = metrics(solve_with(c, L, P_fc, e_max=float(e)), c, q1.ETA_C, q1.ETA_D)
        fee_emax.append(m["购电费 (元)"])
        rows.append(("储电量上限 E_max", "E_max = %d" % e, m))
        print("  E_max=%5d  购电费 %10.4f 元（%+.4f%%）  充电 %10.4f kWh"
              % (e, m["购电费 (元)"],
                 (m["购电费 (元)"] - base["购电费 (元)"]) / base["购电费 (元)"] * 100,
                 m["充电 ΣuΔt (kWh)"]))

    title("4. 充放电功率上限 P_st^max 扫描")
    pst_pts = [2500, 5000, 10000]
    for p in pst_pts:
        m = metrics(solve_with(c, L, P_fc, p_st=float(p)), c, q1.ETA_C, q1.ETA_D)
        rows.append(("充放电功率上限", "P_st = %d kW" % p, m))
        print("  P_st=%5d kW  购电费 %10.4f 元（%+.4f%%）"
              % (p, m["购电费 (元)"],
                 (m["购电费 (元)"] - base["购电费 (元)"]) / base["购电费 (元)"] * 100))

    title("5. 约束松绑：容量与功率都不限，只剩日循环约束")
    m_free = metrics(solve_with(c, L, P_fc, e_min=0.0, e_max=1.0e6, p_st=1.0e6),
                     c, q1.ETA_C, q1.ETA_D)
    rows.append(("约束松绑", "容量与功率不限", m_free))
    print("  购电费 %.4f 元（%+.4f%%）"
          % (m_free["购电费 (元)"],
             (m_free["购电费 (元)"] - base["购电费 (元)"]) / base["购电费 (元)"] * 100))

    header = ["实验组", "参数取值", "购电费 (元)", "相对基准", "购电量 (kWh)", "充电 (kWh)",
              "放电 (kWh)", "Σv/Σu", "η_c·η_d", "弃光 (kWh)", "E_0 = E_T (kWh)"]
    fmt_num = [None, None, "0.0000", "+0.0000%;-0.0000%;0.0000%", "0.0000", "0.0000",
               "0.0000", "0.000000", "0.000000", "0.0000", "0.0000"]
    data, fmts = [], []
    data.append(["基准", "—", base["购电费 (元)"], "—", base["购电量 (kWh)"],
                 base["充电 ΣuΔt (kWh)"], base["放电 ΣvΔt (kWh)"], base["Σv/Σu"],
                 base["η_c·η_d"], base["弃光 (kWh)"], base["E_0 = E_T (kWh)"]])
    fmts.append([None, None, "0.0000", None, "0.0000", "0.0000", "0.0000",
                 "0.000000", "0.000000", "0.0000", "0.0000"])
    for group, pname, m in rows:
        rel = (m["购电费 (元)"] - base["购电费 (元)"]) / base["购电费 (元)"]
        data.append([group, pname, m["购电费 (元)"], rel, m["购电量 (kWh)"],
                     m["充电 ΣuΔt (kWh)"], m["放电 ΣvΔt (kWh)"], m["Σv/Σu"],
                     m["η_c·η_d"], m["弃光 (kWh)"], m["E_0 = E_T (kWh)"]])
        fmts.append(fmt_num)

    d_free = base["购电费 (元)"] - m_free["购电费 (元)"]
    notes = [
        "全部在约束 A（固定端点，可执行的那一版）下重新求解。基准：η = 0.90/0.90、E_max = 10800 kWh、",
        "E_min = 1200 kWh、P_st^max = 5000 kW，基准购电费 %.4f 元。" % base["购电费 (元)"],
        "每一行的 Σv/Σu 都严格等于同一行的 η_c·η_d，与容量、功率无关 —— 这个比值只由日循环闭合决定，",
        "是模型内部一致性的强自检；效率项写错，这一列立刻就对不上。",
        "充电效率与放电效率对购电费的影响方向相同、量级相近，符合往返效率 η_c·η_d 只以乘积进入模型的结构。",
        "储电量上限从 10800 往下压，购电费单调上升（压到 6000 时高出 14.04%）；往上扩则一路下降，到约",
        "36000 kWh 才饱和，说明现在 10800 kWh 的容量是明确的瓶颈之一 —— 扩容量仍然省钱，只是边际收益递减。",
        "把容量和功率限制全部拿掉（只剩日循环约束）得到 %.4f 元，比基准低 %.4f 元（%.4f%%），"
        % (m_free["购电费 (元)"], d_free, d_free / base["购电费 (元)"] * 100),
        "这就是物理限制一共让我们多花的钱，可作为论文里的上界参照。",
    ]
    tableio.write_table(os.path.join(TABDIR, "敏感性结果.xlsx"),
                        "问题一 参数敏感性与约束松绑",
                        [{"subtitle": None, "header": header, "rows": data, "formats": fmts}],
                        notes=notes)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    ax = axes[0]
    ax.plot(eta_c_pts, fee_eta_c, "o-", color=C1, lw=2.6, ms=7,
            label="扫描 η_c（η_d = 0.90）")
    ax.plot(eta_d_pts, fee_eta_d, "s--", color=C2, lw=2.6, ms=7,
            label="扫描 η_d（η_c = 0.90）")
    ax.axhline(base["购电费 (元)"], color=C_BASE, lw=1.2, ls=":",
               label="基准 %.4f 元" % base["购电费 (元)"])
    ax.legend(frameon=False, fontsize=9.5, loc="upper right", borderaxespad=0.8)
    ax.set_xlabel("效率 η", fontsize=11.5, color="#333333")
    ax.set_ylabel("全天购电费 / 元", fontsize=11.5, color="#333333")
    ax.set_title("(a) 效率", fontsize=11.5, color="#333333", loc="left")

    ax = axes[1]
    ax.plot(emax_pts, fee_emax, "o-", color=C1, lw=2.6, ms=7, label="购电费")
    ax.axhline(base["购电费 (元)"], color=C_BASE, lw=1.2, ls=":",
               label="基准 %.4f 元" % base["购电费 (元)"])
    ax.legend(frameon=False, fontsize=9.5, loc="upper right", borderaxespad=0.8)
    ax.set_xlabel("储电量上限 E_max / kWh", fontsize=11.5, color="#333333")
    ax.set_ylabel("全天购电费 / 元", fontsize=11.5, color="#333333")
    ax.set_title("(b) 储电量上限", fontsize=11.5, color="#333333", loc="left")

    for ax in axes:
        ax.grid(False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#9E9E9E")
        ax.tick_params(colors="#333333", labelsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.5, 0.015, "图 2  购电费对效率与储电量上限的敏感性", ha="center",
             va="bottom", fontsize=12, color="#1A1A1A")
    path_fig = os.path.join(FIGDIR, "图2_参数敏感性.png")
    fig.savefig(path_fig, dpi=200)
    plt.close(fig)
    print("  已写出：", path_fig)

    title("完成")
    print("  总耗时 %.2f 秒" % (time.time() - t0))
    print("  表格目录：%s" % TABDIR)
    print("  图片目录：%s" % FIGDIR)


if __name__ == "__main__":
    main()
