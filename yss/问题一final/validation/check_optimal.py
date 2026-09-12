# -*- coding: utf-8 -*-
"""
问题一 · 检验（二）：最优性 —— KKT 条件与强对偶

对约束 A、约束 B 两个口径分别做三步。

1. 原始可行
   A_eq z = b_eq，l ≤ z ≤ u。核对等式残差与变量是否越界。

2. 对偶可行 + 互补松弛（KKT）
   取求解器给出的等式约束对偶值 y，算约化成本 rc = c − A_eqᵀ y。对每个变量 j 必须满足
       rc_j > 0  ⟹  z_j 贴住下界 l_j
       rc_j < 0  ⟹  z_j 贴住上界 u_j
   等价于 μ_l = max(rc, 0)、μ_u = max(−rc, 0) 分别与 (z − l)、(u − z) 互补。
   违反任意一条，就说明还存在能继续降成本的可行扰动方向，解就不是最优。

3. 强对偶
   拉格朗日对偶函数
       g(y) = bᵀy + Σ_{rc_j > 0} rc_j·l_j + Σ_{rc_j < 0} rc_j·u_j
   由强对偶定理，在最优对偶解处 g(y) = cᵀz。两者之差就是对偶间隙，应当为 0。
   这一步给出一个可核对的数值证书：对偶目标与原始目标相等，且两者同时是全局最优。

说明：对偶值直接取自 HiGHS 返回的等式约束 marginals，不另做数值微分，因此本检验是
      "读证书"而不是"重解一遍"，秒级完成。

用法：python check_optimal.py
产出：tables\\最优性检验.xlsx
      figures\\图1_对偶变量与边际电价（约束A）.png
      figures\\图1_对偶变量与边际电价（约束B）.png
      （两个口径各一张，单面板比例约 2.4:1，便于并排或上下插入论文）
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from plot_style import (C_MAIN, C_PURPLE, style_frame_grid,
                        apply_base_style)                       # noqa: E402

apply_base_style()

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
TABDIR = os.path.join(BASE, "tables")
FIGDIR = os.path.join(BASE, "figures")

T = q1.T
DT = q1.DT

TOL_RC = 1e-7      # 约化成本的符号容差，元/kW
TOL_Z = 1e-6       # 变量"是否贴住某个界"的容差
TOL_GAP = 1e-8     # 相对对偶间隙的通过线

# 解向量布局：x(T) | u(T) | v(T) | E(T+1) | w(T)
IDX = [("x 计划购电", 0, T), ("u 充电", T, 2 * T), ("v 放电", 2 * T, 3 * T),
       ("E 储电量", 3 * T, 4 * T + 1), ("w 弃光", 4 * T + 1, 5 * T + 1)]

C_DUAL = C_MAIN           # 对偶变量：#005792 深蓝
C_PRICE = C_PURPLE        # 边际电价：#6e2277 紫色


def title(text):
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


def analyze(solve_fn, c, L, P_fc, tag):
    """对一个口径做完整的 KKT 与强对偶检验"""
    sol = solve_fn(c, L, P_fc, return_lp=True)
    A, b = sol["A_eq"], sol["b_eq"]
    cobj, z, y = sol["c_obj"], sol["z"], sol["y"]
    if y is None:
        sys.exit("当前 scipy 未返回等式约束的对偶值，无法做 KKT 检验。")
    bounds = sol["bounds"]
    lo = np.array([-np.inf if bl[0] is None else float(bl[0]) for bl in bounds])
    hi = np.array([np.inf if bl[1] is None else float(bl[1]) for bl in bounds])

    # ---- 1 原始可行
    pr_res = float(np.max(np.abs(A @ z - b)))
    in_bounds = bool(np.all((z >= lo - TOL_Z) & (z <= hi + TOL_Z)))

    # ---- 2 对偶可行 + 互补松弛
    rc = cobj - A.T @ y
    at_lo = (z - lo) <= TOL_Z
    at_hi = (hi - z) <= TOL_Z
    bad_pos = (rc > TOL_RC) & (~at_lo)        # rc > 0 却没贴下界
    bad_neg = (rc < -TOL_RC) & (~at_hi)       # rc < 0 却没贴上界
    n_bad = int(bad_pos.sum() + bad_neg.sum())

    # ---- 3 强对偶
    dual_ok = True
    g = float(b @ y)
    for j in range(len(rc)):
        if rc[j] > 0:
            if np.isinf(lo[j]):
                dual_ok = False
            else:
                g += rc[j] * lo[j]
        elif rc[j] < 0:
            if np.isinf(hi[j]):
                dual_ok = False
            else:
                g += rc[j] * hi[j]
    gap = sol["C"] - g
    rel_gap = abs(gap) / max(abs(sol["C"]), 1e-12)

    # ---- 分变量块明细
    blocks = []
    for name, a, bnd in IDX:
        sl = slice(a, bnd)
        blocks.append({
            "name": name, "n": bnd - a,
            "n_lo": int(at_lo[sl].sum()), "n_hi": int(at_hi[sl].sum()),
            "n_bad_pos": int(bad_pos[sl].sum()), "n_bad_neg": int(bad_neg[sl].sum()),
            "max_rc": float(np.max(np.abs(rc[sl]))),
        })

    # ---- 购电时段的边际电价一致性（对偶值的经济学读法）
    # 对偶向量长度 = 等式约束条数，前 T 行对应各时段的电量平衡约束
    y_bal = y[:T]
    mg = c * DT
    buy = sol["x"] > 1e-6
    dev = np.abs(y_bal[buy] - mg[buy])
    dev_max = float(dev.max()) if len(dev) else 0.0
    slack_min = float((mg[~buy] - y_bal[~buy]).min()) if (~buy).any() else float("nan")

    ok = (pr_res < 1e-6 and in_bounds and n_bad == 0 and dual_ok and rel_gap < TOL_GAP)
    return {
        "tag": tag, "sol": sol, "y": y_bal, "mg": mg, "buy": buy,
        "pr_res": pr_res, "in_bounds": in_bounds,
        "n_bad": n_bad, "dual_ok": dual_ok,
        "primal": sol["C"], "dual": g, "gap": gap, "rel_gap": rel_gap,
        "blocks": blocks, "dev_max": dev_max, "slack_min": slack_min,
        "pass": ok,
    }


def build_blocks(res_list):
    head1 = ["口径", "原始可行残差 (kWh)", "KKT 违反变量数",
             "原始目标 (元)", "对偶目标 (元)", "对偶间隙 (元)", "结论"]
    rows1, fmts1 = [], []
    for r in res_list:
        rows1.append([r["tag"], r["pr_res"], r["n_bad"],
                      r["primal"], r["dual"], r["gap"],
                      "通过" if r["pass"] else "未通过"])
        fmts1.append([None, "0.00E+00", "0", "0.0000", "0.0000", "0.00E+00", None])
    block1 = {"subtitle": None, "header": head1, "rows": rows1, "formats": fmts1}

    head2 = ["口径", "变量块", "贴下界数", "贴上界数", "互补松弛违反数"]
    rows2, fmts2 = [], []
    for r in res_list:
        for bl in r["blocks"]:
            rows2.append([r["tag"], bl["name"], bl["n_lo"], bl["n_hi"],
                          bl["n_bad_pos"] + bl["n_bad_neg"]])
            fmts2.append([None, None, "0", "0", "0"])
    block2 = {"subtitle": "分变量块的互补松弛明细", "header": head2,
              "rows": rows2, "formats": fmts2}
    return [block1, block2]


def build_notes(res_list):
    rel = max(r["rel_gap"] for r in res_list)
    dev = max(r["dev_max"] for r in res_list)
    slack = min(r["slack_min"] for r in res_list)
    mrc = max(bl["max_rc"] for r in res_list for bl in r["blocks"])
    return [
        "KKT 条件：对每个变量 j，约化成本 rc_j = c_j − (A_eqᵀy)_j 与变量上下界必须互补 ——",
        "   rc_j > 0 时 z_j 必须贴住下界 l_j（μ_l = rc_j > 0），rc_j < 0 时 z_j 必须贴住上界 u_j（μ_u = −rc_j > 0）。",
        "   违反任一条，说明还存在能继续降低购电费的可行扰动方向，解就不是最优。",
        "强对偶：拉格朗日对偶函数 g(y) = bᵀy + Σ_{rc_j>0} rc_j·l_j + Σ_{rc_j<0} rc_j·u_j。",
        "   由强对偶定理，最优对偶解处 g(y) = cᵀz。表中「原始目标」「对偶目标」两列相等、对偶间隙为 0，",
        "   说明原始解与对偶解同时达到最优，这就是全局最优的数值证书。最大相对间隙 %.2e。" % rel,
        "两个口径的原始可行判定均为：等式残差可忽略、变量无越界。分变量块表中 x/u/v/w 各 144 个变量、",
        "   E 为 145 个；「互补松弛违反数」含 rc>0 未贴下界与 rc<0 未贴上界两类，最大 |约化成本| 为 %.4f 元/kW。" % mrc,
        "影子价格读法：购电时段（x_t > 0）每一度电都靠外网多买，边际成本就是电价本身，故影子价格 = c_t·Δt；",
        "   非购电时段（光伏富余、储能顶负荷）可从系统内部腾出一度电，边际成本低于电价。购电时段两值最大相差",
        "   %.2e 元/kW，非购电时段电价至少高出影子价格 %.4f 元/kW。" % (dev, slack),
        "   单位说明：约束右端是功率 kW，故影子价格单位为元/kW，数值上等于 c_t·Δt = c_t/6。",
    ]


def fig(res_list):
    """影子价格图：约束 A、约束 B 各单独一张，单面板比例约 2.4:1"""
    hours = np.arange(T) / 6.0 + DT / 2.0
    for r in res_list:
        fig, ax = plt.subplots(figsize=(9.6, 4.0))
        ax.plot(hours, r["mg"], color=C_PRICE, lw=3.0, zorder=3,
                label="边际电价 $c_t\\Delta t$")
        ax.plot(hours, r["y"], color=C_DUAL, lw=1.4, ls="--", zorder=4,
                label="对偶变量 $y_t$（影子价格）")
        free = ~r["buy"]
        if free.any():
            ax.scatter(hours[free], r["y"][free], s=26, color=C_DUAL, zorder=5,
                       label="购电量为 0 的时段（影子价格低于电价）")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 2))
        ax.set_ylim(0, max(float(r["y"].max()), float(r["mg"].max())) * 1.34)
        style_frame_grid(ax)        # 四周细黑框 + 内部细灰网格
        ax.tick_params(colors="#333333", labelsize=10.5)
        ax.set_xlabel("时刻 / h", fontsize=11.5, color="#333333")   # 图内不写口径，靠文件名区分
        ax.set_ylabel("元/kW", fontsize=11.5, color="#333333")
        ax.legend(frameon=False, fontsize=10, loc="upper left",
                  ncol=2, borderaxespad=0.8, columnspacing=1.8)
        fig.tight_layout()          # 图题不画进图里，标题由文件名承担
        short = r["tag"].split("（")[0]
        path = os.path.join(FIGDIR, "图1_对偶变量与边际电价（%s）.png" % short)
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print("  已写出：", path)


def main():
    os.makedirs(TABDIR, exist_ok=True)
    os.makedirs(FIGDIR, exist_ok=True)

    title("问题一 检验（二）：最优性 —— KKT 条件与强对偶")
    c, L, P_fc, minutes = q1.read_attachment1()
    if len(c) != T:
        sys.exit("附件 1 行数不是 %d" % T)

    t0 = time.time()
    res_list = []
    for solve_fn, tag in ((q1.solve_a, q1.LABEL_A), (q1.solve_b, q1.LABEL_B)):
        r = analyze(solve_fn, c, L, P_fc, tag)
        res_list.append(r)
        print("  【%s】" % tag)
        print("    原始可行残差 %.3e，变量越界：%s" % (r["pr_res"], "有" if r["in_bounds"] is False else "无"))
        print("    KKT 违反变量数 %d（对偶可行：%s）" % (r["n_bad"], "是" if r["dual_ok"] else "否"))
        print("    原始目标 %.6f 元，对偶目标 %.6f 元，间隙 %.3e（相对 %.3e）"
              % (r["primal"], r["dual"], r["gap"], r["rel_gap"]))
        print("    结论：%s" % ("通过" if r["pass"] else "未通过"))

    title("分变量块明细")
    for r in res_list:
        print("  【%s】" % r["tag"])
        for bl in r["blocks"]:
            print("    %-12s 变量 %4d  贴下界 %4d  贴上界 %4d  违反 %d / %d"
                  % (bl["name"], bl["n"], bl["n_lo"], bl["n_hi"],
                     bl["n_bad_pos"], bl["n_bad_neg"]))

    title("写出结果")
    fig(res_list)      # 先出图，避免表文件被 Excel 占用时连图也写不出来
    tableio.write_table(os.path.join(TABDIR, "最优性检验.xlsx"),
                        "问题一 最优性检验（KKT 条件与强对偶）",
                        build_blocks(res_list), notes=build_notes(res_list))

    title("完成")
    print("  用时 %.2f 秒" % (time.time() - t0))
    print("  表格目录：%s" % TABDIR)
    print("  图片目录：%s" % FIGDIR)


if __name__ == "__main__":
    main()
