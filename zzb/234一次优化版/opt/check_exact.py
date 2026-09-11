# -*- coding: utf-8 -*-
"""精确链式结算目标（`hist_min` / `exact` 分支）的**独立**验证。

`run_stage.py` 里 exact 档的诊断量写成 ``proxy = true_df``，因此
``chain_misprice_yuan = 0`` 是**按定义置零**，不能作为正确性证据。
本脚本用三条与实现无关的独立检验替代它：

  1. 闭式对照：固定 $m=80,r=100,c=1$，检查 $\Delta F(x)$ 在 $x=70,90,110$
     处分别为 $5,5,15$；
  2. 望远镜恒等式：随机承诺路径上
     $c\,x_0+\sum_k\Delta F_k=F_{\mathrm{settle}}(x_0,\dots,x_K)$；
  3. 网格穷举：单时段小规模 MILP 的最优值与解析 $\Delta F$ 在可行域上的
     穷举最小值对照。

跑法：``python check_exact.py``
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

OPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OPT_DIR))

import opt_core as oc  # noqa: E402

SEED = 20260911
TOL = 1e-8


def delta_F(m: float, r: float, x: float, c: float) -> float:
    """题面定义的严格阶段增量（与 opt_core 的实现无关的独立写法）。"""
    return (c * (min(m, x) - m)
            + 0.5 * c * max(r - x, 0.0)
            + 1.5 * c * max(x - r, 0.0))


def test_closed_form() -> tuple[bool, list[str]]:
    """检验 1：$m=80,r=100,c=1$ 的闭式值。"""
    cases = [(70.0, 5.0), (90.0, 5.0), (110.0, 15.0)]
    lines = ["| $x$ | 期望 $\\Delta F$ | 实测 $\\Delta F$ | 判定 |", "|---:|---:|---:|---|"]
    ok = True
    for x, want in cases:
        got = delta_F(80.0, 100.0, x, 1.0)
        good = abs(got - want) < TOL
        ok &= good
        lines.append(f"| {x:.0f} | {want:.1f} | {got:.1f} | {'✓' if good else '✗'} |")
    return ok, lines


def test_telescoping(n_trial: int = 500) -> tuple[bool, list[str], float]:
    """检验 2：随机承诺路径上的望远镜恒等式。"""
    rng = np.random.default_rng(SEED)
    worst = 0.0
    worst_case = None
    for _ in range(n_trial):
        K = int(rng.integers(2, 7))                    # 段数
        c = float(rng.uniform(0.2, 1.5))
        path = rng.uniform(0.0, 200.0, size=K + 1)      # 承诺路径 $x_0..x_K$
        acc = c * path[0]
        m = path[0]
        for k in range(1, K + 1):
            r, x = path[k - 1], path[k]
            acc += delta_F(m, r, x, c)
            m = min(m, x)
        settle = oc.settle_period(path.reshape(-1, 1), np.array([c]))["total_yuan"]
        err = abs(acc - settle)
        if err > worst:
            worst, worst_case = err, (K, c, path.copy())
    lines = [
        f"随机路径数：{n_trial}（段数 2–6，价格 $c\\in[0.2,1.5]$ 元/kWh）",
        "",
        f"最大偏差 $|c x_0+\\sum_k\\Delta F_k-F_{{\\mathrm{{settle}}}}|$ = **{worst:.3e}** 元",
    ]
    if worst_case is not None and worst > TOL:
        K, c, path = worst_case
        lines.append(f"最差用例：$K={K}$, $c={c:.4f}$, 路径 "
                     + ", ".join(f"{v:.2f}" for v in path))
    return worst < TOL, lines, worst


def _feasible(x: float, net: float, s0: float) -> bool:
    """单时段、末态储电固定为 $s_0$（`force_end_soc`）时的可行性。

    此时 $\eta_c u=v/\eta_d$，由 $x-u+v-w=\\text{net}$、$w\\ge0$ 得
    $x\\ge\\text{net}$，且 $u,v\\le\\mathrm{STEP\\_MAX}$ 自然满足
    （取 $u=v=0,\\ w=x-\\text{net}$）。
    """
    return x >= net - 1e-9


def test_grid_vs_milp(n_trial: int = 60) -> tuple[bool, list[str], float]:
    """检验 3：单时段 MILP 最优值与解析穷举最小值对照。

    **必须固定末态储电**：`force_end_soc=None` 时目标含日末水值项
    $-\\lambda\\,s_1$（$\\lambda=0.9c$，量级与 $\\Delta F$ 相当甚至更大），
    会把解推向储电侧，而本检验收敛的只是 $\\Delta F$ 部分，两者不可混。
    """
    rng = np.random.default_rng(SEED + 1)
    step = 0.02                                       # 网格步长 kWh
    worst = 0.0
    n_fail = 0
    lines = []
    for _ in range(n_trial):
        c = float(rng.uniform(0.2, 1.5))
        price = np.array([c])
        net = float(rng.uniform(20.0, 400.0))       # 净负荷 kWh
        m0 = float(rng.uniform(0.0, 300.0))
        r0 = float(rng.uniform(0.0, 300.0))
        s0 = float(rng.uniform(oc.E_MIN + 200.0, oc.E_MAX - 200.0))

        x_star = float(oc.stage_lp(
            np.array([net / oc.DT]), np.array([0.0]), price, s0,
            ref=np.array([r0]), settle="correct", hist_min=np.array([m0]),
            force_end_soc=s0,
        )[0])
        got = delta_F(m0, r0, x_star, c)

        # 解析穷举：$x\\ge\\text{net}$，步长 `step`，扫描到两倍 STEP_MAX 之外
        grid = np.arange(net, net + 2 * oc.STEP_MAX + step, step)
        vals = (c * (np.minimum(m0, grid) - m0)
                + 0.5 * c * np.maximum(r0 - grid, 0.0)
                + 1.5 * c * np.maximum(grid - r0, 0.0))
        best = float(vals.min())
        # 穷举为离网格，容差按步长与 $\\Delta F$ 的最大斜率 $1.5c$ 给
        tol = step * 1.5 * c + 1e-9
        err = abs(got - best)
        worst = max(worst, err)
        if err > tol:
            n_fail += 1
            lines.append(f"- ✗ $c={c:.3f}$, net={net:.1f}, $m={m0:.1f}$, $r={r0:.1f}$: "
                         f"MILP $x^*={x_star:.3f}\\to\\Delta F={got:.4f}$，"
                         f"穷举最小 {best:.4f}（容差 {tol:.4f}）")

    head = [
        f"随机场景数：{n_trial}（单时段、末态储电固定，"
        f"初始储电 $\\in[E_{{\\min}}+200,E_{{\\max}}-200]$，网格步长 {step} kWh）",
        "",
        f"$|\\Delta F(x^*) - \\min_x\\Delta F(x)|$ 的最大值 = **{worst:.3e}** 元，"
        f"超出步长容差的场景 {n_fail} 个",
    ]
    return n_fail == 0, head + lines, worst


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ok1, l1 = test_closed_form()
    ok2, l2, e2 = test_telescoping()
    ok3, l3, e3 = test_grid_vs_milp()

    out: list[str] = []
    A = out.append
    A("# exact 分支的独立验证")
    A("")
    A("> `chain_misprice_yuan = 0` 在 exact 档是按定义置零，**不构成正确性证据**。")
    A("> 本文件用三条与实现无关的独立检验替代它。")
    A("")
    A("## 检验 1：闭式增量对照（$m=80,\\ r=100,\\ c=1$）")
    A("")
    out.extend(l1)
    A("")
    A(f"**判定：{'通过' if ok1 else '不通过'}**")
    A("")
    A("## 检验 2：望远镜恒等式")
    A("")
    A("对任意承诺路径 $x_0,\\dots,x_K$，题面链式结算可分解为")
    A("")
    A("$$c\\,x_0+\\sum_{k=1}^{K}\\Delta F_k"
      "=c\\min_k x_k+0.5c\\sum_k \\Delta^-_k+1.5c\\sum_k\\Delta^+_k"
      "=F_{\\mathrm{settle}}(x_0,\\dots,x_K).$$")
    A("")
    out.extend(l2)
    A("")
    A(f"**判定：{'通过' if ok2 else '不通过'}**（阈值 $10^{-8}$）")
    A("")
    A("## 检验 3：单时段网格穷举对照 MILP")
    A("")
    out.extend(l3)
    A("")
    A(f"**判定：{'通过' if ok3 else '不通过'}**"
      "（容差 = 网格步长 $\\times\\Delta F$ 最大斜率 $1.5c$；"
      "网格为离网格点，故不能要求逐位相等）")
    A("")
    A("## 结论")
    A("")
    A(f"三条检验全部{'通过' if (ok1 and ok2 and ok3) else '**未全部通过**'}。")
    A("`exact` 分支的目标函数与题面链式结算等价性由此独立成立，"
      "不再依赖「误价为 0」这一按定义置零的量。")
    A("")

    text = "\n".join(out)
    (OPT_DIR / "exact验证.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\n已写出 {OPT_DIR / 'exact验证.md'}")


if __name__ == "__main__":
    main()
