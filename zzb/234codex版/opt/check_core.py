# -*- coding: utf-8 -*-
"""opt_core 的自检：费用方向（§二.1）+ 贪心执行器等价性 + MPC 可行性。

跑法：``python check_core.py``（在 opt/ 下）。全部为纯计算，不读上游模块。
"""
from __future__ import annotations

import sys

import numpy as np

import opt_core as oc


def hand(x: float, r: float, c: float = 1.0) -> float:
    """题面闭式（含常数项 0.5cr）：0.5cx + 1.0c(x-r)+ + 0.5cr"""
    return 0.5 * c * x + 1.0 * c * max(x - r, 0.0) + 0.5 * c * r


def probe(settle: str) -> list[float]:
    """受控探针：2 时段、净负荷 0、上一版承诺 100、电价 1、SOC 起止 6000。

    正确口径最优 x=(0,0)（边际只有下调侧 0.5c）；上游口径最优 x=(100,100)。
    """
    x = oc.stage_lp(
        np.zeros(2), np.zeros(2), np.ones(2), 6000.0,
        ref=np.array([100.0, 100.0]), settle=settle,
        force_end_soc=6000.0,
    )
    return [round(float(v), 6) for v in x]


def lp_value(settle: str, xvec: np.ndarray, ref: np.ndarray, price: np.ndarray) -> float:
    """在给定 x 下复算 LP 目标（去掉常数项），用于对照手算边际。"""
    if settle == "correct":
        extra = (oc.OVERBUY_RATE - oc.BREACH_RATE) * price * np.maximum(xvec - ref, 0.0)
    else:
        extra = (oc.OVERBUY_RATE - oc.BREACH_RATE) * price * np.maximum(ref - xvec, 0.0)
    return float((oc.BREACH_RATE * price * xvec + extra).sum())


def mpc_feasibility(seed: int = 20260910) -> dict:
    """随机一天，检查 MPC 执行结果满足全部物理恒等式与边界。"""
    rng = np.random.default_rng(seed)
    n = oc.N_SLOT
    price = 0.4 + 1.0 * rng.random(n)
    load = rng.uniform(3000, 9000, n)
    pv = np.clip(rng.uniform(-200, 9000, n), 0.0, None)
    x = rng.uniform(0, 6000, n)
    load_fc = load + rng.normal(0, 400, n)
    pv_fc = np.clip(pv + rng.normal(0, 500, n), 0.0, None)

    res = {}
    for tag, mpc in (("greedy", False), ("mpc", True)):
        o = oc.run_day_single(
            x, load, pv, price, oc.E_INITIAL,
            load_fc_kw=load_fc, pv_fc_kw=pv_fc, horizon=36, mpc=mpc,
        )
        L, P = load * oc.DT, pv * oc.DT
        bal = (o["plan_used"] + o["pv_used"] + o["discharge"] + o["emergency"]
               - (L + o["charge"]))
        pvi = P - (o["pv_used"] + o["curtailment"])
        led = x - (o["plan_used"] + o["unused_plan"])
        s = o["soc"]
        soc = s[1:] - (s[:-1] + oc.ETA_C * o["charge"][1:] - o["discharge"][1:] / oc.ETA_D)
        start = s[0] - oc.ETA_C * o["charge"][0] + o["discharge"][0] / oc.ETA_D
        res[tag] = {
            "能量平衡最大误差": float(np.abs(bal).max()),
            "光伏恒等式最大误差": float(np.abs(pvi).max()),
            "购电恒等式最大误差": float(np.abs(led).max()),
            "SOC递推最大误差": float(np.abs(soc).max()),
            "首时段隐含起始SOC": float(start),
            "SOC越界": int((s < oc.E_MIN - 1e-6).sum() + (s > oc.E_MAX + 1e-6).sum()),
            "功率越界": int((o["charge"] > oc.STEP_MAX + 1e-6).sum()
                            + (o["discharge"] > oc.STEP_MAX + 1e-6).sum()),
            "紧急购电_kWh": float(o["emergency"].sum()),
            "弃光_kWh": float(o["curtailment"].sum()),
            "放电_kWh": float(o["discharge"].sum()),
        }
    return res


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ok = True

    print("=" * 78)
    print("§二.1 单时段手算单元测试   r = 100 kWh, c = 1.0 元/kWh")
    print("=" * 78)
    r, c = 100.0, 1.0
    print(f"{'x':>5} | {'手算闭式':>10} | {'correct目标':>12} | {'legacy目标':>12}")
    for x in (80.0, 120.0):
        xv, ref = np.array([x]), np.array([r])
        pv_ = np.array([c])
        cor = lp_value("correct", xv, ref, pv_)
        leg = lp_value("legacy", xv, ref, pv_)
        # 手算含常数项，LP 目标不含，故比较时给手算减去 0.5cr
        h = hand(x, r, c) - 0.5 * c * r
        print(f"{x:>5.0f} | {h:>10.1f} | {cor:>12.1f} | {leg:>12.1f}")
        if not (abs(cor - h) < 1e-9):
            ok = False
    print(f"  手算含常数项：x=80 → {hand(80, r, c):.1f}（=90c）；"
          f"x=120 → {hand(120, r, c):.1f}（=130c）")

    print()
    print("=" * 78)
    print("受控 LP 探针（净负荷 0，承诺 r=100，SOC 起止 6000）")
    print("=" * 78)
    cor, leg = probe("correct"), probe("legacy")
    print(f"  settle='correct' → x = {cor}   预期 [0.0, 0.0]")
    print(f"  settle='legacy'  → x = {leg}   预期 [100.0, 100.0]（复刻上游冻结值）")
    if cor != [0.0, 0.0] or leg != [100.0, 100.0]:
        ok = False

    print()
    print("=" * 78)
    print("执行器对照（随机一天）")
    print("=" * 78)
    res = mpc_feasibility()
    for tag in ("greedy", "mpc"):
        print(f"  [{tag}]  " + "  ".join(
            f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in res[tag].items()))
    for tag in ("greedy", "mpc"):
        r_ = res[tag]
        for key in ("能量平衡最大误差", "光伏恒等式最大误差", "购电恒等式最大误差",
                    "SOC递推最大误差"):
            if r_[key] > 1e-9:
                ok = False
                print(f"  ★ {tag}.{key} = {r_[key]:g} 超差")
        if r_["SOC越界"] or r_["功率越界"] or abs(r_["首时段隐含起始SOC"] - 6000) > 1e-6:
            ok = False
            print(f"  ★ {tag} 边界越界")

    print()
    print("=" * 78)
    print("自检结果：", "全部通过" if ok else "★存在不通过项")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
