# -*- coding: utf-8 -*-
"""问题3 / 问题4-3 的优化版全年仿真（四个决策时刻的滚动结构）。

上游目录 ``问题3/模型3决策优化版`` 与 ``问题4/重算问题3`` 各自带有同名的
``common.py`` / ``models.py`` / ``dispatch_core.py``，同进程无法共存，故用本脚本
按 ``--problem`` 把其中一个目录插到 ``sys.path`` 最前面，再 import。**预测层、
物理执行器（贪心分支）与结算规则一律沿用上游**，只有 ``plan`` 与执行器来自
``opt_core``，因此各档位之间的费用差只来自被改动的那一处。

档位（``--ladder``）：
  ``legacy-greedy``  上游口径：目标把 1.0c 记在下调量上（方向反）+ 贪心执行
  ``fixed-greedy``   只修费用方向
  ``fixed-mpc``      修方向 + 短时域滚动执行

跑法：
  python run_stage.py --problem Q3
  python run_stage.py --problem Q4-3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OPT_DIR = Path(__file__).resolve().parent
C_DIR = OPT_DIR.parent.parent
DIRS = {
    "Q3": C_DIR / "问题3" / "模型3决策优化版",
    "Q4-3": C_DIR / "问题4" / "重算问题3",
}
OUT_DIR = OPT_DIR / "out"
sys.path.insert(0, str(OPT_DIR))

import joint_stage as js  # noqa: E402
import opt_core as oc  # noqa: E402

STAGES = [0, 6, 12, 18]
REVISION = (6, 12, 18)
HORIZON = 36          # MPC 前瞻窗口 = 6 小时
Q = 0.80

# 输入附件使用“时间点=刚结束时段的右端点”：00:10 对应00:00-00:10。
# 因而6:00发布预报时，t=36已经执行完，只能从t=37（零基36）开始调整。
# 结果附件中的区间表头属于导出层，不能反过来改变物理计算索引。
STAGE_START = {0: 0, 6: 36, 12: 72, 18: 108}


def stage_start(k: int) -> int:
    """Return the first zero-based physical slot adjustable at issue hour ``k``."""
    try:
        return STAGE_START[k]
    except KeyError as exc:
        raise ValueError(f"非题面规定的决策时刻：{k}") from exc


def load_at_stage(data, bundle, day: int, k: int, source: str) -> np.ndarray:
    """日内负载预报；只使用决策时间点之前已经结束的物理时段。"""
    if source == "perfect":
        return data.load_kw[day].copy()
    out = np.array(bundle["load"][day], dtype=float, copy=True)
    done = stage_start(k)
    if done <= 0:
        return out
    realized = float(data.load_kw[day, :done].sum())
    planned = float(out[:done].sum())
    if planned >= 1e-9:
        lam = float(np.clip(1.0 + 0.5 * (realized / planned - 1.0), 0.85, 1.15))
        out[done:] *= lam
    return out


def pv_at_stage(data, bundle, day: int, k: int, source: str) -> np.ndarray:
    """日内光伏预报；官方未覆盖列回退到自有预报。"""
    if source == "perfect":
        return data.pv_kw[day].copy()
    if source == "own":
        return bundle["pv_own"][day].copy()
    if source != "fused":
        raise ValueError(f"未知光伏预报源：{source}")
    out = np.array(bundle["pv_fused"][STAGES.index(k), day], dtype=float, copy=True)
    own = np.asarray(bundle["pv_own"][day], dtype=float)
    missing = ~np.isfinite(out)
    out[missing] = own[missing]
    return out


# --------------------------------------------------------------------------
def load_stack(problem: str):
    """把上游目录插到最前，import 该问题的公共层。"""
    root = DIRS[problem]
    if not root.exists():
        raise FileNotFoundError(root)
    sys.path.insert(0, str(root))
    import common            # noqa: E402
    import dispatch_core     # noqa: E402
    import forecasts         # noqa: E402
    return common, dispatch_core, forecasts


def price_channel(problem: str, common, data, n_days: int):
    """返回 (结算电价, 决策电价)，均为 (n_days, 144)。"""
    if problem == "Q3":
        p = np.tile(data.price, (n_days, 1))
        return p, p
    actual, fc = common.load_price_channel()
    return actual[:n_days], fc[:n_days]


# --------------------------------------------------------------------------
def errors_for(data, bundle, dispatch_core, *, perfect: bool = False) -> dict:
    """各决策时刻的净负荷残差矩阵 ``(n_days, 144-6k)``，单位 kW。

    与 ``reserves_for`` 用同一份残差历史，故储备与情景两个机制口径一致。
    """
    if perfect:
        return {k: np.zeros((len(data.dates), oc.N_SLOT - stage_start(k)))
                for k in STAGES}
    actual_net = data.load_kw - data.pv_kw
    out = {}
    for k in STAGES:
        start = stage_start(k)
        pv = np.asarray(bundle["pv_fused"][STAGES.index(k)], dtype=float).copy()
        own = np.asarray(bundle["pv_own"], dtype=float)
        missing = ~np.isfinite(pv)
        pv[missing] = own[missing]
        out[k] = actual_net[:, start:] - (bundle["load"][:, start:] - pv[:, start:])
    return out


def reserves_for(data, bundle, dispatch_core, n_days: int,
                 *, perfect: bool = False, quantile: float = Q) -> dict:
    """各决策时刻的储能储备 R_d^(k)（kWh），形状 (n_days,)。

    ``perfect=True`` 时用真值取值源，残差恒为 0，储备随之归零。
    """
    errs = errors_for(data, bundle, dispatch_core, perfect=perfect)
    return {k: oc.causal_cumulative_reserve(errs[k][:n_days], quantile)
            for k in STAGES}


def simulate(
    data, bundle, dispatch_core, *,
    settle: str, mpc: bool, reserves: dict, price_s, price_d, days=None,
    perfect: bool = False, exact: bool = False, price_perfect: bool = False,
    joint: bool = False, errs: dict | None = None, no_reserve: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """逐日四个决策时刻的滚动仿真。``settle`` 取 correct/legacy。

    ``perfect=True`` 时负载、光伏、电价全部用真值，得到 $C_1$。
    ``price_perfect=True`` 只把电价换成真值，负载/光伏仍用因果预测，得到 $C_2$。
    ``exact=True`` 时各阶段按严格链式结算目标求解（带历史最小承诺状态）。
    ``joint=True`` 时把承诺对执行后果的定价并入同一支 LP（见 ``joint_stage``），
    ``errs`` 为各决策时刻的因果净负荷残差矩阵，用于生成情景。

    第三个返回值是对**本次运行全部实际执行时段**的在线物理核对累计
    （误差取 max、越界计数求和），口径与 `verify/out/表_物理可行性.csv` 一致。
    """
    load_src = "perfect" if perfect else "own"
    pv_src = "perfect" if perfect else "fused"
    if perfect or price_perfect:
        price_d = price_s
    n_days = len(data.dates) if days is None else min(len(data.dates), days)
    slot_rows = []
    path_days = []
    audit: dict = {}
    energy = oc.E_INITIAL
    stages = [0] + list(REVISION)
    t0 = time.time()

    for day in range(n_days):
        date = data.dates[day]
        start_energy = energy
        commit_path = np.full((4, oc.N_SLOT), np.nan)
        run_min = np.full(oc.N_SLOT, np.inf)
        mis_err = 0.0
        below_min_kwh = 0.0
        prev = None
        ex = {k_: np.zeros(oc.N_SLOT) for k_ in
              ("charge", "discharge", "emergency", "unused_plan",
               "curtailment", "pv_used", "plan_used", "soc")}

        for si, k in enumerate(stages):
            start = stage_start(k)
            is_last = si == len(stages) - 1
            load_full = load_at_stage(data, bundle, day, k, load_src)
            pv_full = pv_at_stage(data, bundle, day, k, pv_src)
            force_end = (oc.E_INITIAL if (is_last and days is None and day == n_days - 1)
                         else None)

            p_dec = price_d[day, start:]
            m_suf = run_min[start:]
            scen, wts = js.causal_scenarios(errs[k], day) if joint else (None, None)
            x = oc.stage_lp(
                load_full[start:], np.nan_to_num(pv_full[start:], nan=0.0),
                p_dec, energy,
                ref=None if prev is None else prev[start:],
                settle="none" if prev is None else settle,
                reserve=(0.0 if no_reserve else float(reserves[k][day])),
                force_end_soc=force_end,
                hist_min=(m_suf if (exact and prev is not None) else None),
                scenarios=scen, weights=wts, emergency=joint,
            )
            if prev is not None:
                r = prev[start:]
                cs = price_s[day][start:]
                dn = np.maximum(r - x, 0.0)
                up = np.maximum(x - r, 0.0)
                true_df = (cs * (np.minimum(m_suf, x) - m_suf)
                           + 0.5 * cs * dn + 1.5 * cs * up)
                if exact:
                    proxy = true_df        # 阶段目标即真实增量
                elif settle == "legacy":
                    proxy = 0.5 * cs * x + 1.0 * cs * dn
                else:
                    proxy = 0.5 * cs * x + 1.0 * cs * up
                # 对齐到 $x=r$ 基准，去掉与 $x$ 无关的常数再比较；
                # 精确档 proxy 与 true_df 同式，残差恒为 0。
                base = 0.0 if exact else 0.5 * cs * r
                mis_err += float((true_df + base - proxy).sum())
                below_min_kwh += float(np.maximum(m_suf - x, 0.0).sum())
            new_commit = np.zeros(oc.N_SLOT) if prev is None else prev.copy()
            new_commit[start:] = x
            commit_path[si] = new_commit
            run_min[start:] = np.minimum(run_min[start:], x)

            exec_end = oc.N_SLOT if is_last else stage_start(stages[si + 1])
            if exec_end > start:
                act_start_energy = energy
                act = oc.run_day_single(
                    new_commit[start:exec_end],
                    data.load_kw[day, start:exec_end],
                    data.pv_kw[day, start:exec_end],
                    p_dec[: exec_end - start],
                    energy,
                    load_fc_kw=load_full[start:exec_end],
                    pv_fc_kw=pv_full[start:exec_end],
                    horizon=HORIZON, force_end_soc=force_end, mpc=mpc,
                )
                energy = act["end_energy"]
                for key in ex:
                    ex[key][start:exec_end] = act[key]
                oc.merge_audit(audit, oc.audit_execution(
                    act, new_commit[start:exec_end],
                    data.load_kw[day, start:exec_end],
                    data.pv_kw[day, start:exec_end], act_start_energy))
            prev = new_commit

        for j in range(1, 4):
            if np.isnan(commit_path[j]).all():
                commit_path[j] = commit_path[j - 1]

        steps = np.diff(commit_path, axis=0)
        has_down = (steps < -1e-9).any(axis=0)
        has_up = (steps > 1e-9).any(axis=0)

        st = oc.settle_period(commit_path, price_s[day])
        emer_cost = float(oc.EMERGENCY_MULTIPLIER * (price_s[day] @ ex["emergency"]))
        if date >= pd.Timestamp("2025-02-01"):
            slot_rows.append({
                "date": date,
                "base_cost_yuan": st["base_yuan"],
                "breach_cost_yuan": st["breach_yuan"],
                "overbuy_cost_yuan": st["overbuy_yuan"],
                "grid_cost_yuan": st["total_yuan"],
                "adjusted_kwh": st["adjusted_kwh"],
                "emergency_kwh": float(ex["emergency"].sum()),
                "emergency_cost_yuan": emer_cost,
                "total_cost_yuan": st["total_yuan"] + emer_cost,
                "planned_purchase_kwh": float(commit_path[0].sum()),
                "final_commitment_kwh": float(commit_path[3].sum()),
                "unused_plan_kwh": float(ex["unused_plan"].sum()),
                "curtailed_pv_kwh": float(ex["curtailment"].sum()),
                "charge_kwh": float(ex["charge"].sum()),
                "discharge_kwh": float(ex["discharge"].sum()),
                "start_soc_kwh": float(start_energy),
                "end_soc_kwh": energy,
                "emergency_slots": int((ex["emergency"] > 1e-9).sum()),
                "reversal_slots": int((has_down & has_up).sum()),
                "down_slots": int(has_down.sum()),
                "path_down_kwh": float(np.maximum(-steps, 0.0).sum()),
                "path_up_kwh": float(np.maximum(steps, 0.0).sum()),
                "below_min_kwh": below_min_kwh,
                "chain_misprice_yuan": mis_err,
            })
            path_days.append(commit_path)
        if (day + 1) % 90 == 0:
            print(f"    {day + 1}/{n_days} 天  储电 {energy:8.0f}  "
                  f"耗时 {time.time() - t0:6.1f}s", flush=True)

    paths = np.stack(path_days) if path_days else np.zeros((0, 4, oc.N_SLOT))
    return pd.DataFrame(slot_rows), paths, audit


def summarize(daily: pd.DataFrame) -> dict:
    costs = daily.total_cost_yuan.to_numpy(float)
    cut = float(np.quantile(costs, 0.95))
    return {
        "total_cost_yuan": float(costs.sum()),
        "grid_cost_yuan": float(daily.grid_cost_yuan.sum()),
        "base_cost_yuan": float(daily.base_cost_yuan.sum()),
        "breach_cost_yuan": float(daily.breach_cost_yuan.sum()),
        "overbuy_cost_yuan": float(daily.overbuy_cost_yuan.sum()),
        "emergency_cost_yuan": float(daily.emergency_cost_yuan.sum()),
        "emergency_kwh": float(daily.emergency_kwh.sum()),
        "days_with_emergency": int((daily.emergency_kwh > 1e-8).sum()),
        "emergency_slots": int(daily.emergency_slots.sum()),
        "planned_purchase_kwh": float(daily.planned_purchase_kwh.sum()),
        "planned_cost_yuan": float(daily.base_cost_yuan.sum()),
        "unused_plan_kwh": float(daily.unused_plan_kwh.sum()),
        "adjusted_kwh": float(daily.adjusted_kwh.sum()),
        "curtailed_pv_kwh": float(daily.curtailed_pv_kwh.sum()),
        "charge_kwh": float(daily.charge_kwh.sum()),
        "discharge_kwh": float(daily.discharge_kwh.sum()),
        "final_soc_kwh": float(daily.end_soc_kwh.iloc[-1]),
        "daily_cost_p95_yuan": cut,
        "daily_cost_cvar95_yuan": float(costs[costs >= cut].mean()),
        "reversal_slots": int(daily.reversal_slots.sum()),
        "reversal_days": int((daily.reversal_slots > 0).sum()),
        "reversal_slot_share": float(daily.reversal_slots.sum() / (len(daily) * oc.N_SLOT)),
        "below_min_kwh": float(daily.below_min_kwh.sum()),
        "chain_misprice_yuan": float(daily.chain_misprice_yuan.sum()),
        "path_down_kwh": float(daily.path_down_kwh.sum()),
        "path_up_kwh": float(daily.path_up_kwh.sum()),
        "days": int(len(daily)),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=list(DIRS), required=True)
    ap.add_argument("--ladder", default="all")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--perfect", action="store_true",
                    help="完美预测口径：真值替代预测、储备归零，得到 C1")
    ap.add_argument("--price-perfect", action="store_true",
                    help="只用真值电价、负载光伏仍为因果预测，得到 C2")
    ap.add_argument("--quantile", type=float, default=Q,
                    help="风险分位数（仅用于可靠性 Pareto 灵敏度展示）")
    args = ap.parse_args()

    common, dispatch_core, forecasts = load_stack(args.problem)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = common.load_inputs()
    # 最终复算禁用预测缓存，避免旧版融合权重或不同长度缓存混入结果。
    bundle = forecasts.forecast_bundle(data, use_cache=False, verbose=False)
    n_days = len(data.dates) if args.days is None else args.days
    price_s, price_d = price_channel(args.problem, common, data, n_days)
    errs = errors_for(data, bundle, dispatch_core, perfect=args.perfect)
    res = reserves_for(data, bundle, dispatch_core, n_days, perfect=args.perfect,
                       quantile=args.quantile)
    print(f"[{args.problem}] 数据就绪：{n_days} 天，储备 R_d^(0) 均值 "
          f"{res[0].mean():.1f} kWh", flush=True)

    ladder = [
        ("legacy-greedy", "legacy", False, False, False, False),
        ("fixed-greedy", "correct", False, False, False, False),
        ("fixed-mpc", "correct", True, False, False, False),
        ("exact-greedy", "correct", False, True, False, False),
        ("exact-mpc", "correct", True, True, False, False),
        ("joint-greedy", "correct", False, False, True, False),
        ("joint-mpc", "correct", True, False, True, False),
        ("joint-exact-greedy", "correct", False, True, True, False),
        # 用联合情景定价**替代**储能储备，而不是叠加：检验风险机制是否重复投保
        ("joint0-greedy", "correct", False, False, True, True),
        ("joint0-exact-greedy", "correct", False, True, True, True),
    ]
    if args.ladder != "all":
        want = set(args.ladder.split(","))
        ladder = [x for x in ladder if x[0] in want]

    out = {}
    suffix = ("_perfect" if args.perfect
              else "_priceperfect" if args.price_perfect else "")
    if args.quantile != Q:
        suffix += f"_q{args.quantile:.2f}"
    for tag, settle, mpc, exact, joint, no_res in ladder:
        t0 = time.time()
        daily, paths, audit = simulate(
            data, bundle, dispatch_core, settle=settle, mpc=mpc,
            reserves=res, price_s=price_s, price_d=price_d,
            days=args.days, perfect=args.perfect,
            exact=exact, price_perfect=args.price_perfect,
            joint=joint, errs=errs, no_reserve=no_res)
        s = summarize(daily)
        s["audit"] = audit
        s["seconds"] = round(time.time() - t0, 1)
        out[tag] = s
        daily.to_csv(OUT_DIR / f"{args.problem}_{tag}{suffix}_逐日.csv",
                     index=False, encoding="utf-8-sig")
        if paths.size:
            n_day, _, n_slot = paths.shape
            # ``paths`` 的轴顺序是 (天, 阶段, 时段)。导出表要求每一行对应
            # (天, 时段)，四列分别为四个阶段，故必须先换成 (天, 时段, 阶段)。
            flat = paths.transpose(0, 2, 1).reshape(n_day * n_slot, 4)
            pd.DataFrame({
                "date": np.repeat(daily.date.to_numpy(), n_slot),
                "slot": np.tile(np.arange(n_slot), n_day),
                "x0_kwh": flat[:, 0], "x1_kwh": flat[:, 1],
                "x2_kwh": flat[:, 2], "x3_kwh": flat[:, 3],
            }).to_csv(OUT_DIR / f"{args.problem}_{tag}{suffix}_承诺路径.csv",
                      index=False, encoding="utf-8-sig", float_format="%.6f")
        print(f"  [{args.problem}] {tag:14s} 全年 {s['total_cost_yuan']:>15,.2f} 元  "
              f"紧急 {s['emergency_kwh']:>10,.1f} kWh  "
              f"下调费 {s['breach_cost_yuan']:>12,.2f}  "
              f"上调费 {s['overbuy_cost_yuan']:>12,.2f}  "
              f"反转时段 {s['reversal_slots']:>7,} "
              f"误价 {s['chain_misprice_yuan']:>12,.0f}  "
              f"({s['seconds']}s)", flush=True)

    (OUT_DIR / f"{args.problem}_阶梯{suffix}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
