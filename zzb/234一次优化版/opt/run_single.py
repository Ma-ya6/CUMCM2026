# -*- coding: utf-8 -*-
"""问题2 / 问题4-2 的优化版全年仿真（每天只在 0:00 决策一次的滚动结构）。

上游 ``问题2/最终优化版`` 与 ``问题4/重算问题2`` 各自带有同名的
``deterministic_baseline`` 与 ``seasonal``，同进程无法共存，故按 ``--problem``
只把其中一个插到 ``sys.path`` 最前面。**预测层一字不动**（同一份
``generate_causal_forecasts``，问题4-2 再加同一份 ``generate_causal_price_forecast``），
改动只在两处：

  §三.1  逐时段分位余量 $m_{d,t}$ → 储能储备 $R_d=Q_{0.80}(\\sum_t\\xi_{j,t}\\Delta t)$
  §三.2  贪心执行 → 短时域滚动执行

档位（``--ladder``）：
  ``legacy-greedy``   逐时段余量 + 贪心（上游口径，须复现冻结值）
  ``reserve-greedy``  储能储备 + 贪心
  ``reserve-mpc``     储能储备 + 滚动 MPC

跑法：
  python run_single.py --problem Q2
  python run_single.py --problem Q4-2
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
    "Q2": C_DIR / "问题2" / "最终优化版",
    "Q4-2": C_DIR / "问题4" / "重算问题2",
}
OUT_DIR = OPT_DIR / "out"
CACHE_DIR = OPT_DIR / "cache"
sys.path.insert(0, str(OPT_DIR))

import joint_stage as js  # noqa: E402
import opt_core as oc  # noqa: E402

Q = 0.80
HORIZON = 36


def load_stack(problem: str):
    root = DIRS[problem]
    if not root.exists():
        raise FileNotFoundError(root)
    sys.path.insert(0, str(root))
    import deterministic_baseline as db      # noqa: E402
    from seasonal import CorrectionConfig, ForecastConfig, generate_causal_forecasts
    return db, CorrectionConfig, ForecastConfig, generate_causal_forecasts


def build_forecasts(problem: str, db, CorrectionConfig, ForecastConfig, gen_fc) -> dict:
    """预测层：与上游 final_model_q2 / final_model_q4_2 逐行同参数，结果落缓存。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{problem}_forecasts.npz"
    if cache.exists():
        z = np.load(cache)
        return {"load": z["load"], "pv": z["pv"], "price": z["price"]}
    data = db.load_inputs()
    cfg = ForecastConfig(correction=CorrectionConfig(
        use_pv_window=True, use_level=True, use_weekly_effect=False))
    fc = gen_fc(data, cfg, verbose=False)
    price = data.price
    if problem == "Q4-2":
        from price_forecast import generate_causal_price_forecast
        price = generate_causal_price_forecast(
            data, fc["diagnostics"], cfg, verbose=False)["price"]
    np.savez(cache, load=fc["load"], pv=fc["pv"], price=price)
    return {"load": fc["load"], "pv": fc["pv"], "price": price}


def simulate(
    db, data, forecasts, *, mode: str, days=None, perfect: bool = False,
    price_perfect: bool = False, quantile: float = Q,
) -> tuple[pd.DataFrame, dict]:
    """``mode`` ∈ {legacy-greedy, reserve-greedy, reserve-mpc}。

    ``perfect=True`` 时以真实负载/光伏/电价替代预测、储备归零，得到
    $C_1$（完美预测 + 当前逐日结构），用于 §四 的二元费用分解。
    ``price_perfect=True`` 只把电价换成真值，负载/光伏仍用因果预测，得到
    $C_2$。问题2 的电价是附件1 固定曲线，本就是已知量，故该档等价于 $C_3$。

    第二个返回值是对**本次运行全部实际执行时段**的在线物理核对累计。
    """
    n_days = len(data.dates) if days is None else min(len(data.dates), days)
    load_fc, pv_fc = forecasts["load"], forecasts["pv"]

    # 问题2：附件1 单条固定曲线，决策价与结算价同一条。
    # 问题4-2：决策价用 0:00 的因果电价预测，结算价用附件4 实际电价。
    if forecasts["price"].ndim == 2:
        price_dec = forecasts["price"][:n_days]
        price_set = data.price[:n_days]
    else:
        price_dec = np.tile(data.price, (n_days, 1))
        price_set = price_dec

    if perfect:
        load_fc, pv_fc = data.load_kw, data.pv_kw
        price_dec = price_set
    elif price_perfect:
        price_dec = price_set

    err = (data.load_kw - data.pv_kw) - (load_fc - pv_fc)
    reserve = oc.causal_cumulative_reserve(err[:n_days], quantile)
    margins = db.causal_residual_quantiles(err[:n_days], quantile)

    rows = []
    audit: dict = {}
    energy = oc.E_INITIAL
    t0 = time.time()
    for day in range(n_days):
        start_energy = energy
        force_end = oc.E_INITIAL if (days is None and day == n_days - 1) else None
        if mode == "legacy-greedy":
            plan = db.solve_plan(
                np.maximum(load_fc[day] + margins[day], 0.0), pv_fc[day],
                price_dec[day], start_energy, db.StorageParameters(),
                force_end_soc=force_end,
            )
            x = plan["purchase"]
            act = oc._greedy_execute(x, data.load_kw[day], data.pv_kw[day], start_energy)
        else:
            joint = mode.startswith("joint")
            # ``joint0``：用联合情景定价**替代**储能储备，而不是与之叠加
            no_res = mode.startswith("joint0")
            scen, wts = js.causal_scenarios(err[:n_days], day) if joint else (None, None)
            x = oc.stage_lp(
                load_fc[day], pv_fc[day], price_dec[day], start_energy,
                settle="none",
                reserve=(0.0 if no_res else float(reserve[day])),
                force_end_soc=force_end,
                scenarios=scen, weights=wts, emergency=joint,
            )
            act = oc.run_day_single(
                x, data.load_kw[day], data.pv_kw[day], price_dec[day], start_energy,
                load_fc_kw=load_fc[day], pv_fc_kw=pv_fc[day],
                horizon=HORIZON, force_end_soc=force_end,
                mpc=(mode.endswith("mpc")),
            )
        oc.merge_audit(audit, oc.audit_execution(
            act, x, data.load_kw[day], data.pv_kw[day], start_energy))
        energy = act["end_energy"]
        c = price_set[day]
        plan_cost = float(c @ x)
        emer_cost = float(oc.EMERGENCY_MULTIPLIER * (c @ act["emergency"]))
        if data.dates[day] >= pd.Timestamp("2025-02-01"):
            rows.append({
                "date": data.dates[day],
                "planned_purchase_kwh": float(x.sum()),
                "planned_cost_yuan": plan_cost,
                "emergency_kwh": float(act["emergency"].sum()),
                "emergency_cost_yuan": emer_cost,
                "total_cost_yuan": plan_cost + emer_cost,
                "unused_plan_kwh": float(act["unused_plan"].sum()),
                "curtailed_pv_kwh": float(act["curtailment"].sum()),
                "charge_kwh": float(act["charge"].sum()),
                "discharge_kwh": float(act["discharge"].sum()),
                "start_soc_kwh": float(start_energy),
                "end_soc_kwh": energy,
                "reserve_kwh": float(reserve[day]),
            })
        if (day + 1) % 90 == 0:
            print(f"    {day + 1}/{n_days} 天  储电 {energy:8.0f}  "
                  f"耗时 {time.time() - t0:6.1f}s", flush=True)
    return pd.DataFrame(rows), audit


def summarize(daily: pd.DataFrame) -> dict:
    costs = daily.total_cost_yuan.to_numpy(float)
    cut = float(np.quantile(costs, 0.95))
    return {
        "total_cost_yuan": float(costs.sum()),
        "planned_cost_yuan": float(daily.planned_cost_yuan.sum()),
        "emergency_cost_yuan": float(daily.emergency_cost_yuan.sum()),
        "emergency_kwh": float(daily.emergency_kwh.sum()),
        "days_with_emergency": int((daily.emergency_kwh > 1e-8).sum()),
        "planned_purchase_kwh": float(daily.planned_purchase_kwh.sum()),
        "unused_plan_kwh": float(daily.unused_plan_kwh.sum()),
        "curtailed_pv_kwh": float(daily.curtailed_pv_kwh.sum()),
        "charge_kwh": float(daily.charge_kwh.sum()),
        "discharge_kwh": float(daily.discharge_kwh.sum()),
        "reserve_mean_kwh": float(daily.reserve_kwh.mean()),
        "final_soc_kwh": float(daily.end_soc_kwh.iloc[-1]),
        "daily_cost_p95_yuan": cut,
        "daily_cost_cvar95_yuan": float(costs[costs >= cut].mean()),
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

    db, CorrectionConfig, ForecastConfig, gen_fc = load_stack(args.problem)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = db.load_inputs()
    forecasts = build_forecasts(args.problem, db, CorrectionConfig,
                                ForecastConfig, gen_fc)
    print(f"[{args.problem}] 预测层就绪：load{forecasts['load'].shape} "
          f"pv{forecasts['pv'].shape}", flush=True)

    ladder = ["legacy-greedy", "reserve-greedy", "reserve-mpc",
              "joint-greedy", "joint-mpc", "joint0-greedy", "joint0-mpc"]
    if args.ladder != "all":
        want = set(args.ladder.split(","))
        ladder = [x for x in ladder if x in want]

    out = {}
    suffix = ("_perfect" if args.perfect
              else "_priceperfect" if args.price_perfect else "")
    if args.quantile != Q:
        suffix += f"_q{args.quantile:.2f}"
    for mode in ladder:
        t0 = time.time()
        daily, audit = simulate(db, data, forecasts, mode=mode, days=args.days,
                                perfect=args.perfect,
                                price_perfect=args.price_perfect,
                                quantile=args.quantile)
        s = summarize(daily)
        s["audit"] = audit
        s["seconds"] = round(time.time() - t0, 1)
        out[mode] = s
        daily.to_csv(OUT_DIR / f"{args.problem}_{mode}{suffix}_逐日.csv",
                     index=False, encoding="utf-8-sig")
        print(f"  [{args.problem}] {mode:16s} 全年 {s['total_cost_yuan']:>15,.2f} 元  "
              f"紧急 {s['emergency_kwh']:>10,.1f} kWh  "
              f"未用 {s['unused_plan_kwh']:>12,.1f} kWh  "
              f"({s['seconds']}s)", flush=True)

    (OUT_DIR / f"{args.problem}_阶梯{suffix}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
