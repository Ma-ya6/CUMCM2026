"""Isolated experiment simulator copied from the frozen default runner.
Only adds risk-q/oracle switches and isolated output paths.
The default source runner and baseline result files are NOT modified.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from run_year import oc, HOURS, forecast_at, predicted_energy, execute

def simulate(a, problem, days=None, *, quantile=.80, oracle_load_pv=False, oracle_price=False, output_dir):
    OUT = Path(output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    def get_forecast(d, k):
        if oracle_load_pv:
            return a["load"][d].copy(), a["pv"][d].copy()
        return forecast_at(a, d, k, problem in ("Q3", "Q4-3"))
    staged = problem in ("Q3", "Q4-3")
    mpc = not staged
    variable = problem.startswith("Q4")
    n = min(days or len(a["load"]), len(a["load"]))
    dates = pd.DatetimeIndex(a["dates"])
    e = oc.E_INITIAL
    carry = 0.
    carry_path = np.zeros(4)
    rows, detail, plans, audit = [], [], [], {}
    error_history = {k: [] for k in HOURS}
    start_time = time.time()
    for d in range(n):
        e_start = e
        last_day = n == 365 and d == 364
        pset = a["price_actual"][d] if variable else a["price_fixed"]
        pdec = (pset if oracle_price else a["price_fc"][d]) if variable else a["price_fixed"]
        lf, pf = get_forecast(d, 0)
        # Both next-midnight endpoints are predicted with information available NOW.
        next_l = a["load"][d+1,0] if oracle_load_pv and d+1 < n else lf[0]
        next_p = a["pv"][d+1,0] if oracle_load_pv and d+1 < n else 0.
        next_c = a["price_actual"][d+1,0] if oracle_price and variable and d+1 < n else pdec[0]
        pl = np.r_[lf[1:], next_l]
        pp = np.r_[pf[1:], next_p]
        pc = np.r_[pdec[1:], next_c]
        pred_e = predicted_energy(e, carry, lf[0], pf[0])
        hist = error_history[0]
        reserve = max(0., float(np.quantile(hist, quantile))) if len(hist) >= 20 else 0.
        x = oc.stage_lp(pl, pp, pc, pred_e, settle="none", reserve=reserve)
        paths = np.tile(x, (4, 1))
        min_path = x.copy()
        ex = {key: np.zeros(144) for key in oc._DAY_KEYS}
        # Actual first interval executes only AFTER the 00:00 new-plan calculation.
        first = execute(np.array([carry]), a["load"][d, :1], a["pv"][d, :1], pdec[:1], e,
                        lf[:1], pf[:1], mpc=mpc, last_day=last_day, physical_start=0)
        oc.merge_audit(audit, oc.audit_execution(first, np.array([carry]), a["load"][d, :1], a["pv"][d, :1], e))
        e = first["end_energy"]
        for key in ex:
            ex[key][0] = first[key][0]
        current_pred = {0: (lf.copy(), pf.copy())}
        effective = np.r_[carry, x[:143]]
        stage_hours = HOURS if staged else (0,)
        for pos, k in enumerate(stage_hours):
            begin = 1 if k == 0 else k*6
            if k:
                lf, pf = get_forecast(d, k)
                current_pred[k] = (lf.copy(), pf.copy())
                slot = begin-1
                next_l = a["load"][d+1,0] if oracle_load_pv and d+1 < n else lf[0]
                lp = np.r_[lf[begin:], next_l]
                ppv = np.r_[pf[begin:], next_p]
                cp = np.r_[pdec[begin:], next_c]
                hist = error_history[k]
                reserve = max(0., float(np.quantile(hist, quantile))) if len(hist) >= 20 else 0.
                revised = oc.stage_lp(lp, ppv, cp, e, ref=paths[pos-1, slot:], settle="correct",
                                      reserve=reserve, hist_min=min_path[slot:])
                x[slot:] = revised
                paths[pos:, :] = x
                min_path[slot:] = np.minimum(min_path[slot:], revised)
                effective[begin:] = x[slot:143]
            end = stage_hours[pos+1]*6 if pos+1 < len(stage_hours) else 144
            c = effective[begin:end]
            act = execute(c, a["load"][d, begin:end], a["pv"][d, begin:end], pdec[begin:end], e,
                          lf[begin:end], pf[begin:end], mpc=mpc, last_day=last_day, physical_start=begin)
            oc.merge_audit(audit, oc.audit_execution(act, c, a["load"][d, begin:end], a["pv"][d, begin:end], e))
            e = act["end_energy"]
            for key in ex:
                ex[key][begin:end] = act[key]
        # Settle each natural-day interval using its full chain, including prior-day last.
        physical_paths = np.column_stack([carry_path, paths[:, :143]])
        settled = oc.settle_period(physical_paths, pset)
        emergency_cost = float(5*pset @ ex["emergency"])
        total = settled["total_yuan"]+emergency_cost
        rows.append(dict(date=str(dates[d].date()), grid_cost_yuan=settled["total_yuan"],
                         base_cost_yuan=settled["base_yuan"], breach_cost_yuan=settled["breach_yuan"],
                         overbuy_cost_yuan=settled["overbuy_yuan"], emergency_cost_yuan=emergency_cost,
                         total_cost_yuan=total, emergency_kwh=float(ex["emergency"].sum()),
                         start_soc_kwh=e_start, end_soc_kwh=e))
        slot_base = pset * physical_paths.min(axis=0)
        changes = np.diff(physical_paths, axis=0)
        slot_cost = slot_base + .5*pset*np.maximum(-changes, 0).sum(axis=0) + 1.5*pset*np.maximum(changes, 0).sum(axis=0) + 5*pset*ex["emergency"]
        for t in range(144):
            detail.append(dict(date=str(dates[d].date()), physical_slot0=t,
                               decision_date=str((dates[d]-pd.Timedelta(days=1) if t == 0 else dates[d]).date()),
                               plan_slot0=143 if t == 0 else t-1,
                               load_kw=a["load"][d,t], pv_kw=a["pv"][d,t], price_actual=pset[t],
                               commitment_kwh=effective[t], total_cost_yuan=slot_cost[t],
                               **{key: ex[key][t] for key in ex}))
        plans.append(paths.copy())
        # Do not observe the next-day first interval in residual history until it occurs.
        for k, (l, p) in current_pred.items():
            b = max(1, k*6)
            error_history[k].append(float(((a["load"][d,b:]-a["pv"][d,b:])-(l[b:]-p[b:])).sum()*oc.DT))
        carry = float(x[-1])
        carry_path = paths[:, -1].copy()
        if (d+1) % 30 == 0 or d+1 == n:
            print(f"{problem}: {d+1}/{n}, elapsed {time.time()-start_time:.1f}s, SOC {e:.2f}", flush=True)
    daily = pd.DataFrame(rows)
    slots = pd.DataFrame(detail)
    daily.to_csv(OUT / f"{problem}_daily.csv", index=False, encoding="utf-8-sig")
    slots.to_csv(OUT / f"{problem}_physical_10min.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(OUT / f"{problem}_plans.npz", dates=a["dates"][:n], paths=np.array(plans))
    if last_day and abs(e-oc.E_INITIAL) > 1e-5:
        raise RuntimeError(f"{problem} actual year-end SOC is {e}, expected 6000")
    for key in ("soc_violations", "power_violations", "simultaneous_charge_discharge", "negative_flow"):
        if audit[key]:
            raise RuntimeError(f"{problem} physical audit failed: {key}={audit[key]}")
    if max(audit[k] for k in audit if k.startswith("max_")) > 1e-5:
        raise RuntimeError(f"{problem} energy identity failed")
    formal = daily[daily.date >= "2025-02-01"]
    result = dict(problem=problem, simulation_days=n, formal_days=len(formal),
                  calendar_year_cost_yuan=float(daily.total_cost_yuan.sum()),
                  formal_period_cost_yuan=float(formal.total_cost_yuan.sum()),
                  final_soc_kwh=e, audit=audit, seconds=time.time()-start_time,
                  quantile=quantile, oracle_load_pv=oracle_load_pv, oracle_price=oracle_price,
                  source="codex isolated experiment; oracle values are NOT implementable forecasts")
    (OUT / f"{problem}_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result

