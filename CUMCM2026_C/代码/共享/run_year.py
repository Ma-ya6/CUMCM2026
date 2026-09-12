"""Independent source-rebuilt annual run; physical and decision axes are separate."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent          # 代码/共享
PROJECT = ROOT.parent.parent                     # 项目根 CUMCM2026_C
sys.path.insert(0, str(ROOT))
import opt_core as oc

HOURS = (0, 6, 12, 18)
DEPS = {"Q2": PROJECT / "代码" / "问题2" / "deps",
        "Q3": PROJECT / "代码" / "问题3" / "deps",
        "Q4-2": PROJECT / "代码" / "问题4" / "问题4-2" / "deps",
        "Q4-3": PROJECT / "代码" / "问题4" / "问题4-3" / "deps"}
RESULT = PROJECT / "结果"
OUT_DIRS = {"Q2": RESULT / "问题2", "Q3": RESULT / "问题3",
            "Q4-2": RESULT / "问题4" / "问题4-2", "Q4-3": RESULT / "问题4" / "问题4-3"}
SUMMARY_DIR = RESULT / "汇总"
CACHE = ROOT / "cache"

def _find_attachments():
    for base in (PROJECT, *PROJECT.parents):
        for cand in (base / "C题" / "附件", base / "附件"):
            if (cand / "附件1.xlsx").exists():
                return cand
    raise FileNotFoundError("未找到 附件1.xlsx，请保持 C题/附件/ 目录完整")

ATTACH = _find_attachments()
CACHE.mkdir(exist_ok=True)
for _d in (*OUT_DIRS.values(), SUMMARY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

OUT_OVERRIDE = None

def out_dir(problem):
    return OUT_OVERRIDE or OUT_DIRS[problem]

def stack(name):
    for mod in list(sys.modules):
        if mod in ("deterministic_baseline", "forecasts", "common", "price_forecast", "seasonal") or mod.startswith("seasonal."):
            sys.modules.pop(mod, None)
    keep = {str(d) for d in DEPS.values()}
    sys.path[:] = [p for p in sys.path if p not in keep]
    sys.path.insert(0, str(DEPS[name]))
    return importlib.import_module("deterministic_baseline")

def fingerprint():
    h = hashlib.sha256()
    for d in DEPS.values():
        for p in sorted(d.rglob("*.py")):
            h.update(p.read_bytes())
    h.update(Path(__file__).read_bytes())
    h.update((ROOT / "opt_core.py").read_bytes())
    for p in sorted(ATTACH.glob("*.xlsx")):
        h.update(p.read_bytes())
    return h.hexdigest()

def prepare():
    """Rebuild once using permitted predictors; no inherited cache."""
    tag = fingerprint()
    path = CACHE / "rebuilt.npz"
    if path.exists():
        with np.load(path) as z:
            if str(z["fingerprint"]) == tag:
                return {k: z[k].copy() for k in z.files if k != "fingerprint"}
    db = stack("Q2")
    from seasonal import CorrectionConfig, ForecastConfig, generate_causal_forecasts
    data = db.load_inputs()
    cfg = ForecastConfig(correction=CorrectionConfig(use_pv_window=True, use_level=True, use_weekly_effect=False))
    print("Rebuilding causal load/PV forecasts from attachments 1,2...", flush=True)
    fc = generate_causal_forecasts(data, cfg, verbose=True)
    arrays = dict(load=data.load_kw, pv=data.pv_kw, price_fixed=data.price,
                  load_fc=fc["load"], pv_fc=fc["pv"],
                  initial_load=data.initial_load, initial_pv=data.initial_pv,
                  dates=data.dates.to_numpy(dtype="datetime64[ns]"))
    diagnostics = fc["diagnostics"]
    # The inherited staged models use ForecastConfig() (weekly correction enabled).
    print("Rebuilding staged-model own forecasts with inherited default configuration...", flush=True)
    staged_fc = generate_causal_forecasts(data, ForecastConfig(), verbose=True)
    arrays["load_stage_fc"] = staged_fc["load"]
    arrays["pv_stage_fc"] = staged_fc["pv"]
    db4 = stack("Q4-2")
    data4 = db4.load_inputs()
    from price_forecast import generate_causal_price_forecast
    from seasonal import CorrectionConfig, ForecastConfig
    cfg4 = ForecastConfig(correction=CorrectionConfig(use_pv_window=True, use_level=True, use_weekly_effect=False))
    print("Rebuilding causal price forecasts from attachment 4...", flush=True)
    pf = generate_causal_price_forecast(data4, diagnostics, cfg4, verbose=True)
    arrays["price_actual"] = data4.price
    arrays["price_fc"] = pf["price"]
    stack("Q2")
    sys.path.insert(0, str(DEPS["Q3"]))
    import forecasts
    off = official_points(arrays)
    projected = SimpleNamespace(official_pv=off, pv_kw=arrays["pv"], dates=pd.DatetimeIndex(arrays["dates"]))
    print("Fitting causal official/own PV fusion...", flush=True)
    arrays["pv_fused"] = forecasts.build_pv_fusion(projected, arrays["pv_stage_fc"], verbose=True)["pv_fused"]
    np.savez_compressed(path, fingerprint=tag, **arrays)
    return arrays

def official_points(a):
    """T+h is a point; interpolate at physical interval RIGHT endpoints."""
    raw = pd.read_excel(ATTACH / "附件3.xlsx")
    raw.columns = ["date", "issue"] + list(range(1, 25))
    raw["date"] = pd.to_datetime(raw["date"].ffill())
    out = np.full((len(a["load"]), 4, 144), np.nan)
    for d, date in enumerate(pd.DatetimeIndex(a["dates"])):
        for kpos, k in enumerate(HOURS):
            rows = raw[(raw["date"] == date) & (raw["issue"].astype(str).str.strip() == f"{k}:00")]
            if len(rows) != 1:
                raise ValueError(f"forecast row missing {date} {k}")
            hourly = rows.iloc[0, 2:].to_numpy(float)
            anchor = a["pv"][d, k*6-1] if k else (a["pv"][d-1, -1] if d else 0.)
            points = np.arange(144)/6 + 1/6
            sl = slice(k*6, 144)
            out[d, kpos, sl] = np.interp(points[sl], k + np.arange(25), np.r_[anchor, hourly])
            # Physical night mask comes only from prior observations / initial given PV.
            prior = a["pv"][max(0, d-30):d]
            daylight = prior.max(axis=0) > 1e-8 if d else a["initial_pv"] > 1e-8
            out[d, kpos, (np.arange(144) >= k*6) & ~daylight] = 0.
    return out

def forecast_at(a, d, k, staged):
    load = a["load_stage_fc" if staged else "load_fc"][d].copy()
    pv = a["pv_fused"][HOURS.index(k), d].copy() if staged else a["pv_fc"][d].copy()
    pv = np.where(np.isfinite(pv), pv, a["pv_stage_fc" if staged else "pv_fc"][d])
    done = k*6
    if done:
        ratio = a["load"][d, :done].sum()/max(load[:done].sum(), 1e-9)
        load[done:] *= np.clip(1+.5*(ratio-1), .85, 1.15)
    prior = a["pv"][max(0, d-30):d]
    daylight = prior.max(axis=0) > 1e-8 if d else a["initial_pv"] > 1e-8
    pv[~daylight] = 0.
    return load, np.maximum(pv, 0)

def predicted_energy(e, x, load, pv):
    net = x + (pv-load)*oc.DT
    if net >= 0:
        return min(oc.E_MAX, e+oc.ETA_C*min(net, oc.STEP_MAX))
    return max(oc.E_MIN, e-min(-net, oc.STEP_MAX)/oc.ETA_D)

def execute(commit, load, pv, price, energy, lfc, pfc, *, mpc, last_day, physical_start):
    """Use original MPC except explicit causally reachable actual year-end policy."""
    if not last_day:
        return oc.run_day_single(commit, load, pv, price, energy, load_fc_kw=lfc, pv_fc_kw=pfc, mpc=mpc, horizon=36)
    out = {key: np.zeros(len(commit)) for key in oc._DAY_KEYS}
    for j, x in enumerate(commit):
        l, p = load[j]*oc.DT, pv[j]*oc.DT
        t = physical_start+j
        remaining = 143-t
        # Never charge above the year-end target; drain excess against real load.
        hi = max(oc.E_INITIAL, energy)
        lo = max(oc.E_MIN, oc.E_INITIAL-remaining*oc.ETA_C*oc.STEP_MAX)
        v = min(oc.STEP_MAX, l, max(0., (energy-oc.E_INITIAL)*oc.ETA_D))
        if v <= 1e-9:
            deficit = max(0., l-x-p)
            v = min(deficit, oc.STEP_MAX, max(0., (energy-lo)*oc.ETA_D))
        e_after = energy-v/oc.ETA_D
        surplus = max(0., x+p+v-l)
        u = 0. if v > 1e-9 else min(oc.STEP_MAX, surplus, max(0., (hi-e_after)/oc.ETA_C))
        if e_after+oc.ETA_C*u < lo-1e-8:
            v = 0.
            e_after = energy
            u = max(u, (lo-energy)/oc.ETA_C)
        if u > oc.STEP_MAX+1e-6:
            raise RuntimeError("year-end target no longer physically reachable")
        consumption = l+u-v
        pv_used = min(p, consumption)
        plan_used = min(x, max(0., consumption-pv_used))
        emergency = max(0., consumption-pv_used-plan_used)
        energy = e_after+oc.ETA_C*u
        values = dict(charge=u, discharge=v, emergency=emergency, unused_plan=x-plan_used,
                      curtailment=p-pv_used, pv_used=pv_used, plan_used=plan_used, soc=energy)
        for key, value in values.items():
            out[key][j] = value
    out["end_energy"] = float(energy)
    return out

def simulate(a, problem, days=None, issue_hours=None):
    staged = problem in ("Q3", "Q4-3")
    mpc = not staged
    variable = problem.startswith("Q4")
    hours = tuple(issue_hours) if issue_hours is not None else HOURS
    if not set(hours) <= set(HOURS):
        raise ValueError(f"issue_hours must be a subset of {HOURS}, got {hours}")
    stage_hours = hours if staged else (0,)
    # 计划数组恒定保留 4 条修订路径：非分阶段问题只发布一次，4 条取值相同。
    # 这样四问的 plans.npz 形状一致，核验与出图可用同一套下标而不必分叉。
    n_stage = len(HOURS)
    n = min(days or len(a["load"]), len(a["load"]))
    dates = pd.DatetimeIndex(a["dates"])
    e = oc.E_INITIAL
    carry = 0.
    carry_path = np.zeros(n_stage)
    rows, detail, plans, audit = [], [], [], {}
    error_history = {k: [] for k in hours}
    start_time = time.time()
    for d in range(n):
        e_start = e
        last_day = n == 365 and d == 364
        pset = a["price_actual"][d] if variable else a["price_fixed"]
        pdec = a["price_fc"][d] if variable else a["price_fixed"]
        lf, pf = forecast_at(a, d, 0, staged)
        # Both next-midnight endpoints are predicted with information available NOW.
        pl = np.r_[lf[1:], lf[0]]
        pp = np.r_[pf[1:], 0.]
        pc = np.r_[pdec[1:], pdec[0]]
        pred_e = predicted_energy(e, carry, lf[0], pf[0])
        hist = error_history[0]
        reserve = max(0., float(np.quantile(hist, .80))) if len(hist) >= 20 else 0.
        x = oc.stage_lp(pl, pp, pc, pred_e, settle="none", reserve=reserve)
        paths = np.tile(x, (n_stage, 1))
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
        for pos, k in enumerate(stage_hours):
            begin = 1 if k == 0 else k*6
            if k:
                lf, pf = forecast_at(a, d, k, staged)
                current_pred[k] = (lf.copy(), pf.copy())
                slot = begin-1
                lp = np.r_[lf[begin:], lf[0]]
                ppv = np.r_[pf[begin:], 0.]
                cp = np.r_[pdec[begin:], pdec[0]]
                hist = error_history[k]
                reserve = max(0., float(np.quantile(hist, .80))) if len(hist) >= 20 else 0.
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
    dest = out_dir(problem)
    daily.to_csv(dest / f"{problem}_daily.csv", index=False, encoding="utf-8-sig")
    slots.to_csv(dest / f"{problem}_physical_10min.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(dest / f"{problem}_plans.npz", dates=a["dates"][:n], paths=np.array(plans))
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
                  source="由 附件1—4 原样重算；不使用任何继承的结果缓存")
    (dest / f"{problem}_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=("Q2", "Q3", "Q4-2", "Q4-3"))
    ap.add_argument("--days", type=int)
    args = ap.parse_args()
    a = prepare()
    results = [simulate(a, p, args.days) for p in ([args.problem] if args.problem else ("Q2", "Q3", "Q4-2", "Q4-3"))]
    if not args.days:
        saved = []
        for p in ("Q2", "Q3", "Q4-2", "Q4-3"):
            f = out_dir(p) / f"{p}_summary.json"
            if f.exists():
                saved.append(json.loads(f.read_text(encoding="utf-8")))
        saved = [s for s in saved if s["simulation_days"] == 365]
        (SUMMARY_DIR / "frozen_numbers.json").write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
        text = "# 全年费用汇总\n\n365天为2025自然年；334天为题目正式输出期（2月1日—12月31日）。两者均按实际执行区间结算。\n\n|模型|365天费用/元|334天费用/元|年终SOC/kWh|\n|---|---:|---:|---:|\n"
        for s in saved:
            text += f"|{s['problem']}|{s['calendar_year_cost_yuan']:,.2f}|{s['formal_period_cost_yuan']:,.2f}|{s['final_soc_kwh']:.6f}|\n"
        (SUMMARY_DIR / "全年费用汇总.md").write_text(text, encoding="utf-8")

if __name__ == "__main__":
    main()
