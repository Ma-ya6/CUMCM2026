"""Independent diagnostics only; do not modify the model or baseline results."""
from __future__ import annotations
import contextlib
import io
import json
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import sparse

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"出图数据"))
import run_year as model
from export_plot_data import load_arrays,physical_paths,clock
from experiment_backend import simulate as experimental_simulate

def save_csv(data,name):
    pd.DataFrame(data).to_csv(HERE/name,index=False,encoding="utf-8-sig")

def fixed_commitment_day_oracle(raw,start):
    """Small, explicitly hindsight-only LP for a fixed last-day commitment."""
    n=len(raw)
    eye=sparse.eye(n,format="csr");z=sparse.csr_matrix((n,n))
    rec=sparse.diags([np.ones(n),-np.ones(n-1)],[0,-1],shape=(n,n),format="csr")
    balance=sparse.hstack([-eye,eye,z,eye,-eye],format="csr")
    state=sparse.hstack([-.9*eye,eye/.9,rec,z,z],format="csr")
    price=raw.price_actual.to_numpy()
    obj=np.r_[np.full(n,1e-6),np.full(n,1e-6),np.zeros(n),5*price,np.full(n,1e-9)]
    bounds=[(0,5000/6)]*n+[(0,min(5000/6,float(l/6))) for l in raw.load_kw]+[(1200,10800)]*n+[(0,None)]*(2*n)
    bounds[3*n-1]=(6000,6000)
    b=np.r_[(raw.load_kw-raw.pv_kw)/6-raw.commitment_kwh,np.r_[start,np.zeros(n-1)]]
    result=linprog(obj,A_eq=sparse.vstack([balance,state],format="csr"),b_eq=b,bounds=bounds,method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    u,v,s,e,w=np.split(result.x,5)
    if ((u>1e-6)&(v>1e-6)).any():
        raise RuntimeError("last-day hindsight LP produced simultaneous charge/discharge")
    return float(5*price@e)

def baseline_checks(a):
    checks={};terminal=[];mask_examples=[]
    for p in ("Q2","Q3","Q4-2","Q4-3"):
        raw=pd.read_csv(ROOT/"results"/f"{p}_physical_10min.csv")
        daily=pd.read_csv(ROOT/"results"/f"{p}_daily.csv")
        with np.load(ROOT/"results"/f"{p}_plans.npz") as z:
            paths=z["paths"]
        chain=physical_paths(paths)
        delta=np.diff(chain,axis=1)
        price=raw.price_actual.to_numpy().reshape(365,144)
        fees=(chain.min(axis=1)*price+.5*price*np.maximum(-delta,0).sum(axis=1)
              +1.5*price*np.maximum(delta,0).sum(axis=1)
              +5*price*raw.emergency.to_numpy().reshape(365,144))
        mapping=np.max(np.abs(chain[:,-1,:].ravel()-raw.commitment_kwh))
        assert mapping<1e-5
        assert np.max(np.abs(fees.ravel()-raw.total_cost_yuan))<1e-5
        assert np.max(np.abs(fees.sum(axis=1)-daily.total_cost_yuan))<1e-5
        assert abs(daily.end_soc_kwh.iloc[-1]-6000)<1e-5
        assert np.max(np.abs(daily.end_soc_kwh.to_numpy()[:-1]-daily.start_soc_kwh.to_numpy()[1:]))<1e-5
        for stage,start in ((1,35),(2,71),(3,107)):
            assert np.max(np.abs(paths[:,stage,:start]-paths[:,stage-1,:start]))<1e-5
        balance=raw.plan_used+raw.emergency+raw.discharge+raw.pv_used-raw.load_kw/6-raw.charge
        soc_rec=np.r_[6000,raw.soc.to_numpy()[:-1]]+.9*raw.charge-raw.discharge/.9
        assert np.max(np.abs(balance))<1e-5
        assert np.max(np.abs(soc_rec-raw.soc))<1e-5
        assert raw.soc.between(1200-1e-5,10800+1e-5).all()
        assert ((raw.charge<=5000/6+1e-5)&(raw.discharge<=5000/6+1e-5)).all()
        assert not ((raw.charge>1e-8)&(raw.discharge>1e-8)).any()
        cost_gap = (chain.min(axis=1)*price+.5*price*np.maximum(-delta,0).sum(axis=1)
                    +1.5*price*np.maximum(delta,0).sum(axis=1))-price*chain[:,-1,:]
        assert cost_gap.min()>-1e-5
        checks[p]=dict(slots=len(raw),days=len(daily),mapping_max_error=float(mapping),
                       daily_loop_equal_days=int(np.isclose(daily.start_soc_kwh,daily.end_soc_kwh,atol=1e-5).sum()),
                       min_chain_cost_above_final_purchase_yuan=float(cost_gap.min()),passed=True)
        last=raw[raw.date=="2025-12-31"].copy()
        normal_short=np.maximum(0,last.load_kw/6-last.commitment_kwh-last.pv_kw/6-last.discharge)
        emergency_for_charge=np.maximum(0,last.emergency-normal_short)
        hindsight=fixed_commitment_day_oracle(last,float(daily.start_soc_kwh.iloc[-1]))
        terminal.append(dict(problem=p,last_day_total_cost_yuan=float(daily.total_cost_yuan.iloc[-1]),
                             last_day_emergency_cost_yuan=float(daily.emergency_cost_yuan.iloc[-1]),
                             min_hindsight_emergency_cost_yuan=hindsight,
                             hindsight_emergency_cost_gap_yuan=float(daily.emergency_cost_yuan.iloc[-1])-hindsight,
                             emergency_for_charge_kwh=float(emergency_for_charge.sum()),
                             emergency_for_charge_cost_yuan=float((5*last.price_actual*emergency_for_charge).sum()),
                             emergency_while_no_load_shortage_slots=int(((normal_short<1e-8)&(last.emergency>1e-8)).sum()),
                             start_soc_kwh=float(daily.start_soc_kwh.iloc[-1]),end_soc_kwh=float(daily.end_soc_kwh.iloc[-1])))
    masks=0;energy=0.
    for d in range(365):
        prior=a["pv"][max(0,d-30):d]
        daylight=prior.max(axis=0)>1e-8 if d else a["initial_pv"]>1e-8
        wrong=(~daylight)&(a["pv"][d]>1e-8)
        masks+=int(wrong.sum());energy+=float(a["pv"][d,wrong].sum()/6)
        if len(mask_examples)<20:
            for t in np.where(wrong)[0]:
                if len(mask_examples)<20:
                    mask_examples.append(dict(date=str(pd.Timestamp(a["dates"][d]).date()),t=int(t+1),
                                              time_end=clock((t+1)*10),actual_pv_kw=float(a["pv"][d,t])))
    save_csv(terminal,"年终保护成本诊断.csv")
    save_csv(mask_examples,"历史昼夜掩码误置零示例.csv")
    return dict(baseline_checks=checks,terminal=terminal,
                historical_night_mask_false_zero_slots=masks,historical_night_mask_false_zero_actual_energy_kwh=energy)

def exact_increment_tests():
    rows=[]
    for start,net,r,m in ((6000,150,50,20),(6000,150,600,100),(1200,100,200,50),(10800,300,200,100)):
        price=1.
        x=float(model.oc.stage_lp(np.array([net*6.]),np.array([0.]),np.array([price]),start,
                                 ref=np.array([r]),hist_min=np.array([m]),settle="correct")[0])
        def objective(x):
            if x>=net:
                u=min(x-net,5000/6,(10800-start)/.9);v=0.;w=x-net-u
            else:
                v=net-x;u=0.;w=0.
                if v>min(5000/6,(start-1200)*.9)+1e-6:
                    return np.inf
            s=start+.9*u-v/.9
            return min(m,x)+.5*max(r-x,0)+1.5*max(x-r,0)-.9*s+1e-7*(u+v)+1e-8*w
        candidates={0.,float(m),float(r),float(net),net+5000/6,
                    net+(10800-start)/.9,max(0.,net-min(5000/6,(start-1200)*.9))}
        best=min(objective(c) for c in candidates)
        residual=objective(x)-best
        assert abs(residual)<1e-4
        rows.append(dict(start_soc=start,load_kwh=net,reference_kwh=r,historical_min_kwh=m,
                         solver_purchase_kwh=x,independent_objective_residual=residual))
    save_csv(rows,"精确链式阶段目标独立检查.csv")
    return rows

def cheap_causality_tests(a):
    results={}
    with TemporaryDirectory(prefix="codex_logic_diagnostic_") as temp:
        temp=Path(temp)
        orig_out=model.OUT
        try:
            model.OUT=temp/"baseline";model.OUT.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                model.simulate(a,"Q4-3",days=3)
                experimental_simulate(a,"Q4-3",days=3,output_dir=temp/"experiment")
            b=pd.read_csv(temp/"baseline"/"Q4-3_daily.csv")
            e=pd.read_csv(temp/"experiment"/"Q4-3_daily.csv")
            assert np.max(np.abs(b.total_cost_yuan-e.total_cost_yuan))<1e-6
            with np.load(temp/"baseline"/"Q4-3_plans.npz") as z:
                original=z["paths"].copy()
            with np.load(temp/"experiment"/"Q4-3_plans.npz") as z:
                assert np.max(np.abs(original-z["paths"]))<1e-6
            changed={k:v.copy() for k,v in a.items()}
            changed["price_actual"][:3] *= 10
            with contextlib.redirect_stdout(io.StringIO()):
                experimental_simulate(changed,"Q4-3",days=3,output_dir=temp/"changed_price")
            with np.load(temp/"changed_price"/"Q4-3_plans.npz") as z:
                assert np.max(np.abs(original-z["paths"]))<1e-6
            results["default_backend_matches_primary_3_days"]=True
            results["future_actual_price_not_used_for_decisions_3_days"]=True
        finally:
            model.OUT=orig_out
    for k in (0,6,12,18):
        before=model.forecast_at(a,100,k,True)
        changed={key:value.copy() for key,value in a.items()}
        changed["load"][100,k*6:]*=100
        changed["pv"][100,k*6:]+=10000
        after=model.forecast_at(changed,100,k,True)
        for lhs,rhs in zip(before,after):
            np.testing.assert_array_equal(lhs,rhs)
    results["intraday_forecasts_invariant_to_future_actuals_all_4_issues"]=True
    return results

def predictor_prefix_tests(a):
    db=model.stack("Q2")
    from seasonal import ForecastConfig,CorrectionConfig,generate_causal_forecasts
    n=365;cut=60
    data=db.load_inputs()
    np.testing.assert_array_equal(data.load_kw,a['load'])
    np.testing.assert_array_equal(data.pv_kw,a['pv'])
    cfg=ForecastConfig(correction=CorrectionConfig(use_pv_window=True,use_level=True,use_weekly_effect=False))
    base=generate_causal_forecasts(data,cfg,verbose=False)
    repeat=generate_causal_forecasts(data,cfg,verbose=False)
    altered=replace(data,load_kw=data.load_kw.copy(order='K'),pv_kw=data.pv_kw.copy(order='K'))
    altered.load_kw[cut:]*=3;altered.pv_kw[cut:]*=.2
    other=generate_causal_forecasts(altered,cfg,verbose=False)
    out={}
    for key in ("load","pv"):
        change=float(np.max(np.abs(base[key][:cut+1]-other[key][:cut+1])))
        cached=float(np.max(np.abs(base[key]-a[f"{key}_fc"][:n])))
        repeat_diff=float(np.max(np.abs(base[key]-repeat[key])))
        idx=np.unravel_index(np.argmax(np.abs(base[key]-a[f"{key}_fc"][:n])),base[key].shape)
        print(f"Predictor {key}: future perturbation={change}, cache difference={cached}",flush=True)
        out[key]=dict(prefix_through_decision_day60_max_difference=change,
                      rebuilt365_days_vs_frozen_cache_max_difference=cached,
                      identical_rebuild_max_difference=repeat_diff,
                      cache_max_difference_at_day_slot0=[int(x) for x in idx],
                      future_perturbation_pass=change<1e-6,cache_match=cached<1e-6)
    return out

def main():
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    a=load_arrays()
    print("Checking frozen annual physical/fee/mapping identities...",flush=True)
    report=baseline_checks(a)
    print("Checking exact MILP increment against independent breakpoints...",flush=True)
    report["exact_increment_cases"]=exact_increment_tests()
    print("Checking causal forecast boundaries and default experiment equivalence...",flush=True)
    report["causality_diagnostics"]=cheap_causality_tests(a)
    print("Checking predictor future-perturbation invariance and full-year cache reproduction...",flush=True)
    report["predictor_prefix_diagnostics"]=predictor_prefix_tests(a)
    (HERE/"logic_evidence.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
