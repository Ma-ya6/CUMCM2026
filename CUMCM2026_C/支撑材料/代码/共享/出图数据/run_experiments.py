"""On-demand ONLY: risk scans (D) and oracle ladders (F). Never autorun on import."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

HERE=Path(__file__).resolve().parent
RERUN=HERE.parents[2]/"图表"/"综合"/"补跑输出"   # 补跑实验产物统一落在 图表/综合/ 下
sys.path.insert(0,str(HERE.parent))
from export_plot_data import ROOT,TABLES,SUMMARY,TAGS,FORMAL,load_arrays,save,cvar95,forecasts_and_reserves
from experiment_backend import simulate

RESULT_DIRS={"Q2":ROOT.parent.parent/"结果"/"问题2","Q3":ROOT.parent.parent/"结果"/"问题3",
             "Q4-2":ROOT.parent.parent/"结果"/"问题4"/"问题4-2",
             "Q4-3":ROOT.parent.parent/"结果"/"问题4"/"问题4-3"}

def signature(a, problem, **settings):
    h=hashlib.sha256()
    for p in (HERE/"experiment_backend.py",HERE/"run_experiments.py",ROOT/"run_year.py",ROOT/"opt_core.py",ROOT/"cache"/"rebuilt.npz"):
        h.update(p.read_bytes())
    h.update(json.dumps(dict(problem=problem,**settings),sort_keys=True).encode())
    return h.hexdigest()

def baseline(problem):
    d=RESULT_DIRS[problem]
    daily=pd.read_csv(d/f"{problem}_daily.csv")
    slots=pd.read_csv(d/f"{problem}_physical_10min.csv")
    assert len(daily)==365 and len(slots)==52560
    return daily,slots

def run_cached(a,problem,label,*,q=.8,oracle_load_pv=False,oracle_price=False):
    folder=RERUN/label/problem
    manifest=folder/"experiment_signature.json"
    sig=signature(a,problem,q=q,oracle_load_pv=oracle_load_pv,oracle_price=oracle_price)
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8"))["sha256"]==sig:
        daily=pd.read_csv(folder/f"{problem}_daily.csv")
        slots=pd.read_csv(folder/f"{problem}_physical_10min.csv")
        if len(daily)==365 and len(slots)==52560:
            print("Resume:",label,problem,flush=True)
            return daily,slots
    simulate(a,problem,quantile=q,oracle_load_pv=oracle_load_pv,oracle_price=oracle_price,output_dir=folder)
    manifest.write_text(json.dumps(dict(sha256=sig,settings=dict(q=q,oracle_load_pv=oracle_load_pv,oracle_price=oracle_price)),indent=2),encoding="utf-8")
    return pd.read_csv(folder/f"{problem}_daily.csv"),pd.read_csv(folder/f"{problem}_physical_10min.csv")

def q_reserve(error,q):
    r=np.zeros(len(error))
    for d in range(20,len(error)):
        r[d]=max(0.,float(np.quantile(error[:d],q)))
    return r

def d_row(problem,q,daily,slots,residuals,period):
    formal=period=="formal_334_days"
    days=daily[daily.date>=FORMAL] if formal else daily
    detail=slots[slots.date>=FORMAL] if formal else slots
    selected=np.arange(365)>=31 if formal else np.ones(365,dtype=bool)
    calibrated=selected & (np.arange(365)>=20)
    r=q_reserve(residuals[0],q)
    row=dict(q=q,problem=problem,tag=TAGS[problem],period=period,
             total_cost_yuan=float(days.total_cost_yuan.sum()),emergency_kwh=float(days.emergency_kwh.sum()),
             days_with_emergency=int((days.emergency_kwh>1e-8).sum()),
             unused_plan_kwh=float(detail.unused_plan.sum()),
             reserve_mean_kwh=float(r[selected].mean()),reserve_effective_mean_kwh=float(np.minimum(r[selected],4800.).mean()),
             daily_cost_cvar95_yuan=cvar95(days.total_cost_yuan),
             coverage=float(np.mean(residuals[0][calibrated]<=r[calibrated])),coverage_samples=int(calibrated.sum()),
             final_soc_kwh=float(days.end_soc_kwh.iloc[-1]))
    for k,error in residuals.items():
        reserve=q_reserve(error,q)
        row[f"coverage_{k:02d}"]=float(np.mean(error[calibrated]<=reserve[calibrated]))
    return row

def run_D(a,problems,qs):
    for problem in problems:
        _,_,_,_,_,_,residuals,_=forecasts_and_reserves(a,problem)
        formal,annual=[],[]
        for q in qs:
            print(f"D: {problem}, q={q:.2f}",flush=True)
            if abs(q-.8)<1e-12:
                daily,slots=baseline(problem)
            else:
                daily,slots=run_cached(a,problem,f"D_q{q:.2f}",q=q)
            formal.append(d_row(problem,q,daily,slots,residuals,"formal_334_days"))
            annual.append(d_row(problem,q,daily,slots,residuals,"calendar_365_days"))
            # Checkpoint after EACH completed q; interrupted runs can resume.
            save(pd.DataFrame(formal),TABLES/f"{problem}_{TAGS[problem]}_表D_分位数扫描_正式期334天.csv")
            save(pd.DataFrame(annual),TABLES/"自然年365天"/f"{problem}_{TAGS[problem]}_表D_分位数扫描_自然年365天.csv")

def oracle_lower_bound(a,variable,formal):
    """Perfect-information physical LP relaxation, not a realizable controller.

    Formal-period start SOC=10800 is explicitly relaxed, so the result is a valid
    common lower bound despite different January warm-up states of the models.
    Annual start/end SOC=6000. Chain fees are nonnegative extras and relaxed away.
    """
    selection=slice(31,365) if formal else slice(0,365)
    load=a["load"][selection].ravel()/6
    pv=a["pv"][selection].ravel()/6
    price=(a["price_actual"][selection] if variable else np.tile(a["price_fixed"],(334 if formal else 365,1))).ravel()
    n=len(price)
    eye=sparse.eye(n,format="csr")
    zero=sparse.csr_matrix((n,n))
    # variables: purchase, charge, discharge, SOC, emergency, discarded energy
    balance=sparse.hstack([eye,-eye,eye,zero,eye,-eye],format="csr")
    recurrence=sparse.diags([np.ones(n),-np.ones(n-1)],[0,-1],shape=(n,n),format="csr")
    state=sparse.hstack([zero,-.9*eye,eye/.9,recurrence,zero,zero],format="csr")
    start=10800. if formal else 6000.
    target=np.r_[start,np.zeros(n-1)]
    obj=np.r_[price,np.full(n,1e-8),np.full(n,1e-8),np.zeros(n),5*price,np.full(n,1e-9)]
    bounds=[(0,None)]*n+[(0,5000/6)]*(2*n)+[(1200,10800)]*n+[(0,None)]*(2*n)
    bounds[4*n-1]=(6000,6000)
    if not formal:
        bounds[0]=(0,0) # keep the defined annual first-slot zero commitment
    print(f"C0 LP: {'variable' if variable else 'fixed'} prices, {n} periods, initial SOC={start}",flush=True)
    result=linprog(obj,A_eq=sparse.vstack([balance,state],format="csr"),
                   b_eq=np.r_[load-pv,target],bounds=bounds,method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    g,u,v,s,e,w=np.split(result.x,6)
    assert abs(s[-1]-6000)<1e-5
    physical_residual=float(np.max(np.abs(g-u+v+e-w-(load-pv))))
    if physical_residual>1e-5:
        raise RuntimeError("C0 LP physical equality failed")
    cost=float(price@g+5*price@e) # omit numerical throughput regularizers
    folder=RERUN/"C0"
    folder.mkdir(parents=True,exist_ok=True)
    stem=f"{'variable' if variable else 'fixed'}_{'formal334' if formal else 'annual365'}"
    save(pd.DataFrame(dict(price=price,purchase_kwh=g,charge_kwh=u,discharge_kwh=v,soc_kwh=s,emergency_kwh=e,
                          discarded_kwh=w)),folder/f"{stem}_solution.csv")
    record=dict(cost_yuan=cost,initial_soc_kwh=start,terminal_soc_kwh=6000.,max_balance_residual=physical_residual,
                bound_type="perfect_information_physical_LP_relaxation",formal_initial_state_relaxed=formal,
                simultaneous_charge_discharge_slots=int(((u>1e-6)&(v>1e-6)).sum()))
    (folder/f"{stem}.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    return record

def run_F(a,problems):
    lower={}
    formal,annual=[],[]
    for problem in problems:
        variable=problem.startswith("Q4")
        if variable not in lower:
            lower[variable]=(oracle_lower_bound(a,variable,True),oracle_lower_bound(a,variable,False))
        c3,_=baseline(problem)
        c1,_=run_cached(a,problem,"F_C1_full_oracle",oracle_load_pv=True,oracle_price=True)
        c2,_=run_cached(a,problem,"F_C2_price_oracle",oracle_price=True) if variable else (c3,None)
        for is_formal,dest,bound in ((True,formal,lower[variable][0]),(False,annual,lower[variable][1])):
            costs=[]
            for data in (c1,c2,c3):
                selected=data[data.date>=FORMAL] if is_formal else data
                costs.append(float(selected.total_cost_yuan.sum())/10000)
            C0=bound["cost_yuan"]/10000
            row={"问题":problem,"C0":C0,"C1":costs[0],"C2":costs[1],"C3":costs[2],
                 "unit":"万元","period":"formal_334_days" if is_formal else "calendar_365_days",
                 "C0_initial_soc_kwh":bound["initial_soc_kwh"],
                 "C0_type":bound["bound_type"],
                 "decision_framework_gap_wanyuan":costs[0]-C0,
                 "load_pv_forecast_gap_wanyuan":costs[1]-costs[0],
                 "price_forecast_gap_wanyuan":costs[2]-costs[1],
                 "monotone_ladder":bool(C0<=costs[0]+1e-7 and costs[0]<=costs[1]+1e-7 and costs[1]<=costs[2]+1e-7),
                 "interpretation":"signed heuristic gaps; NOT independently identified causal contributions"}
            dest.append(row)
        save(pd.DataFrame(formal),TABLES/"表F_全知下界阶梯_正式期334天.csv")
        save(pd.DataFrame(formal),SUMMARY/"表F_全知下界阶梯_正式期334天.csv")
        save(pd.DataFrame(annual),TABLES/"自然年365天"/"表F_全知下界阶梯_自然年365天.csv")
    if any(not r["monotone_ladder"] for r in formal+annual):
        print("WARNING: heuristic oracle ladder is not monotone; keep signed gaps, do not force a positive stacked chart.",flush=True)

def main():
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    ap=argparse.ArgumentParser(description="长任务仅在明确运行本文件时开始，不改变正式结果")
    ap.add_argument("task",choices=("D","F","all"))
    ap.add_argument("--problem",choices=tuple(TAGS))
    ap.add_argument("--q",type=float,nargs="+",default=[.50,.55,.60,.65,.70,.75,.80,.85,.90])
    args=ap.parse_args()
    if any(not 0<q<1 for q in args.q):
        ap.error("q must lie in (0,1)")
    a=load_arrays()
    problems=[args.problem] if args.problem else list(TAGS)
    if args.task in ("D","all"):
        run_D(a,problems,args.q)
    if args.task in ("F","all"):
        run_F(a,problems)
    print("Requested experiment tables completed. Original baseline results unchanged.")

if __name__=="__main__":
    main()
