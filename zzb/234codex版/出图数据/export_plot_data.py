"""Cheap, read-only reconstruction of plot tables from frozen codex results."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import run_year as model

TAGS = {"Q2":"reserve-mpc", "Q3":"exact-greedy", "Q4-2":"reserve-mpc", "Q4-3":"exact-greedy"}
FOUR = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")
FORMAL = "2025-02-01"
TABLES = HERE / "表格"

def load_arrays():
    # Intentionally do NOT call prepare(): exporting tables must not run predictors.
    path = ROOT / "cache" / "rebuilt.npz"
    if not path.exists():
        raise FileNotFoundError("缺少本版预测缓存，请先完成本版全年计算；导出脚本不会自动重跑。")
    with np.load(path) as z:
        return {k:z[k].copy() for k in z.files if k != "fingerprint"}

def save(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")

def clock(minutes):
    day, minute = divmod(int(minutes), 1440)
    return f"{minute//60:02d}:{minute%60:02d}" + (f"+{day}" if day else "")

def cvar95(values):
    values = np.asarray(values, float)
    cut = np.quantile(values, .95)
    return float(values[values >= cut].mean())

def forecasts_and_reserves(a, problem):
    staged = problem in ("Q3", "Q4-3")
    n = len(a["load"])
    fc_l, fc_p = np.zeros((n,144)), np.zeros((n,144))
    commit_l, commit_p = np.zeros((n,144)), np.zeros((n,144))
    pred_issue = np.zeros((n,144), int)
    residuals, reserve = {}, {}
    midnight, by_stage = {}, {}
    for k in (model.HOURS if staged else (0,)):
        ls, ps = [], []
        for d in range(n):
            l, p = model.forecast_at(a, d, k, staged)
            ls.append(l)
            ps.append(p)
        ls, ps = np.array(ls), np.array(ps)
        by_stage[k] = (ls, ps)
        if k == 0:
            midnight = (ls, ps)
        begin = max(1, k*6)
        error = ((a["load"][:,begin:]-a["pv"][:,begin:])-(ls[:,begin:]-ps[:,begin:])).sum(axis=1)/6
        residuals[k] = error
        r = np.zeros(n)
        for d in range(20,n):
            r[d] = max(0., float(np.quantile(error[:d], .8)))
        reserve[k] = r
        end = 144 if not staged or k == 18 else k*6+36
        start = 0 if k == 0 else k*6
        fc_l[:,start:end], fc_p[:,start:end] = ls[:,start:end], ps[:,start:end]
        commit_l[:,start:end], commit_p[:,start:end] = ls[:,start:end], ps[:,start:end]
        pred_issue[:,start:end] = k
    # Current-day midnight forecast is used by the executor for the inherited first slot.
    # Governing purchase commitment instead used the PREVIOUS day's last-stage forecast.
    final_l = by_stage[18 if staged else 0][0]
    commit_l[1:,0] = final_l[:-1,0]
    commit_p[:,0] = 0.
    commit_l[0,0] = a["initial_load"][0]
    return fc_l, fc_p, commit_l, commit_p, pred_issue, reserve, residuals, midnight

def physical_paths(paths):
    previous = np.concatenate([np.zeros((1,4)),paths[:-1,:,-1]],axis=0)
    return np.concatenate([previous[:,:,None],paths[:,:,:143]],axis=2)

def metric_row(problem, b, a, period):
    return {"问题":problem, "档位":TAGS[problem], "period":period,
            "total_cost_yuan":float(b.total_cost_yuan.sum()),
            "校正后总费用":float(b.total_cost_yuan.sum()),
            "emergency_kwh":float(b.emergency_kwh.sum()),
            "days_with_emergency":int((b.emergency_kwh>1e-8).sum()),
            "emergency_ratio_of_load":float(b.emergency_kwh.sum()/(a.L_actual_kw.sum()/6)),
            "unused_plan_kwh":float(b.unused_plan_kwh.sum()),
            "curtailed_pv_kwh":float(b.curtailed_pv_kwh.sum()),
            "final_soc_kwh":float(b.end_soc_kwh.iloc[-1]),
            "daily_cost_cvar95_yuan":cvar95(b.total_cost_yuan),
            "reserve_mean_kwh":float(b.reserve_kwh.mean()),
            "adjusted_kwh":float(b.adjusted_kwh.sum()),
            "emergency_slots":int(b.emergency_slots.sum()), "days":len(b),
            "correction_applied_yuan":0.,
            "correction_note":"actual terminal SOC=6000; no residual-value adjustment"}

def physics_row(problem, a, start_soc, period):
    soc = a.E_end_kwh.to_numpy()
    prev = np.r_[start_soc,soc[:-1]]
    balance = a.y_used_kwh+a.e_emergency_kwh+a.v_discharge_kwh+a.g_pv_used_kwh-a.L_actual_kw/6-a.u_charge_kwh
    return {"问题":problem,"档位":TAGS[problem],"period":period,"slots":len(a),
            "max_balance_err":float(np.abs(balance).max()),
            "max_purchase_identity_err":float(np.abs(a.y_used_kwh+a.r_purchase_waste_kwh-a.x_plan_kwh).max()),
            "max_pv_identity_err":float(np.abs(a.g_pv_used_kwh+a.w_pv_waste_kwh-a.P_act_kw/6).max()),
            "max_soc_recursion_err":float(np.abs(soc-prev-.9*a.u_charge_kwh+a.v_discharge_kwh/.9).max()),
            "soc_violations":int(((soc<1200-1e-5)|(soc>10800+1e-5)).sum()),
            "power_violations":int(((a.u_charge_kwh>5000/6+1e-5)|(a.v_discharge_kwh>5000/6+1e-5)).sum()),
            "simultaneous_charge_discharge":int(((a.u_charge_kwh>1e-8)&(a.v_discharge_kwh>1e-8)).sum())}

def export_problem(arrays, problem, week):
    tag = f"{problem}_{TAGS[problem]}"
    source = ROOT / "results"
    raw = pd.read_csv(source/f"{problem}_physical_10min.csv")
    daily = pd.read_csv(source/f"{problem}_daily.csv")
    with np.load(source/f"{problem}_plans.npz") as z:
        paths = z["paths"]
    assert len(raw)==365*144 and len(daily)==365 and paths.shape==(365,4,144)
    fc_l,fc_p,cl,cp,issue,reserves,residuals,midnight = forecasts_and_reserves(arrays,problem)
    chain = physical_paths(paths)
    changes = np.diff(chain,axis=1)
    rename = {"load_kw":"L_actual_kw","pv_kw":"P_act_kw","commitment_kwh":"x_plan_kwh",
              "plan_used":"y_used_kwh","unused_plan":"r_purchase_waste_kwh",
              "charge":"u_charge_kwh","discharge":"v_discharge_kwh","emergency":"e_emergency_kwh",
              "curtailment":"w_pv_waste_kwh","pv_used":"g_pv_used_kwh","soc":"E_end_kwh"}
    a = raw.rename(columns=rename).copy()
    a.insert(1,"t",a.physical_slot0+1)
    a.insert(2,"time_end",[clock((int(t)+1)*10) for t in raw.physical_slot0])
    a["period_start"] = [clock(int(t)*10) for t in raw.physical_slot0]
    a["period_end"] = a.time_end
    a["L_fc_kw"],a["P_fc_kw"] = fc_l.ravel(),fc_p.ravel()
    a["L_commit_fc_kw"],a["P_commit_fc_kw"] = cl.ravel(),cp.ravel()
    a["forecast_issue_hour"] = issue.ravel()
    a["x0_kwh"],a["x3_kwh"] = chain[:,0,:].ravel(),chain[:,3,:].ravel()
    a["price_fc"] = (arrays["price_fc"] if problem.startswith("Q4") else np.tile(arrays["price_fixed"],(365,1))).ravel()
    starts = np.r_[6000.,a.E_end_kwh.to_numpy()[:-1]]
    a["E_start_kwh"] = starts
    # A.t is always the natural-day PHYSICAL index, not the shifted plan index.
    leading = ["date","t","time_end","L_actual_kw","P_act_kw","L_fc_kw","P_fc_kw",
               "x_plan_kwh","y_used_kwh","r_purchase_waste_kwh","u_charge_kwh","v_discharge_kwh",
               "e_emergency_kwh","w_pv_waste_kwh","g_pv_used_kwh","E_end_kwh"]
    a = a[leading+[c for c in a if c not in leading]]
    b = daily.copy()
    group = a.groupby("date",sort=False)
    mapped = {"unused_plan_kwh":"r_purchase_waste_kwh","curtailed_pv_kwh":"w_pv_waste_kwh",
              "charge_kwh":"u_charge_kwh","discharge_kwh":"v_discharge_kwh"}
    for dst,src in mapped.items():
        b[dst] = group[src].sum().to_numpy()
    b["planned_purchase_kwh"] = chain[:,0,:].sum(axis=1)
    b["planned_cost_yuan"] = (chain[:,0,:]*a.price_actual.to_numpy().reshape(365,144)).sum(axis=1)
    b["decision_day_planned_purchase_kwh"] = paths[:,0,:].sum(axis=1)
    b["final_commitment_kwh"] = chain[:,3,:].sum(axis=1)
    b["reserve_kwh"] = reserves[0]
    b["reserve_effective_kwh"] = np.minimum(reserves[0],4800.)
    for k,r in reserves.items():
        b[f"reserve_{k:02d}_kwh"] = r
        b[f"window_residual_{k:02d}_kwh"] = residuals[k]
    up,down = np.maximum(changes,0),np.maximum(-changes,0)
    b["path_up_kwh"],b["path_down_kwh"] = up.sum(axis=(1,2)),down.sum(axis=(1,2))
    b["adjusted_kwh"] = b.path_up_kwh+b.path_down_kwh
    b["emergency_slots"] = (a.e_emergency_kwh.to_numpy().reshape(365,144)>1e-8).sum(axis=1)
    b["reversal_slots"] = ((changes>1e-8).any(axis=1)&(changes < -1e-8).any(axis=1)).sum(axis=1)
    b["down_slots"] = (changes < -1e-8).any(axis=1).sum(axis=1)
    minimum = chain[:,0,:].copy()
    below = np.zeros(365)
    for pos in range(1,4):
        below += np.maximum(minimum-chain[:,pos,:],0).sum(axis=1)
        minimum = np.minimum(minimum,chain[:,pos,:])
    b["below_min_kwh"] = below
    price = a.price_actual.to_numpy().reshape(365,144)
    independent = (minimum*price+.5*price*down.sum(axis=1)+1.5*price*up.sum(axis=1)).sum(axis=1)
    b["chain_misprice_yuan"] = b.grid_cost_yuan-independent
    b["cost_identity_residual_yuan"] = b.total_cost_yuan-b.grid_cost_yuan-b.emergency_cost_yuan
    assert np.max(np.abs(b.chain_misprice_yuan)) < 1e-5
    assert np.max(np.abs(b.cost_identity_residual_yuan)) < 1e-5
    for d in range(365):
        assert np.max(np.abs(a.x_plan_kwh.to_numpy().reshape(365,144)[d]-chain[d,3,:])) < 1e-5
    ca = {"date":a.date,"t":a.t,"slot":a.t,"physical_slot0":a.physical_slot0,
          "decision_date":a.decision_date,"plan_slot0":a.plan_slot0,
          "period_start":a.period_start,"period_end":a.period_end}
    for pos in range(4):
        ca[f"x{pos}_kwh"] = chain[:,pos,:].ravel()
    c = pd.DataFrame(ca)
    dates = pd.DatetimeIndex(arrays["dates"])
    cp_rows = {"date":np.repeat(dates.strftime("%Y-%m-%d"),144),
               "j":np.tile(np.arange(1,145),365),"plan_slot0":np.tile(np.arange(144),365),
               "period_start":np.tile([clock(10*(j+1)) for j in range(144)],365),
               "period_end":np.tile([clock(10*(j+2)) for j in range(144)],365)}
    for pos in range(4):
        cp_rows[f"x{pos}_kwh"] = paths[:,pos,:].ravel()
    plan_table = pd.DataFrame(cp_rows)
    plan_table["execution_date"] = np.repeat(dates.strftime("%Y-%m-%d"),144)
    mask = plan_table.plan_slot0==143
    plan_table.loc[mask,"execution_date"] = (dates+pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    af,bf,cf = a[a.date>=FORMAL],b[b.date>=FORMAL],c[c.date>=FORMAL]
    for folder,aa,bb,cc in ((TABLES,af,bf,cf),(TABLES/"自然年365天",a,b,c)):
        save(aa,folder/f"{tag}_表A_逐10分钟.csv")
        save(bb,folder/f"{tag}_表B_逐日.csv")
        if problem in ("Q3","Q4-3"):
            save(cc,folder/f"{tag}_表C_承诺路径_自然日对齐.csv")
    save(plan_table[plan_table.date>=FORMAL],TABLES/f"{tag}_计划日144段.csv")
    save(af[af.date.isin(FOUR)],TABLES/f"{tag}_四个指定日期_调度.csv")
    endweek = (pd.Timestamp(week)+pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    save(af[(af.date>=week)&(af.date<=endweek)],TABLES/f"{tag}_典型周_调度.csv")
    # Four designated dates: table1 six periods, table2 six natural-day blocks, table3 events.
    t1,t2,t3 = [],[],[]
    for date in FOUR:
        da = a[a.date==date]
        db = b[b.date==date].iloc[0]
        row1 = {"date":date,"planned_purchase_kwh":db.planned_purchase_kwh,
                "final_commitment_kwh":db.final_commitment_kwh,"total_cost_yuan":db.total_cost_yuan}
        for slot in (60,72,84,96,108,120):
            row1[f"{clock(slot*10)}-{clock(slot*10+10)}_purchase_kwh"] = float(da.iloc[slot].x_plan_kwh)
        t1.append(row1)
        row2 = {"date":date,"soc_00_kwh":db.start_soc_kwh,"soc_24_kwh":db.end_soc_kwh}
        for begin in range(0,144,24):
            label = f"{clock(begin*10)}-{clock((begin+24)*10)}"
            row2[label+"_charge_kwh"] = float(da.iloc[begin:begin+24].u_charge_kwh.sum())
            row2[label+"_discharge_kwh"] = float(da.iloc[begin:begin+24].v_discharge_kwh.sum())
        t2.append(row2)
        events = da[da.e_emergency_kwh>1e-8]
        for _,r in events.iterrows():
            t3.append(dict(date=date,period=f"{r.period_start}-{r.period_end}",emergency_kwh=r.e_emergency_kwh))
    save(pd.DataFrame(t1),TABLES/f"{tag}_指定日期_表1.csv")
    save(pd.DataFrame(t2),TABLES/f"{tag}_指定日期_表2.csv")
    save(pd.DataFrame(t3,columns=["date","period","emergency_kwh"]),TABLES/f"{tag}_指定日期_表3.csv")
    monthly = bf.assign(month=bf.date.str[:7]).groupby("month").agg(
        total_cost_yuan=("total_cost_yuan","sum"), emergency_kwh=("emergency_kwh","sum"),
        path_up_kwh=("path_up_kwh","sum"),path_down_kwh=("path_down_kwh","sum"),
        adjusted_kwh=("adjusted_kwh","sum"),reversal_slots=("reversal_slots","sum"),
        down_slots=("down_slots","sum"),days=("date","size")).reset_index()
    monthly["reversal_ratio_all_slots"] = monthly.reversal_slots/(monthly.days*144)
    monthly["reversal_ratio_down_slots"] = np.where(monthly.down_slots>0,monthly.reversal_slots/monthly.down_slots,0.)
    save(monthly,TABLES/f"{tag}_月度统计.csv")
    soc_band = group.E_end_kwh.agg(["min","max"]).reset_index().rename(columns={"min":"soc_min_kwh","max":"soc_max_kwh"})
    soc_band = soc_band.merge(b[["date","start_soc_kwh","end_soc_kwh","reserve_kwh"]],on="date")
    soc_band["soc_min_kwh"] = np.minimum(soc_band.soc_min_kwh,soc_band.start_soc_kwh)
    soc_band["soc_max_kwh"] = np.maximum(soc_band.soc_max_kwh,soc_band.start_soc_kwh)
    soc_band["E_min_kwh"],soc_band["E_max_kwh"] = 1200.,10800.
    save(soc_band[soc_band.date>=FORMAL],TABLES/f"{tag}_SOC范围与储备.csv")
    # Forecast diagnostics: reconstructed actual execution forecast and independent naive baseline.
    baseline_l = np.vstack([arrays["initial_load"],arrays["load"][:-1]])
    baseline_p = np.vstack([arrays["initial_pv"],arrays["pv"][:-1]])
    errors = af[["date","t"]].copy()
    errors["load_error_kw"] = af.L_actual_kw.to_numpy()-af.L_fc_kw.to_numpy()
    errors["pv_error_kw"] = af.P_act_kw.to_numpy()-af.P_fc_kw.to_numpy()
    errors["net_error_kw"] = errors.load_error_kw-errors.pv_error_kw
    maskd = dates>=pd.Timestamp(FORMAL)
    errors["naive_load_error_kw"] = (arrays["load"]-baseline_l)[maskd].ravel()
    errors["naive_pv_error_kw"] = (arrays["pv"]-baseline_p)[maskd].ravel()
    save(errors,TABLES/f"{tag}_预测误差明细.csv")
    bands=[]
    for t,grp in errors.groupby("t"):
        for channel in ("load","pv","net"):
            value = grp[f"{channel}_error_kw"]
            bands.append(dict(t=t,channel=channel,mean_error_kw=value.mean(),q05_error_kw=value.quantile(.05),
                              q25_error_kw=value.quantile(.25),q50_error_kw=value.quantile(.50),
                              q75_error_kw=value.quantile(.75),q95_error_kw=value.quantile(.95),samples=len(value)))
    save(pd.DataFrame(bands),TABLES/f"{tag}_预测误差分位带.csv")
    forecast_metrics=[]
    for channel in ("load","pv"):
        actual_channel = af.L_actual_kw if channel=="load" else af.P_act_kw
        for name,col in (("model",f"{channel}_error_kw"),("previous_day_naive",f"naive_{channel}_error_kw")):
            err = errors[col]
            forecast_metrics.append(dict(channel=channel,predictor=name,mae_kw=np.abs(err).mean(),
                                         rmse_kw=np.sqrt(np.mean(err**2)),nmae=np.abs(err).mean()/actual_channel.mean()))
    save(pd.DataFrame(forecast_metrics),TABLES/f"{tag}_预测与朴素基准对比.csv")
    # Risk coverage requires scans to compare several q; only the actual final-q point is exported now.
    risk=[]
    for k,r in reserves.items():
        selected = maskd & (np.arange(365)>=20)
        risk.append(dict(problem=problem,issue_hour=k,q=.8,coverage=float(np.mean(residuals[k][selected]<=r[selected])),
                         reserve_mean_kwh=float(r[selected].mean()),samples=int(selected.sum())))
    save(pd.DataFrame(risk),TABLES/f"{tag}_最终分位风险覆盖率.csv")
    result = json.loads((source/f"{problem}_summary.json").read_text(encoding="utf-8"))
    assert abs(bf.total_cost_yuan.sum()-result["formal_period_cost_yuan"]) < 1e-4
    assert abs(b.total_cost_yuan.sum()-result["calendar_year_cost_yuan"]) < 1e-4
    return (metric_row(problem,bf,af,"formal_334_days"),metric_row(problem,b,a,"calendar_365_days"),
            physics_row(problem,af,float(bf.start_soc_kwh.iloc[0]),"formal_334_days"),
            physics_row(problem,a,6000.,"calendar_365_days"))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--week",default="2025-03-17")
    args=ap.parse_args()
    a=load_arrays()
    g,g365,v=[],[],[]
    for problem in TAGS:
        row,row365,formal,annual=export_problem(a,problem,args.week)
        g.append(row);g365.append(row365);v.extend([formal,annual])
        print(problem,"A/B/C and derived plot data exported.",flush=True)
    save(pd.DataFrame(g),TABLES/"表G_最终档位指标汇总_正式期334天.csv")
    save(pd.DataFrame(g365),TABLES/"自然年365天"/"表G_最终档位指标汇总_自然年365天.csv")
    save(pd.DataFrame(v),TABLES/"表V1_物理可行性核对.csv")
    dates=pd.DatetimeIndex(a["dates"]).strftime("%Y-%m-%d")
    price=pd.DataFrame({"date":np.repeat(dates,144),"t":np.tile(np.arange(1,145),365),
                        "time_end":np.tile([clock((t+1)*10) for t in range(144)],365),
                        "price_actual":a["price_actual"].ravel(),"price_fixed":np.tile(a["price_fixed"],365),
                        "price_fc":a["price_fc"].ravel()})
    save(price,TABLES/"Q4_电价热力图_365天_长表.csv")
    wide=pd.DataFrame(a["price_actual"],columns=[clock((t+1)*10) for t in range(144)])
    wide.insert(0,"date",dates)
    save(wide,TABLES/"Q4_电价热力图_365天_矩阵.csv")
    p1=pd.DataFrame(dict(t=np.arange(1,145),time_end=[clock((t+1)*10) for t in range(144)],price_yuan_per_kwh=a["price_fixed"]))
    save(p1,TABLES/"附件1_固定电价144段.csv")
    examples={}
    for problem in TAGS:
        df=pd.read_csv(TABLES/f"{problem}_{TAGS[problem]}_表B_逐日.csv")
        if problem in ("Q3","Q4-3"):
            examples[problem]=str(df.loc[df.adjusted_kwh.idxmax(),"date"])
    (HERE/"典型日期建议.json").write_text(json.dumps(dict(typical_week_start=args.week,four_dates=FOUR,largest_revision_dates=examples),indent=2,ensure_ascii=False),encoding="utf-8")
    causality = ROOT/"time_tests.json"
    if causality.exists():
        (HERE/"既有时间回归检验证据.json").write_text(causality.read_text(encoding="utf-8"),encoding="utf-8")
    checks=[]
    for path in sorted(TABLES.rglob("*.csv")):
        data=pd.read_csv(path)
        if len(data)>0:
            assert not data.isna().any().any(), str(path)
        assert path.read_bytes().startswith(b"\xef\xbb\xbf")
        checks.append(dict(file=str(path.relative_to(HERE)),rows=len(data),columns=len(data.columns),
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (HERE/"数据核验与文件清单.json").write_text(json.dumps(dict(source_results="234codex版/results",files=checks,
                  expensive_experiments_executed=False),indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"Export complete: {len(checks)} CSV tables. No annual rerun or sensitivity scan performed.")

if __name__=="__main__":
    main()
