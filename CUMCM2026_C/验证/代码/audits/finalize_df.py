"""Verify completed isolated D/F artifacts and refresh their inventory only."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
PRJ=ROOT
HERE=PRJ/'代码'/'共享'/'出图数据';TABLES=PRJ/'图表'/'数据表'
RERUN=PRJ/'图表'/'综合'/'补跑输出'
RES=PRJ/'结果'
OUTDIR=Path(__file__).resolve().parent
PROBLEMS=('Q2','Q3','Q4-2','Q4-3')
def _res_dir(res,p):
    f=p.replace('Q','问题')
    return res/'问题4'/f if p.startswith('Q4') else res/f
TAGS={'Q2':'reserve-mpc','Q3':'exact-greedy','Q4-2':'reserve-mpc','Q4-3':'exact-greedy'}
BASELINE_FORMAL=(13864639.364845123,13439044.36781069,14595194.941129487,14158125.73330921)
BASELINE_ANNUAL=(15948651.29096032,15251117.16585094,16975525.14556457,16218310.911334753)

def main():
    evidence={'D':[],'F':[],'experiment_audits':[],'baseline_unchanged':True}
    for formal,label,expected in ((True,'正式期334天',BASELINE_FORMAL),(False,'自然年365天',BASELINE_ANNUAL)):
        folder=TABLES if formal else TABLES/'自然年365天'
        for p,baseline in zip(PROBLEMS,expected):
            path=folder/f'{p}_{TAGS[p]}_表D_分位数扫描_{label}.csv'
            d=pd.read_csv(path)
            assert len(d)==9 and np.allclose(sorted(d.q),np.arange(.50,.91,.05))
            assert np.isfinite(d.select_dtypes('number')).all().all()
            assert (abs(d.final_soc_kwh-6000)<1e-6).all()
            q8=d.loc[np.isclose(d.q,.8)].iloc[0]
            assert abs(q8.total_cost_yuan-baseline)<.02
            best=d.loc[d.total_cost_yuan.idxmin()]
            evidence['D'].append(dict(problem=p,period=label,rows=len(d),best_q=float(best.q),
                best_cost_yuan=float(best.total_cost_yuan),q08_cost_yuan=float(q8.total_cost_yuan),
                best_vs_q08_difference_yuan=float(q8.total_cost_yuan-best.total_cost_yuan),
                q08_coverage=float(q8.coverage),q08_emergency_days=int(q8.days_with_emergency)))
        f=pd.read_csv(folder/f'表F_全知下界阶梯_{label}.csv')
        assert len(f)==4 and set(f['问题'])==set(PROBLEMS)
        assert np.isfinite(f[['C0','C1','C2','C3']]).all().all()
        for p,baseline in zip(PROBLEMS,expected):
            row=f.loc[f['问题']==p].iloc[0]
            assert abs(row.C3*10000-baseline)<.02 and row.C0<=row.C3+1e-7
            evidence['F'].append(dict(problem=p,period=label,C0_wanyuan=float(row.C0),
                C1_wanyuan=float(row.C1),C2_wanyuan=float(row.C2),C3_wanyuan=float(row.C3),
                monotone=bool(row.monotone_ladder)))
    for path in sorted(RERUN.rglob('*_summary.json')):
        summary=json.loads(path.read_text(encoding='utf-8'))
        audit=summary['audit']
        assert summary['simulation_days']==365 and audit['slots']==52560
        assert abs(summary['final_soc_kwh']-6000)<1e-6
        for key in ('soc_violations','power_violations','simultaneous_charge_discharge','negative_flow'):
            assert audit[key]==0,(str(path),key,audit[key])
        for key in ('max_balance_err','max_purchase_identity_err','max_pv_identity_err','max_soc_recursion_err'):
            assert audit[key]<1e-5,(str(path),key,audit[key])
        evidence['experiment_audits'].append(str(path.relative_to(ROOT)))
    assert len(evidence['experiment_audits'])==38, len(evidence['experiment_audits'])
    lp_evidence=[]
    for path in sorted((RERUN/'C0').glob('*.json')):
        row=json.loads(path.read_text(encoding='utf-8'))
        assert row['max_balance_residual']<1e-5
        assert row['simultaneous_charge_discharge_slots']==0
        assert abs(row['terminal_soc_kwh']-6000)<1e-6
        lp_evidence.append(str(path.relative_to(ROOT)))
    assert len(lp_evidence)==4
    evidence['C0_LP_audits']=lp_evidence
    evidence['numerical_adapters']=[dict(file=str(path.relative_to(ROOT)),
        details=json.loads(path.read_text(encoding='utf-8')))
        for path in sorted(RERUN.rglob('numerical_adapter.json'))]
    for p,formal,annual in zip(PROBLEMS,BASELINE_FORMAL,BASELINE_ANNUAL):
        daily=pd.read_csv(_res_dir(RES,p)/f'{p}_daily.csv')
        assert abs(daily.total_cost_yuan.sum()-annual)<1e-5
        assert abs(daily.loc[daily.date>='2025-02-01','total_cost_yuan'].sum()-formal)<1e-5
    manifest=json.loads((HERE/'数据核验与文件清单.json').read_text(encoding='utf-8'))
    files=[]
    for path in sorted(TABLES.rglob('*.csv')):
        df=pd.read_csv(path)
        files.append(dict(file=str(path.relative_to(ROOT)),rows=len(df),columns=len(df.columns),
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest.update(files=files,expensive_experiments_executed=True,
        expensive_experiments_status='D: four problems x nine q; F: four problems; both 334/365-day tables complete',
        experiment_completion_evidence='../audits/DF完成核验.json')
    (HERE/'数据核验与文件清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUTDIR/'DF完成核验.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# D、F补算汇总','', '基线未改动；D最优扫描点仅为2025年同样本事后比较，不自动选作新模型参数。','',
        '|模型|口径|扫描最低费用q|最低费用/万元|q=0.80费用/万元|差额/万元|',
        '|---|---|---:|---:|---:|---:|']
    for r in evidence['D']:
        lines.append(f"|{r['problem']}|{r['period']}|{r['best_q']:.2f}|{r['best_cost_yuan']/10000:.4f}|{r['q08_cost_yuan']/10000:.4f}|{r['best_vs_q08_difference_yuan']/10000:.4f}|")
    lines.extend(['','## 正式期334天全知对照（万元）','','|模型|C0|C1|C2|C3|','|---|---:|---:|---:|---:|'])
    for r in evidence['F'][:4]:
        lines.append(f"|{r['problem']}|{r['C0_wanyuan']:.4f}|{r['C1_wanyuan']:.4f}|{r['C2_wanyuan']:.4f}|{r['C3_wanyuan']:.4f}|")
    lines.extend(['','C0为全知物理LP松弛下界，C1/C2是保留启发式框架的全知对照，不是独立纯因果贡献。',
        '38次新365天仿真物理审计通过；q=0.80复用基线。详细逻辑风险见《逻辑检查报告.md》。',''])
    (OUTDIR/'DF补算汇总.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(evidence['D'],ensure_ascii=False,indent=2))
    print(f'Complete: {len(files)} table CSVs, 38 new annual simulations; original baseline verified unchanged.')

if __name__=='__main__':main()
