"""Explicit isolated runtime adapter for solver roundoff; baseline source unchanged."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'出图数据'))
import run_experiments as r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('problem');ap.add_argument('q',type=float)
    args=ap.parse_args()
    original=r.simulate.__globals__['oc'].stage_lp
    cleaned=[]
    def stage_lp(*a,**kw):
        x=original(*a,**kw)
        if x.min() < -1e-6:raise RuntimeError(f'Material negative purchase: {x.min()}')
        neg=x[x<0]
        if len(neg):cleaned.extend(float(v) for v in neg)
        return np.maximum(x,0.)
    r.simulate.__globals__['oc'].stage_lp=stage_lp
    r.run_cached(r.load_arrays(),args.problem,f'D_q{args.q:.2f}',q=args.q)
    folder=r.HERE/'补跑输出'/f'D_q{args.q:.2f}'/args.problem
    note=dict(adapter='audits/run_d_numeric.py',adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tolerance_kwh=1e-6,negative_purchase_values_clipped=cleaned,
        rationale='Normalize only solver lower-bound roundoff before physical execution; no baseline/model source edit.')
    (folder/'numerical_adapter.json').write_text(json.dumps(note,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(note,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
