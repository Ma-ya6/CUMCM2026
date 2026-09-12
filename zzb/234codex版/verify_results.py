"""Read-only annual result verification, with a reproducibility manifest."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_year import ROOT, OUT, oc

def main():
    report = {}
    for problem in ("Q2", "Q3", "Q4-2", "Q4-3"):
        daily = pd.read_csv(OUT / f"{problem}_daily.csv")
        slots = pd.read_csv(OUT / f"{problem}_physical_10min.csv")
        with np.load(OUT / f"{problem}_plans.npz") as z:
            paths = z["paths"]
        assert len(daily) == 365 and len(slots) == 52560
        assert (daily.date >= "2025-02-01").sum() == 334
        actual = slots.commitment_kwh.to_numpy().reshape(365,144)
        expected = np.column_stack([np.r_[0., paths[:-1,-1,-1]], paths[:,-1,:143]])
        mapping = float(np.abs(actual-expected).max())
        assert mapping < 1e-5
        for pos, begin in ((1,35),(2,71),(3,107)):
            np.testing.assert_allclose(paths[:,pos,:begin], paths[:,pos-1,:begin], atol=1e-7, rtol=0)
        price = slots.price_actual.to_numpy().reshape(365,144)
        previous = np.concatenate([np.zeros((1,4)), paths[:-1,:,-1]], axis=0)
        physical = np.concatenate([previous[:,:,None], paths[:,:,:143]], axis=2)
        delta = np.diff(physical, axis=1)
        independent = (physical.min(axis=1)*price + .5*price*np.maximum(-delta,0).sum(axis=1)
                       + 1.5*price*np.maximum(delta,0).sum(axis=1)
                       + 5*price*slots.emergency.to_numpy().reshape(365,144))
        fee_error = float(np.abs(independent-slots.total_cost_yuan.to_numpy().reshape(365,144)).max())
        assert fee_error < 1e-5
        assert np.max(np.abs(independent.sum(axis=1)-daily.total_cost_yuan)) < 1e-5
        state = slots.soc.to_numpy().reshape(365,144)
        previous_state = np.r_[6000., state.ravel()[:-1]]
        recursion = previous_state + .9*slots.charge.to_numpy()-slots.discharge.to_numpy()/.9
        assert np.max(np.abs(recursion-state.ravel())) < 1e-5
        assert np.max(np.abs(daily.start_soc_kwh.to_numpy()[1:]-daily.end_soc_kwh.to_numpy()[:-1])) < 1e-5
        assert abs(state[-1,-1]-6000.) < 1e-5
        assert not np.any((slots.charge > 1e-8) & (slots.discharge > 1e-8))
        assert np.max(slots.charge) <= oc.STEP_MAX+1e-5
        assert np.max(slots.discharge) <= oc.STEP_MAX+1e-5
        balance = slots.plan_used + slots.emergency + slots.discharge + slots.pv_used - (slots.load_kw/6+slots.charge)
        assert np.max(np.abs(balance)) < 1e-5
        formal = daily[daily.date >= "2025-02-01"]
        summary = json.loads((OUT / f"{problem}_summary.json").read_text(encoding="utf-8"))
        assert abs(summary["formal_period_cost_yuan"]-formal.total_cost_yuan.sum()) < 1e-5
        report[problem] = dict(slots=52560, days=365, formal_days=334,
                               max_mapping_error_kwh=mapping, max_fee_error_yuan=fee_error,
                               final_soc_kwh=float(state[-1,-1]), passed=True)
    manifest = {}
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and (p.suffix in (".py", ".csv", ".npz", ".json")) and "__pycache__" not in p.parts and p.name != "annual_audit.json":
            manifest[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (ROOT / "annual_audit.json").write_text(json.dumps(dict(checks=report, sha256=manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
