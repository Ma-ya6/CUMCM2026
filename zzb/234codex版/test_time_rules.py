"""Small independent regression tests; never replace annual result outputs."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import run_year as ry

def main():
    report = {}
    for initial in (1200., 6000., 10800.):
        for load in (500., 2000., 5000.):
            commitment = np.zeros(144)
            kw = np.full(144, load)
            pv = np.zeros(144)
            act = ry.execute(commitment, kw, pv, np.ones(144), initial, kw, pv,
                             mpc=True, last_day=True, physical_start=0)
            audit = ry.oc.audit_execution(act, commitment, kw, pv, initial)
            assert abs(act["end_energy"]-6000.) < 1e-6
            assert audit["max_balance_err"] < 1e-6
            assert audit["power_violations"] == 0
            assert audit["simultaneous_charge_discharge"] == 0
    report["year_end_reachable_cases"] = 9
    dates = pd.date_range("2025-01-01", periods=3)
    load = np.full((3, 144), 3000.)
    pv = np.zeros((3, 144))
    price = np.tile(np.r_[np.full(60, .4), np.full(60, .9), np.full(24, .5)], (3, 1))
    a = dict(load=load, pv=pv, load_fc=load*.97, pv_fc=pv,
             load_stage_fc=load*.96, pv_stage_fc=pv,
             price_fixed=price[0], price_actual=price, price_fc=price*.95,
             dates=dates.to_numpy(), initial_pv=np.zeros(144),
             pv_fused=np.zeros((4, 3, 144)))
    original_out = ry.OUT
    with TemporaryDirectory(prefix="codex_time_tests_") as tmp:
        ry.OUT = Path(tmp)
        for problem in ("Q2", "Q3", "Q4-2", "Q4-3"):
            ry.simulate(a, problem, days=3)
            slots = pd.read_csv(ry.OUT / f"{problem}_physical_10min.csv")
            daily = pd.read_csv(ry.OUT / f"{problem}_daily.csv")
            with np.load(ry.OUT / f"{problem}_plans.npz") as z:
                paths = z["paths"]
            assert slots.iloc[0].commitment_kwh == 0.
            for d in (1, 2):
                assert abs(slots.iloc[d*144].commitment_kwh-paths[d-1, -1, -1]) < 1e-6
                assert abs(daily.iloc[d].start_soc_kwh-daily.iloc[d-1].end_soc_kwh) < 1e-6
            for d in range(3):
                for t in range(1, 144):
                    assert abs(slots.iloc[d*144+t].commitment_kwh-paths[d, -1, t-1]) < 1e-6
            if problem in ("Q3", "Q4-3"):
                for pos, start in ((1,35), (2,71), (3,107)):
                    np.testing.assert_array_equal(paths[:,pos,:start], paths[:,pos-1,:start])
            assert abs(slots.total_cost_yuan.sum()-daily.total_cost_yuan.sum()) < 1e-5
            report[problem] = "carry, SOC continuity, plan-to-physical mapping, immutable revision prefixes, fee identity passed"
    ry.OUT = original_out
    # At 06:00 the 06:00 observation is available, the 06:10 observation is not.
    before = ry.forecast_at(a, 1, 6, True)
    changed = {k: v.copy() for k,v in a.items()}
    changed["load"][1,36:] *= 100
    changed["pv"][1,36:] += 1000
    after = ry.forecast_at(changed, 1, 6, True)
    for left, right in zip(before, after):
        np.testing.assert_array_equal(left, right)
    report["intraday_future_actual_invariance"] = "passed"
    (ry.ROOT / "time_tests.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("All time regression tests passed.")

if __name__ == "__main__":
    main()
