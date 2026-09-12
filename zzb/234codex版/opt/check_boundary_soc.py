# -*- coding: utf-8 -*-
"""Extract SOC at a calendar-day boundary under Attachment 5 slot labels."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

OPT = Path(__file__).resolve().parent
sys.path.insert(0, str(OPT))

import opt_core as oc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=("Q2", "Q3", "Q4-2", "Q4-3"), required=True)
    args = ap.parse_args()
    cap: list[dict] = []
    original = oc.run_day_single

    def hooked(commitment, load_kw, pv_kw, *a, **kw):
        result = original(commitment, load_kw, pv_kw, *a, **kw)
        cap.append(result)
        return result

    oc.run_day_single = hooked
    try:
        if args.problem in ("Q2", "Q4-2"):
            import run_single as rs
            db, corr, fc_cfg, gen = rs.load_stack(args.problem)
            data = db.load_inputs()
            forecasts = rs.build_forecasts(args.problem, db, corr, fc_cfg, gen)
            rs.simulate(db, data, forecasts, mode="reserve-mpc", days=31)
            jan31 = cap[30]
            soc = np.asarray(jan31["soc"], float)
        else:
            import run_stage as rt
            common, dispatch_core, forecasts = rt.load_stack(args.problem)
            data = common.load_inputs()
            bundle = forecasts.forecast_bundle(data, use_cache=False, verbose=False)
            price_s, price_d = rt.price_channel(args.problem, common, data, 31)
            errs = rt.errors_for(data, bundle, dispatch_core)
            reserves = rt.reserves_for(data, bundle, dispatch_core, 31)
            rt.simulate(
                data, bundle, dispatch_core, settle="correct", mpc=False,
                reserves=reserves, price_s=price_s, price_d=price_d, days=31,
                exact=True, errs=errs,
            )
            # Four calls per day: concatenate the four executed stage segments.
            soc = np.concatenate([np.asarray(x["soc"], float) for x in cap[-4:]])
        if soc.size != 144:
            raise RuntimeError(f"Jan 31 SOC path length is {soc.size}, expected 144")
        print(f"{args.problem},2025-02-01 00:00,{soc[142]:.6f}")
        print(f"{args.problem},2025-02-01 00:10,{soc[143]:.6f}")
    finally:
        oc.run_day_single = original


if __name__ == "__main__":
    main()
