import pandas as pd, numpy as np
from common import load_inputs
from forecasts import forecast_bundle
from dispatch_core import simulate_year, summarize
import models as M
data = load_inputs(); bundle = forecast_bundle(data, verbose=False)
class M10_0(M.StorageReserveLP):
    QUANTILE = 0.70; code = "M10·仅0点"; name = "M10 仅0点"
for stages in ((6,12,18), ()):
    d = simulate_year(data, bundle, M10_0(), revision_hours=stages)
    s = summarize(d, M10_0.code + ("·全" if stages else ""))
    print(f"{s['model']:16s} {s['total_cost_yuan']:>16,.1f}  紧急 {s['emergency_kwh']:>10,.1f}  未用 {s['unused_plan_kwh']:>12,.1f}")
