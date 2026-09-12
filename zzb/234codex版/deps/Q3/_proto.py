"""原型：用真实历史残差日作情景（保留时间相关性），对比逐时段分位叠加。"""
import numpy as np, pandas as pd
from common import load_inputs, DT
from forecasts import forecast_bundle
from dispatch_core import simulate_year, summarize
import models as M

class HistScen(M.DecisionModel):
    S = 24
    def __init__(self):
        self.code = "H"; self.name = "历史情景 SP"; self.paradigm = "随机优化"
        self.risk = f"最近 {self.S} 个历史残差日作情景，紧急购电为追索"
    def plan(self, ctx):
        h = ctx.residual_hist
        if len(h) < self.S:
            sc, w = np.zeros((1, ctx.n)), np.ones(1)
        else:
            idx = np.linspace(0, len(h) - 1, self.S).astype(int)
            sc, w = h[idx], np.full(self.S, 1.0 / self.S)
        return M.solve_scenario_lp(ctx.load_fc, ctx.pv_fc, ctx.price, ctx.start_energy,
            ctx.ref, sc, w, emergency=True, force_end_soc=ctx.force_end_soc)

class QuantScaled(M.QuantileLP):
    Q = 0.5
    def __init__(self):
        pass
    def plan(self, ctx):
        S = self.S
        return M.solve_scenario_lp(ctx.load_fc, ctx.pv_fc, ctx.price, ctx.start_energy,
            ctx.ref, (ctx.margin_ref * S)[None, :], np.ones(1), emergency=False,
            force_end_soc=ctx.force_end_soc)

data = load_inputs(); bundle = forecast_bundle(data, verbose=False)
rows = []
def run(m, tag, stages):
    d = simulate_year(data, bundle, m, revision_hours=stages)
    s = summarize(d, tag)
    rows.append({"方案": tag, "总费用": s["total_cost_yuan"], "紧急kWh": s["emergency_kwh"],
                 "未用kWh": s["unused_plan_kwh"], "电网结算": s["grid_cost_yuan"],
                 "计划购电kWh": s["planned_purchase_kwh"]})
    return s

ALL = (6, 12, 18)
run(HistScen(), "H·全时点", ALL)
run(HistScen(), "H·仅0点", ())
q5 = QuantScaled(); q5.S = 0.5
run(q5, "q=0.40·全时点", ALL)   # margin_ref 是 q=0.80 的，×0.5 近似 q≈0.4 口径
t = pd.DataFrame(rows)
t["相对M2_%"] = 100 * (t.总费用 - 13748164.499748) / 13748164.499748
pd.set_option("display.width", 220); pd.set_option("display.unicode.east_asian_width", True)
print(t.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
