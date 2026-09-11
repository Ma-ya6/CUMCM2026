from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


@dataclass(frozen=True)
class StochasticPlan:
    purchase_kwh: np.ndarray
    objective_yuan: float
    expected_emergency_cost_yuan: float
    emergency_cvar_yuan: float
    solver_iterations: int


def solve_two_stage_saa(
    *,
    load_scenarios_kw: np.ndarray,
    pv_scenarios_kw: np.ndarray,
    probabilities: np.ndarray,
    price: np.ndarray,
    start_energy_kwh: float,
    dt_hours: float,
    eta_c: float,
    eta_d: float,
    e_min_kwh: float,
    e_max_kwh: float,
    p_max_kw: float,
    risk_weight: float = 0.0,
    cvar_alpha: float = 0.95,
    terminal_value_yuan_per_kwh: float | None = None,
    force_terminal_soc_kwh: float | None = None,
) -> StochasticPlan:
    """求解标准期望-CVaR凸组合的两阶段线性规划。

    目标中的紧急购电风险为
    ``(1-lambda) * E[C_em] + lambda * CVaR_alpha(C_em)``，
    避免旧版在完整期望成本上再次叠加CVaR而产生重复风险惩罚。
    """
    load_scenarios_kw = np.asarray(load_scenarios_kw, float)
    pv_scenarios_kw = np.asarray(pv_scenarios_kw, float)
    probabilities = np.asarray(probabilities, float)
    price = np.asarray(price, float)
    if load_scenarios_kw.shape != pv_scenarios_kw.shape:
        raise ValueError("负载和光伏场景维度不一致")
    n_scenarios, n_slot = load_scenarios_kw.shape
    if price.shape != (n_slot,) or probabilities.shape != (n_scenarios,):
        raise ValueError("价格或场景概率维度异常")
    if not np.isclose(probabilities.sum(), 1.0) or (probabilities < 0).any():
        raise ValueError("场景概率必须非负且总和为1")
    if not 0.0 <= risk_weight <= 1.0:
        raise ValueError("risk_weight 必须位于[0,1]")
    if not 0.0 < cvar_alpha < 1.0:
        raise ValueError("cvar_alpha 必须位于(0,1)")

    # 变量：[x] + 每场景[c,d,e,w,r,E] + [VaR] + [z_s]，流量单位均为 kWh/时段。
    n_flow = 6 * n_slot
    scenario0 = n_slot
    eta_idx = scenario0 + n_scenarios * n_flow
    z0 = eta_idx + 1
    n_var = z0 + n_scenarios
    objective = np.zeros(n_var)
    objective[:n_slot] = price
    water_value = float(eta_d * np.mean(price) if terminal_value_yuan_per_kwh is None else terminal_value_yuan_per_kwh)

    def offsets(s: int) -> tuple[int, int, int, int, int, int]:
        base = scenario0 + s * n_flow
        return tuple(base + k * n_slot for k in range(6))  # type: ignore[return-value]

    for s, prob in enumerate(probabilities):
        c0, d0, e0, w0, r0, soc0 = offsets(s)
        objective[e0 : e0 + n_slot] = (1.0 - risk_weight) * prob * 5.0 * price
        objective[c0 : c0 + n_slot] = prob * 1e-7
        objective[d0 : d0 + n_slot] = prob * 1e-7
        objective[w0 : w0 + n_slot] = prob * 1e-8
        if force_terminal_soc_kwh is None:
            objective[soc0 + n_slot - 1] = -prob * water_value
    if risk_weight > 0.0:
        objective[eta_idx] = risk_weight
        objective[z0:] = risk_weight * probabilities / (1.0 - cvar_alpha)

    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_vals: list[float] = []
    eq_rhs = np.empty(2 * n_scenarios * n_slot)
    row = 0
    load_kwh = load_scenarios_kw * dt_hours
    pv_kwh = pv_scenarios_kw * dt_hours
    for s in range(n_scenarios):
        c0, d0, e0, w0, r0, soc0 = offsets(s)
        for t in range(n_slot):
            # x-c+d+e-w-r = L-P
            eq_rows.extend([row] * 6)
            eq_cols.extend([t, c0 + t, d0 + t, e0 + t, w0 + t, r0 + t])
            eq_vals.extend([1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
            eq_rhs[row] = load_kwh[s, t] - pv_kwh[s, t]
            row += 1
            # -eta_c*c+d/eta_d+E_t-E_{t-1}=0；首时段右端为初始电量。
            eq_rows.extend([row] * (4 if t else 3))
            eq_cols.extend([c0 + t, d0 + t, soc0 + t])
            eq_vals.extend([-eta_c, 1.0 / eta_d, 1.0])
            if t:
                eq_cols.append(soc0 + t - 1)
                eq_vals.append(-1.0)
                eq_rhs[row] = 0.0
            else:
                eq_rhs[row] = start_energy_kwh
            row += 1
    a_eq = sparse.coo_matrix((eq_vals, (eq_rows, eq_cols)), shape=(row, n_var)).tocsr()

    # z_s >= emergency_cost_s - VaR。
    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_vals: list[float] = []
    for s in range(n_scenarios):
        _, _, e0, _, _, _ = offsets(s)
        ub_rows.extend([s] * (n_slot + 2))
        ub_cols.extend(list(range(e0, e0 + n_slot)) + [eta_idx, z0 + s])
        ub_vals.extend(list(5.0 * price) + [-1.0, -1.0])
    a_ub = sparse.coo_matrix(
        (ub_vals, (ub_rows, ub_cols)), shape=(n_scenarios, n_var)
    ).tocsr()

    step_max = p_max_kw * dt_hours
    bounds: list[tuple[float | None, float | None]] = [(0.0, None)] * n_var
    for s in range(n_scenarios):
        c0, d0, _, _, _, soc0 = offsets(s)
        for t in range(n_slot):
            bounds[c0 + t] = (0.0, step_max)
            bounds[d0 + t] = (0.0, step_max)
            bounds[soc0 + t] = (e_min_kwh, e_max_kwh)
        if force_terminal_soc_kwh is not None:
            terminal = float(np.clip(force_terminal_soc_kwh, e_min_kwh, e_max_kwh))
            bounds[soc0 + n_slot - 1] = (terminal, terminal)
    bounds[eta_idx] = (0.0, None)

    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=np.zeros(n_scenarios),
        A_eq=a_eq,
        b_eq=eq_rhs,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if not result.success:
        raise RuntimeError(f"SAA-CVaR线性规划失败：{result.message}")

    scenario_emergency_cost = np.empty(n_scenarios)
    for s in range(n_scenarios):
        _, _, e0, _, _, _ = offsets(s)
        scenario_emergency_cost[s] = float(5.0 * price @ result.x[e0 : e0 + n_slot])
    var = float(result.x[eta_idx]) if risk_weight > 0 else float(np.quantile(scenario_emergency_cost, cvar_alpha))
    cvar_mask = scenario_emergency_cost >= var - 1e-8
    cvar = float(scenario_emergency_cost[cvar_mask].mean()) if cvar_mask.any() else var
    return StochasticPlan(
        purchase_kwh=result.x[:n_slot].copy(),
        objective_yuan=float(result.fun),
        expected_emergency_cost_yuan=float(probabilities @ scenario_emergency_cost),
        emergency_cvar_yuan=cvar,
        solver_iterations=int(result.nit),
    )
