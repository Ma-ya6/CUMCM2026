"""决策模型对照的公共底座：物理执行、经济结算、全年滚动仿真循环。

本模块只含**所有决策模型共用**的部分，不含任何决策逻辑：

- ``execute_slots``：给定承诺功率与实际负荷/光伏，逐时段物理执行（充放电、
  紧急购电、弃光、计划电量作废）；
- ``settle_period``：按题目 #15 的链式规则逐时段精确结算；
- ``simulate_year``：逐日、逐决策时刻滚动，把每个阶段的决策**委托给传入的
  决策模型**（``models.DecisionModel`` 的 ``plan``）。

因此"换模型"= 换 ``plan`` 的实现，物理与经济规则一字不动，各模型的费用可直接
比较。决策时刻只能调整尚未执行的时段（#17），储能跨日连续、从 1 月 1 日的
$E_0=6000$ kWh 起算，费用自 2 月 1 日起统计。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common import (
    BREACH_RATE,
    DT,
    EMERGENCY_MULTIPLIER,
    E_INITIAL,
    E_MAX,
    E_MIN,
    FORMAL_START,
    ISSUE_HOURS,
    N_SLOT,
    OVERBUY_RATE,
    P_MAX_KW,
    Q2_MODEL_DIR,
    SLOTS_PER_HOUR,
    DataBundle3,
)

if str(Q2_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(Q2_MODEL_DIR))

import deterministic_baseline as base  # noqa: E402

REVISION_HOURS = (6, 12, 18)


# --------------------------------------------------------------------------
# 物理执行（与 问题2/模型2再优化 的 execute_actual 逐位一致）
# --------------------------------------------------------------------------
def execute_slots(
    commitment: np.ndarray,
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    start_energy: float,
    *,
    p_max_kw: float = P_MAX_KW,
    eta_c: float = 0.90,
    eta_d: float = 0.90,
) -> dict:
    """按承诺功率与实际负载/光伏逐时段执行，返回实际充放电与紧急购电。"""
    load = load_kw * DT
    pv = pv_kw * DT
    step_max = p_max_kw * DT
    energy = float(np.clip(start_energy, E_MIN, E_MAX))
    n = len(commitment)
    out = {
        k: np.zeros(n)
        for k in ("charge", "discharge", "emergency", "unused_plan",
                  "curtailment", "pv_used", "plan_used", "soc")
    }

    for t in range(n):
        available = commitment[t] + pv[t]
        if available >= load[t]:
            surplus = available - load[t]
            charge = min(surplus, step_max, max(0.0, (E_MAX - energy) / eta_c))
            energy += eta_c * charge
            consumption = load[t] + charge
            pv_used = min(pv[t], consumption)
            plan_used = min(commitment[t], max(0.0, consumption - pv_used))
            out["charge"][t] = charge
            out["pv_used"][t] = pv_used
            out["plan_used"][t] = plan_used
            out["curtailment"][t] = max(0.0, pv[t] - pv_used)
            out["unused_plan"][t] = max(0.0, commitment[t] - plan_used)
        else:
            deficit = load[t] - available
            discharge = min(deficit, step_max, max(0.0, (energy - E_MIN) * eta_d))
            energy -= discharge / eta_d
            out["discharge"][t] = discharge
            out["emergency"][t] = max(0.0, deficit - discharge)
            out["pv_used"][t] = pv[t]
            out["plan_used"][t] = commitment[t]
        out["soc"][t] = energy

    out["end_energy"] = float(energy)
    return out


def verify_against_baseline(tol: float = 1e-9, seed: int = 20260910) -> dict[str, float]:
    """机械验证 ``execute_slots`` 与问题2 基准执行器等价。"""
    rng = np.random.default_rng(seed)
    n = N_SLOT
    load = rng.uniform(3000, 8000, n)
    pv = np.clip(rng.uniform(-200, 9000, n), 0.0, None)
    commitment = rng.uniform(0, 6000, n)
    mine = execute_slots(commitment, load, pv, E_INITIAL)
    theirs = base.execute_actual(commitment, load, pv, E_INITIAL, base.StorageParameters())
    keys = ("charge", "discharge", "emergency", "unused_plan",
            "curtailment", "pv_used", "plan_used", "soc")
    dev = {k: float(np.max(np.abs(mine[k] - theirs[k]))) for k in keys}
    dev["end_energy"] = abs(mine["end_energy"] - theirs["end_energy"])
    worst = max(dev.values())
    if worst > tol:
        raise AssertionError(f"execute_slots 与 execute_actual 不一致，最大偏差 {worst:g}")
    return dev


# --------------------------------------------------------------------------
# 链式结算（题目 #15）
# --------------------------------------------------------------------------
def settle_slot(commitments: list[float], price_t: float, settlement: str) -> dict:
    """对单个时段按承诺序列 $x^{(0)},x^{(6)},x^{(12)},x^{(18)}$ 结算。"""
    x0, xf = commitments[0], commitments[-1]
    if settlement == "two_way":
        base_qty = min(x0, xf)
        down = max(x0 - xf, 0.0)
        up = max(xf - x0, 0.0)
    elif settlement == "chain":
        base_qty = min(commitments)
        down = up = 0.0
        prev = commitments[0]
        for cur in commitments[1:]:
            down += max(prev - cur, 0.0)
            up += max(cur - prev, 0.0)
            prev = cur
    else:
        raise ValueError(f"settlement 只能是 'chain' 或 'two_way'，收到 {settlement!r}")

    base_cost = price_t * base_qty
    return {
        "base_yuan": base_cost,
        "breach_yuan": BREACH_RATE * price_t * down,
        "overbuy_yuan": OVERBUY_RATE * price_t * up,
        "total_yuan": base_cost + BREACH_RATE * price_t * down + OVERBUY_RATE * price_t * up,
        "adjusted_kwh": down + up,
    }


def settle_period(commitment_path: np.ndarray, price: np.ndarray, settlement: str) -> dict:
    """对一段时域逐时段结算。``commitment_path`` 形状 ``(4, n_slot)``。"""
    rows = [
        settle_slot([commitment_path[j, t] for j in range(4)], price[t], settlement)
        for t in range(commitment_path.shape[1])
    ]
    return {
        key: float(sum(r[key] for r in rows))
        for key in ("base_yuan", "breach_yuan", "overbuy_yuan", "total_yuan",
                    "adjusted_kwh")
    }


# --------------------------------------------------------------------------
# 喂给决策模型的阶段信息
# --------------------------------------------------------------------------
@dataclass
class StageContext:
    """一个决策时刻的全部已知信息。决策模型只能看这里面的东西。

    ``residual_hist`` 的行是**该日之前**的净负荷残差（$j<d$），故任何从它导出
    的统计量都是因果的，不含当日真值。
    """

    issue_hour: int
    abs_start: int
    load_fc: np.ndarray          # (n,) 负载预报 kW
    pv_fc: np.ndarray            # (n,) 光伏预报 kW
    price: np.ndarray            # (n,) 电价 元/kWh
    start_energy: float
    ref: np.ndarray | None       # (n,) 上一版承诺（每时段电量 kWh）
    residual_hist: np.ndarray    # (m, n) 因果残差样本，可能为空
    margin_ref: np.ndarray       # (n,) 分位安全余量（模型3 口径，q 由 simulate_year 给定）
    force_end_soc: float | None
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.price)

    @property
    def net_fc(self) -> np.ndarray:
        return self.load_fc - self.pv_fc


# --------------------------------------------------------------------------
# 全年滚动仿真
# --------------------------------------------------------------------------
def _error_matrix(
    data: DataBundle3,
    bundle: dict,
    k: int,
    load_source: str,
    pv_source: str,
) -> np.ndarray:
    """发布时刻 ``k`` 覆盖窗口上的净负荷残差矩阵 ``(n_days, n_covered)``。

    第 ``d`` 行只用到第 ``d`` 日的真值，故取前 ``d`` 行即为因果历史。
    """
    start = k * SLOTS_PER_HOUR
    actual_net = data.load_kw - data.pv_kw
    if load_source == "perfect":
        load_ref = data.load_kw
    else:
        load_ref = bundle["load"]
    if pv_source == "fused":
        pv = bundle["pv_fused"][ISSUE_HOURS.index(k)]
    elif pv_source == "own":
        pv = bundle["pv_own"]
    elif pv_source == "perfect":
        return np.zeros((len(data.dates), N_SLOT - start))
    else:
        raise ValueError(f"未知 pv_source：{pv_source}")
    return actual_net[:, start:] - (load_ref[:, start:] - pv[:, start:])


def _pv_at_stage(data, bundle, day, k, pv_source) -> np.ndarray:
    """第 ``day`` 天 ``k`` 时刻持有的光伏预报（覆盖区外为 NaN）。"""
    if pv_source == "fused":
        return bundle["pv_fused"][ISSUE_HOURS.index(k)][day]
    if pv_source == "own":
        return bundle["pv_own"][day]
    if pv_source == "perfect":
        return data.pv_kw[day]
    raise ValueError(f"未知 pv_source：{pv_source}")


def _load_at_stage(data, bundle, day, k, load_source) -> np.ndarray:
    """第 ``day`` 天 ``k`` 时刻持有的负载预报（全长 144）。

    $k>0$ 时用已执行时段做日内修正——这是预测层的一部分，所有决策模型共用，
    换模型不改它。
    """
    if load_source == "perfect":
        return data.load_kw[day].copy()
    daily = bundle["load"][day]
    if k == 0:
        return daily.copy()
    from forecasts import intraday_load_update

    return intraday_load_update(daily, data.load_kw[day], k)


def simulate_year(
    data: DataBundle3,
    bundle: dict,
    model,
    *,
    load_source: str = "own",
    pv_source: str = "fused",
    revision_hours: tuple[int, ...] = REVISION_HOURS,
    quantile: float = 0.80,
    days: int | None = None,
    verbose: bool = False,
    slot_sink=None,
) -> pd.DataFrame:
    """逐日滚动：每个决策时刻把 ``StageContext`` 交给 ``model.plan``。

    ``quantile`` 只用于预计算 ``margin_ref``（模型3 口径的分位余量，供需要它的
    决策模型取用）。``causal_residual_quantiles`` 本身是因果递归，第 $d$ 行只
    依赖 $j<d$ 的样本，故直接取第 $d$ 行不构成信息泄露。

    ``slot_sink`` 为可选的逐时段导出回调（默认 ``None``，不改变任何数值）：
    每天结算后以关键字参数 ``date/commitment/executed/start_energy/end_energy``
    调用一次，用于导出 附件5 ``result3.xlsx`` 所需的逐十分钟明细。
    """
    n_days = len(data.dates) if days is None else min(len(data.dates), days)
    price = data.price
    errors = {
        k: _error_matrix(data, bundle, k, load_source, pv_source) for k in ISSUE_HOURS
    }
    margins = {
        k: base.causal_residual_quantiles(errors[k], quantile) for k in ISSUE_HOURS
    }
    daily_rows: list[dict] = []
    energy = E_INITIAL
    stages = [0] + [k for k in REVISION_HOURS if k in revision_hours]

    for day in range(n_days):
        date = data.dates[day]
        start_energy = energy
        commitment_path = np.full((4, N_SLOT), np.nan)
        prev_commit: np.ndarray | None = None
        executed = {
            key: np.zeros(N_SLOT)
            for key in ("charge", "discharge", "emergency", "unused_plan",
                        "curtailment", "pv_used", "plan_used", "soc")
        }
        model.begin_day(day, start_energy)

        for si, k in enumerate(stages):
            start = k * SLOTS_PER_HOUR
            is_last = si == len(stages) - 1
            load_full = _load_at_stage(data, bundle, day, k, load_source)
            pv_full = _pv_at_stage(data, bundle, day, k, pv_source)

            force_end = (
                E_INITIAL if (is_last and days is None and day == n_days - 1) else None
            )
            ctx = StageContext(
                issue_hour=k,
                abs_start=start,
                load_fc=load_full[start:].copy(),
                pv_fc=np.nan_to_num(pv_full[start:], nan=0.0),
                price=price[start:],
                start_energy=energy,
                ref=None if prev_commit is None else prev_commit[start:].copy(),
                residual_hist=errors[k][:day],
                margin_ref=margins[k][day].copy(),
                force_end_soc=force_end,
            )
            purchase = np.asarray(model.plan(ctx), dtype=float)
            if purchase.shape != (ctx.n,):
                raise ValueError(
                    f"{model.code} 返回的购电向量形状 {purchase.shape}，应为 {(ctx.n,)}"
                )
            if not np.isfinite(purchase).all():
                raise ValueError(f"{model.code} 返回了非有限值")
            if (purchase < -1e-7).any():
                raise ValueError(f"{model.code} 返回了负购电量")

            new_commit = (
                np.zeros(N_SLOT) if prev_commit is None else prev_commit.copy()
            )
            new_commit[start:] = purchase
            commitment_path[ISSUE_HOURS.index(k)] = new_commit

            exec_end = N_SLOT if is_last else stages[si + 1] * SLOTS_PER_HOUR
            if exec_end > start:
                actual = execute_slots(
                    new_commit[start:exec_end],
                    data.load_kw[day, start:exec_end],
                    data.pv_kw[day, start:exec_end],
                    energy,
                )
                energy = actual["end_energy"]
                for key in executed:
                    executed[key][start:exec_end] = actual[key]
            prev_commit = new_commit

        for j in range(1, 4):
            if np.isnan(commitment_path[j]).all():
                commitment_path[j] = commitment_path[j - 1]

        chain = settle_period(commitment_path, price, "chain")
        two_way = settle_period(commitment_path, price, "two_way")
        emergency_cost = float(EMERGENCY_MULTIPLIER * (price @ executed["emergency"]))

        if date >= FORMAL_START:
            daily_rows.append({
                "date": date,
                "model": model.code,
                "base_cost_yuan": chain["base_yuan"],
                "breach_cost_yuan": chain["breach_yuan"],
                "overbuy_cost_yuan": chain["overbuy_yuan"],
                "grid_cost_yuan": chain["total_yuan"],
                "two_way_grid_cost_yuan": two_way["total_yuan"],
                "reversal_gap_yuan": chain["total_yuan"] - two_way["total_yuan"],
                "adjusted_kwh": chain["adjusted_kwh"],
                "emergency_kwh": float(executed["emergency"].sum()),
                "emergency_cost_yuan": emergency_cost,
                "total_cost_yuan": chain["total_yuan"] + emergency_cost,
                "planned_purchase_kwh": float(commitment_path[0].sum()),
                "final_commitment_kwh": float(commitment_path[3].sum()),
                "unused_plan_kwh": float(executed["unused_plan"].sum()),
                "curtailed_pv_kwh": float(executed["curtailment"].sum()),
                "charge_kwh": float(executed["charge"].sum()),
                "discharge_kwh": float(executed["discharge"].sum()),
                "start_soc_kwh": float(start_energy),
                "end_soc_kwh": energy,
                "emergency_slots": int((executed["emergency"] > 1e-9).sum()),
            })

        if slot_sink is not None:
            slot_sink(
                date=date,
                commitment=commitment_path.copy(),
                executed={key: value.copy() for key, value in executed.items()},
                start_energy=start_energy,
                end_energy=energy,
            )

        if verbose and (day + 1) % 60 == 0:
            print(f"    {model.code}: {day + 1}/{n_days} 天  储电 {energy:8.0f}",
                  flush=True)

    return pd.DataFrame(daily_rows)


def summarize(daily: pd.DataFrame, label: str | None = None) -> dict:
    """把一个模型的逐日结果汇总为一行（正式期）。"""
    name = label or (daily.model.iloc[0] if len(daily) else "?")
    total = float(daily.total_cost_yuan.sum())
    return {
        "model": name,
        "days": int(len(daily)),
        "grid_cost_yuan": float(daily.grid_cost_yuan.sum()),
        "breach_cost_yuan": float(daily.breach_cost_yuan.sum()),
        "overbuy_cost_yuan": float(daily.overbuy_cost_yuan.sum()),
        "emergency_cost_yuan": float(daily.emergency_cost_yuan.sum()),
        "total_cost_yuan": total,
        "emergency_kwh": float(daily.emergency_kwh.sum()),
        "emergency_slots": int(daily.emergency_slots.sum()),
        "unused_plan_kwh": float(daily.unused_plan_kwh.sum()),
        "adjusted_kwh": float(daily.adjusted_kwh.sum()),
        "reversal_gap_yuan": float(daily.reversal_gap_yuan.sum()),
        "planned_purchase_kwh": float(daily.planned_purchase_kwh.sum()),
        "final_commitment_kwh": float(daily.final_commitment_kwh.sum()),
        "yuan_per_day": total / max(len(daily), 1),
    }
