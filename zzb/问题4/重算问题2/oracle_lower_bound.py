"""问题4：理论完美最低值（全知下界）与各口径的比较。

四个口径，储能物理约束、初始 SOC=6000 kWh、年末 SOC 回归 6000 kWh、
费用统计区间（2025-02-01 至 2025-12-31）完全一致：

  1. 可实现策略  ：负载/光伏/电价的因果预测 + q=0.80 安全余量（正式结果）
  2. 完美电价    ：负载/光伏仍因果预测，LP 改用当天实际电价
  3. 完美预测    ：负载/光伏/电价全部换成实际值、余量置 0，但保持逐日决策结构
  4. 理论完美最低值：全年统一优化，已知全部实际负载/光伏/电价，允许跨日套利

口径 3 与 4 的差别只来自"逐日决策 + 日末水值近似"造成的次优性；
口径 4 是全知条件下的最小可能费用，任何因果策略都不可能低于它。
全知条件下不会出现缺电，故 3、4 的紧急购电量恒为 0。

运行：python oracle_lower_bound.py
结果：results/理论完美下界对照.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

import deterministic_baseline as db


HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "results"
RESULT_DIR.mkdir(exist_ok=True)


def solve_full_year_oracle(data, params: db.StorageParameters) -> dict:
    """全年统一 LP：全知条件下的最小购电费用（跨日套利已计入）。"""
    n_days = len(data.dates)
    n = n_days * db.N_SLOT
    load = np.maximum(data.load_kw, 0.0).reshape(-1) * db.DT
    pv = np.maximum(data.pv_kw, 0.0).reshape(-1) * db.DT
    price = data.price.reshape(-1)
    step_max = params.p_max_kw * db.DT

    q0, c0, d0, w0, s0 = 0, n, 2 * n, 3 * n, 4 * n
    objective = np.zeros(5 * n)
    objective[q0 : q0 + n] = price
    objective[c0 : c0 + n] = 1e-7
    objective[d0 : d0 + n] = 1e-7
    objective[w0 : w0 + n] = 1e-8

    rows, cols, vals, rhs = [], [], [], []
    row_id = 0
    for t in range(n):
        for offset, coef in ((q0, 1.0), (c0, -1.0), (d0, 1.0), (w0, -1.0)):
            rows.append(row_id)
            cols.append(offset + t)
            vals.append(coef)
        rhs.append(load[t] - pv[t])
        row_id += 1

        rows.append(row_id)
        cols.append(s0 + t)
        vals.append(1.0)
        if t:
            rows.append(row_id)
            cols.append(s0 + t - 1)
            vals.append(-1.0)
        rows.append(row_id)
        cols.append(c0 + t)
        vals.append(-params.eta_c)
        rows.append(row_id)
        cols.append(d0 + t)
        vals.append(1.0 / params.eta_d)
        rhs.append(db.E_INITIAL if t == 0 else 0.0)
        row_id += 1

    a_eq = sparse.coo_matrix((vals, (rows, cols)), shape=(row_id, 5 * n)).tocsc()
    bounds = (
        [(0.0, None)] * n
        + [(0.0, step_max)] * n
        + [(0.0, step_max)] * n
        + [(0.0, None)] * n
        + [(params.e_min, params.e_max)] * (n - 1)
        + [(db.E_INITIAL, db.E_INITIAL)]
    )
    result = linprog(
        objective,
        A_eq=a_eq,
        b_eq=np.asarray(rhs),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"全知下界LP失败：{result.message}")
    x = result.x
    return {
        "purchase_kwh": x[q0 : q0 + n],
        "charge_kwh": x[c0 : c0 + n],
        "discharge_kwh": x[d0 : d0 + n],
        "curtailment_kwh": x[w0 : w0 + n],
    }


def annual_numbers(purchase, charge, discharge, curtailment, price, dates) -> dict:
    """只统计正式评价期（2025-02-01 起）的费用与电量。"""
    formal = (np.asarray(dates) >= db.FORMAL_START).repeat(db.N_SLOT)
    cost = float(np.asarray(price).reshape(-1)[formal] @ purchase[formal])
    return {
        "total_cost_yuan": float(cost),
        "purchase_kwh": float(purchase[formal].sum()),
        "charge_kwh": float(charge[formal].sum()),
        "discharge_kwh": float(discharge[formal].sum()),
        "curtailment_kwh": float(curtailment[formal].sum()),
        "emergency_kwh": 0.0,
    }


def main() -> None:
    data = db.load_inputs()
    params = db.StorageParameters()
    n_days = len(data.dates)
    price_flat = data.price.reshape(-1)

    frozen = json.loads(
        (RESULT_DIR / "最终冻结结果.json").read_text(encoding="utf-8")
    )
    realized = {
        "total_cost_yuan": frozen["dispatch"]["total_cost_yuan"],
        "purchase_kwh": frozen["dispatch"]["planned_purchase_kwh"],
        "charge_kwh": float("nan"),
        "discharge_kwh": float("nan"),
        "curtailment_kwh": frozen["dispatch"]["curtailed_pv_kwh"],
        "emergency_kwh": frozen["dispatch"]["emergency_kwh"],
    }
    perfect_price = {
        "total_cost_yuan": frozen["perfect_price_information"]["total_cost_yuan"],
        "purchase_kwh": float("nan"),
        "charge_kwh": float("nan"),
        "discharge_kwh": float("nan"),
        "curtailment_kwh": float("nan"),
        "emergency_kwh": frozen["perfect_price_information"]["emergency_kwh"],
    }

    # 口径 3：预测完美但保持逐日决策结构，安全余量为 0。
    perfect_forecast_daily, _ = db.run_dispatch(
        data,
        data.load_kw,
        data.pv_kw,
        np.zeros((n_days, db.N_SLOT)),
        price_forecast=data.price,
    )
    perfect_forecast = {
        "total_cost_yuan": float(perfect_forecast_daily.total_cost_yuan.sum()),
        "purchase_kwh": float(perfect_forecast_daily.planned_purchase_kwh.sum()),
        "charge_kwh": float(perfect_forecast_daily.charge_kwh.sum()),
        "discharge_kwh": float(perfect_forecast_daily.discharge_kwh.sum()),
        "curtailment_kwh": float(perfect_forecast_daily.curtailed_pv_kwh.sum()),
        "emergency_kwh": float(perfect_forecast_daily.emergency_kwh.sum()),
    }

    # 口径 4：全年统一优化的全知下界。
    oracle = solve_full_year_oracle(data, params)
    oracle_numbers = annual_numbers(
        oracle["purchase_kwh"],
        oracle["charge_kwh"],
        oracle["discharge_kwh"],
        oracle["curtailment_kwh"],
        price_flat,
        data.dates,
    )

    rows = [
        ("可实现策略", realized),
        ("完美电价（仍是逐日决策）", perfect_price),
        ("完美预测（逐日决策，余量0）", perfect_forecast),
        ("理论完美最低值（全年统一优化）", oracle_numbers),
    ]
    table = pd.DataFrame(
        [
            {
                "口径": name,
                "总费用/元": numbers["total_cost_yuan"],
                "计划购电量/kWh": numbers["purchase_kwh"],
                "充电量/kWh": numbers["charge_kwh"],
                "放电量/kWh": numbers["discharge_kwh"],
                "弃光量/kWh": numbers["curtailment_kwh"],
                "紧急购电量/kWh": numbers["emergency_kwh"],
                "相对理论下界/元": numbers["total_cost_yuan"]
                - oracle_numbers["total_cost_yuan"],
                "相对理论下界/%": 100.0
                * (numbers["total_cost_yuan"] - oracle_numbers["total_cost_yuan"])
                / oracle_numbers["total_cost_yuan"],
            }
            for name, numbers in rows
        ]
    )
    table.to_csv(RESULT_DIR / "理论完美下界对照.csv", index=False, encoding="utf-8-sig")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    with (RESULT_DIR / "理论完美下界.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "note": "储能约束、初始SOC=6000、年末SOC=6000、统计区间(2025-02-01起)四口径完全一致",
                "realized": realized,
                "perfect_price": perfect_price,
                "perfect_forecast_daily_decision": perfect_forecast,
                "oracle_full_year_joint": oracle_numbers,
                "realized_minus_oracle_yuan": realized["total_cost_yuan"]
                - oracle_numbers["total_cost_yuan"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
