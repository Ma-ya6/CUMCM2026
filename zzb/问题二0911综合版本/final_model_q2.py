"""C题问题2最终模型：季节水平预测 + 因果分位数安全余量 + 确定性线性规划。

运行后生成全年逐日/逐时段结果、四个指定日期表格、冻结数字与论文图。
所有第 d 天的预测和安全余量只使用 j<d 的历史数据。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import deterministic_baseline as db
from seasonal import CorrectionConfig, ForecastConfig, generate_causal_forecasts


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "results"
FIGURE_DIR = HERE / "figures"
RESULT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

Q = 0.75
SPECIFIED_DATES = pd.to_datetime(
    ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
)
REPORT_SLOTS = {
    "10:00-10:10": 61,
    "12:00-12:10": 73,
    "14:00-14:10": 85,
    "16:00-16:10": 97,
    "18:00-18:10": 109,
    "20:00-20:10": 121,
}

# 与 C 题统一符号规定对应的结果字段：x/y/r/g/w/u/v/e/E/c/L/P/m。
DAILY_EXPORT_NAMES = {
    "planned_purchase_kwh": "x_plan_kwh",
    "planned_cost_yuan": "C_grid_yuan",
    "emergency_kwh": "e_emergency_kwh",
    "emergency_cost_yuan": "C_emer_yuan",
    "total_cost_yuan": "C_total_yuan",
    "unused_plan_kwh": "r_purchase_waste_kwh",
    "curtailed_pv_kwh": "w_pv_waste_kwh",
    "charge_kwh": "u_charge_kwh",
    "discharge_kwh": "v_discharge_kwh",
    "start_soc_kwh": "E_start_kwh",
    "end_soc_kwh": "E_end_kwh",
    "load_kwh": "L_energy_kwh",
    "pv_kwh": "P_act_energy_kwh",
}
INTERVAL_EXPORT_NAMES = {
    "slot": "t",
    "load_actual_kw": "L_actual_kw",
    "pv_actual_kw": "P_act_kw",
    "load_forecast_kw": "L_fc_kw",
    "pv_forecast_kw": "P_fc_kw",
    "net_safety_margin_kw": "m_safety_kw",
    "planned_purchase_kwh": "x_plan_kwh",
    "plan_used_kwh": "y_used_kwh",
    "unused_plan_kwh": "r_purchase_waste_kwh",
    "charge_kwh": "u_charge_kwh",
    "discharge_kwh": "v_discharge_kwh",
    "emergency_kwh": "e_emergency_kwh",
    "curtailed_pv_kwh": "w_pv_waste_kwh",
    "pv_used_kwh": "g_pv_used_kwh",
    "soc_kwh": "E_end_kwh",
}


def summary(daily: pd.DataFrame) -> dict:
    costs = daily["total_cost_yuan"].to_numpy(float)
    cutoff = np.quantile(costs, 0.95)
    return {
        "evaluation_start": str(daily.date.min().date()),
        "evaluation_end": str(daily.date.max().date()),
        "n_days": int(len(daily)),
        "planned_purchase_kwh": float(daily.planned_purchase_kwh.sum()),
        "planned_cost_yuan": float(daily.planned_cost_yuan.sum()),
        "emergency_kwh": float(daily.emergency_kwh.sum()),
        "emergency_cost_yuan": float(daily.emergency_cost_yuan.sum()),
        "total_cost_yuan": float(daily.total_cost_yuan.sum()),
        "unused_plan_kwh": float(daily.unused_plan_kwh.sum()),
        "curtailed_pv_kwh": float(daily.curtailed_pv_kwh.sum()),
        "days_with_emergency": int((daily.emergency_kwh > 1e-8).sum()),
        "daily_cost_p95_yuan": float(cutoff),
        "daily_cost_cvar95_yuan": float(costs[costs >= cutoff].mean()),
        "final_soc_kwh": float(daily.end_soc_kwh.iloc[-1]),
    }


def forecast_summary(data, forecasts: dict) -> dict:
    mask = np.asarray(data.dates >= db.FORMAL_START)
    actual_load = data.load_kw[mask]
    actual_pv = data.pv_kw[mask]
    pred_load = forecasts["load"][mask]
    pred_pv = forecasts["pv"][mask]
    actual_net = actual_load - actual_pv
    pred_net = pred_load - pred_pv

    def metrics(actual, pred):
        err = pred - actual
        return {
            "mae_kw": float(np.abs(err).mean()),
            "rmse_kw": float(np.sqrt(np.mean(err**2))),
            "nmae": float(np.abs(err).mean() / np.mean(np.abs(actual))),
            "bias_kw": float(err.mean()),
        }

    return {
        "load": metrics(actual_load, pred_load),
        "pv": metrics(actual_pv, pred_pv),
        "net_load": metrics(actual_net, pred_net),
    }


def make_specified_tables(daily: pd.DataFrame, interval: pd.DataFrame):
    selected_daily = daily[daily.date.isin(SPECIFIED_DATES)].copy()
    selected_interval = interval[interval.date.isin(SPECIFIED_DATES)].copy()

    table1 = []
    table2 = []
    for date in SPECIFIED_DATES:
        day = selected_interval[selected_interval.date == date]
        drow = selected_daily[selected_daily.date == date].iloc[0]
        row1 = {"date": date}
        for label, slot in REPORT_SLOTS.items():
            row1[label] = float(day.loc[day.slot == slot, "planned_purchase_kwh"].iloc[0])
        row1["daily_purchase_kwh"] = float(drow.planned_purchase_kwh)
        row1["daily_planned_cost_yuan"] = float(drow.planned_cost_yuan)
        table1.append(row1)

        row2 = {"date": date}
        for start in range(0, 24, 4):
            lo, hi = start * 6 + 1, (start + 4) * 6
            block = day[(day.slot >= lo) & (day.slot <= hi)]
            label = f"{start:02d}:00-{start + 4:02d}:00"
            row2[f"charge_{label}_kwh"] = float(block.charge_kwh.sum())
            row2[f"discharge_{label}_kwh"] = float(block.discharge_kwh.sum())
        row2["soc_00_kwh"] = float(drow.start_soc_kwh)
        row2["soc_24_kwh"] = float(drow.end_soc_kwh)
        table2.append(row2)

    emergency = selected_daily[
        ["date", "emergency_kwh", "emergency_cost_yuan", "total_cost_yuan"]
    ].copy()
    return pd.DataFrame(table1), pd.DataFrame(table2), emergency, selected_interval


def plot_specified(interval: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, date in zip(axes.flat, SPECIFIED_DATES):
        day = interval[interval.date == date]
        hour = (day.slot.to_numpy() - 0.5) / 6
        ax.plot(hour, day.load_actual_kw * db.DT, label="负载", lw=1.4)
        ax.plot(hour, day.pv_actual_kw * db.DT, label="光伏", lw=1.4)
        ax.step(hour, day.planned_purchase_kwh, where="mid", label="计划购电", lw=1.2)
        ax.fill_between(hour, 0, day.emergency_kwh, step="mid", alpha=0.3, label="紧急购电")
        ax.set_title(str(date.date()))
        ax.set_ylabel("每10分钟电量 / kWh")
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("时刻 / h")
    axes[-1, 1].set_xlabel("时刻 / h")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.985))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURE_DIR / "指定日期调度.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def risk_sensitivity(data, forecasts: dict) -> pd.DataFrame:
    """固定预测，只改变风险分位数；开发期用于选参，留出期仅用于评价。"""
    actual_net = data.load_kw - data.pv_kw
    predicted_net = forecasts["load"] - forecasts["pv"]
    rows = []
    for q in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        margins = db.causal_residual_quantiles(actual_net - predicted_net, q)
        daily, _ = db.run_dispatch(data, forecasts["load"], forecasts["pv"], margins)
        for split, chosen in (
            ("全年正式期", daily),
            ("开发期", daily[daily.date <= pd.Timestamp("2025-08-31")]),
            ("留出期", daily[daily.date >= pd.Timestamp("2025-09-01")]),
        ):
            row = summary(chosen)
            rows.append({"quantile": q, "split": split, **row})
    return pd.DataFrame(rows)


def main() -> None:
    data = db.load_inputs()
    correction = CorrectionConfig(
        use_pv_window=True,
        use_level=True,
        use_weekly_effect=False,
    )
    config = ForecastConfig(correction=correction)
    forecasts = generate_causal_forecasts(data, config, verbose=True)
    actual_net = data.load_kw - data.pv_kw
    predicted_net = forecasts["load"] - forecasts["pv"]
    margins = db.causal_residual_quantiles(actual_net - predicted_net, Q)
    daily, interval = db.run_dispatch(
        data, forecasts["load"], forecasts["pv"], margins, keep_intervals=True
    )

    table1, table2, emergency, specified_interval = make_specified_tables(daily, interval)
    sensitivity = risk_sensitivity(data, forecasts)
    dispatch_numbers = summary(daily)
    forecast_numbers = forecast_summary(data, forecasts)
    validation = db.validate_physical_results(interval, db.StorageParameters())
    frozen = {
        "model": {
            "name": "季节水平预测+因果分位数安全余量+确定性LP",
            "risk_quantile": Q,
            "weekly_effect_enabled": False,
            "causal_information_set": "第d天0:00仅使用j<d历史实际数据",
            "initial_state": "E_{1,0}=E_0=6000 kWh；随后逐日跨日连续递推",
            "symbol_convention": [
                {"symbol": "c", "meaning": "外网电价"},
                {"symbol": "L", "meaning": "小区负载功率"},
                {"symbol": "P", "meaning": "光伏功率"},
                {"symbol": "x/y/r", "meaning": "计划购电/实际使用计划购电/购电浪费功率"},
                {"symbol": "g/w", "meaning": "光伏实际使用/光伏浪费功率"},
                {"symbol": "u/v", "meaning": "储能充电/放电功率"},
                {"symbol": "e", "meaning": "紧急购电功率"},
                {"symbol": "E", "meaning": "储电量"},
                {"symbol": "m", "meaning": "净负荷安全余量"},
            ],
            "storage": asdict(db.StorageParameters()),
        },
        "forecast": forecast_numbers,
        "dispatch": dispatch_numbers,
        "specified_dates": emergency.assign(date=emergency.date.astype(str)).to_dict("records"),
        "physical_validation": validation,
        "development_selected_quantile": float(
            sensitivity[sensitivity.split == "开发期"]
            .sort_values("total_cost_yuan")
            .iloc[0]["quantile"]
        ),
    }

    daily.rename(columns=DAILY_EXPORT_NAMES).to_csv(
        RESULT_DIR / "全年逐日结果.csv", index=False, encoding="utf-8-sig"
    )
    interval.rename(columns=INTERVAL_EXPORT_NAMES).to_csv(
        RESULT_DIR / "全年逐10分钟策略.csv", index=False, encoding="utf-8-sig"
    )
    table1.to_csv(RESULT_DIR / "四天表1购电结果.csv", index=False, encoding="utf-8-sig")
    table2.to_csv(RESULT_DIR / "四天表2储能结果.csv", index=False, encoding="utf-8-sig")
    emergency.to_csv(RESULT_DIR / "四天紧急购电结果.csv", index=False, encoding="utf-8-sig")
    specified_interval.rename(columns=INTERVAL_EXPORT_NAMES).to_csv(
        RESULT_DIR / "四天逐10分钟明细.csv", index=False, encoding="utf-8-sig"
    )
    sensitivity.to_csv(RESULT_DIR / "风险参数敏感性.csv", index=False, encoding="utf-8-sig")
    forecasts["diagnostics"].to_csv(
        RESULT_DIR / "预测诊断.csv", index=False, encoding="utf-8-sig"
    )
    with (RESULT_DIR / "最终冻结结果.json").open("w", encoding="utf-8") as f:
        json.dump(frozen, f, ensure_ascii=False, indent=2)
    plot_specified(interval)

    print(json.dumps(frozen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
