"""按附件5 模板导出 result4-2.xlsx（波动电价下问题2的完整策略）。

数据来源全部为 ``results/`` 下已冻结的模型输出，本脚本不重算任何决策。
填表口径与问题3的 export_result3.py 保持一致：模板第 t 个时段列填第 t 个
十分钟时段（00:00–00:10 起）的计划购电量，忽略模板列标签的一格偏移。

- **计划购电量**：每格填 0:00 制定的计划购电 $x_{d,t}$（kWh）；全天购电量为全天
  合计，全天购电费为计划购电费用 $\\sum_t c_{d,t}x_{d,t}\\Delta t$（按当天实际电价）。
- **充放电量**：按模板 0:00-4:00 … 20:00-24:00 六段汇总，并填当日 0:00、24:00 储电量。
- **紧急购电量**：把逐十分钟紧急购电合并为连续区间，格式同题目表4 示例。
"""

from __future__ import annotations

import shutil

import numpy as np
import openpyxl
import pandas as pd

import deterministic_baseline as db

TEMPLATE_PATH = db.ATTACHMENT_DIR / "附件5" / "result4-2.xlsx"
OUTPUT_PATH = db.RESULT_DIR / "result4-2.xlsx"

BLOCKS = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]
N_DAYS = 334
N_SLOT = db.N_SLOT
SLOTS_PER_HOUR = 6
EMERGENCY_TOL = 1e-9


def _label(minutes: int) -> str:
    return f"{minutes // 60}:{minutes % 60:02d}"


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    interval = pd.read_csv(db.RESULT_DIR / "全年逐10分钟策略.csv", parse_dates=["date"])
    daily = pd.read_csv(db.RESULT_DIR / "全年逐日结果.csv", parse_dates=["date"])
    interval = interval.sort_values(["date", "t"]).reset_index(drop=True)
    daily = daily.sort_values("date").reset_index(drop=True)
    if len(daily) != N_DAYS or len(interval) != N_DAYS * N_SLOT:
        raise ValueError(f"明细行数异常：daily={len(daily)}, interval={len(interval)}")
    return interval, daily


def _daily_series(interval: pd.DataFrame, column: str) -> np.ndarray:
    """把逐时段列整形成 (334, 144)。"""
    return interval[column].to_numpy(float).reshape(N_DAYS, N_SLOT)


def write_result_workbook(interval: pd.DataFrame, daily: pd.DataFrame) -> None:
    x_plan = _daily_series(interval, "x_plan_kwh")
    charge = _daily_series(interval, "u_charge_kwh")
    discharge = _daily_series(interval, "v_discharge_kwh")
    emergency = _daily_series(interval, "e_emergency_kwh")
    planned_cost = daily["C_grid_yuan"].to_numpy(float)
    dates = daily["date"].to_numpy()

    shutil.copyfile(TEMPLATE_PATH, OUTPUT_PATH)
    wb = openpyxl.load_workbook(OUTPUT_PATH)

    ws = wb["计划购电量"]
    for i in range(N_DAYS):
        row = i + 2
        ws.cell(row, 1).value = pd.Timestamp(dates[i]).strftime("%Y-%m-%d")
        for t in range(N_SLOT):
            ws.cell(row, 2 + t).value = round(float(x_plan[i, t]), 4)
        ws.cell(row, 2 + N_SLOT).value = round(float(x_plan[i].sum()), 4)
        ws.cell(row, 3 + N_SLOT).value = round(float(planned_cost[i]), 2)

    ws = wb["充放电量"]
    for row in range(2, ws.max_row + 1):
        for col in range(1, 7):
            ws.cell(row, col).value = None
    row = 2
    for i in range(N_DAYS):
        day_start = float(daily["E_start_kwh"].iloc[i])
        day_end = float(daily["E_end_kwh"].iloc[i])
        for b, (h0, h1) in enumerate(BLOCKS):
            sl = slice(h0 * SLOTS_PER_HOUR, h1 * SLOTS_PER_HOUR)
            ws.cell(row, 1).value = pd.Timestamp(dates[i]).strftime("%Y-%m-%d") if b == 0 else None
            ws.cell(row, 2).value = f"{h0}:00-{h1}:00"
            ws.cell(row, 3).value = round(float(charge[i, sl].sum()), 4)
            ws.cell(row, 4).value = round(float(discharge[i, sl].sum()), 4)
            if b == 0:
                ws.cell(row, 5).value = "0:00"
                ws.cell(row, 6).value = round(day_start, 4)
            elif b == 1:
                ws.cell(row, 5).value = "24:00"
                ws.cell(row, 6).value = round(day_end, 4)
            row += 1

    ws = wb["紧急购电量"]
    for row in range(ws.max_row, 1, -1):
        for col in range(1, 4):
            ws.cell(row, col).value = None
    row = 2
    for i in range(N_DAYS):
        slots = np.flatnonzero(emergency[i] > EMERGENCY_TOL) + 1
        if slots.size == 0:
            continue
        breaks = np.flatnonzero(np.diff(slots) > 1)
        starts = np.concatenate(([slots[0]], slots[breaks + 1]))
        ends = np.concatenate(([slots[breaks], [slots[-1]]]))
        first = True
        for s, e in zip(starts, ends):
            ws.cell(row, 1).value = pd.Timestamp(dates[i]).strftime("%Y-%m-%d") if first else None
            ws.cell(row, 2).value = f"{_label((s - 1) * 10)}-{_label(e * 10)}"
            ws.cell(row, 3).value = round(float(emergency[i, s - 1 : e].sum()), 4)
            row += 1
            first = False

    wb.save(OUTPUT_PATH)


def main() -> None:
    interval, daily = load_tables()
    write_result_workbook(interval, daily)
    print(f"已写出 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
