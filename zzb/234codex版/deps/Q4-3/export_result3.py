"""按 附件5 模板导出问题4 重算问题3 的最终提交结果，全部基于模型 M9（q=0.80）。

相对 ``问题3/模型3决策优化版`` 的唯一差别是**电价通道**：由附件1 的单条固定日内
曲线换成附件4 的逐日逐时段波动电价，且电价与负载、光伏同为不可预知量。
电价预测沿用 ``问题4/重算问题2`` 的实现（见 ``gen_price_channel.py`` 的缓存），
正文不重写预测器；本脚本只负责把它接进四时点滚动决策。

与 ``run_models.py`` 的区别：本脚本只跑**一个**模型（$M_9$ 储能追索型 LP），
但通过 ``simulate_year`` 的 ``slot_sink`` 回调把逐十分钟明细留下来，从而能填
附件5 的四张工作表、生成题目表1/表2/表3 格式的四个指定日期结果、以及四个指定
日期的调度图。预测层、执行器、结算规则与 ``run_models.py`` 完全共用。

产出：
    results/result4-3.xlsx            附件5 模板（计划购电/调整购电/充放电/紧急购电）
    results/全年逐10分钟策略.csv        全年 48,096 个时段的完整策略（符号命名）
    results/全年逐日结果.csv           全年逐日汇总（正式期 334 天）
    results/四天表1购电结果.csv        题目表1 格式
    results/四天表2储能结果.csv        题目表2 格式
    results/四天表3紧急购电结果.csv     题目表3 格式
    results/四天逐10分钟明细.csv        四个指定日期的逐时段明细
    results/预测诊断.csv               逐日储备与两种分位口径的对照
    results/风险参数敏感性.csv          q 网格 × 开发期/留出期/全年
    results/电价预测权重.csv            正式期逐日的三模型集成权重（沿用重算问题2）
    results/最终冻结结果.json           论文全部数字的唯一冻结入口
    results/最终结果汇总.md            一页纸汇总
    figures/指定日期调度.png           四个指定日期调度图
    figures/电价预测.png              波动电价实际值与因果预测值
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import openpyxl
import pandas as pd

from common import (ATTACHMENT_DIR, CACHE_DIR, DT, FORMAL_START, N_SLOT, RESULT_DIR,
                    SLOTS_PER_HOUR, load_inputs, load_price_channel,
                    load_price_metrics)
from dispatch_core import _error_matrix, simulate_year, summarize
from forecasts import forecast_bundle
from models import StorageReserveLP

TEMPLATE_PATH = ATTACHMENT_DIR / "附件5" / "result4-3.xlsx"
OUTPUT_XLSX = RESULT_DIR / "result4-3.xlsx"
FIGURE_PATH = RESULT_DIR.parent / "figures" / "指定日期调度.png"
PRICE_FIGURE_PATH = RESULT_DIR.parent / "figures" / "电价预测.png"

SPECIFIED_DATES = pd.to_datetime(["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"])
BLOCKS = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]
TABLE1_MINUTES = [600, 720, 840, 960, 1080, 1200]   # 10:00 12:00 14:00 16:00 18:00 20:00
Q_GRID = [0.60, 0.70, 0.80, 0.90]
EMERGENCY_TOL = 1e-9
N_FORMAL = 334
DEV_END = pd.Timestamp("2025-08-31")
MARGIN_Q = 0.80          # 模型3 口径的逐时段分位余量，仅用于预测诊断对照


def _label(minutes: int) -> str:
    return f"{minutes // 60}:{minutes % 60:02d}"


def _slot_label(slot: int) -> str:
    """时段编号（1-based）→ "6:00-6:10" 形式的标签。"""
    return f"{_label((slot - 1) * 10)}-{_label(slot * 10)}"


def _merge_runs(slots: np.ndarray) -> list[tuple[int, int]]:
    """把不连续的时段编号合并为连续区间 ``[(start, end), ...]``。"""
    if slots.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(slots) > 1)
    starts = np.concatenate(([slots[0]], slots[breaks + 1]))
    ends = np.concatenate(([slots[breaks], [slots[-1]]]))
    return list(zip(starts.tolist(), ends.tolist()))


def _runs_label(runs) -> str:
    return "；".join(f"{_label((s - 1) * 10)}-{_label(e * 10)}" for s, e in runs)


# --------------------------------------------------------------------------
# 跑模型并把逐时段明细留下
# --------------------------------------------------------------------------
def run_capture(data, bundle, quantile: float, price_day=None, price_fc=None):
    """跑一年 $M_9$，返回 (逐日结果 DataFrame, {日期: 逐时段明细})。

    ``price_day``/``price_fc`` 为波动电价通道；为 ``None`` 时即问题3 的固定电价口径。
    """
    model = StorageReserveLP()
    model.QUANTILE = quantile
    store: dict[pd.Timestamp, dict] = {}

    def sink(**kw):
        store[pd.Timestamp(kw["date"])] = kw

    return simulate_year(data, bundle, model, slot_sink=sink,
                         price_day=price_day, price_fc=price_fc), store


class Slots:
    """把 sink 里存下的逐日明细整形成 ``(n_days, ...)`` 的数组容器。"""

    KEYS = ("charge", "discharge", "emergency", "unused_plan", "curtailment",
            "pv_used", "plan_used", "soc")

    def __init__(self, store: dict, dates: pd.DatetimeIndex) -> None:
        self.commitment = np.stack([np.asarray(store[d]["commitment"], float)
                                    for d in dates])              # (n, 4, 144)
        self.executed = {key: np.stack([np.asarray(store[d]["executed"][key], float)
                                        for d in dates]) for key in self.KEYS}
        self.soc_start = np.array([float(store[d]["start_energy"]) for d in dates])
        self.soc_end = np.array([float(store[d]["end_energy"]) for d in dates])

    @property
    def x_plan(self) -> np.ndarray:
        return self.commitment[:, 0, :]

    @property
    def x_final(self) -> np.ndarray:
        return self.commitment[:, 3, :]

    @property
    def x_move(self) -> np.ndarray:
        """三次调整的变动绝对值之和（不区分上调下调）。"""
        c = self.commitment
        return np.abs(c[:, 1] - c[:, 0]) + np.abs(c[:, 2] - c[:, 1]) + np.abs(c[:, 3] - c[:, 2])


# --------------------------------------------------------------------------
# 附件5 模板
# --------------------------------------------------------------------------
def write_result_workbook(dates, slots: Slots, price, planned_cost, adjust_cost) -> None:
    shutil.copyfile(TEMPLATE_PATH, OUTPUT_XLSX)
    wb = openpyxl.load_workbook(OUTPUT_XLSX)
    # 模板第 2..145 列共 144 个时段列（表头文字按区间右端点命名，与时段编号一一
    # 对应），第 146/147 列分别为全天购电量与全天购电费。
    for sheet, series, cost in (("计划购电量", slots.x_plan, planned_cost),
                                ("调整购电量", slots.x_move, adjust_cost)):
        ws = wb[sheet]
        for i in range(len(dates)):
            row = i + 2
            ws.cell(row, 1).value = pd.Timestamp(dates[i]).strftime("%Y-%m-%d")
            for t in range(N_SLOT):
                ws.cell(row, 2 + t).value = round(float(series[i, t]), 4)
            ws.cell(row, 2 + N_SLOT).value = round(float(series[i].sum()), 4)
            ws.cell(row, 3 + N_SLOT).value = round(float(cost[i]), 2)

    ws = wb["充放电量"]
    for row in range(2, ws.max_row + 1):
        for col in range(1, 7):
            ws.cell(row, col).value = None
    row = 2
    for i in range(len(dates)):
        for b, (h0, h1) in enumerate(BLOCKS):
            sl = slice(h0 * SLOTS_PER_HOUR, h1 * SLOTS_PER_HOUR)
            ws.cell(row, 1).value = (pd.Timestamp(dates[i]).strftime("%Y-%m-%d")
                                     if b == 0 else None)
            ws.cell(row, 2).value = f"{h0}:00-{h1}:00"
            ws.cell(row, 3).value = round(float(slots.executed["charge"][i, sl].sum()), 4)
            ws.cell(row, 4).value = round(float(slots.executed["discharge"][i, sl].sum()), 4)
            if b == 0:
                ws.cell(row, 5).value = "0:00"
                ws.cell(row, 6).value = round(float(slots.soc_start[i]), 4)
            elif b == 1:
                ws.cell(row, 5).value = "24:00"
                ws.cell(row, 6).value = round(float(slots.soc_end[i]), 4)
            row += 1

    ws = wb["紧急购电量"]
    for row in range(ws.max_row, 1, -1):
        for col in range(1, 4):
            ws.cell(row, col).value = None
    row = 2
    for i in range(len(dates)):
        runs = _merge_runs(np.flatnonzero(slots.executed["emergency"][i] > EMERGENCY_TOL) + 1)
        for j, (s, e) in enumerate(runs):
            ws.cell(row, 1).value = (pd.Timestamp(dates[i]).strftime("%Y-%m-%d")
                                     if j == 0 else None)
            ws.cell(row, 2).value = f"{_label((s - 1) * 10)}-{_label(e * 10)}"
            ws.cell(row, 3).value = round(
                float(slots.executed["emergency"][i, s - 1:e].sum()), 4)
            row += 1

    wb.save(OUTPUT_XLSX)


# --------------------------------------------------------------------------
# 逐时段 / 逐日 / 四天 明细
# --------------------------------------------------------------------------
def write_interval_csv(data, bundle, dates, slots: Slots, day_indices, reserve,
                       price_day=None, price_fc=None) -> None:
    k0_load, k0_pv = bundle["load"], np.nan_to_num(bundle["pv_fused"][0], nan=0.0)
    frames = []
    for i, day in enumerate(dates):
        di = day_indices[i]
        frames.append(pd.DataFrame({
            "date": day.strftime("%Y-%m-%d"),
            "t": np.arange(1, N_SLOT + 1),
            "time_end": [_label(t * 10) for t in range(1, N_SLOT + 1)],
            "L_actual_kw": data.load_kw[di],
            "P_act_kw": data.pv_kw[di],
            "L_fc_kw": k0_load[di],
            "P_fc_kw": k0_pv[di],
            # 波动电价口径：c_actual 用于结算，c_fc 进入 LP 目标与储能套利。
            "c_actual_yuan_per_kwh": (data.price if price_day is None
                                      else np.asarray(price_day, float)[di]),
            "c_fc_yuan_per_kwh": (data.price if price_fc is None
                                  else np.asarray(price_fc, float)[di]),
            "R_reserve_kwh": float(reserve[i]),
            # LP 变量与执行器输出都已经是"每时段电量(kWh)"（平衡式右侧为
            # ``(L−P)·Δt``，SOC 递推无 Δt 因子），故此处不得再乘 DT。
            "x_plan_kwh": slots.x_plan[i],
            "x_final_kwh": slots.x_final[i],
            "y_used_kwh": slots.executed["plan_used"][i],
            "r_purchase_waste_kwh": slots.executed["unused_plan"][i],
            "u_charge_kwh": slots.executed["charge"][i],
            "v_discharge_kwh": slots.executed["discharge"][i],
            "e_emergency_kwh": slots.executed["emergency"][i],
            "w_pv_waste_kwh": slots.executed["curtailment"][i],
            "g_pv_used_kwh": slots.executed["pv_used"][i],
            "E_end_kwh": slots.executed["soc"][i],
        }))
    pd.concat(frames, ignore_index=True).to_csv(
        RESULT_DIR / "全年逐10分钟策略.csv", index=False, encoding="utf-8-sig")


def write_daily_csv(data, daily: pd.DataFrame, dates, day_indices) -> None:
    pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "x_plan_kwh": daily["planned_purchase_kwh"].to_numpy(),
        "C_plan_yuan": daily["base_cost_yuan"].to_numpy(),
        "C_adjust_yuan": (daily["breach_cost_yuan"] + daily["overbuy_cost_yuan"]).to_numpy(),
        "C_grid_yuan": daily["grid_cost_yuan"].to_numpy(),
        "e_emergency_kwh": daily["emergency_kwh"].to_numpy(),
        "C_emer_yuan": daily["emergency_cost_yuan"].to_numpy(),
        "C_total_yuan": daily["total_cost_yuan"].to_numpy(),
        "r_purchase_waste_kwh": daily["unused_plan_kwh"].to_numpy(),
        "w_pv_waste_kwh": daily["curtailed_pv_kwh"].to_numpy(),
        "u_charge_kwh": daily["charge_kwh"].to_numpy(),
        "v_discharge_kwh": daily["discharge_kwh"].to_numpy(),
        "E_start_kwh": daily["start_soc_kwh"].to_numpy(),
        "E_end_kwh": daily["end_soc_kwh"].to_numpy(),
        "L_energy_kwh": (data.load_kw[day_indices] * DT).sum(axis=1),
        "P_act_energy_kwh": (data.pv_kw[day_indices] * DT).sum(axis=1),
    }).to_csv(RESULT_DIR / "全年逐日结果.csv", index=False, encoding="utf-8-sig")


def write_four_day(data, daily, dates, slots: Slots, day_indices, reserve):
    cost = dict(zip(dates, daily["total_cost_yuan"].to_numpy()))
    emer_cost = dict(zip(dates, daily["emergency_cost_yuan"].to_numpy()))
    t1, t2, t3, detail = [], [], [], []
    for date in SPECIFIED_DATES:
        i = int(np.flatnonzero(dates == date)[0])
        di = day_indices[i]

        rec = {"date": date.strftime("%Y-%m-%d")}
        for minutes in TABLE1_MINUTES:
            rec[_slot_label(minutes // 10 + 1)] = round(float(slots.x_final[i, minutes // 10]), 4)
        rec["daily_purchase_kwh"] = round(float(slots.x_final[i].sum()), 4)
        rec["daily_cost_yuan"] = round(float(cost[date]), 2)
        t1.append(rec)

        rec = {"date": date.strftime("%Y-%m-%d")}
        for h0, h1 in BLOCKS:
            sl = slice(h0 * SLOTS_PER_HOUR, h1 * SLOTS_PER_HOUR)
            rec[f"charge_{h0:02d}:00-{h1:02d}:00_kwh"] = round(
                float(slots.executed["charge"][i, sl].sum()), 4)
            rec[f"discharge_{h0:02d}:00-{h1:02d}:00_kwh"] = round(
                float(slots.executed["discharge"][i, sl].sum()), 4)
        rec["soc_00_kwh"] = round(float(slots.soc_start[i]), 4)
        rec["soc_24_kwh"] = round(float(slots.soc_end[i]), 4)
        t2.append(rec)

        runs = _merge_runs(np.flatnonzero(slots.executed["emergency"][i] > EMERGENCY_TOL) + 1)
        t3.append({
            "date": date.strftime("%Y-%m-%d"),
            "count": len(runs),
            "intervals": _runs_label(runs),
            "emergency_kwh": round(float(slots.executed["emergency"][i].sum()), 4),
            "emergency_cost_yuan": round(float(emer_cost[date]), 2),
            "total_cost_yuan": round(float(cost[date]), 2),
        })

        detail.append(pd.DataFrame({
            "date": date.strftime("%Y-%m-%d"),
            "t": np.arange(1, N_SLOT + 1),
            "time_end": [_label(t * 10) for t in range(1, N_SLOT + 1)],
            "L_actual_kw": data.load_kw[di],
            "P_act_kw": data.pv_kw[di],
            "R_reserve_kwh": float(reserve[i]),
            "x_plan_kwh": slots.x_plan[i],
            "x_final_kwh": slots.x_final[i],
            "u_charge_kwh": slots.executed["charge"][i],
            "v_discharge_kwh": slots.executed["discharge"][i],
            "e_emergency_kwh": slots.executed["emergency"][i],
            "E_end_kwh": slots.executed["soc"][i],
        }))

    t1, t2, t3 = pd.DataFrame(t1), pd.DataFrame(t2), pd.DataFrame(t3)
    t1.to_csv(RESULT_DIR / "四天表1购电结果.csv", index=False, encoding="utf-8-sig")
    t2.to_csv(RESULT_DIR / "四天表2储能结果.csv", index=False, encoding="utf-8-sig")
    t3.to_csv(RESULT_DIR / "四天表3紧急购电结果.csv", index=False, encoding="utf-8-sig")
    pd.concat(detail, ignore_index=True).to_csv(
        RESULT_DIR / "四天逐10分钟明细.csv", index=False, encoding="utf-8-sig")
    return t1, t2, t3


# --------------------------------------------------------------------------
# 诊断与敏感性
# --------------------------------------------------------------------------
def reserve_series(errors0: np.ndarray, day_indices, quantile: float) -> np.ndarray:
    """因果日储备 $R_d=Q_q(\\sum_t \\xi_{j,t}\\Delta t)$，只用 $j<d$ 的样本。"""
    return np.asarray([
        0.0 if errors0[:di].shape[0] < 20
        else float(np.quantile(errors0[:di].sum(axis=1) * DT, quantile))
        for di in day_indices])


def write_diagnostics(data, bundle, dates, day_indices, reserve) -> None:
    errors0 = _error_matrix(data, bundle, 0, "own", "fused")
    k0_load = bundle["load"]
    k0_pv = np.nan_to_num(bundle["pv_fused"][0], nan=0.0)
    per_slot, load_mae, pv_mae, net_mae = [], [], [], []
    for di in day_indices:
        hist = errors0[:di]
        per_slot.append(0.0 if hist.shape[0] < 20
                        else float(np.quantile(hist, MARGIN_Q, axis=0).sum() * DT))
        load_mae.append(float(np.mean(np.abs(data.load_kw[di] - k0_load[di]))))
        pv_mae.append(float(np.mean(np.abs(data.pv_kw[di] - k0_pv[di]))))
        net_mae.append(float(np.mean(np.abs(
            (data.load_kw[di] - data.pv_kw[di]) - (k0_load[di] - k0_pv[di])))))
    per_slot = np.asarray(per_slot)
    with np.errstate(divide="ignore", invalid="ignore"):
        infl = np.where(reserve > 0, per_slot / reserve, np.nan)
    pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "R_reserve_kwh": reserve,
        "per_slot_q80_sum_kwh": per_slot,
        "inflation_ratio": infl,
        "load_mae_kw": load_mae,
        "pv_mae_kw": pv_mae,
        "netload_mae_kw": net_mae,
    }).to_csv(RESULT_DIR / "预测诊断.csv", index=False, encoding="utf-8-sig")


def split_row(daily: pd.DataFrame, label: str) -> dict:
    if label == "全年正式期":
        sub = daily
    elif label == "开发期(2-8月)":
        sub = daily[daily.date <= DEV_END]
    else:
        sub = daily[daily.date > DEV_END]
    cost = sub.total_cost_yuan
    p95 = float(np.quantile(cost, 0.95))
    return {
        "split": label,
        "evaluation_start": sub.date.min().strftime("%Y-%m-%d"),
        "evaluation_end": sub.date.max().strftime("%Y-%m-%d"),
        "n_days": int(len(sub)),
        "planned_purchase_kwh": float(sub.planned_purchase_kwh.sum()),
        "planned_cost_yuan": float(sub.base_cost_yuan.sum()),
        "adjust_cost_yuan": float(sub.breach_cost_yuan.sum() + sub.overbuy_cost_yuan.sum()),
        "grid_cost_yuan": float(sub.grid_cost_yuan.sum()),
        "emergency_kwh": float(sub.emergency_kwh.sum()),
        "emergency_cost_yuan": float(sub.emergency_cost_yuan.sum()),
        "total_cost_yuan": float(cost.sum()),
        "unused_plan_kwh": float(sub.unused_plan_kwh.sum()),
        "curtailed_pv_kwh": float(sub.curtailed_pv_kwh.sum()),
        "days_with_emergency": int((sub.emergency_kwh > EMERGENCY_TOL).sum()),
        "daily_cost_p95_yuan": p95,
        "daily_cost_cvar95_yuan": float(cost[cost >= p95].mean()),
        "final_soc_kwh": float(sub.end_soc_kwh.iloc[-1]),
    }


def write_sensitivity(data, bundle, daily_main: pd.DataFrame,
                      price_day=None, price_fc=None) -> pd.DataFrame:
    rows = []
    for q in Q_GRID:
        daily = (daily_main if abs(q - 0.80) < 1e-12
                 else run_capture(data, bundle, q, price_day, price_fc)[0])
        for label in ("开发期(2-8月)", "留出期(9-12月)", "全年正式期"):
            rows.append({"quantile": q, **split_row(daily, label)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / "风险参数敏感性.csv", index=False, encoding="utf-8-sig")
    return out


# --------------------------------------------------------------------------
# 图
# --------------------------------------------------------------------------
def plot_four_days(data, dates, slots: Slots, day_indices, reserve) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    hours = np.arange(N_SLOT) / SLOTS_PER_HOUR
    for ax, date in zip(axes, SPECIFIED_DATES):
        di = int(day_indices[int(np.flatnonzero(dates == date)[0])])
        i = int(np.flatnonzero(dates == date)[0])
        ax.plot(hours, data.load_kw[di] - data.pv_kw[di], lw=1.2, color="#2b6cb0",
                label="实际净负荷 $L-P$")
        ax.plot(hours, slots.x_final[i] / DT, lw=1.4, color="#c53030",
                label="最终承诺购电 $x^{(18)}$")
        ax.fill_between(hours, 0, slots.x_final[i] / DT, color="#c53030", alpha=0.10)
        ax.set_ylabel("功率 / kW")
        ax.set_title(f"{date:%Y-%m-%d}　储能储备 $R_d$ = {reserve[i]:,.0f} kWh　"
                     f"承诺电量 {slots.x_final[i].sum():,.0f} kWh",
                     fontsize=10)
        ax.grid(alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(hours, slots.executed["soc"][i], lw=1.0, ls="--", color="#2f855a",
                 label="储电量 $E$")
        ax2.axhline(1200 + reserve[i], lw=0.8, ls=":", color="#975a16",
                    label="$E_{\\min}+R_d$")
        ax2.set_ylim(0, 12000)
        ax2.set_ylabel("储电量 / kWh")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    axes[-1].set_xlabel("时刻 / h")
    fig.suptitle("四个指定日期的调度结果（$M_9$ 储能追索型 LP，$q=0.80$）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(FIGURE_PATH, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------
def plot_price(dates, day_indices, price_day, price_fc) -> None:
    """波动电价的实际值与因果预测值（正式期逐日均价 + 日内平均形状）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    PRICE_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    act = np.asarray(price_day, float)[day_indices]
    fc = np.asarray(price_fc, float)[day_indices]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5))
    axes[0].plot(dates, act.mean(axis=1), lw=1.0, label="实际日均价")
    axes[0].plot(dates, fc.mean(axis=1), lw=1.0, alpha=0.8, label="预测日均价")
    axes[0].set_ylabel("电价 / (元/kWh)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    slot_hour = (np.arange(N_SLOT) + 0.5) / SLOTS_PER_HOUR
    axes[1].plot(slot_hour, act.mean(axis=0), lw=1.5, label="实际日内均价")
    axes[1].plot(slot_hour, fc.mean(axis=0), lw=1.5, alpha=0.8, label="预测日内均价")
    axes[1].set_xlabel("时刻 / h")
    axes[1].set_ylabel("电价 / (元/kWh)")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    fig.suptitle("波动电价的实际值与因果预测值（2025-02-01—12-31）")
    fig.tight_layout()
    fig.savefig(PRICE_FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(full, t1, t3, reserve, sens) -> None:
    # 必须用 sens["quantile"] / sens["split"] 取列：``sens.quantile`` 会解析到
    # DataFrame.quantile 方法、``sens.split`` 会解析到 str.split 方法，都不等于
    # 数据列，结果会静默得到空筛选。
    sel = sens["quantile"] == 0.80
    dev = sens[sel & (sens["split"] == "开发期(2-8月)")].iloc[0]
    hold = sens[sel & (sens["split"] == "留出期(9-12月)")].iloc[0]
    lines = [
        "# 问题4 重算问题3 最终结果汇总（波动电价）", "",
        f"正式评价期：2025-02-01—2025-12-31，共 {full['n_days']} 天"
        f"（{full['n_days'] * N_SLOT:,} 个时段）。", "",
        "最终方案为 $M_9$ **储能追索型 LP**：计划购电按点预报下达、不加任何分位余量，",
        "不确定性全部由储能的能量下界 $E_{\\min}+R_d$ 承担。",
        "风险分位数取 $q=0.80$，直接来自 5 倍紧急电价下的报童临界比 $4c/(4c+c)$，",
        "是事前给定的理论值，不由任何评价期回测挑选。", "",
        "**波动电价口径**：电价取自附件4，与负载、光伏同为不可预知量。",
        "第 $d$ 天 $k$ 时刻只能用 $j<d$ 的历史实际电价因果预测当天电价；",
        "**预测电价只进入 LP 目标与储能套利，费用一律按当天实际电价结算**",
        "（计划购电 $c_t\\min_kx^{(k)}$、调整 $0.5c_t\\Delta^-$ 与 $1.5c_t\\Delta^+$、",
        "紧急购电 $5c_t$）。", "",
        "| 指标 | 最终值 |", "|---|---:|",
        f"| 计划购电量 | {full['planned_purchase_kwh']:,.2f} kWh |",
        f"| 电网结算费 | {full['grid_cost_yuan']:,.2f} 元 |",
        f"| 紧急购电量 | {full['emergency_kwh']:,.2f} kWh |",
        f"| 紧急购电费 | {full['emergency_cost_yuan']:,.2f} 元 |",
        f"| **全年正式期总费用** | **{full['total_cost_yuan']:,.2f} 元** |",
        f"| 未用计划电量 | {full['unused_plan_kwh']:,.2f} kWh |",
        f"| 弃光量 | {full['curtailed_pv_kwh']:,.2f} kWh |",
        f"| 紧急购电天数 | {full['days_with_emergency']} 天 |",
        f"| 单日费用 95% 分位数 | {full['daily_cost_p95_yuan']:,.2f} 元 |",
        f"| 单日费用 CVaR$_{{95}}$ | {full['daily_cost_cvar95_yuan']:,.2f} 元 |",
        f"| 年末储电量 | {full['final_soc_kwh']:,.2f} kWh |",
        f"| 储能日储备 $R_d$ 均值 | {reserve.mean():,.2f} kWh |", "",
        "## 四个指定日期", "",
        "| 日期 | 全天购电量/kWh | 紧急购电量/kWh | 当日总费用/元 |",
        "|---|---:|---:|---:|",
    ]
    for rec, em in zip(t1.to_dict("records"), t3.to_dict("records")):
        lines.append(f"| {rec['date']} | {rec['daily_purchase_kwh']:,.4f} | "
                     f"{em['emergency_kwh']:.4f} | {rec['daily_cost_yuan']:,.2f} |")
    lines += [
        "", "## 分段稳健性", "",
        "| 时段 | 天数 | 总费用/元 |", "|---|---:|---:|",
        f"| 开发期 2025-02-01—08-31 | {int(dev.n_days)} | {dev.total_cost_yuan:,.2f} |",
        f"| 留出期 2025-09-01—12-31 | {int(hold.n_days)} | {hold.total_cost_yuan:,.2f} |",
        f"| 全年正式期 | {full['n_days']} | {full['total_cost_yuan']:,.2f} |", "",
        "完整逐十分钟策略见 `全年逐10分钟策略.csv`，附件5 格式见 `result4-3.xlsx`，",
        "四个指定日期规定格式见同目录三个四天结果 CSV。",
    ]
    (RESULT_DIR / "最终结果汇总.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    bundle = forecast_bundle(data, verbose=False)
    price_day, price_fc = load_price_channel()
    price_metrics = load_price_metrics()

    daily, store = run_capture(data, bundle, 0.80, price_day, price_fc)
    if len(daily) != N_FORMAL:
        raise ValueError(f"正式期天数异常：{len(daily)}")
    dates = pd.DatetimeIndex(daily.date)
    day_indices = [int(np.flatnonzero(data.dates == d)[0]) for d in dates]
    slots = Slots(store, dates)

    errors0 = _error_matrix(data, bundle, 0, "own", "fused")
    reserve = reserve_series(errors0, day_indices, 0.80)

    # 对照口径：电价可预知（完美电价信息），同一套因果负载/光伏预测与同一储备
    # 规则下的费用。不属于可实现的策略，仅用于量化"电价不可预知"的代价。
    perfect_daily, _ = run_capture(data, bundle, 0.80, price_day, None)
    perfect_full = split_row(perfect_daily, "全年正式期")

    # 计划购电费 = 链式结算的基础项 Σ price·min_k x^(k)（与 全年逐日结果.csv 的
    # C_plan_yuan 同口径），加上 调整购电费 即得 C_grid_yuan。
    planned_cost = daily["base_cost_yuan"].to_numpy(float)
    adjust_cost = (daily.breach_cost_yuan + daily.overbuy_cost_yuan).to_numpy(float)

    write_result_workbook(dates, slots, price_day, planned_cost, adjust_cost)
    write_interval_csv(data, bundle, dates, slots, day_indices, reserve,
                       price_day, price_fc)
    write_daily_csv(data, daily, dates, day_indices)
    t1, t2, t3 = write_four_day(data, daily, dates, slots, day_indices, reserve)
    write_diagnostics(data, bundle, dates, day_indices, reserve)
    sens = write_sensitivity(data, bundle, daily, price_day, price_fc)
    plot_four_days(data, dates, slots, day_indices, reserve)
    plot_price(dates, day_indices, price_day, price_fc)
    shutil.copyfile(CACHE_DIR / "price_forecast_weights.csv",
                    RESULT_DIR / "电价预测权重.csv")

    full = split_row(daily, "全年正式期")
    frozen = {
        "model": {
            "name": "储能追索型 LP（M9）",
            "code": "M9",
            "risk_quantile": 0.80,
            "quantile_source": "报童临界比 C_u/(C_u+C_o)=4c/(4c+c)=0.80，事前给定，不由评价期挑选",
            "purchase_rule": "按点预报下达计划购电，不加任何分位余量",
            "risk_carrier": "储能能量下界 E_min + R_d，R_d 为因果历史残差日累计量的 0.80 分位；不设充电空间上界",
            "causal_information_set": "第d天第k时刻仅使用 j<d 的历史残差与日内已发生时段的实际值",
            "price_information_set": "电价取自附件4，与负载/光伏同为不可预知量；第d天k时刻只用 j<d 的历史实际电价预测当天电价",
            "price_rule": "预测电价只进入 LP 目标与储能套利；费用按当天实际电价结算（计划 c·min_k x，调整 0.5c·Δ⁻ 与 1.5c·Δ⁺，紧急 5c·e）",
            "price_forecast_namespace": "问题4/重算问题2 的 price_forecast.generate_causal_price_forecast，只做缓存不重写",
            "initial_state": "E_{1,0}=E_0=6000 kWh；随后逐日跨日连续递推",
            "storage": {"eta_c": 0.9, "eta_d": 0.9, "e_min": 1200.0,
                        "e_max": 10800.0, "p_max_kw": 5000.0},
        },
        "evaluation": {
            "formal_start": FORMAL_START.strftime("%Y-%m-%d"),
            "formal_end": dates[-1].strftime("%Y-%m-%d"),
            "n_days": int(len(daily)),
            "n_slots": int(len(daily) * N_SLOT),
            "dev_end": DEV_END.strftime("%Y-%m-%d"),
        },
        "annual": full,
        "summary": summarize(daily, label="M9"),
        "price_forecast": price_metrics,
        "perfect_price_information": {
            **perfect_full,
            "perfect_minus_forecast_yuan": float(
                perfect_full["total_cost_yuan"] - full["total_cost_yuan"]
            ),
            "note": "电价可预知的对照口径，不属于可实现的策略；其绝对值即电价不可预知的代价",
        },
        "four_days": t1.to_dict("records"),
        "four_days_emergency": t3.to_dict("records"),
        "reserve_kwh": {
            "mean": float(reserve.mean()),
            "min": float(reserve.min()),
            "max": float(reserve.max()),
        },
        "sensitivity": sens.to_dict("records"),
    }
    (RESULT_DIR / "最终冻结结果.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(full, t1, t3, reserve, sens)

    print(f"已写出 {OUTPUT_XLSX}")
    print(t1.to_string(index=False))
    print(t2.to_string(index=False))
    print(t3.to_string(index=False))
    print(f"\n全年正式期 {full['total_cost_yuan']:,.2f} 元"
          f"（完美电价对照 {perfect_full['total_cost_yuan']:,.2f} 元，"
          f"电价不可预知的代价 {full['total_cost_yuan'] - perfect_full['total_cost_yuan']:,.2f} 元）")


if __name__ == "__main__":
    main()
