# -*- coding: utf-8 -*-
"""导出四个最终档位的逐十分钟明细与三个指定日期结果表。

现有 ``run_single.py`` / ``run_stage.py`` 只落逐日结果，逐时段数组在仿真循环内
用完即弃。本脚本**不改动这两个文件**：在 ``opt_core.run_day_single`` 上挂一层
记录钩子捕获每次执行的 (承诺, 执行结果)，复用它们各自的 ``simulate`` 跑一次
最终档位，再按上游 ``问题2/最终优化版/results`` 的格式导出：

  ``<问题>_<档位>_逐10分钟.csv``      全年逐十分钟策略
  ``<问题>_四天表1购电结果.csv``      表1 格式（六个时点购电量 + 全天）
  ``<问题>_四天表2储能结果.csv``      表2 格式（六段充放电 + 首末储电）
  ``<问题>_四天紧急购电结果.csv``     表3 格式

跑法：
  python export_final.py --problem Q2
  python export_final.py --problem Q3
  python export_final.py --problem Q4-2
  python export_final.py --problem Q4-3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OPT_DIR = Path(__file__).resolve().parent
PKG_DIR = OPT_DIR.parent
sys.path.insert(0, str(OPT_DIR))

import opt_core as oc  # noqa: E402

DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
SIX_SLOTS = [60, 72, 84, 96, 108, 120]          # 10:00/12:00/14:00/16:00/18:00/20:00
SIX_LABELS = ["10:00-10:10", "12:00-12:10", "14:00-14:10",
              "16:00-16:10", "18:00-18:10", "20:00-20:10"]
BLOCKS = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144)]
BLOCK_LABELS = ["00:00-04:00", "04:00-08:00", "08:00-12:00",
                "12:00-16:00", "16:00-20:00", "20:00-24:00"]

FINAL = {"Q2": "reserve-mpc", "Q3": "exact-greedy",
         "Q4-2": "reserve-mpc", "Q4-3": "exact-greedy"}
OUT_SUBDIR = {"Q2": "问题2最终模型", "Q3": "问题3最终模型",
              "Q4-2": "问题4最终模型", "Q4-3": "问题4最终模型"}

_KEYS = ("charge", "discharge", "emergency", "unused_plan",
         "curtailment", "pv_used", "plan_used", "soc")

_CAP: list = []
_ORIG = oc.run_day_single


def _hooked(commitment, load_kw, pv_kw, *args, **kw):
    out = _ORIG(commitment, load_kw, pv_kw, *args, **kw)
    _CAP.append((np.asarray(commitment, float).copy(), out))
    return out


def _merge(per_day: int) -> list[dict]:
    """把钩子捕获的调用序列按每天 ``per_day`` 段拼成整天的逐时段记录。"""
    assert len(_CAP) % per_day == 0, f"捕获条数 {len(_CAP)} 不是 {per_day} 的倍数"
    days = []
    for i in range(0, len(_CAP), per_day):
        day = {k: np.full(oc.N_SLOT, np.nan) for k in ("x",) + _KEYS}
        pos = 0
        for x, act in _CAP[i:i + per_day]:
            n = len(x)
            day["x"][pos:pos + n] = x
            for k in _KEYS:
                day[k][pos:pos + n] = act[k][:n]
            pos += n
        assert pos == oc.N_SLOT, f"第 {i // per_day} 天拼接长度 {pos} != {oc.N_SLOT}"
        days.append(day)
    return days


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=list(FINAL), required=True)
    args = ap.parse_args()
    problem, tag = args.problem, FINAL[args.problem]

    _CAP.clear()
    oc.run_day_single = _hooked
    try:
        if problem in ("Q2", "Q4-2"):
            import run_single as rs
            db, CorrCfg, FcCfg, gen_fc = rs.load_stack(problem)
            data = db.load_inputs()
            fc = rs.build_forecasts(problem, db, CorrCfg, FcCfg, gen_fc)
            daily, _ = rs.simulate(db, data, fc, mode=tag)
            load_fc, pv_fc = fc["load"], fc["pv"]
            per_day = 1
        else:
            import run_stage as rt
            common, dispatch_core, forecasts = rt.load_stack(problem)
            data = common.load_inputs()
            bundle = forecasts.forecast_bundle(data, use_cache=False, verbose=False)
            n_days = len(data.dates)
            price_s, price_d = rt.price_channel(problem, common, data, n_days)
            errs = rt.errors_for(data, bundle, dispatch_core)
            res = rt.reserves_for(data, bundle, dispatch_core, n_days)
            spec = dict((t[0], t[1:]) for t in [
                ("legacy-greedy", "legacy", False, False, False, False),
                ("fixed-greedy", "correct", False, False, False, False),
                ("fixed-mpc", "correct", True, False, False, False),
                ("exact-greedy", "correct", False, True, False, False),
                ("exact-mpc", "correct", True, True, False, False)])
            settle, mpc, exact, joint, no_res = spec[tag]
            daily, _, _ = rt.simulate(
                data, bundle, dispatch_core, settle=settle, mpc=mpc, reserves=res,
                price_s=price_s, price_d=price_d, exact=exact,
                joint=joint, errs=errs, no_reserve=no_res)
            load_fc = np.full((n_days, oc.N_SLOT), np.nan)
            pv_fc = np.full((n_days, oc.N_SLOT), np.nan)
            for day in range(n_days):
                for k in rt.STAGES:
                    s = k * oc.SLOTS_PER_HOUR
                    load_fc[day, s:] = dispatch_core._load_at_stage(
                        data, bundle, day, k, "own")[s:]
                    pv_fc[day, s:] = np.nan_to_num(
                        dispatch_core._pv_at_stage(data, bundle, day, k, "fused")[s:])
            per_day = len(rt.STAGES)
    finally:
        oc.run_day_single = _ORIG

    days = _merge(per_day)
    assert len(days) == len(data.dates), f"天数不符：{len(days)} != {len(data.dates)}"

    # 评价期 2025-02-01 起
    keep = [i for i, d in enumerate(data.dates) if d >= pd.Timestamp("2025-02-01")]
    assert len(keep) == len(daily), f"评价期天数不符：{len(keep)} != {len(daily)}"
    assert abs(float(daily.total_cost_yuan.sum())
               - _frozen_cost(problem)) < 1.0, "重跑总费用与冻结值不一致"

    # ---- 逐十分钟明细 ----
    rows = []
    for j, i in enumerate(keep):
        day = days[i]
        for t in range(oc.N_SLOT):
            rows.append({
                "date": data.dates[i], "t": t + 1,
                "time_end": f"{(t + 1) * 10 // 60:02d}:{(t + 1) * 10 % 60:02d}",
                "L_actual_kw": float(data.load_kw[i, t]),
                "P_act_kw": float(data.pv_kw[i, t]),
                "L_fc_kw": float(load_fc[i, t]),
                "P_fc_kw": float(pv_fc[i, t]),
                "x_plan_kwh": float(day["x"][t]),
                "y_used_kwh": float(day["plan_used"][t]),
                "r_purchase_waste_kwh": float(day["unused_plan"][t]),
                "u_charge_kwh": float(day["charge"][t]),
                "v_discharge_kwh": float(day["discharge"][t]),
                "e_emergency_kwh": float(day["emergency"][t]),
                "w_pv_waste_kwh": float(day["curtailment"][t]),
                "g_pv_used_kwh": float(day["pv_used"][t]),
                "E_end_kwh": float(day["soc"][t]),
            })
    slots = pd.DataFrame(rows)

    price2 = data.price if data.price.ndim == 2 else np.tile(data.price, (len(data.dates), 1))

    # ---- 三个指定日期表 ----
    t1, t2, t3 = [], [], []
    for j, ds in enumerate(DATES):
        d = pd.Timestamp(ds)
        idx = int(np.where(data.dates == d)[0][0])
        v = slots[slots.date == d].sort_values("t")
        if len(v) != oc.N_SLOT:
            raise ValueError(f"{problem}: 指定日期 {ds} 不在评价期内")
        purch = v.x_plan_kwh.to_numpy(float)
        ps = price2[idx]
        t1.append({"date": ds,
                   **{lab: float(purch[s]) for lab, s in zip(SIX_LABELS, SIX_SLOTS)},
                   "daily_purchase_kwh": float(purch.sum()),
                   "daily_planned_cost_yuan": float(ps @ purch)})
        ch, di = v.u_charge_kwh.to_numpy(float), v.v_discharge_kwh.to_numpy(float)
        row = {"date": ds}
        for (a, b), lab in zip(BLOCKS, BLOCK_LABELS):
            row[f"charge_{lab}_kwh"] = float(ch[a:b].sum())
            row[f"discharge_{lab}_kwh"] = float(di[a:b].sum())
        row["soc_00_kwh"] = float(daily.start_soc_kwh.iloc[j])
        row["soc_24_kwh"] = float(daily.end_soc_kwh.iloc[j])
        t2.append(row)
        em = v.e_emergency_kwh.to_numpy(float)
        t3.append({"date": ds, "emergency_kwh": float(em.sum()),
                   "emergency_cost_yuan": float(oc.EMERGENCY_MULTIPLIER * (ps @ em))})

    out_dir = PKG_DIR / OUT_SUBDIR[problem] / "结果"
    out_dir.mkdir(parents=True, exist_ok=True)
    slots.to_csv(out_dir / f"{problem}_{tag}_逐10分钟.csv",
                 index=False, encoding="utf-8-sig")
    pd.DataFrame(t1).to_csv(out_dir / f"{problem}_四天表1购电结果.csv",
                            index=False, encoding="utf-8-sig")
    pd.DataFrame(t2).to_csv(out_dir / f"{problem}_四天表2储能结果.csv",
                            index=False, encoding="utf-8-sig")
    pd.DataFrame(t3).to_csv(out_dir / f"{problem}_四天紧急购电结果.csv",
                            index=False, encoding="utf-8-sig")

    print(f"[{problem}] {tag}  总费用 {daily.total_cost_yuan.sum():,.2f} 元  "
          f"紧急 {daily.emergency_kwh.sum():,.1f} kWh  "
          f"未用 {daily.unused_plan_kwh.sum():,.1f} kWh")
    print(f"  逐十分钟 {len(slots):,} 行 -> {out_dir}")
    print(f"  四个指定日期合计购电 "
          f"{sum(x['daily_purchase_kwh'] for x in t1):,.2f} kWh")


def _frozen_cost(problem: str) -> float:
    """``结果冻结.json`` 中该问题最终档位的**原始**总费用，用于重跑一致性核对。"""
    import json
    d = json.loads((OPT_DIR / "结果冻结.json").read_text(encoding="utf-8"))
    tag = FINAL[problem]
    return float(d["各档位实跑"][f"{problem}/{tag}"]["total_cost_yuan"])


if __name__ == "__main__":
    main()
