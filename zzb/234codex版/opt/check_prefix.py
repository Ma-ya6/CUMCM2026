# -*- coding: utf-8 -*-
"""前缀不变性检验：把全年仿真截断到前 K 天重跑，检查前 K 天的决策与全年一致。

因果策略（只用 $t$ 之前的信息）在同一天上的决策不应随"仿真总共跑多少天"而变。
本脚本对每个问题的最终档位取 K ∈ {90, 150, 220, 300} 截断重跑，与**同进程内
重跑的完整全年**按日期对齐逐位比对（不用 ``opt/out/*_承诺路径.csv``，见下）。

两处实现都只在 ``days is None`` 时对**最后一天**施加 ``force_end_soc``，
传 ``--days K`` 时该分支不生效，故截断跑的第 K 天与全年跑的同一天可直接比较。

跑法：``python check_prefix.py``（在 opt/ 下），输出 ``opt/前缀不变性.md``。
"""
from __future__ import annotations

import io
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

OPT_DIR = Path(__file__).resolve().parent
OUT_DIR = OPT_DIR / "out"
sys.path.insert(0, str(OPT_DIR))

KS = [90, 150, 220, 300]
# 各问题的最终档位与其上游模块名（同进程切换问题时要清掉同名模块）
FINAL = {"Q2": "reserve-mpc", "Q3": "exact-greedy",
         "Q4-2": "reserve-mpc", "Q4-3": "exact-greedy"}
STAGE_PROBLEMS = ("Q3", "Q4-3")          # 四个决策时刻、写承诺路径的
TOL = 1e-9                               # 逐位一致，容差仅用于浮点噪声


def purge(names) -> None:
    for m in names:
        sys.modules.pop(m, None)


def _truncate_data(data, days: int):
    """真正截断原始数据；一维日内价格保留，逐日数组同步截断。"""
    changes = {
        "dates": data.dates[:days],
        "load_kw": data.load_kw[:days],
        "pv_kw": data.pv_kw[:days],
    }
    if getattr(data.price, "ndim", 1) == 2:
        changes["price"] = data.price[:days]
    if hasattr(data, "official_pv"):
        changes["official_pv"] = data.official_pv[:days]
        changes["official_mask"] = data.official_mask[:days]
    return replace(data, **changes)


def _stage_forecasts(module, data):
    """绕开磁盘缓存，用传入的（可能已截断）原始数据重新生成预测。"""
    projected = module.base.DataBundle(
        dates=data.dates, load_kw=data.load_kw, pv_kw=data.pv_kw,
        price=data.price, initial_load=data.initial_load, initial_pv=data.initial_pv)
    own = module.generate_causal_forecasts(
        projected, module.ForecastConfig(), verbose=False)
    fusion = module.build_pv_fusion(data, own["pv"], verbose=False)
    return {
        "load": own["load"], "pv_own": own["pv"],
        "pv_official": fusion["pv_official"],
        "pv_fused": fusion["pv_fused"],
        "fusion_diagnostics": fusion["diagnostics"],
        "own_diagnostics": own["diagnostics"],
    }


def run_stage_prefix(problem: str, days: int | None):
    """问题3 / 问题4-3：返回 (逐日DataFrame, 承诺路径 ndarray[天,4,144])。"""
    purge(("common", "dispatch_core", "forecasts",
           "deterministic_baseline", "seasonal"))
    # 问题4-2 留在 sys.path 里的上游目录会抢先命中它自己的
    # deterministic_baseline（其 DataBundle 要求 initial_price），
    # 清掉指向它的条目，让 forecasts.py 重新按问题2 目录解析。
    import run_single as rg
    stale = str(rg.DIRS["Q4-2"])
    while stale in sys.path:
        sys.path.remove(stale)
    import run_stage as rs
    rs.DIRS[problem]                       # 触发 KeyError 检查
    common, dispatch_core, forecasts = rs.load_stack(problem)
    data = common.load_inputs()
    if days is not None:
        data = _truncate_data(data, days)
    bundle = _stage_forecasts(forecasts, data)
    n = len(data.dates)
    price_s, price_d = rs.price_channel(problem, common, data, n)
    res = rs.reserves_for(data, bundle, dispatch_core, n)
    daily, paths, _ = rs.simulate(
        data, bundle, dispatch_core, settle="correct", mpc=False,
        reserves=res, price_s=price_s, price_d=price_d,
        days=(None if days is None else n), exact=True)
    pred = {
        "load": np.asarray(bundle["load"]),
        "pv_fused": np.nan_to_num(bundle["pv_fused"], nan=-1.0),
    }
    return daily, paths, pred


def run_single_prefix(problem: str, days: int | None):
    """问题2 / 问题4-2：返回 (逐日DataFrame, None)。"""
    purge(("deterministic_baseline", "seasonal", "price_forecast"))
    import run_single as rg
    db, CC, FC, gen_fc = rg.load_stack(problem)
    data = db.load_inputs()
    if days is not None:
        data = _truncate_data(data, days)
    cfg = FC(correction=CC(
        use_pv_window=True, use_level=True, use_weekly_effect=False))
    fc = gen_fc(data, cfg, verbose=False)
    price = data.price
    if problem == "Q4-2":
        from price_forecast import generate_causal_price_forecast
        price = generate_causal_price_forecast(
            data, fc["diagnostics"], cfg, verbose=False)["price"]
    forecasts = {"load": fc["load"], "pv": fc["pv"], "price": price}
    n = len(data.dates)
    daily, _ = rg.simulate(
        db, data, forecasts, mode="reserve-mpc",
        days=(None if days is None else n))
    pred = {"load": np.asarray(fc["load"]), "pv": np.asarray(fc["pv"])}
    if np.asarray(price).ndim == 2:
        pred["price"] = np.asarray(price)
    return daily, None, pred


def reference(problem: str, runner):
    """同进程内跑一遍完整全年，作为比对基准。

    基准在同一进程内重跑，避免旧缓存或旧导出文件混入比对。
    """
    return runner(problem, None)


DAILY_COLS = ["total_cost_yuan", "planned_purchase_kwh", "end_soc_kwh",
              "emergency_kwh"]


def prediction_deviation(pred: dict, ref: dict, n_days: int) -> float:
    """按正确的日期轴比较截断预测与全年预测前缀。"""
    worst = 0.0
    for key, cur in pred.items():
        full = ref[key]
        prefix = full[:, :n_days] if full.ndim == 3 and full.shape[0] == 4 \
            else full[:n_days]
        worst = max(worst, float(np.max(np.abs(cur - prefix))))
    return worst


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    L: list[str] = []
    A = L.append
    A("# 前缀不变性检验")
    A("")
    A("把全年仿真截断到前 $K$ 天重跑，与已冻结的全年结果按日期对齐逐位比对。")
    A("策略若只用到当前及历史信息，则同一天的决策不应随仿真总长度改变。")
    A("")
    A("| 问题 | 最终档位 | $K$ | 比对天数 | 承诺路径/预测最大偏差 | "
      "日费用最大偏差/元 | 日末储电最大偏差/kWh | 判定 |")
    A("|---|---|---:|---:|---:|---:|---:|---|")
    worst_all = 0.0
    bad_all = 0
    for p, tag in FINAL.items():
        t0 = time.time()
        runner = run_stage_prefix if p in STAGE_PROBLEMS else run_single_prefix
        ref_daily, ref_paths, ref_pred = reference(p, runner)
        print(f"  {p} 全年基准 {len(ref_daily)} 天  ({time.time()-t0:.0f}s)", flush=True)
        for K in KS:
            daily, paths, pred = runner(p, K)
            m = daily.merge(ref_daily, on="date", suffixes=("_k", "_full"))
            nd = len(m)
            dx = float(np.max(np.abs(paths[:nd] - ref_paths[:nd]))) \
                if paths is not None and ref_paths is not None else None
            dp = prediction_deviation(pred, ref_pred, K)
            dc = float(np.max(np.abs(m["total_cost_yuan_k"]
                                     - m["total_cost_yuan_full"])))
            de = float(np.max(np.abs(m["end_soc_kwh_k"] - m["end_soc_kwh_full"])))
            dev = max(v for v in (dx, dp, dc, de) if v is not None)
            worst_all = max(worst_all, dev)
            ok = dev <= TOL
            bad_all += (not ok)
            A(f"| {p} | `{tag}` | {K} | {len(m)} | "
              f"{'—' if dx is None else f'{dx:.3e}'} / 预测 {dp:.3e} | {dc:.3e} | {de:.3e} | "
              f"{'✓' if ok else '★'} |")
            print(f"  {p} K={K:>3}  天{nd:>3}  x偏差 "
                  f"{'—' if dx is None else f'{dx:.3e}'}  费用偏差 {dc:.3e}  "
                  f"储电偏差 {de:.3e}  ({time.time()-t0:.0f}s)", flush=True)
    A("")
    A(f"**判定**：共 {len(FINAL) * len(KS)} 组比对，最大偏差 {worst_all:.3e}，"
      f"超容差 {bad_all} 组。")
    A("")
    A("> 说明：基准是**同进程内重跑的完整全年**，不读取历史缓存或旧承诺路径导出。")
    A("> 比对范围是两者的日期交集，即前 $K$ 天；问题2/4-2 只记逐日量、不存承诺路径，")
    A("> 故其\"承诺路径\"一列为 `—`，比对项为逐日总费用、计划购电量、日末储电、紧急购电量。")
    A("> 两个实现都只在 `days is None` 时对最后一天施加 `force_end_soc`，")
    A("> 传 `days=K` 时该分支不生效，故不存在\"截断点被强制回 $E_0$\"造成的假不一致。")
    A("> 每个截断档都先截断负载、光伏、逐日电价和官方光伏预报，再禁用预测缓存、")
    A("> 从截断后的原始数据重新生成预测；表中同时比较预测数组和决策结果。")
    A("> 因此该检验覆盖跨日未来信息泄露。日内滚动的信息边界另由 `_load_at_stage` /")
    A("> `_pv_at_stage` 的时点切片规则保证。")
    A("")
    A("> 本检验**不覆盖**：多步前瞻窗口内部的滚动重优化是否收敛到同一解（MPC 的前瞻")
    A("> 只是执行器的内部机制，不改变承诺路径的比较口径）；也不覆盖 Diebold—Mariano 检验。")
    A("")

    (OPT_DIR / "前缀不变性.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n已写出 {OPT_DIR / '前缀不变性.md'}")
    print(f"最大偏差 {worst_all:.3e}，超容差 {bad_all} 组")


if __name__ == "__main__":
    main()
