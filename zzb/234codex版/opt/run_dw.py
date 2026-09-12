# -*- coding: utf-8 -*-
"""五(a)/五(b) 的 A/B 实跑：决策加权损失、电价–净负荷联合建模。

对每个问题构造四个预测层变体，**同一套决策模型、同一套执行器**下比较：

  ``orig``     上游预测层（``seasonal.generate_causal_forecasts`` / ``price_forecast``）
  ``replica``  本模块 ``dec_weight.generate_forecasts``，三个开关全关
  ``dw``       决策加权：价格权重 + 危险方向惩罚 $\\beta$
  ``joint``    仅问题4：电价集成额外接入当日净负荷预测（五b）

``replica`` 是**实现自检**：它与 ``orig`` 逐位一致才说明差异来自加权本身，
而不是我把上游算法重写错了。不一致就直接报错停下。

评价两套指标：
  · 普通 MAE / nMAE —— 上游口径，用于说明"加权没有把普通精度搞坏"；
  · 决策加权 nMAE   —— 高价与危险方向时段加权的误差，说明"钱花在了该花的地方"。

最后跑该问题的**最终档位**，给出预测层变体带来的全年费用差。

跑法：
  python run_dw.py --problem Q3
  python run_dw.py --problem Q2 --no-sim          # 只比预测精度
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

OPT_DIR = Path(__file__).resolve().parent
OUT_DIR = OPT_DIR / "out"
sys.path.insert(0, str(OPT_DIR))

BETA = 0.50            # 危险方向惩罚系数
NULL = {"use_price_weight": False, "beta": 0.0}
STAGE_PROBLEMS = ("Q3", "Q4-3")
# 电价为未知量的问题（决策权重取前一日已实现曲线）
PRICE_UNKNOWN = ("Q4-2", "Q4-3")
# 电价预测层可以重算的问题：只有问题4-2 有 price_forecast.py；
# 问题4-3 的决策电价通道由 gen_price_channel.py 预先冻结，本 A/B 不改它。
REGEN_PRICE = ("Q4-2",)


def purge(names) -> None:
    for m in names:
        sys.modules.pop(m, None)


# --------------------------------------------------------------------------
def build_variants(problem: str, data) -> tuple[dict, np.ndarray]:
    """返回 ``(变体字典, 逐日决策电价曲线)``。

    变体字典的键与各 runner 的预测层 bundle 一致（问题3/4-3 为
    ``load/pv_own/pv_fused/...``，问题2/4-2 为 ``load/pv/price``）。
    """
    import dec_weight as dw

    cfg = _forecast_config(problem)
    price_bd = dw.price_by_day(data, use_forecast_price=problem in PRICE_UNKNOWN)
    projected = _project(data)
    base = _base_module()

    variants: dict = {}
    if problem in STAGE_PROBLEMS:
        import run_stage as rs
        _, _, forecasts = rs.load_stack(problem)

        def bundle(fc, price=None):
            fusion = forecasts.build_pv_fusion(data, fc["pv"], verbose=False)
            out = {"load": fc["load"], "pv_own": fc["pv"],
                   "pv_official": fusion["pv_official"],
                   "pv_fused": fusion["pv_fused"],
                   "fusion_diagnostics": fusion["diagnostics"],
                   "own_diagnostics": None}
            if price is not None:
                out["price"] = price
            return out

        t0 = time.time()
        variants["orig"] = forecasts.forecast_bundle(data, use_cache=False, verbose=False)
        if problem in PRICE_UNKNOWN:
            variants["orig"]["price"] = _upstream_price_channel(problem)
        print(f"  [orig] 上游预测层 ({time.time()-t0:.0f}s)", flush=True)

        diag0 = variants["orig"].get("own_diagnostics")
    else:
        import run_single as rg
        db, _, _, gen_fc = rg.load_stack(problem)

        def bundle(fc, price=None):
            p = data.price if price is None else price
            return {"load": fc["load"], "pv": fc["pv"], "price": p}

        t0 = time.time()
        ufc = gen_fc(projected, cfg, verbose=False)
        variants["orig"] = {"load": ufc["load"], "pv": ufc["pv"],
                            "price": fc_price(problem, data, ufc, cfg)}
        print(f"  [orig] 上游预测层 ({time.time()-t0:.0f}s)", flush=True)
        diag0 = ufc["diagnostics"]
    del base

    t0 = time.time()
    rep = dw.generate_forecasts(projected, cfg, price_by_day=price_bd, **NULL)
    variants["replica"] = bundle(rep, fc_price(problem, data, rep, cfg)
                                 if problem in REGEN_PRICE else None)
    _assert_same(problem, "replica", variants["orig"], variants["replica"])
    _assert_diag(problem, "replica", diag0, rep["diagnostics"])
    print(f"  [replica] 开关全关，与上游逐位一致 ✓ ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    dwn = dw.generate_forecasts(projected, cfg, price_by_day=price_bd,
                                use_price_weight=True, beta=BETA)
    dprice = (dw.generate_price_forecast(data, dwn["diagnostics"], cfg,
                                         price_by_day=price_bd,
                                         use_price_weight=True, beta=BETA)["price"]
              if problem in REGEN_PRICE else None)
    variants["dw"] = bundle(dwn, dprice)
    print(f"  [dw] 决策加权 ({time.time()-t0:.0f}s)", flush=True)

    if problem in REGEN_PRICE:
        # 五(b)：电价集成额外接入当日净负荷预测（用自有预测，因果）
        net = dwn["net"]
        t0 = time.time()
        pj = dw.generate_price_forecast(
            data, dwn["diagnostics"], cfg, price_by_day=price_bd,
            use_price_weight=True, beta=BETA,
            extra_feature=lambda d: net[d])["price"]
        jn = dw.generate_forecasts(projected, cfg, price_by_day=price_bd,
                                   use_price_weight=True, beta=BETA)
        variants["joint"] = bundle(jn, pj)
        print(f"  [joint] 电价–净负荷联合 ({time.time()-t0:.0f}s)", flush=True)

    return variants, price_bd


# --------------------------------------------------------------------------
def _base_module():
    import deterministic_baseline as base
    return base


def _project(data):
    """投影成上游 ``deterministic_baseline`` 认识的 DataBundle（多问题共用）。

    问题4 目录的 DataBundle 多一个必填的 ``initial_price``，按需带上。
    """
    kw = {}
    init = getattr(data, "initial_price", None)
    if init is not None:
        kw["initial_price"] = np.array(init, dtype=float, copy=True)
    return _base_module().DataBundle(
        dates=data.dates, load_kw=data.load_kw, pv_kw=data.pv_kw,
        price=data.price, initial_load=data.initial_load,
        initial_pv=data.initial_pv, **kw)


def fc_price(problem, data, fc, cfg):
    """问题4-2 的上游电价因果预测（其余问题直接返回固定曲线）。"""
    if problem not in REGEN_PRICE:
        return data.price
    from price_forecast import generate_causal_price_forecast
    return generate_causal_price_forecast(data, fc["diagnostics"], cfg,
                                          verbose=False)["price"]


def _price_actual(problem: str, data):
    """电价真值通道 ``(n_days, 144)``；固定电价（问题2/3）返回 ``None`` 即不评。"""
    if problem == "Q4-3":
        import common
        actual, _ = common.load_price_channel()
        return np.asarray(actual, dtype=float)
    p = np.asarray(data.price)
    return p if p.ndim == 2 else None


def _upstream_price_channel(problem: str) -> np.ndarray:
    """问题4-3 上游已冻结的决策电价通道（由 ``gen_price_channel.py`` 生成）。"""
    import run_stage as rs
    _, _, _ = rs.load_stack(problem)
    import common
    _, fc = common.load_price_channel()
    return np.asarray(fc, dtype=float)


def _assert_diag(problem: str, tag: str, a, b) -> None:
    """季节应力诊断也必须逐位一致，否则说明滚动状态被改动了。"""
    if a is None or b is None:
        raise AssertionError(f"{problem}/{tag}: 诊断表缺失")
    x = a.sort_values("day_index")["season_stress"].to_numpy(float)
    y = b.sort_values("day_index")["season_stress"].to_numpy(float)
    dev = float(np.max(np.abs(x - y)))
    if dev > 1e-12:
        raise AssertionError(f"{problem}/{tag}: 季节应力最大偏差 {dev:.3e}")


def _assert_same(problem: str, tag: str, a: dict, b: dict) -> None:
    for key in ("load", "pv", "pv_fused", "price"):
        if key not in a or key not in b:
            continue
        x, y = np.asarray(a[key], float), np.asarray(b[key], float)
        if x.shape != y.shape:
            raise AssertionError(f"{problem}/{tag}: {key} 形状 {x.shape} != {y.shape}")
        both_nan = np.isnan(x) & np.isnan(y)
        dev = float(np.max(np.abs(np.nan_to_num(x - y, nan=0.0))[
            ~both_nan])) if (~both_nan).any() else 0.0
        if dev > 1e-9:
            raise AssertionError(
                f"{problem}/{tag}: {key} 与上游最大偏差 {dev:.3e}，"
                f"说明重写实现有误，A/B 结论不成立")


# --------------------------------------------------------------------------
def simulate_final(problem: str, data, bundle, *, days=None) -> dict:
    """在该预测层下跑最终档位，返回全年费用汇总。"""
    import opt_core as oc
    if problem in STAGE_PROBLEMS:
        import run_stage as rs
        common, dispatch_core, forecasts = rs.load_stack(problem)
        n = len(data.dates) if days is None else days
        price_s, price_d = rs.price_channel(problem, common, data, n)
        if bundle.get("price") is not None:
            # 问题4-3 的决策电价本身也是预测量：变体给的是自己的因果电价预测，
            # 结算价仍用实际电价通道。
            price_d = np.asarray(bundle["price"], dtype=float)[:n]
        res = rs.reserves_for(data, bundle, dispatch_core, n)
        daily, _, _ = rs.simulate(
            data, bundle, dispatch_core, settle="correct", mpc=False,
            reserves=res, price_s=price_s, price_d=price_d, days=days, exact=True)
    else:
        import run_single as rg
        db, _, _, _ = rg.load_stack(problem)
        daily, _ = rg.simulate(db, data, bundle, mode="reserve-mpc", days=days)
    costs = daily.total_cost_yuan.to_numpy(float)
    cut = float(np.quantile(costs, 0.95))
    return {
        "total_cost_yuan": float(costs.sum()),
        "emergency_kwh": float(daily.emergency_kwh.sum()),
        "planned_purchase_kwh": float(daily.planned_purchase_kwh.sum()),
        "final_soc_kwh": float(daily.end_soc_kwh.iloc[-1]),
        "cvar95_yuan": float(costs[costs >= cut].mean()),
        "days": int(len(daily)),
    }


def evaluate(problem: str, data, variants: dict, price_bd) -> dict:
    """预测精度指标：普通口径 + 决策加权口径。"""
    import dec_weight as dw
    base = _base_module()
    n = len(data.dates)
    mask = np.asarray(data.dates >= base.FORMAL_START)
    # 逐日取价格权重：固定电价（问题2/3）每天同一条曲线，逐日电价（问题4）
    # 每天不同，必须按天归一，不能用全年均值 `price_weight(price_bd)`。
    w = np.stack([dw.price_weight(price_bd[d]) for d in range(n)])
    # 电价真值：问题4-3 的 ``data.price`` 仍是附件1 固定曲线，实际逐日电价
    # 在 ``price_channel.npz`` 里。
    price_actual = _price_actual(problem, data)
    out = {}
    for tag, b in variants.items():
        row = {}
        for ch, actual in (("load", data.load_kw), ("pv", data.pv_kw)):
            # 问题3/4-3 的预测层 bundle 里光伏分 ``pv_own``（自有预测）与
            # ``pv_fused``（融合后，实际喂给决策的那一路），逐个通道都评。
            keys = [ch] if ch in b else [k for k in b if k.startswith(ch + "_")]
            for key in keys:
                pred = np.asarray(b[key], dtype=float)
                # 融合通道是逐决策时刻的 (天, 阶段, 时段)；取第 0 阶段
                # （当天 0:00 那一次）作为"日前"预报与真值同轴比较。
                if pred.ndim == 3 and pred.shape[1] == 4:
                    pred = pred[:, 0, :]
                if pred.shape != np.shape(actual):
                    continue
                row[key] = dw.accuracy(actual, pred, mask, weight=w)
        if "price" in b and np.asarray(b["price"]).ndim == 2:
            row["price"] = dw.accuracy(
                price_actual, np.asarray(b["price"])[:n], mask, weight=w)
        out[tag] = row
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["Q2", "Q3", "Q4-2", "Q4-3"], required=True)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--no-sim", action="store_true", help="只比预测精度，不跑仿真")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    purge(("common", "dispatch_core", "forecasts", "models",
           "deterministic_baseline", "seasonal", "price_forecast",
           "dec_weight", "opt_core", "joint_stage"))
    import dec_weight  # noqa: F401  提前确认路径
    problem = args.problem

    data = _load(problem)
    t0 = time.time()
    variants, price_bd = build_variants(problem, data)
    print(f"  [{problem}] 预测层变体就绪 ({time.time()-t0:.0f}s)", flush=True)

    metrics = evaluate(problem, data, variants, price_bd)
    print(f"\n  [{problem}] 正式期预测精度")
    print(f"  {'变体':8s} {'通道':6s} {'MAE':>10s} {'nMAE%':>8s} {'决策加权nMAE%':>14s}")
    for tag, row in metrics.items():
        for ch, m in row.items():
            print(f"  {tag:8s} {ch:6s} {m['mae']:10.3f} {100*m['nmae']:8.3f} "
                  f"{100*m.get('dw_nmae', float('nan')):14.3f}", flush=True)

    payload = {"problem": problem, "beta": BETA, "metrics": metrics}
    if not args.no_sim:
        sims = {}
        for tag, b in variants.items():
            t = time.time()
            sims[tag] = simulate_final(problem, data, b, days=args.days)
            sims[tag]["seconds"] = round(time.time() - t, 1)
            print(f"  [{problem}] 最终档位 @ {tag:8s} 全年 "
                  f"{sims[tag]['total_cost_yuan']:>15,.2f} 元  "
                  f"紧急 {sims[tag]['emergency_kwh']:>10,.1f} kWh  "
                  f"({sims[tag]['seconds']}s)", flush=True)
        ref = sims["orig"]["total_cost_yuan"]
        for tag, s in sims.items():
            s["相对orig_yuan"] = s["total_cost_yuan"] - ref
            s["相对orig_%"] = 100 * s["相对orig_yuan"] / ref
        payload["sim"] = sims

    (OUT_DIR / f"{problem}_预测层AB.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {OUT_DIR / f'{problem}_预测层AB.json'}")


def _load(problem: str):
    if problem in STAGE_PROBLEMS:
        import run_stage as rs
        common, _, _ = rs.load_stack(problem)
        return common.load_inputs()
    import run_single as rg
    db, _, _, _ = rg.load_stack(problem)
    return db.load_inputs()


def _forecast_config(problem: str):
    if problem in STAGE_PROBLEMS:
        from seasonal import ForecastConfig
        return ForecastConfig()
    import run_single as rg
    db, CC, FC, _ = rg.load_stack(problem)
    return FC(correction=CC(use_pv_window=True, use_level=True,
                            use_weekly_effect=False))


if __name__ == "__main__":
    main()
