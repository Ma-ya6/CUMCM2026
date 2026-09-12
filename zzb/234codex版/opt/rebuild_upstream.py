# -*- coding: utf-8 -*-
"""用**修正后的预测层**重算问题3 / 问题4-3 的上游基线（M9，q=0.80）。

背景：前缀不变性检验发现上游 ``forecasts.py:_pooled_coef`` 用全年实测光伏
识别昼夜列，属跨日未来信息泄露；已改为只用决策日之前的历史。上游自身的
``results/最终冻结结果.json`` 是泄露版预测层跑出来的，与本次各档位不再同口径。

本脚本 import 上游 ``models.StorageReserveLP`` 与 ``dispatch_core.simulate_year``
重跑一遍基线，**只读上游代码、不覆盖上游 ``results/``**，结果写进本目录。

跑法：``python rebuild_upstream.py``，输出
``opt/out/{Q3,Q4-3}_上游重算.json`` 与 ``..._逐日.csv``。
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

Q = 0.80
DIRS = {"Q3": OPT_DIR.parent.parent / "问题3" / "模型3决策优化版",
        "Q4-3": OPT_DIR.parent.parent / "问题4" / "重算问题3"}


def run_upstream(problem: str):
    """跑上游 M9（q=0.80），返回逐日 DataFrame。"""
    for m in ("common", "dispatch_core", "forecasts", "models"):
        sys.modules.pop(m, None)
    root = DIRS[problem]
    sys.path.insert(0, str(root))
    import common                                        # noqa: E402
    import dispatch_core                                 # noqa: E402
    import forecasts                                     # noqa: E402
    from models import StorageReserveLP                  # noqa: E402

    data = common.load_inputs()
    # 冻结结果必须由当前源码从原始数据重建，不能悄悄复用历史预测缓存。
    bundle = forecasts.forecast_bundle(data, use_cache=False, verbose=False)
    model = StorageReserveLP()
    model.QUANTILE = Q
    # 问题4-3 的电价与负载/光伏同为不可预知量：结算用附件4 实际电价，
    # 决策用第 d 天 0:00 可得的因果预测电价。问题3 为固定电价，两参数为 None。
    kw = {}
    if problem == "Q4-3":
        price_day, price_fc = common.load_price_channel()
        kw = {"price_day": price_day, "price_fc": price_fc}
    return dispatch_core.simulate_year(data, bundle, model, **kw)


def fields(daily) -> dict:
    """与 ``verify/out/表_年末口径统一.csv`` 同口径的一行。"""
    costs = daily.total_cost_yuan.to_numpy(float)
    cut = float(np.quantile(costs, 0.95))
    return {
        "原始总费用/元": float(costs.sum()),
        "实际年末SOC/kWh": float(daily.end_soc_kwh.iloc[-1]),
        "计划购电量/kWh": float(daily.planned_purchase_kwh.sum()),
        "紧急购电量/kWh": float(daily.emergency_kwh.sum()),
        "紧急天数/天": int((daily.emergency_kwh > 1e-8).sum()),
        "未用计划电量/kWh": float(daily.unused_plan_kwh.sum()),
        "单日费用CVaR95/元": float(costs[costs >= cut].mean()),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=list(DIRS), default=None)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out = {}
    for p in ([args.problem] if args.problem else list(DIRS)):
        t0 = time.time()
        daily = run_upstream(p)
        f = fields(daily)
        old = json.loads((DIRS[p] / "results" / "最终冻结结果.json")
                         .read_text(encoding="utf-8"))["annual"]
        f["旧基线-原始总费用/元"] = old["total_cost_yuan"]
        f["旧基线-实际年末SOC/kWh"] = old["final_soc_kwh"]
        f["费用变化/元"] = f["原始总费用/元"] - old["total_cost_yuan"]
        f["费用变化/%"] = f["费用变化/元"] / old["total_cost_yuan"] * 100
        out[p] = f
        daily.to_csv(OUT_DIR / f"{p}_上游重算_逐日.csv",
                     index=False, encoding="utf-8-sig")
        print(f"[{p}] 重算 {f['原始总费用/元']:,.2f} 元  "
              f"旧 {old['total_cost_yuan']:,.2f} 元  "
              f"差 {f['费用变化/元']:+,.2f} ({f['费用变化/%']:+.4f}%)  "
              f"({time.time()-t0:.0f}s)", flush=True)

    (OUT_DIR / "上游重算.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {OUT_DIR / '上游重算.json'}")


if __name__ == "__main__":
    main()
