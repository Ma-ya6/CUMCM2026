"""生成波动电价通道并缓存，供重算问题3 使用。

问题4 中电价与负载、光伏同为不可预知量。本脚本**完全沿用 问题4/重算问题2
的电价预测实现**（同一 ``price_forecast``、同一 ``seasonal`` 集成、
同一 ``CorrectionConfig``），不重写、不改参，只把结果缓存下来供问题3 的四时点
滚动决策使用。

之所以单独开一个进程：本目录与 ``问题4/重算问题2`` 各自持有一份名为
``deterministic_baseline`` / ``seasonal`` 的模块（两者是不同的分支），
在同一进程里按名字导入会互相覆盖。电价预测本身与问题3 的决策模型完全解耦，
因此最干净的做法就是在重算问题2 的模块环境下跑一次、把数组落盘。

输出（写入 重算问题3/cache/）：

- ``price_channel.npz``：``price_actual``（365×144，附件4 实际电价）与
  ``price_fc``（365×144，第 d 天 0:00 可用的因果预测电价）；
- ``price_channel_metrics.json``：正式期预测精度与冷启动说明；
- ``price_forecast_weights.csv``：正式期逐日的三模型集成权重。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
Q4_Q2_DIR = HERE.parents[1] / "问题4-2" / "deps"
CACHE_DIR = HERE / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

if str(Q4_Q2_DIR) not in sys.path:
    sys.path.insert(0, str(Q4_Q2_DIR))

import deterministic_baseline as db4  # noqa: E402
from price_forecast import generate_causal_price_forecast  # noqa: E402
from seasonal import (  # noqa: E402
    CorrectionConfig,
    ForecastConfig,
    generate_causal_forecasts,
)


def main() -> None:
    data = db4.load_inputs()
    # 与 final_model_q4_2.py 逐字一致，保证电价通道与重算问题2 可比。
    correction = CorrectionConfig(
        use_pv_window=True,
        use_level=True,
        use_weekly_effect=False,
    )
    config = ForecastConfig(correction=correction)

    print("重算问题2 的负载/光伏因果预测（提供季节应力）...", flush=True)
    forecasts = generate_causal_forecasts(data, config, verbose=True)
    print("因果电价预测...", flush=True)
    price_result = generate_causal_price_forecast(
        data, forecasts["diagnostics"], config, verbose=True
    )

    price_fc = np.asarray(price_result["price"], dtype=float)
    price_actual = np.asarray(data.price, dtype=float)
    if price_fc.shape != price_actual.shape:
        raise ValueError(f"电价预测形状 {price_fc.shape} 与实际 {price_actual.shape} 不一致")

    np.savez(
        CACHE_DIR / "price_channel.npz",
        price_actual=price_actual,
        price_fc=price_fc,
    )

    mask = np.asarray(data.dates >= db4.FORMAL_START)
    metrics = {
        **price_result["metrics"],
        "formal_start": str(db4.FORMAL_START.date()),
        "n_days_total": int(len(data.dates)),
        "n_days_formal": int(mask.sum()),
        "cold_start": "第0天（2025-01-01）用附件1的单条电价曲线冷启动",
        "namespace": "问题4/重算问题2 的 price_forecast.generate_causal_price_forecast",
        "mean_actual_formal_yuan_per_kwh": float(price_actual[mask].mean()),
        "mean_forecast_formal_yuan_per_kwh": float(price_fc[mask].mean()),
    }
    (CACHE_DIR / "price_channel_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows = []
    for day, weights in enumerate(price_result["weights"]):
        if data.dates[day] >= db4.FORMAL_START:
            rows.append({"date": data.dates[day], **weights})
    pd.DataFrame(rows).fillna(0.0).to_csv(
        CACHE_DIR / "price_forecast_weights.csv", index=False, encoding="utf-8-sig"
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"已写出 {CACHE_DIR / 'price_channel.npz'}")


if __name__ == "__main__":
    main()
