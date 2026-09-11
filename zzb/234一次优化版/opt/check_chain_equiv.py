# -*- coding: utf-8 -*-
"""二（续）：M2 ≡ M6 为什么在上游成立，而 ``opt`` 的 fixed/exact 却分岔。

``rerun_models.py`` 已确认：在**修正后的预测层**上 M2 与 M6 仍然逐位相同
（差 $-1.9\\times10^{-9}$ 元），且两者的"链式减两元差额"都 $\approx 0$。
所以等价性与预测层无关，差异只可能来自**执行器/储备口径**。

假说：上游的锚定凸化之所以"精确"，是因为它的分位余量足够大，使承诺序列
**全程单调非降**；此时链式结算目标

$$ 0.5c\\,\\Delta^- + 1.5c\\,\\Delta^+ \\quad(\\Delta^\\pm=\\max(\\pm(r-x),0)) $$

的拐点 $x=r$ 从不被跨越，分段线性函数在实际可行域上退化为线性，
凸化自然无损。``opt`` 的储能储备 $R_d$（均值 1355.6 kWh）比上游的逐时段
分位余量紧，承诺路径出现 5,272 个**反转时段**（先降后升），拐点被激活，
于是 exact 比 fixed 便宜约 2.71 万元。

本脚本直接量它：用 `dispatch_core.simulate_year` 的 ``slot_sink`` 钩子取
M2/M6 的逐阶段承诺，统计反转时段数。若为 0，假说成立。

跑法：``python check_chain_equiv.py``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OPT_DIR = Path(__file__).resolve().parent
C_DIR = OPT_DIR.parent.parent
OUT_DIR = OPT_DIR / "out"
sys.path.insert(0, str(OPT_DIR))

ROOT = C_DIR / "问题3" / "模型3决策优化版"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for m in ("common", "dispatch_core", "forecasts", "models",
              "deterministic_baseline", "seasonal"):
        sys.modules.pop(m, None)
    sys.path.insert(0, str(ROOT))
    import common                                        # noqa: E402
    import dispatch_core                                 # noqa: E402
    import forecasts                                     # noqa: E402
    from models import all_models                        # noqa: E402

    data = common.load_inputs()
    bundle = forecasts.forecast_bundle(data, use_cache=False, verbose=False)
    picked = {m.code: m for m in all_models() if m.code in ("M2", "M6")}

    report = {}
    for code, model in picked.items():
        per_day: list[np.ndarray] = []

        def sink(*, commitment, _p=per_day, **__):
            _p.append(np.asarray(commitment, dtype=float))

        daily = dispatch_core.simulate_year(data, bundle, model, slot_sink=sink)
        if not per_day:
            report[code] = {"错误": "未取到承诺路径"}
            continue

        rev_slots = 0
        rev_days = 0
        down_slots = 0
        for path in per_day:
            steps = np.diff(path, axis=0)
            has_down = (steps < -1e-9).any(axis=0)
            has_up = (steps > 1e-9).any(axis=0)
            rev_slots += int((has_down & has_up).sum())
            rev_days += int((has_down & has_up).any())
            down_slots += int(has_down.sum())
        n_day, _, n_slot = np.array(per_day).shape
        report[code] = {
            "天数": int(n_day),
            "全年费用/元": float(daily.total_cost_yuan.sum()),
            "单调非降天数": int(n_day - rev_days),
            "含反转天数": int(rev_days),
            "反转时段数": int(rev_slots),
            "总时段数": int(n_day * n_slot),
            "反转时段占比": float(rev_slots / (n_day * n_slot)),
            "含下调时段数": int(down_slots),
        }
        r = report[code]
        print(f"  {code}  全年 {r['全年费用/元']:>15,.2f}  "
              f"反转时段 {r['反转时段数']:>7,} / {r['总时段数']:,} "
              f"({r['反转时段占比']*100:.4f}%)  "
              f"含反转天数 {r['含反转天数']}", flush=True)

    (OUT_DIR / "Q3_链式等价性诊断.json").write_text(
        json.dumps({"说明": "M2/M6 承诺路径单调性诊断（修正后预测层）",
                    "上游": report},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {OUT_DIR / 'Q3_链式等价性诊断.json'}")


if __name__ == "__main__":
    main()
