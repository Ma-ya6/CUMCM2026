"""波动电价下的理论最优（全知下界）及其在问题3 链式结构下的可达性。

问题3 的结算规则是链式的：同一时段的四次承诺按
$c_t\\min_kx^{(k)}+0.5c_t\\Delta^-+1.5c_t\\Delta^+$ 结算，缺口再加 $5c_t e_t$。
在**全知**条件下这个结构是松的——既然第 $d$ 天 0:00 就已经知道当天全部实际
负载、光伏与电价，把四次承诺取成同一个值 $x_t$ 即可令 $\\Delta^\\pm=0$，
费用退化为 $\\sum_t c_tx_t\\Delta t$，与四时点结构无关。因此

$$
\\text{理论最优}=\\min\\sum_t c_tx_t\\Delta t\\quad
\\text{s.t. 功率平衡与储能递推},
$$

**这个 LP 与问题2 的全知下界是同一个**，故问题4 中问题2 与问题3 的理论最优
必然相等。本脚本既求解该 LP，也**构造性地验证可达性**：把 LP 解出的每个时段
购电量原样作为四时点承诺下达，用真实执行器与链式结算跑一遍，检查实际费用是否
等于 LP 值。

为与 问题4/重算问题2 的口径逐位一致，LP 直接复用其 ``solve_full_year_oracle``
（按路径加载，不复制代码）。之所以单独开进程：本目录与 ``重算问题2`` 各持一份
同名但不同分支的 ``deterministic_baseline``，同进程按名导入会互相覆盖。

输出：``results/理论最优.json``。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
Q4_Q2_DIR = HERE.parents[1] / "问题4-2" / "deps"
RESULT_DIR = HERE / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

if str(Q4_Q2_DIR) not in sys.path:
    sys.path.insert(0, str(Q4_Q2_DIR))

import deterministic_baseline as db4  # noqa: E402


def _load_q42_oracle():
    """按文件路径加载 重算问题2 的 oracle_lower_bound（避免与本文件名冲突）。"""
    spec = importlib.util.spec_from_file_location(
        "q42_oracle_lower_bound", Q4_Q2_DIR / "oracle_lower_bound.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    oracle = _load_q42_oracle()
    data = db4.load_inputs()
    params = db4.StorageParameters()

    sol = oracle.solve_full_year_oracle(data, params)
    n_days = len(data.dates)
    purchase = sol["purchase_kwh"].reshape(n_days, db4.N_SLOT)
    price = np.asarray(data.price, dtype=float)
    formal = np.asarray(data.dates >= db4.FORMAL_START)

    lp_cost = float((price[formal] * purchase[formal]).sum())

    # ---- 可达性验证：四时点承诺全部取 LP 解，跑真实执行器 + 链式结算 ----
    energy = db4.E_INITIAL
    realized_plan = 0.0
    realized_emer = 0.0
    emer_kwh = 0.0
    for day in range(n_days):
        commitment = purchase[day]
        actual = db4.execute_actual(
            commitment, data.load_kw[day], data.pv_kw[day], energy, params
        )
        energy = actual["end_energy"]
        if not formal[day]:
            continue
        c = price[day]
        # 四时点承诺相同 → min_k x = x，Δ± = 0，链式结算退化为 c·x
        realized_plan += float(c @ commitment)
        # 重算问题2 把 5 倍写成字面量（deterministic_baseline.py:675），此处沿用
        realized_emer += float(5.0 * (c @ actual["emergency"]))
        emer_kwh += float(actual["emergency"].sum())

    realized = realized_plan + realized_emer

    # 与 重算问题2 的冻结值对照，确认复现
    frozen = json.loads(
        (Q4_Q2_DIR / "results" / "理论完美下界.json").read_text(encoding="utf-8")
    )
    frozen_oracle = float(frozen["oracle_full_year_joint"]["total_cost_yuan"])

    out = {
        "note": "全知条件下问题3 的链式结构是松的，故理论最优与问题2 相同",
        "price_channel": "附件4 实际电价（365×144）",
        "evaluation": "2025-02-01—2025-12-31，334 天 / 48,096 时段",
        "oracle_full_year_joint_yuan": lp_cost,
        "reproduces_q4_2_frozen": bool(abs(lp_cost - frozen_oracle) < 1e-6),
        "q4_2_frozen_yuan": frozen_oracle,
        "attainability": {
            "rule": "四时点承诺取同值，Δ±=0，链式结算退化为 Σ c·x",
            "realized_plan_cost_yuan": realized_plan,
            "realized_emergency_cost_yuan": realized_emer,
            "realized_total_yuan": realized,
            "emergency_kwh": emer_kwh,
            "gap_vs_lp_yuan": realized - lp_cost,
        },
        "purchase_kwh_formal": float(purchase[formal].sum()),
        "comparison_yuan": {
            "theoretical_optimum": lp_cost,
            "m9_realized": 14138543.962760434,
            "m9_minus_optimum": 14138543.962760434 - lp_cost,
            "m9_minus_optimum_pct": 100.0 * (14138543.962760434 - lp_cost) / lp_cost,
            "perfect_price_m9": 14072142.491091568,
            "perfect_price_minus_optimum": 14072142.491091568 - lp_cost,
        },
    }
    (RESULT_DIR / "理论最优.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
