# -*- coding: utf-8 -*-
"""模型解析核对（一）：调整购电量的费用方向与链式结算。

对应 `模型评价建议.md` §六.1（单时段手算单元测试）与 §六.2（链式结算路径测试）。

题面规则（C题.md）：相对上一版承诺 r，新版 x
    x <= r : 边际 0.5c
    x >  r : 边际 1.5c
等价闭式： C(x) = 0.5c*x + 1.0c*(x-r)+ + 0.5c*r

本脚本并列三处实现：手算闭式 / 结算函数 settle_slot / 优化目标 solve_scenario_lp，
并额外做一次**受控 LP 探针**：构造一个两种口径给出不同最优解的算例，直接看
真实函数选哪个解。

只读，不修改上游任何文件。
"""
from __future__ import annotations

import json
import sys

import numpy as np

from paths import OUT, Q3

sys.path.insert(0, str(Q3))
from dispatch_core import settle_slot  # noqa: E402
from models import STEP_MAX, solve_scenario_lp  # noqa: E402

BREACH, OVERBUY = 0.5, 1.5


def hand_single(x: float, r: float, c: float = 1.0) -> float:
    """闭式：0.5c*x + 1.0c*(x-r)+ + 0.5c*r"""
    return BREACH * c * x + (OVERBUY - BREACH) * c * max(x - r, 0.0) + BREACH * c * r


def hand_chain(path, c: float = 1.0) -> dict:
    """链式手算：基准量取路径最小值；每次相邻变动分别计下调 0.5c / 上调 1.5c。"""
    base_qty = min(path)
    down = up = 0.0
    for prev, cur in zip(path[:-1], path[1:]):
        down += max(prev - cur, 0.0)
        up += max(cur - prev, 0.0)
    return {
        "base_yuan": c * base_qty,
        "breach_yuan": BREACH * c * down,
        "overbuy_yuan": OVERBUY * c * up,
        "total_yuan": c * base_qty + BREACH * c * down + OVERBUY * c * up,
        "down_kwh": down, "up_kwh": up,
    }


def lp_obj_correct(x: float, r: float, c: float = 1.0) -> float:
    """正确口径（去掉常数项 0.5c*r）：0.5c*x + 1.0c*(x-r)+"""
    return BREACH * c * x + (OVERBUY - BREACH) * c * max(x - r, 0.0)


def lp_obj_asis(x: float, r: float, c: float = 1.0) -> float:
    """复刻 models.solve_scenario_lp 在 ref is not None 分支的目标口径。

    该函数内约束为 eb - eo = r - x，故 eb = (r-x)+（下调量）、eo = (x-r)+（上调量）；
    目标为 0.5c*x + (OVERBUY-BREACH)*c*eb，即 1.0c 的附加费记在**下调量**上。
    """
    eb = max(r - x, 0.0)
    return BREACH * c * x + (OVERBUY - BREACH) * c * eb


def lp_probe() -> dict:
    """受控 LP 探针：两种口径给出不同最优解，直接看真实函数选哪个。

    构造：2 个时段，净负荷 0，上一版承诺均为 100 kWh，电价 1 元/kWh，SOC 起止均 6000。
    储蓄电池可充放（允许同充放，惩罚仅 1e-7），故 x=0 与 x=100 都可行。
      · 正确口径：φ = 0.5(x0+x1) + (x0-100)+ + (x1-100)+   → 最优 x = (0, 0)
      · 现行口径：φ = 0.5(x0+x1) + (100-x0)+ + (100-x1)+   → 最优 x = (100, 100)
    """
    x = solve_scenario_lp(
        np.zeros(2), np.zeros(2), np.ones(2), 6000.0,
        np.array([100.0, 100.0]), np.zeros((1, 2)), np.ones(1),
        emergency=False, force_end_soc=6000.0,
    )
    return {
        "x": [float(v) for v in x],
        "正确口径预测": [0.0, 0.0],
        "现行口径预测": [100.0, 100.0],
        "命中": "现行口径（方向已反）" if float(x.sum()) > 50 else "正确口径",
        "STEP_MAX": float(STEP_MAX),
    }


def main() -> dict:
    c = 1.0
    out: dict = {}

    # ---------- §六.1 单时段手算 ----------
    r = 100.0
    rows = []
    for x in (80.0, 120.0):
        got = settle_slot([r, x], c, "chain")
        rows.append({
            "x": x,
            "手算结算": hand_single(x, r, c),
            "settle_slot": got["total_yuan"],
            "现行LP目标": lp_obj_asis(x, r, c),
            "正确LP目标": lp_obj_correct(x, r, c) + BREACH * c * r,
        })
    out["single_slot"] = rows

    # ---------- §六.2 链式结算路径 ----------
    path = [100.0, 120.0, 110.0, 130.0]
    h = hand_chain(path, c)
    code = settle_slot(path, c, "chain")
    two = settle_slot(path, c, "two_way")
    out["chain_path"] = {
        "path": path, "手算": h,
        "settle_chain": {k: code[k] for k in
                         ("base_yuan", "breach_yuan", "overbuy_yuan", "total_yuan", "adjusted_kwh")},
        "settle_two_way": {k: two[k] for k in
                           ("base_yuan", "breach_yuan", "overbuy_yuan", "total_yuan", "adjusted_kwh")},
    }

    # ---------- 受控 LP 探针 ----------
    out["lp_probe"] = lp_probe()

    out["verdict"] = {
        "settle_slot_方向正确": all(abs(x["手算结算"] - x["settle_slot"]) < 1e-9 for x in rows),
        "现行LP目标_方向正确": all(abs(x["手算结算"] - x["现行LP目标"]) < 1e-9 for x in rows),
        "链式settle_与手算一致": abs(h["total_yuan"] - code["total_yuan"]) < 1e-9,
        "链式能识别重复调整": code["total_yuan"] > two["total_yuan"] + 1e-9,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "核对_调整费用方向.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 打印 ----------
    print("=" * 80)
    print("§六.1 单时段手算单元测试   r = 100 kWh, c = 1.0 元/kWh")
    print("=" * 80)
    print(f"{'x':>5} | {'手算':>8} | {'settle_slot':>11} | {'现行LP目标':>10} | {'正确LP目标':>10}")
    for x in rows:
        print(f"{x['x']:>5.0f} | {x['手算结算']:>8.1f} | {x['settle_slot']:>11.1f} "
              f"| {x['现行LP目标']:>10.1f} | {x['正确LP目标']:>10.1f}")

    print("\n" + "=" * 80)
    print("§六.2 链式结算路径测试   100 → 120 → 110 → 130")
    print("=" * 80)
    print(f"  手算 chain      : {h['base_yuan']:>7.1f} + 下调 {h['breach_yuan']:>7.1f} "
          f"+ 上调 {h['overbuy_yuan']:>7.1f} = {h['total_yuan']:>8.1f}")
    print(f"  settle(chain)   : {code['base_yuan']:>7.1f} + 下调 {code['breach_yuan']:>7.1f} "
          f"+ 上调 {code['overbuy_yuan']:>7.1f} = {code['total_yuan']:>8.1f}")
    print(f"  settle(two_way) : {two['base_yuan']:>7.1f} + 下调 {two['breach_yuan']:>7.1f} "
          f"+ 上调 {two['overbuy_yuan']:>7.1f} = {two['total_yuan']:>8.1f}")
    print(f"  链式相对两元的重复调整成本 = {code['total_yuan'] - two['total_yuan']:.1f} 元")

    print("\n" + "=" * 80)
    print("受控 LP 探针（净负荷 0，承诺 r=100，SOC 起止 6000，x 自由）")
    print("=" * 80)
    p = out["lp_probe"]
    print(f"  solve_scenario_lp 返回 x = {p['x']}")
    print(f"  正确口径预测 x = {p['正确口径预测']}，现行口径预测 x = {p['现行口径预测']}")
    print(f"  → 实际命中：{p['命中']}")

    print("\n" + "=" * 80)
    print("判定")
    print("=" * 80)
    for k, v in out["verdict"].items():
        print(f"  {k:26s} : {'通过' if v else '★不通过'}")
    return out


if __name__ == "__main__":
    main()
