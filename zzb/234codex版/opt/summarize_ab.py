# -*- coding: utf-8 -*-
"""汇总决策加权/联合预测 A/B，作为是否替换最终预测层的消融证据。"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def main() -> None:
    rows = []
    for problem in ("Q2", "Q3", "Q4-2", "Q4-3"):
        path = OUT / f"{problem}_预测层AB.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        sim = data.get("sim", {})
        orig = sim.get("orig")
        replica = sim.get("replica")
        if not orig or not replica:
            continue
        if abs(replica["total_cost_yuan"] - orig["total_cost_yuan"]) > 1e-6:
            raise RuntimeError(f"{problem} replica 未逐位复现 orig")
        for variant in ("dw", "joint"):
            if variant not in sim:
                continue
            v = sim[variant]
            rows.append((problem, variant,
                         (v["total_cost_yuan"] - orig["total_cost_yuan"]) / 1e4,
                         v["emergency_kwh"] - orig["emergency_kwh"],
                         (v["cvar95_yuan"] - orig["cvar95_yuan"]) / 1e4))

    lines = ["# 预测层增强消融", "",
             "> `orig` 与独立重写的 `replica` 费用逐位一致，先证明 A/B 实现口径一致。",
             "> `dw` 为决策加权损失，`joint` 为问题4可用的电价—净负荷联合特征。",
             "", "| 问题 | 变体 | 年费用变化/万元 | 紧急购电变化/kWh | 日费用CVaR95变化/万元 | 是否替换最终预测层 |",
             "|---|---|---:|---:|---:|---|"]
    for p, v, dc, de, dr in rows:
        adopt = "否" if dc >= 0 else "候选"
        lines.append(f"| {p} | `{v}` | {dc:+.2f} | {de:+,.0f} | {dr:+.3f} | {adopt} |")
    lines += ["", "## 结论", "",
              "决策加权或联合特征可能降低紧急购电量/尾部费用，但若全年总费用上升，",
              "就不替换当前最终预测层。该实验是负向消融，不用于重新选择理论分位数 $q=0.80$。", ""]
    (HERE / "预测层增强消融.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"已写出 {HERE / '预测层增强消融.md'}，共 {len(rows)} 个变体")


if __name__ == "__main__":
    main()
