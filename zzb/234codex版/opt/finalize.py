# -*- coding: utf-8 -*-
"""校验并同步三个最终模型文件夹，生成封版清单。

本脚本不重跑模型。应在 ``check_extra.py``、``check_pareto.py``、
``summarize_all.py`` 和 ``freeze.py`` 之后执行。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import pandas as pd

OPT = Path(__file__).resolve().parent
ROOT = OPT.parent
OUT = OPT / "out"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def check_prefix() -> None:
    text = (OPT / "前缀不变性.md").read_text(encoding="utf-8")
    m = re.search(r"共 (\d+) 组比对，最大偏差 ([\d.eE+-]+)，超容差 (\d+) 组", text)
    if not m or int(m.group(1)) != 16 or float(m.group(2)) != 0 or int(m.group(3)) != 0:
        raise RuntimeError("前缀不变性检验未形成 16 组全零偏差的完整证据")


def check_ladders() -> None:
    for problem, tag in (("Q2", "reserve-mpc"), ("Q3", "exact-greedy"),
                         ("Q4-2", "reserve-mpc"), ("Q4-3", "exact-greedy")):
        data = json.loads((OUT / f"{problem}_阶梯.json").read_text(encoding="utf-8"))
        row = data[tag]
        audit = row.get("audit", {})
        violations = sum(int(audit.get(k, 0)) for k in
                         ("soc_violations", "power_violations",
                          "simultaneous_charge_discharge", "negative_flow"))
        if violations:
            raise RuntimeError(f"{problem}/{tag} 存在 {violations} 个物理越界")
        if problem in ("Q3", "Q4-3") and abs(float(row["chain_misprice_yuan"])) > 1e-8:
            raise RuntimeError(f"{problem}/{tag} 链式目标残差非零")


def check_path(problem: str) -> None:
    p = pd.read_csv(OUT / f"{problem}_exact-greedy_承诺路径.csv")
    d = pd.read_csv(OUT / f"{problem}_exact-greedy_逐日.csv")
    sums = p.groupby("date", sort=False)["x0_kwh"].sum()
    daily = d.set_index("date")["planned_purchase_kwh"]
    common = sums.index.intersection(daily.index)
    if len(common) != len(daily):
        raise RuntimeError(f"{problem} 承诺路径与逐日表日期不完整")
    err = float((sums.loc[common] - daily.loc[common]).abs().max())
    if err > 1e-4:
        raise RuntimeError(f"{problem} 承诺路径导出列错位，最大偏差 {err}")


def main() -> None:
    check_prefix()
    check_ladders()
    check_path("Q3")
    check_path("Q4-3")

    mapping = {
        OUT / "Q2_阶梯.json": ROOT / "问题2最终模型" / "结果" / "Q2_阶梯.json",
        OUT / "Q2_reserve-mpc_逐日.csv": ROOT / "问题2最终模型" / "结果" / "Q2_reserve-mpc_逐日.csv",
        OUT / "Q3_阶梯.json": ROOT / "问题3最终模型" / "结果" / "Q3_阶梯.json",
        OUT / "Q3_exact-greedy_逐日.csv": ROOT / "问题3最终模型" / "结果" / "Q3_exact-greedy_逐日.csv",
        OUT / "Q3_exact-greedy_承诺路径.csv": ROOT / "问题3最终模型" / "结果" / "Q3_exact-greedy_承诺路径.csv",
        OUT / "Q3_上游重算_逐日.csv": ROOT / "问题3最终模型" / "结果" / "Q3_上游重算_逐日.csv",
        OUT / "Q4-2_阶梯.json": ROOT / "问题4最终模型" / "结果" / "Q4-2_阶梯.json",
        OUT / "Q4-2_reserve-mpc_逐日.csv": ROOT / "问题4最终模型" / "结果" / "Q4-2_reserve-mpc_逐日.csv",
        OUT / "Q4-3_阶梯.json": ROOT / "问题4最终模型" / "结果" / "Q4-3_阶梯.json",
        OUT / "Q4-3_exact-greedy_逐日.csv": ROOT / "问题4最终模型" / "结果" / "Q4-3_exact-greedy_逐日.csv",
        OUT / "Q4-3_exact-greedy_承诺路径.csv": ROOT / "问题4最终模型" / "结果" / "Q4-3_exact-greedy_承诺路径.csv",
        OUT / "Q4-3_上游重算_逐日.csv": ROOT / "问题4最终模型" / "结果" / "Q4-3_上游重算_逐日.csv",
    }
    for src, dst in mapping.items():
        copy(src, dst)

    authoritative = [
        ROOT / "README.md",
        OPT / "结果冻结.json", OPT / "结果冻结.md", OPT / "汇总对比.md",
        OPT / "补充检验.md", OPT / "可靠性Pareto.md", OPT / "前缀不变性.md",
        OPT / "exact验证.md", OPT / "模型对比重跑.md", OPT / "预测层增强消融.md",
        ROOT / "问题2最终模型" / "模型评价_问题2.md",
        ROOT / "问题3最终模型" / "模型评价_问题3.md",
        ROOT / "问题4最终模型" / "模型评价_问题4.md",
        *sorted(OPT.glob("*.py")),
        ROOT.parent / "问题3" / "模型3决策优化版" / "forecasts.py",
        ROOT.parent / "问题4" / "重算问题3" / "forecasts.py",
        *mapping.values(),
    ]
    manifest = {
        "说明": "这些文件共同构成问题2/3/4的最终封版；SHA-256用于检查复制与后续漂移。",
        "前缀不变性": "16组逐位一致，最大偏差0",
        "文件": [
            {"路径": os.path.relpath(p, ROOT), "字节": p.stat().st_size, "sha256": sha256(p)}
            for p in authoritative
        ],
    }
    (ROOT / "最终文件清单.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 封版核验", "",
             "- 前缀不变性：16 组逐位一致，最大偏差 0。",
             "- 最终档位：物理约束越界计数均为 0。",
             "- 问题3/4-3：exact 链式目标残差为 0。",
             "- 承诺路径：逐日 `x0_kwh` 合计与计划购电量一致，导出列未错位。",
             "- 三个最终模型文件夹已从 `opt/out` 同步。",
             "- 完整 SHA-256 见 `最终文件清单.json`。", ""]
    (ROOT / "封版核验.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"封版同步完成，共冻结 {len(authoritative)} 个文件")


if __name__ == "__main__":
    main()
