# -*- coding: utf-8 -*-
"""把所有文档引用的数字冻结到单一结果文件，避免文档与 ``out/*.json`` 继续漂移。

读：``opt/out/*.json``（各档位实跑）、``verify/out/表_年末口径统一.csv``（上游冻结值）、
``opt/exact验证.md``（exact 分支独立检验）、``opt/补充检验.md``（Bootstrap / TOST）。
写：``opt/结果冻结.json``（机器可读）与 ``opt/结果冻结.md``（人读，每条附来源）。

**本脚本不重跑任何仿真**，只做汇总。跑了新的仿真后重新运行本脚本即可。
跑法：``python freeze.py``
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

OPT_DIR = Path(__file__).resolve().parent
OUT_DIR = OPT_DIR / "out"
sys.path.insert(0, str(OPT_DIR))

from summarize_all import (PROBLEMS, LADDER, FINAL4, C0_Q4, adj,  # noqa: E402
                           load_ladder, load_perfect, load_priceperfect,
                           load_upstream)

FINAL = {"Q2": "reserve-mpc", "Q3": "exact-greedy",
         "Q4-2": "reserve-mpc", "Q4-3": "exact-greedy"}
# 各问题最终档位相对上游的全年差额，出现在三份模型评价文档的 §一
UPSTREAM_KEY = {"Q2": "Q2", "Q3": "Q3", "Q4-2": "Q4-2", "Q4-3": "Q4-3"}


def read_exact_checks() -> dict:
    """从 ``exact验证.md`` 里抽出三条检验的最大偏差。"""
    txt = (OPT_DIR / "exact验证.md").read_text(encoding="utf-8")
    out = {}
    m = re.search(r"最大偏差 .*?= \*\*(.+?)\*\* 元", txt)
    if m:
        out["望远镜恒等式最大偏差/元"] = m.group(1)
    m = re.search(r"最大值 = \*\*(.+?)\*\* 元", txt)
    if m:
        out["网格穷举最大偏差/元"] = m.group(1)
        out["网格穷举超容差场景数"] = int(re.search(r"容差的场景 (\d+) 个", txt).group(1))
    out["三条检验是否全部通过"] = "是" if "三条检验全部通过" in txt else "否"
    return out


def read_tost() -> dict:
    """从 ``补充检验.md`` 抽出 TOST 与 exact−fixed Bootstrap 的结论行。"""
    txt = (OPT_DIR / "补充检验.md").read_text(encoding="utf-8")
    out = {}
    for p in PROBLEMS:
        m = re.search(rf"\| {p} \| `?\S+`? \| ±([\d.]+) \| \[(.+?), (.+?)\] \| (\S+?) \|", txt)
        if m:
            out[f"{p} 实践等价带半宽/万元"] = m.group(1)
            out[f"{p} 90%区间/万元"] = f"[{m.group(2)}, {m.group(3)}]"
            out[f"{p} 90%区间是否落入实践等价带"] = "是" in m.group(4)
    for p in ("Q3", "Q4-3"):
        m = re.search(rf"\| {p} \| (-[\d.]+) \| \[(-[\d.]+), (-[\d.]+)\] \| (\S+?) \| ([\d.]+) \|", txt)
        if m:
            out[f"{p} exact−fixed 点估计/万元"] = m.group(1)
            out[f"{p} exact−fixed 95%CI/万元"] = f"[{m.group(2)}, {m.group(3)}]"
    return out


def read_upstream_recompute() -> dict:
    """上游基线在修正预测层下的重算结果。"""
    f = OUT_DIR / "上游重算.json"
    if not f.exists():
        return {}
    out = {}
    for p, v in json.loads(f.read_text(encoding="utf-8")).items():
        out[p] = {
            "重算后原始总费用/元": v["原始总费用/元"],
            "重算后年末SOC/kWh": v["实际年末SOC/kWh"],
            "旧基线原始总费用/元": v["旧基线-原始总费用/元"],
            "费用变化/元": v["费用变化/元"],
            "费用变化/%": v["费用变化/%"],
        }
    return out


def read_prefix() -> dict:
    """从 ``前缀不变性.md`` 抽出比对组数与最大偏差。"""
    txt = (OPT_DIR / "前缀不变性.md").read_text(encoding="utf-8")
    out = {}
    m = re.search(r"共 (\d+) 组比对，最大偏差 ([\d.e+-]+)，超容差 (\d+) 组", txt)
    if m:
        out["比对组数"] = int(m.group(1))
        out["最大偏差"] = float(m.group(2))
        out["超容差组数"] = int(m.group(3))
        out["是否全部逐位一致"] = "是" if float(m.group(2)) == 0.0 else "否"
    return out


def read_audit_qa() -> dict:
    """从 ``汇总对比.md`` 表9 抽出各档位在线物理核对的最大误差与越界计数。"""
    txt = (OPT_DIR / "汇总对比.md").read_text(encoding="utf-8")
    out = {}
    for line in txt.splitlines():
        m = re.match(r"\| (Q[\d-]+) \| (\S+) \| ([\d,]+) \| ([\d.e+-]+) \| ([\d.e+-]+) \| "
                     r"([\d.e+-]+) \| ([\d.e+-]+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|", line)
        if m:
            out[f"{m.group(1)}/{m.group(2)}"] = {
                "核对时段数": int(m.group(3).replace(",", "")),
                "最大平衡误差/kWh": float(m.group(4)),
                "SOC越界": int(m.group(8)), "功率越界": int(m.group(9)),
                "同时充放电": int(m.group(10)), "负流量": int(m.group(11)),
            }
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    up = load_upstream()
    lad = load_ladder()
    perf = load_perfect()
    ppf = load_priceperfect()

    frozen: dict = {"_说明": "本文件由 opt/freeze.py 生成，是文档引用数字的唯一来源。",
                    "上游冻结值": {}, "各档位实跑": {}, "关键数字": {},
                    "exact分支独立检验": read_exact_checks(),
                    "等价性与补充检验": read_tost(),
                    "前缀不变性检验": read_prefix(),
                    "上游基线重算": read_upstream_recompute(),
                    "在线物理核对": read_audit_qa()}

    for p in PROBLEMS:
        u = up[p]
        frozen["上游冻结值"][p] = {
            "原始总费用/元": u["total_cost_yuan"],
            "实际年末SOC/kWh": u["final_soc_kwh"],
            "校正后总费用/元": u["adjusted_cost_yuan"],
            "计划购电量/kWh": u["planned_purchase_kwh"],
            "紧急购电量/kWh": u["emergency_kwh"],
            "紧急天数/天": u["emergency_days"],
            "未用计划电量/kWh": u["unused_plan_kwh"],
            "单日费用CVaR95/元": u["cvar95_yuan"],
        }
        for tag, s in lad[p].items():
            rec = dict(s)
            rec["校正后总费用/元"] = adj(s["total_cost_yuan"], s["final_soc_kwh"])
            rec["相对上游校正后/元"] = rec["校正后总费用/元"] - u["adjusted_cost_yuan"]
            frozen["各档位实跑"][f"{p}/{tag}"] = rec

    key = frozen["关键数字"]
    for p in PROBLEMS:
        u = up[p]
        t = FINAL[p]
        s = lad[p][t]
        c_adj = adj(s["total_cost_yuan"], s["final_soc_kwh"])
        key[f"{p} 最终档位"] = t
        key[f"{p} 最终档位校正后费用/元"] = c_adj
        key[f"{p} 相对上游校正后差额/元"] = c_adj - u["adjusted_cost_yuan"]
        key[f"{p} 相对上游/%"] = (c_adj - u["adjusted_cost_yuan"]) / u["adjusted_cost_yuan"] * 100
        key[f"{p} 紧急购电量/kWh"] = s["emergency_kwh"]
        key[f"{p} 年末SOC/kWh"] = s["final_soc_kwh"]
        key[f"{p} 单日CVaR95/元"] = s["daily_cost_cvar95_yuan"]

    for p in ("Q4-3",):                        # 问题3 用固定电价，C2 ≡ C3，无该档
        t = FINAL[p]
        c1 = adj(perf[p][t]["total_cost_yuan"], perf[p][t]["final_soc_kwh"]) / 1e4
        c2 = adj(ppf[p][t]["total_cost_yuan"], ppf[p][t]["final_soc_kwh"]) / 1e4
        c3 = key[f"{p} 最终档位校正后费用/元"] / 1e4
        # 占比只在有 C2 档的问题4-3 上给出
        key[f"{p} C1/万元"] = c1
        key[f"{p} C2/万元"] = c2
        key[f"{p} C3/万元"] = c3
        key[f"{p} 决策结构占比/%"] = (c1 - C0_Q4) / (c3 - C0_Q4) * 100
        key[f"{p} 预测与风险占比/%"] = (c2 - c1) / (c3 - C0_Q4) * 100
        key[f"{p} 电价不可预知占比/%"] = (c3 - c2) / (c3 - C0_Q4) * 100

    for p in ("Q2", "Q4-2"):
        t = FINAL[p]
        key[f"{p} 完美预测C1/元"] = perf[p][t]["total_cost_yuan"]

    (OPT_DIR / "结果冻结.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")

    L: list[str] = []
    A = L.append
    A("# 结果冻结表")
    A("")
    A("> 由 `opt/freeze.py` 从 `opt/out/*.json`、`verify/out/表_年末口径统一.csv`、")
    A("> `opt/exact验证.md`、`opt/补充检验.md`、`opt/汇总对比.md` 汇总生成，**不重跑仿真**。")
    A("> **文档中引用的数字以本表为准**；改了模型后重跑仿真，再重跑本脚本即可同步。")
    A("")
    A("## 一、各问题最终档位")
    A("")
    A("| 问题 | 最终档位 | 校正后费用/元 | 相对上游/元 | 相对上游/% |")
    A("|---|---|---:|---:|---:|")
    for p in PROBLEMS:
        A(f"| {p} | `{key[f'{p} 最终档位']}` | {key[f'{p} 最终档位校正后费用/元']:,.2f} | "
          f"{key[f'{p} 相对上游校正后差额/元']:+,.2f} | {key[f'{p} 相对上游/%']:+.4f}% |")
    A("")
    A("## 二、全部档位实跑值（校正后费用）")
    A("")
    A("| 问题 | 档位 | 原始总费用/元 | 年末SOC/kWh | 校正后总费用/元 | 相对上游/元 |")
    A("|---|---|---:|---:|---:|---:|")
    for p in PROBLEMS:
        A(f"| {p} | **上游** | {up[p]['total_cost_yuan']:,.2f} | {up[p]['final_soc_kwh']:,.2f} | "
          f"{up[p]['adjusted_cost_yuan']:,.2f} | — |")
        for tag in LADDER:
            r = frozen["各档位实跑"].get(f"{p}/{tag}")
            if r is None:
                continue
            A(f"| {p} | `{tag}` | {r['total_cost_yuan']:,.2f} | {r['final_soc_kwh']:,.2f} | "
              f"{r['校正后总费用/元']:,.2f} | {r['相对上游校正后/元']:+,.2f} |")
    A("")
    A("## 三、exact 分支的独立检验")
    A("")
    for k, v in frozen["exact分支独立检验"].items():
        A(f"- {k}：**{v}**")
    A("")
    A("## 四、实践等价带敏感性与补充 Bootstrap")
    A("")
    for k, v in frozen["等价性与补充检验"].items():
        A(f"- {k}：{v}")
    A("")
    A("## 五、在线物理核对（本次实跑全部档位）")
    A("")
    A("| 档位 | 核对时段数 | 最大平衡误差/kWh | SOC越界 | 功率越界 | 同时充放电 | 负流量 |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    bad = 0
    for k, v in frozen["在线物理核对"].items():
        A(f"| {k} | {v['核对时段数']:,} | {v['最大平衡误差/kWh']:.3e} | {v['SOC越界']} | "
          f"{v['功率越界']} | {v['同时充放电']} | {v['负流量']} |")
        bad += (v["SOC越界"] + v["功率越界"] + v["同时充放电"] + v["负流量"])
    A("")
    A(f"**越界计数合计 {bad}**。")
    A("")
    A("## 六、前缀不变性检验")
    A("")
    for k, v in frozen["前缀不变性检验"].items():
        A(f"- {k}：{v}")
    A("")
    A("> 截断重跑（`--days K`，$K\\in\\{90,150,220,300\\}$）与全年跑逐位一致，")
    A("> 说明最终策略在\"是否偷看未来\"这一维度上是干净的。明细见 `opt/前缀不变性.md`。")
    A("")
    A("## 七、上游基线在修正预测层下的重算")
    A("")
    A("> 前缀不变性检验发现上游 `forecasts.py:_pooled_coef` 用全年实测光伏识别昼夜列")
    A("> （跨日未来信息泄露），已改为只用决策日之前的历史。上游自带的冻结值是泄露版")
    A("> 预测层跑出来的，与本次各档位不再同口径，故用 `opt/rebuild_upstream.py` 重算。")
    A("> 问题2 / 问题4-2 的预测层未改动，基线仍取 `verify/out/表_年末口径统一.csv`。")
    A("")
    A("| 问题 | 重算后基线/元 | 旧基线/元 | 变化/元 | 变化/% |")
    A("|---|---:|---:|---:|---:|")
    for p, v in frozen["上游基线重算"].items():
        A(f"| {p} | {v['重算后原始总费用/元']:,.2f} | {v['旧基线原始总费用/元']:,.2f} | "
          f"{v['费用变化/元']:+,.2f} | {v['费用变化/%']:+.4f}% |")
    A("")

    (OPT_DIR / "结果冻结.md").write_text("\n".join(L), encoding="utf-8")
    print(f"已写出 {OPT_DIR / '结果冻结.json'} 与 {OPT_DIR / '结果冻结.md'}")
    print(f"档位数 {len(frozen['各档位实跑'])}，越界计数合计 {bad}")


if __name__ == "__main__":
    main()
