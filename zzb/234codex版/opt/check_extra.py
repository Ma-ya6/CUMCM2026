# -*- coding: utf-8 -*-
"""补充检验：月度分组、7 日区块 Bootstrap 置信区间、q=0.80 覆盖率检验。

只用已经跑出来的 ``opt/out/*_逐日.csv`` 与各问题的因果预测层，不重跑仿真。

跑法：``python check_extra.py``，输出 ``opt/补充检验.md``。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OPT_DIR = Path(__file__).resolve().parent
OUT_DIR = OPT_DIR / "out"
sys.path.insert(0, str(OPT_DIR))

import opt_core as oc  # noqa: E402
from run_single import build_forecasts, load_stack  # noqa: E402
from run_stage import DIRS as STAGE_DIRS            # noqa: E402

Q = 0.80
BLOCK = 7
BLOCKS = (3, 7, 14, 28)     # 区块长度敏感性
N_BOOT = 2000
SEED = 20260911

# 各问题的最终策略档位（上游对照统一用 legacy-greedy，它逐位复现上游冻结值）
FINAL = {"Q2": "reserve-mpc", "Q3": "exact-greedy",
         "Q4-2": "reserve-mpc", "Q4-3": "exact-greedy"}


def load_daily(problem: str, tag: str) -> pd.DataFrame:
    f = OUT_DIR / f"{problem}_{tag}_逐日.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f, encoding="utf-8-sig")
    d["date"] = pd.to_datetime(d["date"])
    return d


def monthly_table(problem: str, up: pd.DataFrame, new: pd.DataFrame) -> list[str]:
    a = up.set_index("date").total_cost_yuan.resample("MS").sum()
    b = new.set_index("date").total_cost_yuan.resample("MS").sum()
    lines = [f"| 月份 | 上游/万元 | {FINAL[problem]}/万元 | 差额/万元 | 差额/% |",
             "|---|---:|---:|---:|---:|"]
    for m in a.index:
        x, y = float(a[m]), float(b.get(m, np.nan))
        lines.append(f"| {m:%Y-%m} | {x/1e4:,.2f} | {y/1e4:,.2f} | "
                     f"{(y-x)/1e4:+,.2f} | {(y-x)/x*100:+.3f}% |")
    bad = (b - a) < 0
    n_neg = int(bad.sum())
    lines.append("")
    lines.append(f"共 {len(a)} 个月，其中新策略更省的月份 {n_neg} 个、"
                 f"更贵的 {len(a)-n_neg} 个。")
    return lines


def block_bootstrap(up: pd.DataFrame, new: pd.DataFrame,
                    block: int = BLOCK, seed: int = SEED,
                    qs=(0.025, 0.975)) -> tuple:
    """对"每日**原始**费用差额"做 block 日区块 Bootstrap，返回 (点估计, 下界, 上界, p_不省)。

    注：这里用的是未做年末残值校正的原始费用，与主表的校正口径不同（见 §一 说明）。
    ``qs`` 可换成 (0.05, 0.95) 得到 TOST 用的 90% 区间。
    """
    a = up.total_cost_yuan.to_numpy(float)
    b = new.total_cost_yuan.to_numpy(float)
    n = len(a)
    total = float((b - a).sum())
    rng = np.random.default_rng(seed)
    n_blk = int(np.ceil(n / block))
    starts = np.arange(0, n - block + 1)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        s = rng.choice(starts, size=n_blk, replace=True)
        idx = (s[:, None] + np.arange(block)[None, :]).ravel()[:n]
        boots[i] = float((b[idx] - a[idx]).sum())
    lo, hi = np.quantile(boots, list(qs))
    return total, float(lo), float(hi), float((boots >= 0).mean())


def quantile_checks(problem: str) -> list[str]:
    """q=0.80 的日累计覆盖率 / Pinball Loss / 超限连续性 / 逐时段对比。"""
    db, CorrectionConfig, ForecastConfig, gen_fc = load_stack(problem)
    data = db.load_inputs()
    fc = build_forecasts(problem, db, CorrectionConfig, ForecastConfig, gen_fc)
    err = (data.load_kw - data.pv_kw) - (fc["load"] - fc["pv"])
    n = len(data.dates)
    eval_mask = np.asarray(data.dates) >= pd.Timestamp("2025-02-01")

    # 逐时段分位（上游口径） vs 日累计分位（本次口径），均为因果
    per_slot = db.causal_residual_quantiles(err, Q)                 # (n, 144) kW
    daily_R = oc.causal_cumulative_reserve(err, Q)                  # (n,) kWh
    cum = err.sum(axis=1) * oc.DT                                   # 当日累计残差 kWh

    e = eval_mask
    hit = cum[e] <= daily_R[e]
    cov = float(hit.mean())
    # Pinball Loss：分位预测 R 对实现值 cum 的损失
    pin = float(np.where(cum[e] >= daily_R[e], Q * (cum[e] - daily_R[e]),
                         (1 - Q) * (daily_R[e] - cum[e])).mean())
    # 逐时段口径的覆盖率（对每个时段的残差看是否被余量覆盖）
    slot_cov = float((err[e] <= per_slot[e]).mean())
    # 超限连续性：条件概率 P(次日也超限 | 今日超限)
    ex = ~hit
    idx = np.flatnonzero(e)
    nxt = np.array([ex[i + 1] for i in range(len(ex) - 1)
                    if idx[i + 1] == idx[i] + 1])
    cur = ex[:-1][[i for i in range(len(ex) - 1) if idx[i + 1] == idx[i] + 1]]
    p_next = float(nxt[cur].mean()) if cur.sum() else float("nan")
    # 独立假设下的期望值
    indep = 1.0 - Q

    return [
        f"| 指标 | 数值 | 理论/参照 |", "|---|---:|---:|",
        f"| 日累计储备覆盖率 P(ΣξΔt ≤ R_d) | {cov:.4f} | 0.80 |",
        f"| 逐时段余量覆盖率 P(ξ_t ≤ m_t) | {slot_cov:.4f} | 0.80 |",
        f"| 日累计 Pinball Loss | {pin:,.2f} kWh | 越小越好 |",
        f"| 超限后次日再超限的条件概率 | {p_next:.4f} | 独立时 {indep:.2f} |",
        f"| 评价期天数 | {int(e.sum())} | — |",
    ]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    lines: list[str] = []
    A = lines.append
    A("# 补充检验（月度分组 / 区块 Bootstrap / 分位数覆盖）")
    A("")
    A(f"> 数据来源：`opt/out/*_逐日.csv` 与各问题上游预测层。"
      f"区块 Bootstrap 用 {BLOCK} 日区块、{N_BOOT} 次重采样、seed={SEED}。")
    A("")

    A("## 一、7 日区块 Bootstrap 置信区间（新策略 − 上游，负值表示更省）")
    A("")
    A("| 问题 | 最终策略 | 全年差额/万元 | 95% CI 下界/万元 | 上界/万元 | "
      "CI 是否含 0 | Bootstrap P(差额≥0) |")
    A("|---|---|---:|---:|---:|---|---:|")
    boot_rows = []
    for p, tag in FINAL.items():
        up, new = load_daily(p, "legacy-greedy"), load_daily(p, tag)
        if up.empty or new.empty:
            A(f"| {p} | {tag} | — | — | — | 缺数据 | — |")
            continue
        t, lo, hi, pr = block_bootstrap(up, new)
        boot_rows.append((p, tag, t, lo, hi, pr))
        A(f"| {p} | {tag} | {t/1e4:+,.2f} | {lo/1e4:+,.2f} | {hi/1e4:+,.2f} | "
          f"{'是' if lo <= 0 <= hi else '否'} | {pr:.3f} |")
    A("")
    A("> **口径**：`legacy-greedy` 档位与上游最终模型逐位一致，故直接以其逐日费用作上游对照。"
      "Bootstrap 检验的是**原始费用差额**（未做年末残值校正），"
      "与主表的年末校正口径相差约 0.3 万元："
      "问题3 主表 +1.37 万元 / Bootstrap +1.05 万元；问题4-3 主表 +0.62 万元 / Bootstrap +0.30 万元；"
      "问题2 与问题4-2 的最终档位年末 SOC 接近 $E_0$，两口径差异 $<0.03$ 万元。"
      "两条口径的结论一致，但不应混用具体数值。")
    A("")

    A("### 区块长度敏感性")
    A("")
    A("7 日区块假定年内的日费用差额近似平稳，但本数据存在明显季节变化，"
      "故对区块长度做敏感性检验。**下表仍为原始费用口径。**")
    A("")
    A("| 问题 | 区块长度 | 点估计/万元 | 95% CI/万元 | 是否含 0 |")
    A("|---|---:|---:|---|---|")
    for p, tag in FINAL.items():
        up, new = load_daily(p, "legacy-greedy"), load_daily(p, tag)
        if up.empty or new.empty:
            continue
        for blk in BLOCKS:
            t, lo, hi, _ = block_bootstrap(up, new, block=blk)
            A(f"| {p} | {blk} 日 | {t/1e4:+,.2f} | "
              f"[{lo/1e4:+,.2f}, {hi/1e4:+,.2f}] | {'**是**' if lo <= 0 <= hi else '否'} |")
    A("")
    A("> 结论：区块长度在 3–28 日之间变化时，四个问题的\"CI 是否含 0\"判断均不改变——"
      "**全部 16 个组合的 95% 区间都覆盖 0**。因此四个问题的费用差都无法与 0 区分，"
      "只能表述为\"在该区块设定下，费用差未被检出\"，**不应表述为显著上升或显著下降**。"
      "另需注意：区块 Bootstrap 默认年内日费用差额平稳，而本数据存在明显季节变化，"
      "该假设本身是近似的。")
    A("")

    A("## 二、月度分组（改进是否集中在个别月份）")
    A("")
    monthly_all = {}
    for p, tag in FINAL.items():
        up, new = load_daily(p, "legacy-greedy"), load_daily(p, tag)
        if up.empty or new.empty:
            continue
        A(f"### {p}（{tag}）")
        A("")
        lines.extend(monthly_table(p, up, new))
        A("")
        a = up.set_index("date").total_cost_yuan.resample("MS").sum()
        b = new.set_index("date").total_cost_yuan.resample("MS").sum()
        monthly_all[p] = (b - a) / 1e4
    if monthly_all:
        A("### 各月差额汇总（万元，负=新策略更省）")
        A("")
        tbl = pd.DataFrame(monthly_all)
        A("| 月份 | " + " | ".join(tbl.columns) + " |")
        A("|---" * (len(tbl.columns) + 1) + "|")
        for m, row in tbl.iterrows():
            A(f"| {m:%Y-%m} | " + " | ".join(f"{v:+,.2f}" for v in row) + " |")
        A("")

    A("## 三、q=0.80 分位数覆盖检验")
    A("")
    for p in ("Q2", "Q4-2"):
        A(f"### {p}")
        A("")
        lines.extend(quantile_checks(p))
        A("")
    A("> Q3/Q4-3 的储备按四个决策时刻分别计算，覆盖检验口径与 Q2/Q4-2 不同，"
      "此处只列按日的两个问题。")
    A("")

    A("## 四、exact 相对 fixed 的区块 Bootstrap（仅 Q3 / Q4-3）")
    A("")
    A("> 上文 Bootstrap 比较的是 exact 与 **legacy**，不能用来支撑\"exact 全面优于 fixed\"。"
      "此处对 exact−fixed 的逐日费用差另做一次 7 日区块 Bootstrap。**原始费用口径。**")
    A("")
    A("| 问题 | 点估计/万元 | 7 日区块 95% CI/万元 | 含 0？ | P(差额≥0) |")
    A("|---|---:|---|---:|---:|")
    stage_coverage = {}
    for p in ("Q3", "Q4-3"):
        fx, ex_ = load_daily(p, "fixed-greedy"), load_daily(p, "exact-greedy")
        if fx.empty or ex_.empty:
            A(f"| {p} | — | 缺数据 | — | — |")
            continue
        t, lo, hi, pr = block_bootstrap(fx, ex_)
        A(f"| {p} | {t/1e4:+,.2f} | [{lo/1e4:+,.2f}, {hi/1e4:+,.2f}] | "
          f"{'**是**' if lo <= 0 <= hi else '否'} | {pr:.3f} |")
    A("")
    A("> 与前面 exact−legacy 的结果不同，exact−fixed 的区间**完全落在 0 以下**"
      "（$P(\\text{差额}\\ge0)=0.000$），即该费用改善在区块 Bootstrap 下被检出，"
      "不是样本内的偶发波动。这支持\"exact 相对 fixed 有实质改善\"的说法。"
      "但要注意：这只说明**换掉目标函数**有效，"
      "不代表 exact 相对 上游 `legacy` 也更省（那是另一张表，区间覆盖 0）。")
    A("")

    A("## 五、实践等价带敏感性（探索性 TOST 口径，±0.5%）")
    A("")
    A("\"费用差未被检出\"**不等于**\"两模型等价\"。此处把 ±0.5%"
      "（相对上游 `legacy-greedy` 全年费用）作为**实践等价尺度的敏感性设定**，"
      "用区块 Bootstrap 的 **90%** 区间作探索性双单侧检验：90% 区间完全落在"
      " $(-\\delta,+\\delta)$ 内时，只能说差异落入该实践等价带。该界限不是赛前"
      "预注册值，因此不把结果表述为严格的确认性等价结论。")
    A("")
    A("| 问题 | 最终策略 | 实践等价带 ±δ/万元 | 90% CI/万元 | 区间是否落入等价带 |")
    A("|---|---|---:|---|---|")
    tost_equiv = 0
    tost_n = 0
    for p, tag in FINAL.items():
        up, new = load_daily(p, "legacy-greedy"), load_daily(p, tag)
        if up.empty or new.empty:
            continue
        delta = 0.005 * float(up.total_cost_yuan.sum())
        _, lo90, hi90, _ = block_bootstrap(up, new, qs=(0.05, 0.95))
        ok = (lo90 > -delta) and (hi90 < delta)
        tost_n += 1
        tost_equiv += int(ok)
        A(f"| {p} | {tag} | ±{delta/1e4:,.2f} | [{lo90/1e4:+,.2f}, {hi90/1e4:+,.2f}] | "
          f"{'**是**' if ok else '否'} |")
    A("")
    A(f"> {tost_n} 个问题中，{tost_equiv} 个的 90% 区间落入事后设定的 ±0.5% 实践等价带。"
      "未落入不等于证明存在实质差异，只表示该样本下区间宽于此尺度；落入也不作为"
      "确认性等价证明。"
      "据此，文档中一律避免使用\"两模型相同\"\"效果等价\"的表述，"
      "只写\"费用差未被检出\"，并注明区间本身覆盖了经济上不小的范围。")
    A("")

    A("## 六、问题3/4-3 四个决策时刻的储备覆盖率")
    A("")
    A("> §三 的覆盖检验只覆盖按日决策的问题2/4-2。问题3/4-3 的储备按四个决策时刻"
      "（0:00 / 6:00 / 12:00 / 18:00）分别取值，此处按同一因果口径补做覆盖检验："
      "对每个时刻 $k$，把该时刻起至日末的残差累计 $\\sum_{t\\ge t_k}\\xi_t\\Delta t$ "
      "与该时刻的储备 $R_d^{(k)}$ 比较。")
    A("")
    for p in ("Q3", "Q4-3"):
        # 两个问题的上游目录带同名模块（common/dispatch_core/forecasts），
        # 同进程内必须先清掉已导入的模块再换路径，否则会拿到上一个问题的实现。
        sys.path.insert(0, str(STAGE_DIRS[p]))
        for _m in ("common", "dispatch_core", "forecasts"):
            sys.modules.pop(_m, None)
        import common, dispatch_core, forecasts            # noqa: E402
        data = common.load_inputs()
        bundle = forecasts.forecast_bundle(data, use_cache=False, verbose=False)
        mask = np.asarray(data.dates) >= pd.Timestamp("2025-02-01")
        A(f"### {p}")
        A("")
        A("| 决策时刻 | 储备均值/kWh | 覆盖率 $P(\\sum\\xi\\Delta t\\le R)$ | Pinball Loss/kWh |")
        A("|---|---:|---:|---:|")
        for k in (0, 6, 12, 18):
            err = dispatch_core._error_matrix(data, bundle, k, "own", "fused")
            R = oc.causal_cumulative_reserve(err, Q)
            cum = err.sum(axis=1) * oc.DT
            e = mask
            cov = float((cum[e] <= R[e]).mean())
            stage_coverage[(p, k)] = cov
            pin = float(np.where(cum[e] >= R[e], Q * (cum[e] - R[e]),
                                 (1 - Q) * (R[e] - cum[e])).mean())
            A(f"| {k}:00 | {R[e].mean():,.1f} | {cov:.4f} | {pin:,.2f} |")
        A("")
    c = [stage_coverage[("Q3", k)] for k in (0, 6, 12, 18)]
    A(f"> **结果并不完全符合名义水平**：0:00、6:00、12:00、18:00 的覆盖率依次为 "
      f"**{c[0]:.4f}** / {c[1]:.4f} / {c[2]:.4f} / {c[3]:.4f}。"
      "即储备在**日初**偏薄、午间略偏厚，其余时点接近名义水平。"
      "原因是 $R_d^{(k)}$ 由该时刻起至日末的残差累计分位数给出，"
      "而残差在一天内的分布并不平稳（日初窗口长、尾部事件多），"
      "单一分位数无法同时校准四个窗口。"
      "这只影响**储备厚度的时刻分配**，不改变各档位之间的比较结果"
      "（四个档位用的是同一套储备）。")
    A("")
    A("> 另注：Q3 与 Q4-3 两表数值完全相同，因为 `_error_matrix(..., \"own\", \"fused\")` "
      "只依赖预测层，而两个上游目录的 `forecasts.py` 是同一份文件"
      "（`common.py` / `dispatch_core.py` 不同）。这不是加载串档。")
    A("")

    out = OPT_DIR / "补充检验.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
