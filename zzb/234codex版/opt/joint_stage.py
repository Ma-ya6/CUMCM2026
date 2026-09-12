# -*- coding: utf-8 -*-
"""六、把承诺路径本身纳入（滚动）优化：单阶段 LP 的情景化紧急购电。

## 问题：两段式解耦的损失在哪

现状（``run_stage`` / ``run_single`` 的 ``reserve-*`` 档）在每个决策时刻解一次
单阶段 LP 定下承诺路径 $x$，执行器随后只负责"怎么把 $x$ 落地"。这个 LP 的目标
只含**结算口径**的费用：

$$ \\varphi(x)=0.5c\\,x+1.0c\\,(x-r)^+ ,$$

它对"承诺买少了会怎样"完全没有定价——承诺量买得不够，实际执行时的缺口只能由
储能放电或 **5 倍电价的紧急购电**补上，而这两项都不在 $\\varphi$ 里。风险因此
只能靠一条外生的储能储备约束 $E_t\\ge E_{\\min}+R$ 硬扛，$x$ 本身并不是
"权衡过执行后果"的最优承诺。

## 改法：让 $x$ 与物理执行在同一支 LP 里共同定价

``opt_core.stage_lp`` 已经支持 ``emergency=True`` 与 ``scenarios``：它在
净负荷平衡式里加入每情景的紧急购电量 $e_{s,t}$（目标记 $5c_t$）与弃光
$w_{s,t}$，而承诺 $x$、充放电 $u/v$、储电 $s$ 仍**全情景共用**（一份承诺应对
多种可能的净负荷实现）。于是目标变成

$$ \\min\\;\\underbrace{0.5c\\,x+1.0c\\,(x-r)^+}_{\\text{结算}}
   \\;+\\;\\sum_s \\pi_s \\sum_t 5c_t\\,e_{s,t}
   \\;-\\;\\lambda\\,E_T ,$$

承诺量于是"知道自己买少了会挨 5 倍电价"，而不是靠外生储备兜底。

## 情景从哪来（因果）

第 $d$ 天第 $k$ 个决策时刻只能用 $j<d$ 的净负荷残差
$\\varepsilon_{j,t}=\\text{实际净负荷}-\\text{预测净负荷}$，取最近
``N_SCEN`` 天、等权 $\\pi_s=1/S$。这既是因果的，也与
``causal_cumulative_reserve`` 用的是同一份残差历史，两个机制口径一致、可叠加。

样本不足（$d<S$）时用已有的行；一行都没有时退化为单个零情景，即与不传情景
等价，不会因冷启动而产生不存在的风险定价。

## 本模块只提供情景

LP 本身、执行器、储备、结算一律不改：``run_stage`` / ``run_single`` 只是把
``scenarios``/``weights``/``emergency`` 透传给 ``opt_core.stage_lp``。因此
``joint-*`` 档与 ``fixed-*`` 档之间的费用差**只来自"承诺是否对执行后果定价"**
这一处。
"""
from __future__ import annotations

import numpy as np

N_SCEN = 5          # 情景个数：最近 5 天的净负荷残差，等权


def causal_scenarios(
    err: np.ndarray,
    day: int,
    n_scen: int = N_SCEN,
) -> tuple[np.ndarray, np.ndarray]:
    """取第 ``day`` 天之前最近 ``n_scen`` 天的净负荷残差作为情景。

    ``err`` 形状 ``(n_days, n_covered)``，单位 kW，行 $j$ 为第 $j$ 天的
    "实际净负荷 − 预测净负荷"。只取 $j<day$ 的行，故是因果的。

    返回 ``(scenarios, weights)``：``scenarios`` 形状 ``(S, n_covered)``，
    ``weights`` 形状 ``(S,)``，等权。样本为空时返回单个零情景，使 LP 与
    "不传情景"逐位等价。
    """
    hist = np.asarray(err[max(0, day - n_scen):day], dtype=float)
    if hist.shape[0] == 0:
        return np.zeros((1, err.shape[1])), np.ones(1)
    hist = np.nan_to_num(hist, nan=0.0)
    return hist, np.full(hist.shape[0], 1.0 / hist.shape[0])
