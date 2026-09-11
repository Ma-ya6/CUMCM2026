"""问题3模型3的公共数据层与官方预报解码器。

本模块只做两件事：把附件读成数组，以及把附件3的**整点小时预报**解码成
决策与结算所需的**十分钟分辨率**序列。所有口径在下面逐条说明。

## 附件3 的行列语义（经数值验证）

附件3 每天 0:00 / 6:00 / 12:00 / 18:00 各发布一次未来 24 小时整点预报，
共 ``365 × 4 = 1460`` 行、24 个预报列（预报1小时 … 预报24小时）。
对发布时刻 $k$，第 $j$ 列的物理含义是**第 $k+j-1$ 至 $k+j$ 小时的功率平均值**：

$$ f^{(k)}_j \\;=\\; \\frac{1}{1\\text{h}}\\int_{k+j-1}^{k+j} P(t)\\,\\mathrm dt ,
\\qquad j=1,\\dots,24 . $$

因此 $f^{(k)}_j$ 覆盖的绝对整点为 $[k+j-1,\\;k+j)$。

**这个"落在小时左端还是右端"的判定不是约定，是测出来的。** 用两个独立证据：

1. 发布时刻衔接：6:00 发布的"预报1小时"应等于 0:00 发布的"预报7小时"
   （同为 06:00–07:00 那个小时）。实测 2025-01-01 为 3.0919 与 3.1173，
   量级一致；夜间小值也对得上。
2. 同小时互相关：取 0:00 发布的 $f_7\\dots f_{24}$ 与 6:00 发布的 $f_1\\dots f_{18}$，
   二者覆盖**同一批绝对整点**（06:00–24:00）。若映射正确，两条序列应几乎重合。
   实测相关系数 **0.9871**（左端对齐），而"右端对齐"（$f_j$ 覆盖 $[k+j,k+j+1)$）
   只有 0.9205。判别度足够大，映射唯一确定。

## 由整点到十分钟：分段常数而非分段线性

$f^{(k)}_j$ 是该小时内的**平均功率**，故把同一个值平铺到该小时覆盖的 6 个
十分钟时段，是对该平均量的忠实还原（分段常数）。``问题3.md`` 假设 #11 写的
是"分段线性插值"，但两节点都是"小时均值"、置于小时中点时，线性插值会把
信号整体前移约半小时，实测 MAE 由 371.22 kW 恶化到 639.17 kW。
故本实现采用**分段常数**作为主口径，并在 ``accuracy.py`` 中把分段线性
作为敏感性对照一并报告——这是一个有数据支撑的修正，而非口径松动。

## 覆盖范围

发布时刻 $k$ 的预报只能覆盖当天 $t \\ge 6k+1$ 的时段（$t$ 为十分钟时段编号，
$t=1$ 表示 00:00–00:10）。更早的时段在发布时已经执行完毕，预报里也没有它们。
本模块把未覆盖的时段填 ``NaN``，并用配套的布尔掩码显式区分"预报值为 0（夜间）"
与"没有预报（未覆盖）"——这两者在准确度统计中绝不能混为一谈。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 物理常数（与 问题2/模型2再优化 保持逐位一致，保证跨问可比）
# --------------------------------------------------------------------------
DT = 1.0 / 6.0
N_SLOT = 144
SLOTS_PER_HOUR = 6
ETA_C = 0.90
ETA_D = 0.90
E_MIN = 1200.0
E_MAX = 10800.0
E_CAP = 12000.0
E_INITIAL = 6000.0
P_MAX_KW = 5000.0
EMERGENCY_MULTIPLIER = 5.0
BREACH_RATE = 0.5      # 下调承诺（违约）：交易时刻电价的 50%
OVERBUY_RATE = 1.5     # 上调承诺（超购）：交易时刻电价的 150%

FORMAL_START = pd.Timestamp("2025-02-01")
SPECIFIED_DATES = pd.to_datetime(
    ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
)
ISSUE_HOURS = (0, 6, 12, 18)
SEED = 20260910

MODEL_DIR = Path(__file__).resolve().parent
Q3_DIR = MODEL_DIR.parent
C_DIR = Q3_DIR.parent
ATTACHMENT_DIR = C_DIR / "附件"
Q2_MODEL_DIR = C_DIR / "问题2" / "模型2再优化"
Q2_M1_DIR = C_DIR / "问题2" / "模型1"
RESULT_DIR = MODEL_DIR / "results"
FIGURE_DIR = MODEL_DIR / "figures"
CACHE_DIR = MODEL_DIR / "cache"
for _d in (RESULT_DIR, FIGURE_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class DataBundle3:
    """问题3 的输入数据包。

    ``official_pv`` 形状 ``(365, 4, 144)``：第 0 维为日期，第 1 维对应
    ``ISSUE_HOURS`` 的四个发布时刻，第 2 维为十分钟时段。
    未覆盖的时段为 ``NaN``，由 ``official_mask`` 标记。
    """

    dates: pd.DatetimeIndex
    load_kw: np.ndarray          # (365, 144)
    pv_kw: np.ndarray            # (365, 144) 实际光伏
    price: np.ndarray            # (144,) 日内电价，每天相同（问题3.md #13）
    official_pv: np.ndarray      # (365, 4, 144)
    official_mask: np.ndarray    # (365, 4, 144) bool
    initial_load: np.ndarray     # (144,)
    initial_pv: np.ndarray       # (144,)


def load_inputs() -> DataBundle3:
    """读取附件1/2/3，构造问题3 的数据包。"""
    # ---- 附件1：电价（每天相同）+ 问题1 用的初始负荷/光伏 ----
    base_raw = pd.read_excel(ATTACHMENT_DIR / "附件1.xlsx")
    price = pd.to_numeric(base_raw.iloc[:, 1], errors="raise").to_numpy(float)
    initial_load = pd.to_numeric(base_raw.iloc[:, 2], errors="raise").to_numpy(float)
    initial_pv = pd.to_numeric(base_raw.iloc[:, 3], errors="raise").to_numpy(float)

    # ---- 附件2：全年实际负荷与光伏 ----
    attachment2 = ATTACHMENT_DIR / "附件2.xlsx"
    load_raw = pd.read_excel(attachment2, sheet_name=0)
    pv_raw = pd.read_excel(attachment2, sheet_name=1)
    dates = pd.DatetimeIndex(pd.to_datetime(load_raw.iloc[:, 0]))
    load_kw = load_raw.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(float)
    pv_kw = pv_raw.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(float)

    # ---- 附件3：官方滚动光伏预报 ----
    official_pv, official_mask = decode_official_forecast(dates)

    if load_kw.shape != (365, N_SLOT) or pv_kw.shape != (365, N_SLOT):
        raise ValueError(f"附件2 维度异常：load={load_kw.shape}, pv={pv_kw.shape}")
    if len(price) != N_SLOT:
        raise ValueError(f"附件1 电价长度异常：{len(price)}")
    if not np.isfinite(load_kw).all() or not np.isfinite(pv_kw).all():
        raise ValueError("附件2 存在缺失或非数值数据")
    if (load_kw < 0).any() or (pv_kw < 0).any():
        raise ValueError("负载或光伏存在负值")

    return DataBundle3(
        dates=dates,
        load_kw=load_kw,
        pv_kw=pv_kw,
        price=price,
        official_pv=official_pv,
        official_mask=official_mask,
        initial_load=initial_load,
        initial_pv=initial_pv,
    )


def decode_official_forecast(
    dates: pd.DatetimeIndex,
    mode: str = "step",
) -> tuple[np.ndarray, np.ndarray]:
    """把附件3 解码为 ``(365, 4, 144)`` 的十分钟预报与覆盖掩码。

    ``mode="step"``：小时均值平铺到该小时的 6 个时段（主口径）。
    ``mode="linear"``：相邻小时均值之间线性插值（``问题3.md`` #11 的口径，
    作为敏感性对照；实测更差，见模块文档）。
    """
    if mode not in ("step", "linear"):
        raise ValueError("mode 只能是 'step' 或 'linear'")

    raw = pd.read_excel(ATTACHMENT_DIR / "附件3.xlsx", header=0)
    raw.columns = ["date", "ftime"] + [f"f{j}" for j in range(1, 25)]
    raw["date"] = pd.to_datetime(raw["date"].ffill().astype(str), format="%Y-%m-%d")
    raw["ftime"] = raw["ftime"].astype(str).str.strip()

    n_days = len(dates)
    day_index = {d: i for i, d in enumerate(dates)}
    out = np.full((n_days, len(ISSUE_HOURS), N_SLOT), np.nan)
    mask = np.zeros((n_days, len(ISSUE_HOURS), N_SLOT), dtype=bool)

    for k_pos, k in enumerate(ISSUE_HOURS):
        sub = raw[raw.ftime == f"{k}:00"]
        if len(sub) != n_days:
            raise ValueError(f"附件3 在 {k}:00 只有 {len(sub)} 行，应为 {n_days}")
        for _, row in sub.iterrows():
            day = row["date"]
            if day not in day_index:
                raise ValueError(f"附件3 日期 {day} 不在附件2 的日期序列中")
            di = day_index[day]
            f = row[[f"f{j}" for j in range(1, 25)]].to_numpy(float)
            for j in range(1, 25):
                hour = k + j - 1          # f_j 覆盖绝对整点 [hour, hour+1)
                if hour >= 24:
                    break                 # 超出当天 24:00 的部分不再使用（问题3.md #10）
                sl = slice(hour * SLOTS_PER_HOUR, (hour + 1) * SLOTS_PER_HOUR)
                if mode == "step":
                    out[di, k_pos, sl] = f[j - 1]
                else:
                    nxt = f[j] if j < 24 else f[j - 1]
                    frac = (np.arange(SLOTS_PER_HOUR) + 0.5) / SLOTS_PER_HOUR
                    out[di, k_pos, sl] = (1.0 - frac) * f[j - 1] + frac * nxt
                mask[di, k_pos, sl] = True

    if not np.isfinite(out[mask]).all():
        raise ValueError("附件3 解码后覆盖区存在非数值")
    return out, mask


def coverage_start_slot(k: int) -> int:
    """发布时刻 ``k`` 的预报所覆盖的最早时段编号（1-based，问题3.md #16）。

    6:00 只能调整 $t\\ge 37$，12:00 为 $t\\ge 73$，18:00 为 $t\\ge 109$。
    """
    return k * SLOTS_PER_HOUR + 1


def formal_split(dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """开发期 / 留出期 / 全年正式期 的日期掩码（与模型2再优化同口径）。"""
    formal = np.asarray(dates >= FORMAL_START)
    return {
        "开发期(2-8月)": formal & np.asarray(dates <= pd.Timestamp("2025-08-31")),
        "留出期(9-12月)": np.asarray(dates >= pd.Timestamp("2025-09-01")),
        "全年正式期": formal,
    }


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Q2_MODEL_DIR))
    data = load_inputs()
    print(f"日期      : {data.dates[0].date()} → {data.dates[-1].date()}  ({len(data.dates)} 天)")
    print(f"负载      : {data.load_kw.shape}  均值 {data.load_kw.mean():.1f} kW")
    print(f"实际光伏  : {data.pv_kw.shape}  均值 {data.pv_kw.mean():.1f} kW")
    print(f"电价      : {data.price.shape}  均值 {data.price.mean():.4f} 元/kWh")
    print(f"官方预报  : {data.official_pv.shape}  覆盖率 {data.official_mask.mean():.4f}")
    for k_pos, k in enumerate(ISSUE_HOURS):
        cov = data.official_mask[:, k_pos, :].mean()
        print(f"   {k:2d}:00 覆盖 {cov:6.1%}  最早时段 t={coverage_start_slot(k)}")
