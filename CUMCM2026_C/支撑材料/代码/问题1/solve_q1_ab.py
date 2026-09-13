# -*- coding: utf-8 -*-
"""
C 题 问题一 · 约束 A（固定端点）与约束 B（首尾自由）合并求解
=============================================================================
【模型】

    索引与集合
        t = 1,2,…,T 日内十分钟时段序号（t = 1 为 00:00–00:10），T = 144
        Δt = 1/6 h

    参数
        c_t              第 t 时段外网电价，元/kWh
        L_t              小区负载功率，kW
        P_t^fc           光伏发电功率，kW
        E_min, E_max     储电量下限与上限，1200 / 10800 kWh
        P_st^max         最大充放电功率，5000 kW
        η_c, η_d         充电、放电效率，0.90 / 0.90

    决策变量（均为功率，kW）
        x_t ≥ 0                       外网计划购电功率
        0 ≤ u_t ≤ P_st^max            储能充电功率
        0 ≤ v_t ≤ P_st^max            储能放电功率
        w_t ≥ 0                       弃光功率
    状态变量
        E_t                           第 t 时段末储电量，kWh（E_0 为 0:00 储电量）

    目标函数
        min  C = Σ_{t=1}^{T} c_t · x_t · Δt

    共用的约束
        (1) 电量平衡      x_t + P_t^fc + v_t = L_t + u_t + w_t
        (2) 储能动态      E_t = E_{t-1} + η_c·u_t·Δt − v_t·Δt/η_d
        (3) 储电量范围    E_min ≤ E_t ≤ E_max
        (5) 变量取值范围  见上表

    唯一的差别 —— 约束 (4) 日末回归
        约束 A（固定）   E_T = E_0 = 6000
        约束 B（自由）   E_T − E_0 = 0，共同取值由优化决定

    因往返效率 η_c·η_d = 0.81 < 1，最优解自动满足 min(u_t, v_t) = 0，
    无需引入 0-1 变量，模型保持纯线性，用 HiGHS 一次求出全局最优。

【与 A 的关系】
    约束 A 要求 E_T = E_0 = 6000。把 E_0 取成 6000，A 的任意可行解在 B 中均可行，
    故 B 的可行域更大，必有 C(B) ≤ C(A)。两个函数除第 (4) 条约束外逐行相同。

【输出】
    同一个表格文件 问题一结果_约束AB.xlsx，三张 sheet：
        约束A（固定）   144 行逐时段明细
        约束B（自由）   144 行逐时段明细
        汇总            两口径指标对比
    另导出两份 csv（供出图脚本读取）与两份题目模板格式的 result1。

运行：python solve_q1_ab.py
依赖：pandas、numpy、scipy、openpyxl

=============================================================================
"""

import os
import sys
import time
import datetime as dt
import numpy as np
import pandas as pd


from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix

import openpyxl
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ===========================================================================
# 参数与路径
# ===========================================================================
BASE = os.path.dirname(os.path.abspath(__file__))         
ROOT = os.path.dirname(BASE)                               
def _locate_attachments():
    base = os.path.abspath(BASE)
    while True:
        for cand in (os.path.join(base, "C题", "附件"), os.path.join(base, "附件")):
            if os.path.exists(os.path.join(cand, "附件1.xlsx")):
                return cand
        parent = os.path.dirname(base)
        if parent == base:
            break
        base = parent
    raise FileNotFoundError("未找到 附件1.xlsx，请保持 C题/附件/ 目录完整")

_ATTACH = _locate_attachments()
ATT1 = os.path.join(_ATTACH, "附件1.xlsx")
TEMPLATE = os.path.join(_ATTACH, "附件5", "result1.xlsx")
PROJECT = os.path.dirname(os.path.dirname(BASE))
OUTDIR = os.path.join(PROJECT, "结果", "问题1")

T = 144                     # 一天内的时段总数
DT = 1.0 / 6.0              # 时段时长 Δt，单位 h
E_MIN = 1200.0              # 储电量下限 E_min，kWh
E_MAX = 10800.0             # 储电量上限 E_max，kWh
E_ENDPOINT = 6000.0         # 约束 A 的固定端点储电量，kWh
P_ST_MAX = 5000.0           # 最大充放电功率 P_st^max，kW
ETA_C = 0.9                 # 充电效率 η_c
ETA_D = 0.9                 # 放电效率 η_d
EPS_W = 1e-6                # 仅为消除弃光量不唯一的数值项，不影响费用

LABEL_A = "约束A（固定）"
LABEL_B = "约束B（自由）"
XLSX_NAME = "问题一结果_约束AB.xlsx"

ROW_MODE = "sequential"

def title(text):
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


# ===========================================================================
# 读取附件 1：c_t、L_t、P_t^fc
# ===========================================================================
def cell_to_minutes(v):
    """
    附件 1 的时间列混用了时间型与文本型（最后一个为 '0:00+1'），
    统一折算成距 0 点的分钟数，用于校验时间轴。
    """
    if isinstance(v, (dt.time, dt.datetime, pd.Timestamp)):
        return v.hour * 60 + v.minute
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        x = float(v)
        return int(round(x * 1440)) if x <= 1.0000001 else int(round(x))
    if isinstance(v, str):
        s = v.strip().replace("：", ":")
        plus = s.endswith("+1")
        if plus:
            s = s[:-2].strip()
        hh, mm = s.split(":")[:2]
        return int(hh) * 60 + int(mm) + (1440 if plus else 0)
    raise ValueError("无法解析的时间值：%r" % v)


def read_attachment1():
    """返回 (c, L, P_fc, minutes)：电价 元/kWh、负载功率 kW、光伏预测功率 kW"""
    df = pd.read_excel(ATT1, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)
    col_c = next(c for c in cols if "电价" in c)
    col_L = next(c for c in cols if "负载" in c)
    col_P = next(c for c in cols if "光伏" in c)
    c = df[col_c].astype(float).values
    L = df[col_L].astype(float).values
    P_fc = df[col_P].astype(float).values
    minutes = [cell_to_minutes(v) for v in df[cols[0]]]
    return c, L, P_fc, minutes


# ===========================================================================
# 两个线性规划函数：只有约束 (4) 不同
# ===========================================================================
def solve_a(c, L, P_fc, dload=None, return_lp=False):
    """
    约束 A（固定）：日循环两端固定为 6000 kWh

        min  Σ c_t x_t Δt
        s.t. (1) x_t + P_t^fc + v_t = L_t + u_t + w_t
             (2) E_t − E_{t−1} − η_c Δt u_t + (Δt/η_d) v_t = 0
             (3) E_min ≤ E_t ≤ E_max
             (4) E_0 = 6000 且 E_T = 6000        ← 与 B 的唯一差别
             (5) x_t ≥ 0, 0 ≤ u_t, v_t ≤ P_st^max, w_t ≥ 0

    dload：可选的长度 T 数组，给第 t 时段额外加 dload[t] kW 负载。
           仅供检验脚本做数值微分求影子价格用，正常求解传 None 即可，
           此时本函数与不带该参数时完全一致。
    return_lp：置 True 时，返回值里额外带上 A_eq、b_eq、bounds、c_obj、
           解向量 z 与等式约束的对偶值 y，供最优性检验（KKT + 强对偶）用。
    """
    # 解向量布局：x(T) | u(T) | v(T) | E(T+1) | w(T)
    iX, iU, iV, iE, iW = 0, T, 2 * T, 3 * T, 4 * T + 1
    nv = 5 * T + 1

    # ---- 目标函数：min Σ c_t x_t Δt
    c_obj = np.zeros(nv)
    c_obj[iX:iX + T] = c * DT
    c_obj[iW:iW + T] = EPS_W

    # ---- 等式约束 (1)(2)(4)
    A_eq = lil_matrix((2 * T + 2, nv))
    b_eq = np.zeros(2 * T + 2)
    for t in range(T):
        # (1) x_t − u_t + v_t − w_t = L_t − P_t^fc
        A_eq[t, iX + t] = 1.0
        A_eq[t, iU + t] = -1.0
        A_eq[t, iV + t] = 1.0
        A_eq[t, iW + t] = -1.0
        b_eq[t] = L[t] - P_fc[t] + (0.0 if dload is None else float(dload[t]))
        # (2) E_t − E_{t−1} − η_c Δt u_t + (Δt/η_d) v_t = 0
        A_eq[T + t, iE + t] = -1.0
        A_eq[T + t, iE + t + 1] = 1.0
        A_eq[T + t, iU + t] = -ETA_C * DT
        A_eq[T + t, iV + t] = DT / ETA_D
    # (4) 日末回归：E_0 = 6000 且 E_T = 6000（两条独立等式）
    A_eq[2 * T, iE] = 1.0
    b_eq[2 * T] = E_ENDPOINT
    A_eq[2 * T + 1, iE + T] = 1.0
    b_eq[2 * T + 1] = E_ENDPOINT

    # ---- 变量边界 (3)(5)
    bounds = ([(0.0, None)] * T                 # x_t ≥ 0
              + [(0.0, P_ST_MAX)] * T           # 0 ≤ u_t ≤ P_st^max
              + [(0.0, P_ST_MAX)] * T           # 0 ≤ v_t ≤ P_st^max
              + [(E_MIN, E_MAX)] * (T + 1)      # E_min ≤ E_t ≤ E_max
              + [(0.0, None)] * T)              # w_t ≥ 0

    t0 = time.time()
    res = linprog(c_obj, A_eq=csr_matrix(A_eq), b_eq=b_eq, bounds=bounds, method="highs")
    elapsed = time.time() - t0
    if not res.success:
        raise RuntimeError("约束 A 求解失败：" + str(res.message))

    z = res.x
    out = {
        "x": z[iX:iX + T], "u": z[iU:iU + T], "v": z[iV:iV + T],
        "E": z[iE:iE + T + 1], "w": z[iW:iW + T],
        "C": float(res.fun), "seconds": elapsed, "status": res.message,
        "label": LABEL_A,
    }
    if return_lp:
        out.update({
            "z": z, "c_obj": c_obj, "A_eq": csr_matrix(A_eq), "b_eq": b_eq,
            "bounds": bounds,
            "y": (np.asarray(res.eqlin.marginals)
                  if hasattr(res, "eqlin") else None),
        })
    return out


def solve_b(c, L, P_fc, dload=None, return_lp=False):
    """
    约束 B（自由）：日循环两端相等、共同取值由优化决定

        min  Σ c_t x_t Δt
        s.t. (1) x_t + P_t^fc + v_t = L_t + u_t + w_t
             (2) E_t − E_{t−1} − η_c Δt u_t + (Δt/η_d) v_t = 0
             (3) E_min ≤ E_t ≤ E_max
             (4) E_T − E_0 = 0                    ← 与 A 的唯一差别
             (5) x_t ≥ 0, 0 ≤ u_t, v_t ≤ P_st^max, w_t ≥ 0

    dload / return_lp：同 solve_a。
    """
    # 解向量布局：x(T) | u(T) | v(T) | E(T+1) | w(T)
    iX, iU, iV, iE, iW = 0, T, 2 * T, 3 * T, 4 * T + 1
    nv = 5 * T + 1

    # ---- 目标函数：min Σ c_t x_t Δt
    c_obj = np.zeros(nv)
    c_obj[iX:iX + T] = c * DT
    c_obj[iW:iW + T] = EPS_W

    # ---- 等式约束 (1)(2)(4)（比 A 少一条：两端不各自固定）
    A_eq = lil_matrix((2 * T + 1, nv))
    b_eq = np.zeros(2 * T + 1)
    for t in range(T):
        # (1) x_t − u_t + v_t − w_t = L_t − P_t^fc
        A_eq[t, iX + t] = 1.0
        A_eq[t, iU + t] = -1.0
        A_eq[t, iV + t] = 1.0
        A_eq[t, iW + t] = -1.0
        b_eq[t] = L[t] - P_fc[t] + (0.0 if dload is None else float(dload[t]))
        # (2) E_t − E_{t−1} − η_c Δt u_t + (Δt/η_d) v_t = 0
        A_eq[T + t, iE + t] = -1.0
        A_eq[T + t, iE + t + 1] = 1.0
        A_eq[T + t, iU + t] = -ETA_C * DT
        A_eq[T + t, iV + t] = DT / ETA_D
    # (4) 日末回归：E_T − E_0 = 0（只约束两端相等）
    A_eq[2 * T, iE] = 1.0
    A_eq[2 * T, iE + T] = -1.0
    b_eq[2 * T] = 0.0

    # ---- 变量边界 (3)(5)
    bounds = ([(0.0, None)] * T                 # x_t ≥ 0
              + [(0.0, P_ST_MAX)] * T           # 0 ≤ u_t ≤ P_st^max
              + [(0.0, P_ST_MAX)] * T           # 0 ≤ v_t ≤ P_st^max
              + [(E_MIN, E_MAX)] * (T + 1)      # E_min ≤ E_t ≤ E_max
              + [(0.0, None)] * T)              # w_t ≥ 0

    t0 = time.time()
    res = linprog(c_obj, A_eq=csr_matrix(A_eq), b_eq=b_eq, bounds=bounds, method="highs")
    elapsed = time.time() - t0
    if not res.success:
        raise RuntimeError("约束 B 求解失败：" + str(res.message))

    z = res.x
    out = {
        "x": z[iX:iX + T], "u": z[iU:iU + T], "v": z[iV:iV + T],
        "E": z[iE:iE + T + 1], "w": z[iW:iW + T],
        "C": float(res.fun), "seconds": elapsed, "status": res.message,
        "label": LABEL_B,
    }
    if return_lp:
        out.update({
            "z": z, "c_obj": c_obj, "A_eq": csr_matrix(A_eq), "b_eq": b_eq,
            "bounds": bounds,
            "y": (np.asarray(res.eqlin.marginals)
                  if hasattr(res, "eqlin") else None),
        })
    return out


# ===========================================================================
# 约束校验
# ===========================================================================
def validate(sol, L, P_fc):
    x, u, v, E, w = sol["x"], sol["u"], sol["v"], sol["E"], sol["w"]
    return {
        "① 电量平衡残差最大值(kW)": float(np.max(np.abs(x + P_fc + v - L - u - w))),
        "② 储能动态残差最大值(kWh)": float(np.max(np.abs(
            E[1:] - E[:-1] - ETA_C * DT * u + DT / ETA_D * v))),
        "③ 储电量最小值(kWh)": float(np.min(E)),
        "③ 储电量最大值(kWh)": float(np.max(E)),
        "③ 储电量越界": bool(np.min(E) < E_MIN - 1e-6 or np.max(E) > E_MAX + 1e-6),
        "④ E_0 / E_T(kWh)": (round(float(E[0]), 4), round(float(E[-1]), 4)),
        "④ |E_T − E_0|(kWh)": float(abs(E[-1] - E[0])),
        "⑤ x_t 出现负值": bool(np.min(x) < -1e-6),
        "⑤ u_t 或 v_t 越限": bool(np.max(u) > P_ST_MAX + 1e-6 or np.max(v) > P_ST_MAX + 1e-6),
        "互斥 min(u_t,v_t) 最大值(kW)": float(np.max(np.minimum(u, v))),
    }


# ===========================================================================
# 数据预处理
# ===========================================================================
def preprocess(c, L, P_fc):
    """
    把附件 1 整理成一张标准表格。

    行数 145，时间列取 0,1,…,144（单位 10 min，0 表示 00:00，144 表示 24:00）：
        时间 0 ~ 143：每行是一个时段，给出以该时刻为起点的 10 分钟时段的输入数据；
        时间 144    ：24:00 的收口行，只保留时间信息，用于对齐 E_144。

    派生列「净负荷」= L_t − P_t^fc，单位 kW：
        净负荷 > 0  该时段必须购电；
        净负荷 < 0  光伏有富余，只能充电或弃光（不能上网）。
    净负荷是无储能对照与后续所有分析的基准量。
    """
    net = L - P_fc
    rows = []
    for k in range(T):
        rows.append({
            "时间": k,
            "时间标签": "%02d:%02d" % ((k * 10) // 60, (k * 10) % 60),
            "时段 t": k + 1,
            "时段区间": "%02d:%02d-%02d:%02d" % ((k * 10) // 60, (k * 10) % 60,
                                              (k * 10 + 10) // 60, (k * 10 + 10) % 60),
            "c_t (元/kWh)": round(float(c[k]), 6),
            "L_t (kW)": round(float(L[k]), 4),
            "P_t^fc (kW)": round(float(P_fc[k]), 4),
            "净负荷 (kW)": round(float(net[k]), 4),
        })
    rows.append({
        "时间": T,
        "时间标签": "24:00",
        "时段 t": None,
        "时段区间": "",
        "c_t (元/kWh)": None,
        "L_t (kW)": None,
        "P_t^fc (kW)": None,
        "净负荷 (kW)": None,
    })
    df = pd.DataFrame(rows)
    df["时段 t"] = df["时段 t"].astype("Int64")     # 末行无时段，保持整数显示
    return df, net


def attach_results(pre, sol):
    """
    把求解结果并回预处理表。
    时间 k 行同时给出：该时刻的储电量 E_k，以及以该时刻为起点的时段 k+1 的
    购电功率 x、充电功率 u、放电功率 v、弃光功率 w。
    """
    df = pre.copy()
    x, u, v, w = list(sol["x"]), list(sol["u"]), list(sol["v"]), list(sol["w"])
    df["x_t (kW)"] = [round(q, 4) for q in x] + [None]
    df["u_t (kW)"] = [round(q, 4) for q in u] + [None]
    df["v_t (kW)"] = [round(q, 4) for q in v] + [None]
    df["w_t (kW)"] = [round(q, 4) for q in w] + [None]
    df["x_t·Δt (kWh)"] = [round(q * DT, 4) for q in x] + [None]
    df["E_t (kWh)"] = [round(e, 4) for e in sol["E"]]
    return df


# ===========================================================================
# 汇总指标
# ===========================================================================
def summarize(c, L, P_fc, net, sol):
    """把两个口径要对比的指标算成一张定序字典"""
    x, u, v, w, E = sol["x"], sol["u"], sol["v"], sol["w"], sol["E"]
    Q = float((x * DT).sum())                       # 全天购电量
    C = float((c * x * DT).sum())                   # 全天购电费
    Q_u = float((u * DT).sum())                     # 充电总量
    Q_v = float((v * DT).sum())                     # 放电总量
    Q_pv = float((P_fc * DT).sum())                 # 光伏发电量
    Q_w = float((w * DT).sum())                     # 弃光电量
    C_base = float((c * np.maximum(net, 0.0) * DT).sum())   # 无储能基准购电费
    return {
        "端点约束": ("E_0 = E_T = %.0f kWh" % E_ENDPOINT) if sol["label"] == LABEL_A
                    else "E_T − E_0 = 0（共同取值自由）",
        "全天购电量 Q (kWh)": Q,
        "全天购电费 C (元)": C,
        "平均购电价 (元/kWh)": C / Q,
        "充电总量 ΣuΔt (kWh)": Q_u,
        "放电总量 ΣvΔt (kWh)": Q_v,
        "往返损耗 (kWh)": Q_u - Q_v,
        "光伏发电量 ΣP^fcΔt (kWh)": Q_pv,
        "弃光电量 ΣwΔt (kWh)": Q_w,
        "弃光率 (%)": Q_w / Q_pv * 100.0,
        "E_0 (kWh)": float(E[0]),
        "E_T (kWh)": float(E[-1]),
        "无储能基准购电费 (元)": C_base,
        "储能节省 (元)": C_base - C,
        "储能节省率 (%)": (C_base - C) / C_base * 100.0,
        "求解耗时 (s)": sol["seconds"],
    }


# ===========================================================================
# 输出：一个表格文件、三张 sheet
# ===========================================================================
def _write_detail_sheet(ws, full):
    """写一张 145 行逐时段明细表"""
    def clean(v):
        """把 pandas 的缺失值（None / NaN / pd.NA）统一成 Excel 的空单元格"""
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return float(v)
        return v

    ws.append(list(full.columns))
    for _, r in full.iterrows():
        ws.append([clean(v) for v in r.tolist()])
    ws.freeze_panes = "A2"


def _write_summary_sheet(ws, met_a, met_b):
    """写一张两口径对比表：指标 / A / B / 差值(B−A)"""
    ws.append(["指标", LABEL_A, LABEL_B, "差值 (B − A)"])
    for k in met_a:
        va, vb = met_a[k], met_b[k]
        if isinstance(va, str):
            ws.append([k, va, vb, "—"])
        else:
            ws.append([k, round(va, 4), round(vb, 4), round(vb - va, 4)])
    ws.column_dimensions["A"].width = 28
    for col in ("B", "C", "D"):
        ws.column_dimensions[col].width = 22
    ws.freeze_panes = "A2"


def write_workbook(path, pre, sol_a, sol_b, met_a, met_b):
    """同一个表格文件的三张 sheet：约束A（固定）/ 约束B（自由）/ 汇总"""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = LABEL_A
    _write_detail_sheet(ws, attach_results(pre, sol_a))
    _write_detail_sheet(wb.create_sheet(LABEL_B), attach_results(pre, sol_b))
    _write_summary_sheet(wb.create_sheet("汇总"), met_a, met_b)
    wb.save(path)
    return path


def build_row_order():
    """模板第 j 行（0 起）应填入的时段下标（0 起）"""
    if ROW_MODE == "label":
        return [(j + 1) % T for j in range(T)]
    return [j for j in range(T)]


def write_template_result(sol):
    """按附件 5 的 result1.xlsx 模板格式，输出该口径的结果文件"""
    wb = openpyxl.load_workbook(TEMPLATE)
    order = build_row_order()

    ws1 = wb["计划购电量"]
    for j, t in enumerate(order):
        ws1.cell(row=2 + j, column=2).value = round(float(sol["x"][t] * DT), 4)

    ws2 = wb["充放电量"]
    for k in range(6):
        sl = slice(k * 24, (k + 1) * 24)
        ws2.cell(row=2 + k, column=2).value = round(float(sol["u"][sl].sum() * DT), 4)
        ws2.cell(row=2 + k, column=3).value = round(float(sol["v"][sl].sum() * DT), 4)
    ws2.cell(row=2, column=5).value = round(float(sol["E"][0]), 4)
    ws2.cell(row=3, column=5).value = round(float(sol["E"][-1]), 4)

    out = os.path.join(OUTDIR, "result1_%s.xlsx" % sol["label"])
    wb.save(out)
    return out


def write_csv(pre, sol):
    """导出供出图脚本读取的明细 csv"""
    full = attach_results(pre, sol)
    path = os.path.join(OUTDIR, "%s_完整结果.csv" % sol["label"])
    full.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ===========================================================================
# 主流程：读一次数据，分别调用 A、B，写进同一个表文件
# ===========================================================================
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    title("问题一：约束 A（固定端点）与约束 B（首尾自由）合并求解")

    c, L, P_fc, minutes = read_attachment1()
    if len(c) != T:
        sys.exit("附件 1 行数不是 %d，实际 %d" % (T, len(c)))

    title("1. 数据预处理（两口径共用）")
    pre, net = preprocess(c, L, P_fc)
    print("  表形状：%d 行 × %d 列（时间 0 ~ 144，单位 10 min）" % pre.shape)
    print("  列名：", list(pre.columns))
    print("  时间轴校验：首 %d 末 %d，相邻间隔唯一值 %s"
          % (pre["时间"].iloc[0], pre["时间"].iloc[-1],
             sorted(set(np.diff(minutes).tolist()))))
    print("  净负荷：最小 %.2f kW，最大 %.2f kW，为正的时段 %d 个（需购电）"
          % (net.min(), net.max(), int((net > 0).sum())))
    pre_path = os.path.join(OUTDIR, "预处理表.csv")
    pre.to_csv(pre_path, index=False, encoding="utf-8-sig")
    print("  已写出：", pre_path)

    title("2. 求解")
    print("  调用 solve_a()：约束 (4)  E_0 = E_T = %.0f kWh" % E_ENDPOINT)
    sol_a = solve_a(c, L, P_fc)
    print("    状态 %s，耗时 %.4f 秒" % (sol_a["status"], sol_a["seconds"]))
    print("  调用 solve_b()：约束 (4)  E_T − E_0 = 0，共同取值自由")
    sol_b = solve_b(c, L, P_fc)
    print("    状态 %s，耗时 %.4f 秒" % (sol_b["status"], sol_b["seconds"]))
    print("    优化得到的循环储电量 E_0 = E_T = %.4f kWh" % sol_b["E"][0])

    title("3. 约束校验")
    for name, sol in ((LABEL_A, sol_a), (LABEL_B, sol_b)):
        print("  【%s】" % name)
        for k, v in validate(sol, L, P_fc).items():
            print("    %-28s %s" % (k, round(v, 8) if isinstance(v, float) else v))

    title("4. 结果汇总与对比")
    met_a = summarize(c, L, P_fc, net, sol_a)
    met_b = summarize(c, L, P_fc, net, sol_b)
    print("  %-28s %18s %18s %16s" % ("指标", LABEL_A, LABEL_B, "差值(B−A)"))
    for k in met_a:
        va, vb = met_a[k], met_b[k]
        if isinstance(va, str):
            print("  %-28s %18s %18s %16s" % (k, va, vb, "—"))
        else:
            print("  %-28s %18.4f %18.4f %16.4f" % (k, va, vb, vb - va))
    if met_b["全天购电费 C (元)"] > met_a["全天购电费 C (元)"] + 1e-6:
        print("  警告：约束 B 的可行域包含 A，理论上不应更贵，请检查建模。")

    title("5. 写出结果（同一个表文件的两张 sheet + 汇总）")
    xlsx = write_workbook(os.path.join(OUTDIR, XLSX_NAME), pre, sol_a, sol_b, met_a, met_b)
    print("  已写出：%s（sheet：%s / %s / 汇总）" % (xlsx, LABEL_A, LABEL_B))
    for sol in (sol_a, sol_b):
        print("  已写出：", write_csv(pre, sol))
        print("  已写出：", write_template_result(sol))

    title("6. 论文表（指定时段购电量 x_t·Δt，kWh）")
    print("  %-14s %14s %14s" % ("时段", LABEL_A, LABEL_B))
    for m in [600, 720, 840, 960, 1080, 1200]:
        t = (m - 10) // 10
        print("  %02d:%02d-%02d:%02d %14.4f %14.4f"
              % (m // 60, m % 60, (m + 10) // 60, (m + 10) % 60,
                 sol_a["x"][t] * DT, sol_b["x"][t] * DT))
    print("  %-11s %14.4f %14.4f" % ("全天", met_a["全天购电量 Q (kWh)"],
                                     met_b["全天购电量 Q (kWh)"]))
    print("  %-11s %14.4f %14.4f" % ("全天购电费", met_a["全天购电费 C (元)"],
                                     met_b["全天购电费 C (元)"]))

    title("7. 论文表（分时段充放电量，kWh）")
    names = ["0:00-4:00", "4:00-8:00", "8:00-12:00",
             "12:00-16:00", "16:00-20:00", "20:00-24:00"]
    for name, sol in ((LABEL_A, sol_a), (LABEL_B, sol_b)):
        print("  【%s】" % name)
        for k, nm in enumerate(names):
            sl = slice(k * 24, (k + 1) * 24)
            print("    %-12s ΣuΔt %11.4f   ΣvΔt %11.4f"
                  % (nm, sol["u"][sl].sum() * DT, sol["v"][sl].sum() * DT))
        print("    E_0 = %.4f kWh，E_T = %.4f kWh" % (sol["E"][0], sol["E"][-1]))

    title("8. 完成")
    print("  产出目录：%s" % OUTDIR)


if __name__ == "__main__":
    main()
