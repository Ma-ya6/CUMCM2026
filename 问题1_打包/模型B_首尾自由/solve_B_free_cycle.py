# -*- coding: utf-8 -*-
"""
C 题 问题一 · 口径 B：日循环两端相等、共同取值自由
（代码与《C 题符号规定》《C 题 问题一 模型》逐条对应）

=============================================================================
【模型】

    索引与集合
        t = 1,2,…,T 日内十分钟时段序号（t = 1 为 00:00–00:10），T = 144
        Δt = 1/6 h

    参数（取自附件 1 与附录 1）
        c_t              第 t 时段外网电价，元/kWh
        L_t              小区负载功率，kW
        P_t^fc           光伏发电功率预测值，kW
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

    约束条件
        (1) 电量平衡      x_t + P_t^fc + v_t = L_t + u_t + w_t
        (2) 储能动态      E_t = E_{t-1} + η_c·u_t·Δt − v_t·Δt/η_d
        (3) 储电量范围    E_min ≤ E_t ≤ E_max
        (4) 日末回归      E_T = E_0               ← 本文件的口径，共同取值自由
        (5) 变量取值范围  见上表

    说明：往返效率 η_c·η_d = 0.81 < 1，最优解自动满足 min(u_t, v_t) = 0，
          无需引入 0-1 变量，模型保持纯线性。

【与口径 A 的关系】
    口径 A 要求 E_T = E_0 = 6000。把 E_0 取成 6000，A 的任意可行解在 B 中均可行，
    故 B 的可行域更大，必有 C(B) ≤ C(A)。两份代码除第 (4) 条约束外完全相同。
    本口径不指定 0:00 的储电量，其取值由优化决定，含义是把这一天看成纯粹的
    循环调度周期，考察储能容量的充分套利潜力；代价是计划起点可能与实际初始
    状态（6000 kWh）不一致，执行时需先补足差额。

【求解方式】
    线性规划，用 HiGHS 求解，一次得到全局最优。

运行：python solve_B_free_cycle.py
依赖：pandas、numpy、scipy、openpyxl
产出：output/问题一_口径B_首尾自由/
原附件只读。
=============================================================================
"""

import os
import sys
import time
import datetime as dt
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix, csr_matrix
except Exception:
    sys.exit("缺少 scipy，请先执行：python -m pip install scipy")

try:
    import openpyxl
except Exception:
    sys.exit("缺少 openpyxl，请先执行：python -m pip install openpyxl")


# ===========================================================================
# 参数与路径
# ===========================================================================
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ATT1 = os.path.join(ROOT, "CUMCM2026", "C题", "附件", "附件1.xlsx")
TEMPLATE = os.path.join(ROOT, "CUMCM2026", "C题", "附件", "附件5", "result1.xlsx")
OUTDIR = os.path.join(ROOT, "output", "问题一_口径B_首尾自由")

T = 144                     # 一天内的时段总数
DT = 1.0 / 6.0              # 时段时长 Δt，单位 h
E_MIN = 1200.0              # 储电量下限 E_min，kWh
E_MAX = 10800.0             # 储电量上限 E_max，kWh
P_ST_MAX = 5000.0           # 最大充放电功率 P_st^max，kW
ETA_C = 0.9                 # 充电效率 η_c
ETA_D = 0.9                 # 放电效率 η_d

# 结果文件填行口径，与口径 A 保持一致，保证差异只来自端点约束
ROW_MODE = "sequential"


def title(text):
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


# ===========================================================================
# 读取附件 1：c_t、L_t、P_t^fc
# ===========================================================================
def cell_to_minutes(v):
    """附件 1 的时间列混用了时间型与文本型（最后一个为 '0:00+1'），统一折算成分钟数"""
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
# 建立并求解线性规划
# ===========================================================================
def solve_q1(c, L, P_fc):
    """
    min  Σ c_t x_t Δt
    s.t. (1) x_t + P_t^fc + v_t = L_t + u_t + w_t
         (2) E_t - E_{t-1} - η_c Δt u_t + (Δt/η_d) v_t = 0
         (3) E_min ≤ E_t ≤ E_max
         (4) E_T - E_0 = 0        ← 共同取值自由
         (5) x_t ≥ 0, 0 ≤ u_t,v_t ≤ P_st^max, w_t ≥ 0
    """
    # 解向量布局：x(T) | u(T) | v(T) | E(T+1) | w(T)
    iX, iU, iV, iE, iW = 0, T, 2 * T, 3 * T, 4 * T + 1
    nv = 5 * T + 1

    # ---- 目标函数：min Σ c_t x_t Δt
    c_obj = np.zeros(nv)
    c_obj[iX:iX + T] = c * DT
    c_obj[iW:iW + T] = 1e-6          # 仅为消除弃光量不唯一的数值项，不影响费用

    # ---- 等式约束（比口径 A 少一条：两端不各自固定）
    A_eq = lil_matrix((2 * T + 1, nv))
    b_eq = np.zeros(2 * T + 1)
    for t in range(T):
        # (1) x_t - u_t + v_t - w_t = L_t - P_t^fc
        A_eq[t, iX + t] = 1.0
        A_eq[t, iU + t] = -1.0
        A_eq[t, iV + t] = 1.0
        A_eq[t, iW + t] = -1.0
        b_eq[t] = L[t] - P_fc[t]
        # (2) E_t - E_{t-1} - η_c Δt u_t + (Δt/η_d) v_t = 0
        A_eq[T + t, iE + t] = -1.0
        A_eq[T + t, iE + t + 1] = 1.0
        A_eq[T + t, iU + t] = -ETA_C * DT
        A_eq[T + t, iV + t] = DT / ETA_D
    # (4) 日末回归：E_T - E_0 = 0
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
        raise RuntimeError("求解失败：" + str(res.message))

    z = res.x
    return {
        "x": z[iX:iX + T], "u": z[iU:iU + T], "v": z[iV:iV + T],
        "E": z[iE:iE + T + 1], "w": z[iW:iW + T],
        "C": float(res.fun), "seconds": elapsed, "status": res.message,
    }


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
        "④ 日末回归 |E_T - E_0|(kWh)": float(abs(E[-1] - E[0])),
        "④ 优化得到的循环储电量(kWh)": float(E[0]),
        "⑤ x_t 出现负值": bool(np.min(x) < -1e-6),
        "⑤ u_t 或 v_t 越限": bool(np.max(u) > P_ST_MAX + 1e-6 or np.max(v) > P_ST_MAX + 1e-6),
        "互斥 min(u_t,v_t) 最大值(kW)": float(np.max(np.minimum(u, v))),
    }


# ===========================================================================
# 输出：result1.xlsx 与明细
# ===========================================================================
def build_row_order():
    """模板第 j 行（0 起）应填入的时段下标（0 起）"""
    if ROW_MODE == "label":
        return [(j + 1) % T for j in range(T)]
    return [j for j in range(T)]


def read_template_labels():
    wb = openpyxl.load_workbook(TEMPLATE, data_only=True)
    ws = wb["计划购电量"]
    return [str(ws.cell(row=2 + i, column=1).value) for i in range(T)]


def write_result1(sol):
    """计划购电量表填 x_t Δt；充放电量表按四个小时段汇总 u_t Δt、v_t Δt"""
    os.makedirs(OUTDIR, exist_ok=True)
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

    out = os.path.join(OUTDIR, "result1.xlsx")
    wb.save(out)
    return out


def preprocess(c, L, P_fc):
    """
    【数据预处理】把附件 1 整理成一张标准表格。

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
# 主流程
# ===========================================================================
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    title("口径 B：日循环两端相等、共同取值自由")

    c, L, P_fc, minutes = read_attachment1()
    if len(c) != T:
        sys.exit("附件 1 行数不是 %d，实际 %d" % (T, len(c)))

    title("1. 数据预处理")
    pre, net = preprocess(c, L, P_fc)
    print("  表形状：%d 行 × %d 列（时间 0 ~ 144，单位 10 min）" % pre.shape)
    print("  列名：", list(pre.columns))
    print("  时间轴校验：首 %d 末 %d，相邻间隔唯一值 %s"
          % (pre["时间"].iloc[0], pre["时间"].iloc[-1],
             sorted(set(np.diff(minutes).tolist()))))
    print("  净负荷：最小 %.2f kW，最大 %.2f kW，为正的时段 %d 个（需购电）"
          % (net.min(), net.max(), int((net > 0).sum())))
    print("  预处理表前 3 行：")
    print(pre.head(3).to_string(index=False))
    print("  预处理表末 2 行：")
    print(pre.tail(2).to_string(index=False))
    pre_path = os.path.join(OUTDIR, "预处理表.csv")
    pre.to_csv(pre_path, index=False, encoding="utf-8-sig")
    print("  已写出：", pre_path)

    title("2. 求解")
    print("  约束 (4)：E_T - E_0 = 0，共同取值由优化决定，且落在 [%.0f, %.0f]" % (E_MIN, E_MAX))
    sol = solve_q1(c, L, P_fc)
    print("  状态：%s" % sol["status"])
    print("  耗时：%.4f 秒" % sol["seconds"])
    print("  优化得到的循环储电量 E_0 = E_T = %.4f kWh" % sol["E"][0])

    title("3. 约束校验")
    for k, v in validate(sol, L, P_fc).items():
        print("  %-28s %s" % (k, round(v, 8) if isinstance(v, float) else v))

    Q = float((sol["x"] * DT).sum())
    C = float((c * sol["x"] * DT).sum())
    Q_u = float((sol["u"] * DT).sum())
    Q_v = float((sol["v"] * DT).sum())
    Q_pv = float((P_fc * DT).sum())
    Q_w = float((sol["w"] * DT).sum())
    C_base = float((c * np.maximum(net, 0.0) * DT).sum())

    title("4. 结果汇总（口径 B）")
    print("  全天购电量 Q        %.4f kWh" % Q)
    print("  全天购电费 C        %.4f 元" % C)
    print("  平均购电价          %.4f 元/kWh" % (C / Q))
    print("  充电总量 ΣuΔt       %.4f kWh" % Q_u)
    print("  放电总量 ΣvΔt       %.4f kWh" % Q_v)
    print("  往返损耗            %.4f kWh" % (Q_u - Q_v))
    print("  光伏发电量 ΣP^fcΔt  %.4f kWh" % Q_pv)
    print("  弃光电量 ΣwΔt       %.4f kWh（%.4f%%）" % (Q_w, Q_w / Q_pv * 100))
    print("  E_0 / E_T           %.4f / %.4f kWh" % (sol["E"][0], sol["E"][-1]))
    print("  无储能基准购电费    %.4f 元" % C_base)
    print("  储能节省            %.4f 元（%.4f%%）" % (C_base - C, (C_base - C) / C_base * 100))

    title("5. 写出结果")
    print("  已写出：", write_result1(sol))
    full = attach_results(pre, sol)
    full_path = os.path.join(OUTDIR, "问题一_完整结果.csv")
    full.to_csv(full_path, index=False, encoding="utf-8-sig")
    print("  已写出：", full_path)

    title("6. 论文表 1（指定时段购电量 x_t·Δt）")
    for m in [600, 720, 840, 960, 1080, 1200]:
        t = (m - 10) // 10
        print("   %02d:%02d-%02d:%02d   %12.4f kWh"
              % (m // 60, m % 60, (m + 10) // 60, (m + 10) % 60, sol["x"][t] * DT))
    print("   全天购电量 Q          %12.4f kWh" % Q)
    print("   全天购电费 C          %12.4f 元" % C)

    title("7. 论文表 2（分时段充放电量）")
    names = ["0:00-4:00", "4:00-8:00", "8:00-12:00",
             "12:00-16:00", "16:00-20:00", "20:00-24:00"]
    for k, nm in enumerate(names):
        sl = slice(k * 24, (k + 1) * 24)
        print("   %-12s ΣuΔt %11.4f   ΣvΔt %11.4f kWh"
              % (nm, sol["u"][sl].sum() * DT, sol["v"][sl].sum() * DT))
    print("   E_0 = %.4f kWh，E_T = %.4f kWh" % (sol["E"][0], sol["E"][-1]))

    title("8. 完成")
    print("  产出目录：%s" % OUTDIR)


if __name__ == "__main__":
    main()
