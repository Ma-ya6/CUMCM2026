# -*- coding: utf-8 -*-
"""
问题一 · 检验（一）：模型检验表 —— 可行性 + 全天能量守恒

【设计原则】只读 results\\ 下现成的两份结果 csv，不调用求解器。
            这样它才是对解的外部验证，而不是拿求解器给自己打分。
            需要求解器的两项检验见 check_optimal.py 与 sensitivity.py。

检验项（约束 A、约束 B 各一列）：
    ① 电量平衡残差最大值
    ② 储能递推残差最大值
    ③ 储电量上下限是否越界
    ④ 充放电功率是否越限
    ⑤ 购电/弃光是否非负、是否同时充放电
    ⑥ 全天能量守恒（144 个时段的平衡式求和）
    ⑦ 充放电量比 Σv/Σu 是否等于往返效率 η_c·η_d

用法：python check_model.py
产出：tables\\模型检验表.xlsx
"""

import os
import sys
import numpy as np
import pandas as pd

import tableio

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------- 常量与路径
BASE = os.path.dirname(os.path.abspath(__file__))          # …\问题1\validation
Q1 = os.path.dirname(BASE)                                 # …\问题1
RESULTS = os.path.join(Q1, "results")
TABDIR = os.path.join(BASE, "tables")

T = 144
DT = 1.0 / 6.0
E_MIN, E_MAX = 1200.0, 10800.0
E_ENDPOINT = 6000.0                # 约束 A 的固定端点储电量
P_ST_MAX = 5000.0
ETA_C = ETA_D = 0.9
ETA_ROUND = ETA_C * ETA_D          # 往返效率 0.81
TOL = 1e-6

LABEL_A = "约束A（固定）"
LABEL_B = "约束B（自由）"


def title(text):
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


# ---------------------------------------------------------------- 读结果
def load_result(label):
    """只读一份结果 csv，返回各条曲线"""
    path = os.path.join(RESULTS, "%s_完整结果.csv" % label)
    if not os.path.exists(path):
        sys.exit("找不到结果文件：%s\n请先运行 solve_q1_ab.py" % path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    seg = df[df["x_t (kW)"].notna()].reset_index(drop=True)
    if len(seg) != T:
        sys.exit("%s 的时段数不是 %d，实际 %d" % (label, T, len(seg)))
    return {
        "label": label,
        "c": seg["c_t (元/kWh)"].to_numpy(float),
        "L": seg["L_t (kW)"].to_numpy(float),
        "P": seg["P_t^fc (kW)"].to_numpy(float),
        "net": seg["净负荷 (kW)"].to_numpy(float),
        "x": seg["x_t (kW)"].to_numpy(float),
        "u": seg["u_t (kW)"].to_numpy(float),
        "v": seg["v_t (kW)"].to_numpy(float),
        "w": seg["w_t (kW)"].to_numpy(float),
        "E": df["E_t (kWh)"].to_numpy(float),      # 145 个点，0:00 ~ 24:00
    }


# ---------------------------------------------------------------- 检验计算
def feasibility(d):
    """可行性检验：残差、越界、退化、全天守恒"""
    x, u, v, w, E = d["x"], d["u"], d["v"], d["w"], d["E"]
    L, P = d["L"], d["P"]

    res_bal = float(np.max(np.abs(x + P + v - L - u - w)))
    res_soc = float(np.max(np.abs(E[1:] - E[:-1] - ETA_C * DT * u + DT / ETA_D * v)))

    lhs = float((x.sum() + P.sum() + v.sum()) * DT)     # Σx·Δt + ΣP·Δt + Σv·Δt
    rhs = float((L.sum() + u.sum() + w.sum()) * DT)     # ΣL·Δt + Σu·Δt + Σw·Δt
    rel = abs(lhs - rhs) / max(abs(rhs), 1e-12)

    return {
        "电量平衡残差最大值 (kW)": res_bal,
        "储能递推残差最大值 (kWh)": res_soc,
        "储电量最小值 (kWh)": float(E.min()),
        "储电量最大值 (kWh)": float(E.max()),
        "储电量越界": bool(E.min() < E_MIN - TOL or E.max() > E_MAX + TOL),
        "充电功率最大值 (kW)": float(u.max()),
        "放电功率最大值 (kW)": float(v.max()),
        "充放电功率越限": bool(u.max() > P_ST_MAX + TOL or v.max() > P_ST_MAX + TOL),
        "购电功率出现负值": bool(x.min() < -TOL),
        "弃光功率出现负值": bool(w.min() < -TOL),
        "同时充放电 min(u,v) 最大值 (kW)": float(np.max(np.minimum(u, v))),
        "全天能量守恒 左边 (kWh)": lhs,
        "全天能量守恒 右边 (kWh)": rhs,
        "全天能量守恒 相对误差": rel,
        "充放电量比 Σv/Σu": float(v.sum() / u.sum()),
        "往返效率 η_c·η_d": ETA_ROUND,
    }


def fmt(v):
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, int):
        return "%d" % v
    return "%.4f" % v


def block_feasibility(fa, fb):
    """
    返回一个"块"：表头 + 数据行 + 逐格数字格式。

    容差说明：结果表 csv 只保留四位小数，残差由舍入后的数算出来必然有
    1e-4 量级的尾巴，所以判据放宽到 1e-3。求解器内存中的残差是精确的 0，
    solve_q1_ab.py 的 validate() 打的就是那个值。
    """
    spec = [
        ("①", "电量平衡残差最大值 (kW)", "< 1e-3", "%.2e", lambda v: v < 1e-3),
        ("②", "储能递推残差最大值 (kWh)", "< 1e-3", "%.2e", lambda v: v < 1e-3),
        ("③", "储电量最小值 (kWh)", "≥ 1200", "%.4f", lambda v: v >= E_MIN - TOL),
        ("③", "储电量最大值 (kWh)", "≤ 10800", "%.4f", lambda v: v <= E_MAX + TOL),
        ("③", "储电量越界", "应为 否", None, lambda v: not v),
        ("④", "充电功率最大值 (kW)", "≤ 5000", "%.4f", lambda v: v <= P_ST_MAX + TOL),
        ("④", "放电功率最大值 (kW)", "≤ 5000", "%.4f", lambda v: v <= P_ST_MAX + TOL),
        ("④", "充放电功率越限", "应为 否", None, lambda v: not v),
        ("⑤", "购电功率出现负值", "应为 否", None, lambda v: not v),
        ("⑤", "弃光功率出现负值", "应为 否", None, lambda v: not v),
        ("⑤", "同时充放电 min(u,v) 最大值 (kW)", "应为 0", "%.2e", lambda v: v < 1e-6),
        ("⑥", "全天能量守恒 左边 (kWh)", "= 右边", "%.4f", lambda v: True),
        ("⑥", "全天能量守恒 右边 (kWh)", "= 左边", "%.4f", lambda v: True),
        ("⑥", "全天能量守恒 相对误差", "< 1e-9", "%.2e", lambda v: v < 1e-9),
        ("⑦", "充放电量比 Σv/Σu", "= η_c·η_d = 0.81", "%.6f", lambda v: abs(v - ETA_ROUND) < 1e-9),
        ("⑦", "往返效率 η_c·η_d", "与上一行相等", "%.6f", lambda v: True),
    ]
    data, fmts = [], []
    for num, name, crit, f, check in spec:
        ok = "通过" if (check(fa[name]) and check(fb[name])) else "未通过"
        data.append([num, name, crit, fa[name], fb[name], ok])
        fmts.append([None, None, None, f or "General", f or "General", None])
    return {
        "subtitle": None,
        "header": ["编号", "检验项", "判据", LABEL_A, LABEL_B, "结论"],
        "rows": data,
        "formats": fmts,
    }


NOTES_FEASIBILITY = [
    "数据来源：results\\约束A（固定）_完整结果.csv、results\\约束B（自由）_完整结果.csv。",
    "本表由 validation\\check_model.py 只读结果 csv 生成，不重新求解，属于对解的外部验证。",
    "① 电量平衡与 ② 储能递推是模型的硬约束，残差为 0 说明解严格可行。本表数值由结果表 csv（四位小数）反算，",
    "   故残差判据放宽到 1e-3；求解器内存中的残差是精确的 0，见 solve_q1_ab.py 的 validate() 输出。",
    "③④⑤ 对应储电量上下限、充放电功率上限、变量非负与充放电互斥。",
    "⑥ 全天能量守恒是把 144 个时段的平衡式求和后得到的整体校验：左边 Σx·Δt + ΣP^fc·Δt + Σv·Δt，",
    "   右边 ΣL·Δt + Σu·Δt + Σw·Δt。它比逐时段残差更严格，能查出 24:00 收口行算重、算漏这类整体性错误。",
    "⑦ 只要当天首尾储电量相同（E_T = E_0），放电总量与充电总量之比就必然等于往返效率 η_c·η_d = 0.81，",
    "   与电价、负载、容量都无关。这是很灵敏的自检：储能递推式或效率项写错，这一行立刻就对不上。",
]


def main():
    os.makedirs(TABDIR, exist_ok=True)

    title("问题一 检验（一）：模型检验表（可行性 + 全天能量守恒）")
    da, db = load_result(LABEL_A), load_result(LABEL_B)
    print("  已读入两份结果：%s、%s（各 %d 个时段）" % (LABEL_A, LABEL_B, T))

    fa, fb = feasibility(da), feasibility(db)
    for k in fa:
        print("  %-32s A: %-22s B: %s" % (k, fmt(fa[k]), fmt(fb[k])))

    tableio.write_table(os.path.join(TABDIR, "模型检验表.xlsx"),
                        "问题一 模型检验表", [block_feasibility(fa, fb)],
                        notes=NOTES_FEASIBILITY)

    title("完成")
    print("  表格目录：%s" % TABDIR)


if __name__ == "__main__":
    main()
