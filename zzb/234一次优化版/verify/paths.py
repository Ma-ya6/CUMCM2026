# -*- coding: utf-8 -*-
"""只读路径配置。本目录下所有脚本**只读**上游结果，不写回上游目录。"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parent                      # 234一次优化版/
C_DIR = WORK.parent                     # C题/
OUT = HERE / "out"

DT = 1.0 / 6.0                          # 小时
E_INITIAL = 6000.0                      # kWh
ETA_C = ETA_D = 0.9
E_MIN, E_MAX = 1200.0, 10800.0
P_MAX = 5000.0

# 四个工作目录（只读）
Q2 = C_DIR / "问题2" / "最终优化版"
Q3 = C_DIR / "问题3" / "模型3决策优化版"
Q4_2 = C_DIR / "问题4" / "重算问题2"
Q4_3 = C_DIR / "问题4" / "重算问题3"

FROZEN = {
    "Q2": Q2 / "results" / "最终冻结结果.json",
    "Q3": Q3 / "results" / "最终冻结结果.json",
    "Q4-2": Q4_2 / "results" / "最终冻结结果.json",
    "Q4-3": Q4_3 / "results" / "最终冻结结果.json",
}

SLOTS = {
    "Q2": Q2 / "results" / "全年逐10分钟策略.csv",
    "Q3": Q3 / "results" / "全年逐10分钟策略.csv",
    "Q4-2": Q4_2 / "results" / "全年逐10分钟策略.csv",
    "Q4-3": Q4_3 / "results" / "全年逐10分钟策略.csv",
}

# 上游已有的理论下界口径文件（只读取其数值，不重算）
ORACLE_Q2Q4 = Q4_2 / "results" / "问题2问题4下界对照.csv"
ORACLE_Q4_2 = Q4_2 / "results" / "理论完美下界.json"
ORACLE_Q4_3 = Q4_3 / "results" / "理论最优.json"

# 问题3 / 问题4-3 的决策代码（只 import，不修改）
sys_q3 = Q3
