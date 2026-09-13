# -*- coding: utf-8 -*-
"""全项目共用的 matplotlib 配色与坐标轴样式。

本文件原先是项目外层的 ``数模/plot_style.py``（不在任何一份源材料里），
收进仓库后所有出图脚本均可独立运行，不再依赖仓库外的文件。

配色（与 `代码/问题1/README.md` 第六节的表一致）：

    C_MAIN      #005792  深蓝   净负荷、计划购电量、对偶变量、SOC 曲线、Q2
    C_SECOND    #00A3FF  亮蓝   小区负载、Q4-2
    C_FOURTH    #009473  青绿   光伏、充电功率、Q3
    C_RED_LIGHT #E57373  浅红   放电功率
    C_GUIDE     #9AA4B2  灰蓝   参考线（零线、SOC 上下限、基准线）
    C_PURPLE    #6E2277  深紫   紧急购电、Q4-3
    C_TEXT      #333333  深灰   坐标轴刻度与标签
    C_FRAME     #333333  深灰   坐标框线
    C_GRID      #DDDDDD  浅灰   内部网格

两个函数：

    apply_base_style()         设置全局 rcParams（中文字体、负号、字号、导出参数）
    style_frame_grid(ax, ...)  单个坐标轴：四周细框 + 内部细灰网格

该模块被以下脚本引用，请保持上述常量名不变：
    `代码/问题1/plot_q1.py`
    `验证/代码/问题1/check_optimal.py`
    `验证/代码/问题1/sensitivity.py`
    `代码/共享/出图数据/plot_result_figures.py`
"""
from __future__ import annotations

import matplotlib.pyplot as plt

# ---------------------------------------------------------------- 配色
C_MAIN      = "#005792"     # 深蓝 · 主色
C_SECOND    = "#00A3FF"     # 亮蓝 · 次色
C_FOURTH    = "#009473"     # 青绿 · 第四档
C_RED_LIGHT = "#E57373"     # 浅红
C_GUIDE     = "#9AA4B2"     # 灰蓝 · 参考线
C_PURPLE    = "#6E2277"     # 深紫
C_TEXT      = "#333333"     # 刻度与标签
C_FRAME     = "#333333"     # 坐标框线
C_GRID      = "#DDDDDD"     # 内部网格

# ---------------------------------------------------------------- 线宽
FRAME_LW = 0.8              # 坐标框线宽
GRID_LW  = 0.6              # 网格线宽


def apply_base_style() -> None:
    """全局样式：中文字体、负号、字号、图内不写标题。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False      # 负号用 ASCII，避免中文字体缺字形
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11.5
    plt.rcParams["xtick.labelsize"] = 10.5
    plt.rcParams["ytick.labelsize"] = 10.5
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"


def style_frame_grid(ax, grid: bool = True, frame: bool = True) -> None:
    """四周细框 + 内部细灰网格。

    grid=False  只画框不画网格；
    frame=False 只画网格不画框（用于 twinx 的右轴，避免与左轴重复描边）。
    """
    if grid:
        ax.grid(True, color=C_GRID, linewidth=GRID_LW)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)
    for side, spine in ax.spines.items():
        spine.set_visible(frame)
        if frame:
            spine.set_color(C_FRAME)
            spine.set_linewidth(FRAME_LW)
    ax.tick_params(colors=C_TEXT)
