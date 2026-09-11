# C 题 问题一 · 约束 A / 约束 B 合并求解

一个求解脚本、一个结果表文件、一套图、一套检验。两个约束口径写在同一个脚本里的两个
线性规划函数中，`main()` 读一次数据、先后调用，结果写进同一个表格文件的两张 sheet。

检验部分、判据与结论另见 **[validation\README.md](validation/README.md)**。

---

## 一、目录

```
问题1/
├── solve_q1_ab.py            求解脚本（solve_a / solve_b 两个 LP 函数 + main）
├── plot_q1_three.py          出图脚本（三张图：负载光伏净负荷 / 储能 / 计划购电量）
├── requirements.txt
├── results/
│   ├── 问题一结果_约束AB.xlsx    ← 主交付：三张 sheet
│   │      约束A（固定）  145 行逐时段明细
│   │      约束B（自由）  145 行逐时段明细
│   │      汇总           两口径指标对比 + 差值(B−A)
│   ├── 约束A（固定）_完整结果.csv   供出图脚本读取
│   ├── 约束B（自由）_完整结果.csv   供出图脚本读取
│   ├── result1_约束A（固定）.xlsx   题目模板格式
│   ├── result1_约束B（自由）.xlsx   题目模板格式
│   └── 预处理表.csv                附件 1 整理后的标准表（145 行 × 8 列）
├── figures/
│   ├── 约束A（固定）/  图1 负载光伏净负荷、图2 储能SOC轨迹、图3 计划购电量
│   └── 约束B（自由）/  同名三张
├── validation/
│   ├── check_model.py        只读结果 csv：可行性 + 全天能量守恒 + Σv/Σu 自检
│   ├── check_optimal.py      读对偶证书：KKT 条件 + 强对偶
│   ├── sensitivity.py        重解 19 次：参数敏感性与约束松绑
│   ├── tableio.py            Excel 写表工具（三个脚本共用）
│   ├── tables/               模型检验表、最优性检验、敏感性结果（三张 Excel）
│   └── figures/              图1 对偶变量与边际电价、图2 参数敏感性
└── docs/
    ├── 论文_约束A（固定）.md
    ├── 论文_约束B（自由）.md
    ├── 约束A，,对比.md
    └── 问题1_题目与思路.md
```

---

## 二、两个约束

目标函数与约束 (1)(2)(3)(5) 完全相同，唯一差别是约束 (4) 日末回归：

| | 约束 A（固定） | 约束 B（自由） |
|---|---|---|
| 约束 (4) | E_0 = E_T = 6000 | E_T − E_0 = 0，共同取值由优化决定 |
| 含义 | 日循环两端钉死在 6000 kWh | 把这一天看成纯粹循环调度周期，循环电量自己定 |
| 代价 | 计划起点与实际初始状态绑定 | 起点可能与实际 6000 kWh 不一致，执行时要先补差额 |

约束 A 的任意可行解在 B 中都可行，所以 B 的可行域更大，必有 C(B) ≤ C(A)。

---

## 三、结果

| 指标 | 约束 A（固定） | 约束 B（自由） | 差值 |
|---|---|---|---|
| 全天购电量 Q (kWh) | 59482.6990 | 59482.6990 | 0 |
| 全天购电费 C (元) | 35126.9486 | 35118.5986 | −8.35 |
| 优化得到的循环储电量 E_0 = E_T (kWh) | 6000 | 8550 | +2550 |
| 充电 ΣuΔt / 放电 ΣvΔt (kWh) | 20740.6661 / 16799.9396 | 同左 | 0 |
| 弃光 (kWh) | 0 | 0 | 0 |
| 无储能基准购电费 (元) | 48052.0466 | 48052.0466 | 0 |
| 储能节省 (元) | 12925.0980 | 12933.4480 | +8.35 |

两个口径购电量、充放电量完全相同，只差起点储电量；把循环电量从 6000 抬到 8550 只省
8.35 元（0.024%），说明日循环电量对购电费几乎没有影响 —— 费用差异完全来自电价时段
而非储电量水平。

---

## 四、复现步骤

### 环境

Python 3.12，依赖写在 `requirements.txt`：

```
pandas
numpy
scipy
openpyxl
matplotlib
```

安装：`python -m pip install -r requirements.txt`

出图用 `Microsoft YaHei` / `SimHei` 中文字体，Windows 自带，不用另装。求解器用的是 scipy
封装好的 HiGHS（`scipy.optimize.linprog` 的 `method="highs"`），随 scipy 一起装，无需单独配置。

### 六步

| 步骤 | 命令 | 产出 | 用时 |
|---|---|---|---|
| 1 求解 | `python solve_q1_ab.py` | `results\` 六个文件 | 约 2 秒 |
| 2 出图（约束A） | `python plot_q1_three.py "results\约束A（固定）_完整结果.csv" "figures\约束A（固定）" "（约束A）"` | 3 张图 | 
| 3 出图（约束B） | `python plot_q1_three.py "results\约束B（自由）_完整结果.csv" "figures\约束B（自由）" "（约束B）"` | 3 张图 |
| 4 模型检验 | `python validation\check_model.py` | `validation\tables\模型检验表.xlsx` |
| 5 最优性检验 | `python validation\check_optimal.py` | `validation\tables\最优性检验.xlsx` + 图 1 | 
| 6 敏感性 | `python validation\sensitivity.py` | `validation\tables\敏感性结果.xlsx` + 图 2 | 


```powershell
cd C:\Users\21187\Desktop\数模\问题1
python solve_q1_ab.py
python plot_q1_three.py "results\约束A（固定）_完整结果.csv" "figures\约束A（固定）" "（约束A）"
python plot_q1_three.py "results\约束B（自由）_完整结果.csv" "figures\约束B（自由）" "（约束B）"
python validation\check_model.py
python validation\check_optimal.py
python validation\sensitivity.py
```

脚本内部用自身位置推算路径，不依赖当前工作目录；但上面这几条命令里的 csv 参数是相对
路径，所以还是先 `cd` 到 `问题1\` 比较稳妥。

### 复现是否成功：跑完核对下表，全部对上就说明复现无误。

| 指标 | 期望值 | 在哪看 |
|---|---|---|
| 全天购电量 Q | 59482.6990 kWh | 两个口径相同，`问题一结果_约束AB.xlsx` 汇总 sheet |
| 全天购电费（约束A 固定） | 35126.9486 元 | 同上 |
| 全天购电费（约束B 自由） | 35118.5986 元 | 同上 |
| 两者之差 | 8.3500 元 | 同上 |
| 约束B 优化出的 E_0 = E_T | 8550 kWh | 同上 |
| 充电 / 放电总量 | 20740.6661 / 16799.9396 kWh | 同上 |
| 弃光总量 | 0 kWh | 同上 |
| 无储能基准购电费 | 48052.0466 元 | 同上 |
| 储能节省 | 12925.0980 元（26.90%） | 同上 |
| 充放电量比 Σv/Σu | 0.810000 = η_c·η_d | 模型检验表第 ⑦ 组 |
| 全天能量守恒两端 | 131765.4742 kWh | 模型检验表第 ⑥ 组 |
| KKT 违反变量数 | 0（五个变量块各 0） | 最优性检验表 |
| 对偶目标 | 与原始目标相等，间隙 7.3e-12 元 | 最优性检验表 |
| 敏感性实验行数 | 基准 1 行 + 实验 19 行 | 敏感性结果表 |

另外，`plot_q1_three.py` 每次跑完会在终端打印一行汇总，那是它直接从 csv 重算的，
可以拿来跟 xlsx 的汇总 sheet 交叉核对：

```
全天购电量 59482.7 kWh，平均购电价 0.5905 元/kWh
充电 20740.7 kWh，放电 16799.9 kWh，弃光 0.0 kWh
```

---

## 五、可以改的地方

| 想改什么 | 改哪 |
|---|---|
| 储电量上下限、充放电功率上限、充放电效率、约束 A 的固定端点 | `solve_q1_ab.py` 顶部的 `E_MIN`、`E_MAX`、`P_ST_MAX`、`ETA_C`、`ETA_D`、`E_ENDPOINT` |
| 敏感性扫描的取值点 | `sensitivity.py` 里的 `eta_c_pts`、`eta_d_pts`、`emax_pts`、`pst_pts` |
| 图的配色、网格开关 | `plot_q1_three.py` 顶部的颜色常量和 `GRID_ON` |
| Excel 表头底色、列宽上限 | `tableio.py` 顶部的 `HEADER_FILL` 与 `write_table` 里的列宽那一行 |

改完记得把 `results\`、`figures\` 重新跑一遍，检验那三个脚本也要跟着重跑，否则表和
图会对不上。

---


---

## 六、固定约定

- 一天 144 个十分钟时段，`Δt = 1/6 h`。时间列统一折算成距 0 点的分钟数，附件 1 末尾的
  `0:00+1` 单独处理。
- 附件 1 是全年同日刻的平均值（三列与"按同时刻取平均"的相关系数均为 1.0000），所以它
  对应的是一条典型日曲线，不是某一天。
- 不允许光伏上网，富余只能充电或弃光；本解中弃光为 0。
- 求解得到的对偶向量长度等于等式约束条数（约束 A 是 290），前 144 个对应各时段的电量
  平衡约束。
