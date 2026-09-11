# verify —— 只读核对包

只读取上游 `问题2/最终优化版`、`问题3/模型3决策优化版`、`问题4/重算问题2`、`问题4/重算问题3`
的已冻结结果，不写回、不重跑仿真。

```bash
python check_adjust.py        # 模型解析核对：§六.1 单时段 + §六.2 链式路径
python core_indicators.py     # 主要指标：四张表 -> out/*.csv|md
python normalize_terminal.py  # 年末储能口径统一 -> ../年末口径统一对比.md
python report.py              # 汇总 -> ../核对报告.md   （须在 core_indicators 之后）
```

| 文件 | 作用 |
|---|---|
| `paths.py` | 全部路径与常量，改这里换目录 |
| `check_adjust.py` | import 上游 `models.py` 做受控 LP 探针 + 结算函数手算对照 |
| `core_indicators.py` | 表1 核心指标 / 表2 预测精度 / 表3 物理可行性 / 表4 费用差距分解 |
| `report.py` | 组装 `../核对报告.md`，含 §1 结论摘要与二元/三元费用分解 |
| `out/` | 机器可读输出（CSV + JSON），论文数字应从这些文件取 |
