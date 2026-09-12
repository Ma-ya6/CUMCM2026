# 核验包

运行 `python report.py` 检查四个基线结果、时间回归及D/F完成状态，不重跑全年优化。
追加 `--deep` 重做预测未来扰动、独立链式目标及年终费用诊断，耗时较长。

证据保存在根目录 `annual_audit.json`、`time_tests.json` 与 `audits/`。通过物理和结算检查不代表未解决的模型假设已经获批，详情见 `../audits/逻辑检查报告.md`。
