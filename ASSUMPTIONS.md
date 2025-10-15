# 假设与权衡

- Python 版本 ≥ 3.10，支持 `match-case`、类型提示联合等语法。
- 默认在 `conda activate stock_prediction` 环境内运行，依赖统一由 `requirements.txt` 管理。
- 现阶段保留 `init.py` 中的全局常量与队列逻辑，后续重构再迁移至配置模块。
- 新增的 `TemporalHybridNet` 默认使用 `INPUT_DIMENSION`=30 的窗口特征，训练前需确保数据列数与常量一致。
