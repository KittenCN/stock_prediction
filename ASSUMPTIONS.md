# 假设与权衡

- Python 版本保持在 3.10 及以上，可使用 `match-case`、类型联合等语法。
- 默认在 `conda activate stock_prediction` 环境内运行，依赖统一由 `requirements.txt` 锁定。
- 暂时保留 `init.py` 中的全局常量、队列与设备设定，后续重构再拆分为模块化配置。
- `feature_engineering.py` 假设行情数据包含 `trade_date`（八位字符串）与 `ts_code` 字段，外生特征放在 `config/external/` 并需在 `config/config.yaml` 的 `features.external_sources` 白名单显式启用。
- 默认窗口特征维度为 30（`INPUT_DIMENSION=30`），特征工程自动补齐或截断；新增模型需确保输入列数匹配。
- 训练/对比绘图假定数据按日期倒序存储（最新样本在顶部），直接截取即可与模型输出对齐。
- `Trainer` 假设批量数据结构为 `(inputs, targets)`，遇到 `None` 批次会跳过；训练曲线基于批次级损失。
- `metrics.py` 假设 `y_true`、`y_pred` 为数值数组，可计算 RMSE/MAPE/VaR/CVaR 等指标；VaR/CVaR 默认分位点 5%。
- 股票嵌入容量默认 4096，可通过 `SYMBOL_EMBED_MAX` 调整；嵌入维度由 `features.symbol_embedding_dim` 控制。
- `HybridLoss` 默认激活振幅/极值/均值/收益约束：`volatility_weight=0.12`、`extreme_weight=0.02`、`mean_weight=0.05`、`return_weight=0.08`，如需关闭可在训练脚本中设置为 0。
- `branch_config` 支持 bool/float/dict 三种形式；当提供 `{"enabled": bool, "weight": float}` 时会转换为门控先验（log-scale），默认权重为 1。
- Hybrid 门控温度通过软正则控制，如需固定可在训练脚本中手动调整。

## Symbol Embedding 一致性假设（2025-10-20）
- 训练与推理必须使用一致的 symbol embedding 配置；若训练阶段启用 `use_symbol_embedding`，推理阶段必须提供有效的 `symbol_index`，以保证输入维度一致（原始特征 + 嵌入维度）。
- Symbol 映射由训练数据动态构建，假设训练数据覆盖所有推理股票；缺失股票默认映射到 ID 0，可能影响嵌入质量。
- 若加载旧模型需要调整输入维度，会随机初始化新增权重，预测精度可能下降，建议重新训练。

## 归一化一致性假设（2025-10-21 更新）
- 训练与推理必须共用同一套归一化统计：`Stock_Data(mode=0)` 计算的 `mean_list/std_list` 会在推理前自动加载，确保反归一化尺度一致。
- 模型保存的 `*_norm_params*.json` 会包含 `per_symbol` 映射（键为零填充 6 位的 `ts_code`），推理阶段优先按该映射加载；若缺失才回退全局统计。
- `config.yaml` 的 `enable_symbol_normalization` 保持为 `false`，避免特征工程与 `Stock_Data` 重复标准化导致尺度错误。
- `scripts/verify_normalization.py` 已支持 `--stock_code/--ts_code` 与 `--norm_file` 参数，可对指定股票进行训练/测试/持久化三方校验。
- Hybrid 模型使用自适应配置，按样本量自动选择 tiny/small/medium/large/full，避免小样本过拟合；也可通过 `--hybrid_size` 手动覆盖。
- 训练结束的自检目标：`contrast_lines()` 复现训练阶段 RMSE < 5%，并确保 `scripts/verify_normalization.py` 针对目标股票校验通过。

## 工具与脚本假设（2025-10-22）
- 新增 `scripts/analyze_predictions.py` 用于批量诊断 png/test、png/predict 中的 CSV，默认阈值 `std_ratio<0.8`、`|bias|>0.5`。
- 诊断输出依赖 `distribution_report`，脚本要求 CSV 提供 `Date`、`Actual`、`Forecast` 三列。
