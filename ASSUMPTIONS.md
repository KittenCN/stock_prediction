# 假设与权衡

- Python 版本 ≥ 3.10，支持 `match-case`、类型提示联合等语法。
- 默认在 `conda activate stock_prediction` 环境内运行，依赖统一由 `requirements.txt` 管理。
- 现阶段保留 `init.py` 中的全局常量与队列逻辑，后续重构再迁移至配置模块。
- 新增的 `feature_engineering.py` 假设所有行情数据包含 `trade_date`（八位字符串）与 `ts_code` 字段；外生特征以 CSV 形式放置在 `config/external/` 并通过 `config/config.yaml` 的 `features.external_sources` 白名单启用。
- 新增的 `TemporalHybridNet` 及其它模型默认使用 `INPUT_DIMENSION`=30 的窗口特征，`feature_engineering` 会自动截断/填充到该尺寸，训练前需确保配置与数据列数一致。
- 预测/对比绘图假设输入数据按日期倒序存储，最新样本位于顶部，因此按队列顺序截取即可与模型输出一一对应。
- `Trainer` 类假设批次数据为 (inputs, targets) 元组，支持 None 批次跳过；损失记录为批次级别，用于绘制训练曲线。
- `metrics.py` 假设 y_true 与 y_pred 为数值数组，支持金融指标计算；VaR/CVaR 默认使用 5% 分位数，适用于收益分布评估。

- 股票嵌入默认容量 4096（可通过环境变量 SYMBOL_EMBED_MAX 调整），eatures.symbol_embedding_dim 控制嵌入维度。

- HybridLoss/PTFTVSSMLoss 默认使用适度的波动度与极值权重（0.02），如需关闭可在训练脚本参数中置 0。

- `branch_config` 支持 bool / float / dict 形式；当提供 `{"enabled": bool, "weight": float}` 时，权重会被转换为门控先验（log-scale），默认值为 1。
- Hybrid 门控温度通过软正则参数控制，如需固定可在训练脚本中手动设定。

## 归一化一致性假设（2025-10-20 新增）

- **训练与推理必须使用相同的归一化参数**：`Stock_Data(mode=0)` 训练时计算的 `mean_list`/`std_list` 必须在推理和反归一化时复用，避免分布偏移。
- **禁用符号级归一化避免重复标准化**：`config.yaml` 中 `enable_symbol_normalization` 默认设为 `false`，防止特征工程与 `Stock_Data` 两次归一化导致尺度错误。
- **自适应模型容量匹配数据规模**：Hybrid 模型使用自适应配置系统，根据训练样本数自动选择模型规模（tiny/small/medium/large/full），避免小数据集过拟合。
- **模型配置可手动覆盖**：用户可通过 `--hybrid_size` 参数手动指定配置级别，覆盖自动选择的配置。
- **训练集推理用于验证过拟合**：`contrast_lines()` 现使用 `mean_list`/`std_list` 反归一化，确保训练集推理 RMSE < 5% 作为基线。
- **归一化参数可通过 `scripts/verify_normalization.py` 验证**：该脚本对比训练/测试模式的归一化输出，确保修复生效。

## 自适应配置策略（2025-10-20 新增）

Hybrid 模型根据训练样本数自动选择配置级别：

| 样本数范围 | 配置级别 | hidden_dim | 启用分支 | 适用场景 |
|-----------|---------|-----------|---------|---------|
| < 500 | tiny | 32 | legacy | 单股票、短历史 |
| 500-1000 | small | 64 | legacy | 小规模训练 |
| 1000-5000 | medium | 128 | legacy + ptft | 标准单股票或小批量多股票 |
| 5000-10000 | large | 160 | legacy + ptft + vssm | 多股票联合训练 |
| >= 10000 | full | 160 | 所有分支 | 大规模多股票训练 |

**默认行为**：使用 `auto` 模式自动选择，可通过 `--hybrid_size` 参数手动覆盖。

