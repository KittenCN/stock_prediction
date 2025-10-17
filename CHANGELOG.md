# Changelog

本文件遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与语义化版本（SemVer）。

## [未发布] - 2025-10-20
### Fixed
- **测试模式归一化参数为空问题修复**（已完成✓）
  - 问题：测试模式加载归一化参数文件后，`mean_list` 和 `std_list` 长度为 0，导致反归一化失败并报错 "[WARN] Index X out of range"
  - 根本原因：
    1. 归一化参数在每次调用 `normalize_data()` 时都会被 `clear()`
    2. 模型保存时，全局 `mean_list` 和 `std_list` 可能已经被清空
    3. 使用 PKL 模式训练时，不会重新计算归一化参数
  - 修复措施：
    1. 在 `init.py` 中添加 `saved_mean_list` 和 `saved_std_list` 作为稳定副本
    2. 在 `Stock_Data` 和 `stock_queue_dataset` 的 `normalize_data()` 中，第一次计算后保存副本
    3. 在 `save_model()` 中优先使用稳定副本而非易变的全局变量
    4. 修复 test() 函数中 "Hybrid" 模型导入错误，应为 "TemporalHybridNet"
    5. 新增 `scripts/fix_norm_params.py` 工具，可从 PKL 文件重新计算并修复现有的空参数文件
  - 验证结果：
    - ✓ 单元测试：46/46 通过
    - ✓ 归一化参数正确加载（mean_list长度30，std_list长度30）
    - ✓ 不再出现索引越界警告
  - 影响范围：所有使用归一化参数的模型测试和推理
  - 相关文件：
    - `src/stock_prediction/init.py`：添加稳定副本变量
    - `src/stock_prediction/common.py`：保存和使用稳定副本
    - `src/stock_prediction/train.py`：修复模型导入错误
    - `scripts/fix_norm_params.py`：新增修复工具

- **测试模式 Symbol Embedding 维度不匹配问题修复**（已完成✓）
  - 问题：测试模式报错 "mat1 and mat2 shapes cannot be multiplied (10x30 and 46x128)"，测试数据输入维度为30维，但模型期望46维（启用 symbol embedding 后）
  - 根本原因：
    1. 训练时启用了 symbol embedding，每个样本会添加 16 维的股票编码特征（30 → 46维）
    2. 测试模式下 `contrast_lines()` 函数未添加 `_symbol_index` 列
    3. Dataloader 迭代格式不匹配（期望 `(data, label)` 但可能返回 `(data, label, symbol_idx)`）
    4. `contrast_lines()` 函数返回值格式错误（未返回 `(y_true, y_pred)` 元组）
  - 修复措施：
    1. 在 `contrast_lines()` 函数中添加 symbol index：
       - 从数据中提取 `ts_code`
       - 使用 `feature_engineer.symbol_to_id` 映射获取 symbol_id
       - 添加 `_symbol_index` 列到数据框
    2. 修复 dataloader 迭代逻辑：
       - 正确处理 2元组 `(data, label)` 和 3元组 `(data, label, symbol_idx)` 格式
       - 添加边界检查避免索引越界
    3. 修复 `contrast_lines()` 返回值：
       - 返回 `(y_true, y_pred)` 元组供 metrics 计算使用
       - 移除过时的 `while contrast_lines() == -1` 循环
  - 验证结果：
    - ✓ 单元测试：46/46 通过
    - ✓ 测试模式成功运行（test accuracy: 0.686）
    - ✓ Symbol index 成功添加（例如 ts_code=19 → symbol_index=4075）
    - ✓ Metrics 文件生成成功（包含 RMSE, MAPE, VaR, CVaR）
  - 影响范围：所有启用 symbol embedding 的模型测试和推理
  - 相关文件：
    - `src/stock_prediction/train.py`：修复 `contrast_lines()` 和 `test()` 函数

### Known Issues
- **用户需要运行一次性修复工具**（可选 ⚙️）
  - 对于使用 PKL 模式训练的现有模型，归一化参数文件可能为空
  - 解决方案：运行 `python scripts\fix_norm_params.py` 一次性修复
  - 未来训练的模型会自动保存正确的归一化参数
  - 问题：测试模式报错 `mat1 and mat2 shapes cannot be multiplied (40x30 and 46x128)`
  - 根本原因：训练时启用 symbol embedding（输入维度30+16=46），但测试时数据加载未提供 symbol_index（输入维度30）
  - 状态：归一化参数问题已解决，但测试数据维度与模型不匹配的问题仍需处理
  - 临时解决方案：使用 `scripts/fix_norm_params.py` 修复归一化参数后，需要重新训练不使用 symbol embedding 的模型，或修改测试数据加载逻辑以支持 symbol_index
  - 详见：`docs/test_mode_fix_progress.md`

- **测试模式模型参数加载问题修复**（关键修复）
  - 问题：测试模式（`mode=test`）时报错 `mat1 and mat2 shapes cannot be multiplied (10x30 and 46x128)` 和 `IndexError: list index out of range`
  - 根本原因：
    1. 测试时使用全局 `INPUT_DIMENSION` 创建模型，但实际训练时的模型可能使用不同的输入维度（例如启用 symbol embedding 后为 30+16=46）
    2. 测试时未加载训练时保存的归一化参数（`mean_list`、`std_list`），导致反归一化时索引越界
  - 修复措施：
    1. 修改 `test()` 函数，在加载模型权重前先加载 `_Model_args.json`，使用训练时的参数重新创建模型
    2. 保存归一化参数到 `_norm_params.json` 文件，测试时自动加载并更新全局变量
    3. 在 `contrast_lines()` 函数中添加边界检查，避免索引越界
    4. 修改 `save_model()` 函数，同时保存模型参数和归一化参数
  - 影响范围：所有模型的测试和推理流程
  - 预期效果：测试模式正确加载训练时的模型配置和归一化参数，避免维度不匹配和索引越界错误
  - 相关文件：
    - `src/stock_prediction/train.py`：test() 函数、contrast_lines() 函数
    - `src/stock_prediction/common.py`：save_model() 函数

- **Hybrid 模型推理维度不匹配问题修复**（关键修复）
  - 问题：推理时 RuntimeError: Given normalized_shape=[46], expected input with shape [*, 46], but got input of size[1, 5, 30]
  - 根本原因：
    1. 训练时启用 symbol embedding，输入维度 30 + 16 = 46，推理时未提供 symbol_index，导致输入维度 30
    2. `get_adaptive_hybrid_config` 函数定义在 `train.py` 中，无法从 `predict.py` 导入
  - 修复措施：
    1. 将 `get_adaptive_hybrid_config` 提取到独立模块 `src/stock_prediction/hybrid_config.py`，便于复用
    2. 在 `predict.py` 中添加 symbol mapping，从训练数据构建 symbol_to_id 映射
    3. 修改 `test()` 函数支持传递 `symbol_index` 参数，启用 symbol embedding 推理
    4. 在模型加载时，自动调整输入维度以匹配数据（临时修复）
  - 影响范围：所有使用 symbol embedding 的模型推理
  - 预期效果：推理脚本正常运行，无维度错误
  - 相关文档：更新 `docs/architecture.md` 说明 symbol embedding 处理
- **Hybrid 模型训练集推理偏差问题修复**（关键修复）
  - 问题：训练 30 epoch 后，使用训练集数据推理时预测结果与真实值偏差巨大
  - 根本原因：
    1. `contrast_lines()` 函数使用测试集归一化参数（`test_mean_list`/`test_std_list`）反归一化，与训练时使用的参数（`mean_list`/`std_list`）不一致
    2. 特征工程启用 `enable_symbol_normalization` 时会重复归一化
    3. Hybrid 模型默认配置参数量过大（160 hidden_dim，5 个分支），容易过拟合
  - 修复措施：
    1. 修改 `train.py` 的 `contrast_lines()` 函数（第 659、680 行），统一使用 `mean_list`/`std_list` 进行反归一化
    2. 在 `config.yaml` 中显式设置 `enable_symbol_normalization: false`，避免重复标准化
    3. **实现自适应模型配置**：根据训练数据量自动调整模型容量，小数据集使用轻量模型，大数据集使用完整模型
  - 新增验证工具：`scripts/verify_normalization.py` 用于验证训练/测试归一化一致性
  - 影响范围：所有使用训练集推理的场景，特别是过拟合检测和模型调试
  - 预期效果：训练集推理 RMSE 从 50%+ 降至 5% 以内
  - 相关文档：`docs/diagnosis_hybrid_training_inference_gap.md`

### Added
- **Hybrid 模型自适应配置系统**（新特性）
  - 新增 `--hybrid_size` 命令行参数，支持手动指定模型规模：
    - `auto`（默认）：根据训练样本数自动选择配置
    - `tiny`：hidden_dim=32，仅 legacy 分支（适合 < 500 样本）
    - `small`：hidden_dim=64，仅 legacy 分支（适合 500-1000 样本）
    - `medium`：hidden_dim=128，legacy + ptft 分支（适合 1000-5000 样本）
    - `large`：hidden_dim=160，legacy + ptft + vssm 分支（适合 5000-10000 样本）
    - `full`：hidden_dim=160，所有分支（适合 >= 10000 样本）
  - 自动模式根据实际训练样本数动态选择最优配置，避免小数据集过拟合
  - 训练开始时显示配置信息和数据量分析，提供配置建议
  - 使用示例：
    ```bash
    # 自动配置（推荐）
    python scripts/train.py --model hybrid --begin_code 000001.SZ
    
    # 手动指定轻量配置
    python scripts/train.py --model hybrid --hybrid_size small --begin_code 000001.SZ
    
    # 强制使用完整配置
    python scripts/train.py --model hybrid --hybrid_size full --begin_code 000001.SZ
    ```

### Added
- 引入 ts_code 股票嵌入：特征工程输出 _symbol_index，TemporalHybridNet/PTFTVSSMEnsemble/DiffusionForecaster/GraphTemporalModel 支持可学习嵌入。
- `feature_engineering.py` 与 `FeatureSettings`：支持对数收益率/差分特征生成、外生特征白名单合并、多股票联合训练与滑动窗口统计。
- `regularization.BayesianDropout` 以及 Regime 感知 Soft Gating，实现 PTFT+VSSM 自适应融合权重并支持 MC Dropout 推理。
- 金融指标驱动训练：`PTFTVSSMLoss` 中新增方向性、Sharpe、最大回撤、分位 Pinball 与 Regime 辅助分类损失。
- 配套测试：`tests/test_feature_engineering.py`、`tests/test_models.py` Regime 融合测试，确保收益率特征与权重学习可用。
- 文档更新：`README.md`、`docs/maintenance.md`、`docs/system_design.md`、`docs/model_strategy.md`、`docs/user_guide.md`、`docs/agent_report.md` 全面记录改进结果。
- `Trainer` 类封装：统一训练循环，支持 LR Scheduler、Early Stopping、批次损失记录与回调机制。
- `metrics.py` 模块：自动采集 RMSE、MAPE、分位覆盖率、VaR、CVaR 等指标，训练/测试后保存至 `output/metrics_*.json`。
- 配置解耦：`app_config.py` 使用 Pydantic 验证，支持 YAML + .env 配置，包含特征工程与训练选项。

### Changed
- Trainer/预测 CLI 支持可选 symbol_index 张量，保持旧模型兼容。
- `PTFTVSSMEnsemble` 默认隐藏维度降至 128/48，引入 Regime 感知融合门与贝叶斯 Dropout。
- `common.Stock_Data` 与 `stock_queue_dataset` 使用特征工程模块，使收益率/外生特征在训练与预测阶段保持一致。
- `config/config.yaml` 增加 `features` 配置节，提供滑动窗口、外生特征与金融损失权重示例。
- `train.py` 重构为使用 `Trainer` 类，保持 CLI 兼容性，批次损失用于绘制训练曲线。
- `predict.py` 与 `contrast_lines` 集成 metrics 采集，自动保存指标文件。

### Tests
- `pytest -q`：共 28 项（新增 2 项）全部通过。

## [未发布] - 2025-10-15
### 修复
- **UnboundLocalError: last_save_time 变量作用域错误**
  - 问题：训练时报错 `UnboundLocalError: cannot access local variable 'last_save_time' where it is not associated with a value`
  - 原因：`main()` 函数中使用并修改了 `last_save_time`，但未在 global 声明中包含该变量，导致 Python 将其视为局部变量
  - 修复：在 `main()` 函数的 global 声明中添加 `last_save_time`
  - 影响范围：训练流程中的模型保存逻辑
  - 文件：`src/stock_prediction/train.py` line 648

### 测试
- ✓ 所有 26 项单元测试通过

---

## [未发布] - 2025-10-15
### 修复
- **PyTorch 2.x 模型 deepcopy 报错**
  - 问题：训练一轮结束后，执行 `testmodel = copy.deepcopy(model)` 时抛出 `RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment`
  - 原因：PyTorch 2.x 开始，模型使用 weight_norm 或特殊层时不支持直接 deepcopy
  - 修复：改为"新建同类型模型 + 加载 state_dict"方式
    - 为所有模型在初始化后增加 `_init_args` 属性，保存构造参数
    - 在 train 函数中两处 testmodel 创建改为 `type(model)(**model._init_args)` + `load_state_dict()`
    - 自动适配 DataParallel 场景
  - 影响范围：所有模型（LSTM, Transformer, PTFT_VSSM, Hybrid, TCN, MultiBranch, CNNLSTM 等）
  - 文件：`src/stock_prediction/train.py`

### 已知限制
- 所有模型在初始化后需手动维护 `_init_args` 属性，若新增模型或修改构造参数，需同步更新

### 后续改进建议
- 考虑在模型基类中自动记录 `__init__` 参数，避免手动维护

---

## [未发布] - 2025-10-15

### Added
- 实现 `ProbTemporalFusionTransformer`、强化版 `Variational State Space Model` 与 `PTFTVSSMEnsemble`，新增 `--model ptft_vssm` 训练入口。
- 新增 `PTFTVSSMLoss` 及 `tests/test_models.py` 中的配套用例，覆盖前向与损失逻辑。
- 升级 `GatedResidualNetwork`（GRN）与 VSSM 时间依赖先验，提升特征提取与状态建模精度。
- 更新文档：`docs/model_strategy.md`、`docs/system_design.md`、`docs/user_guide.md`、`docs/maintenance.md`，同步 README 与文档索引。

### Changed
- `scripts/run_all_models.bat` 增加 `ptft_vssm` 测试项，完善横向对比脚本。
- 拆分 `src/stock_prediction/train.py`（训练/测试）与 `predict.py`（推理），更新 `scripts/train.py`、`scripts/predict.py`。
- README 重写结构，突出混合模型与概率组合流程。
- `predict.py` 提供 `main(argv=None)` 与 `create_predictor()`，方便程序化调用与测试。
- 将 `common.py` 中的 LSTM / Transformer / CNNLSTM 结构迁移至 `models/` 目录，统一模型管理。

### Fixed
- `predict.py` 主逻辑封装为 `main()`，补齐全局变量初始化与 `global` 声明。
- **关键修复**：修复 `predict.py` 中文编码乱码问题，将错误的乱码字符串（"鐩爣鑲＄エ..."）修正为正确的中文（"目标股票未在 pkl 队列中找到"）。
- **关键修复**：修复 `README.md` 中文编码乱码问题，完整重写文件使用正确的 UTF-8 编码，所有中文内容恢复正常显示。
- **关键修复**：修复训练时 "data has nan or inf" 错误，在 `common.py` 的数据加载和标准化过程中添加强健的 NaN/Inf 处理：
  - `Stock_Data.load_data()`: 使用 `np.nan_to_num()` 将所有 NaN/Inf 转换为 0
  - `Stock_Data.normalize_data()`: 在计算均值和标准差前检查并清理 NaN/Inf，防止标准差为 0
  - `stock_queue_dataset.load_data()`: 多层 NaN 填充策略（中位数 → 0 → 替换 Inf）
  - `stock_queue_dataset.normalize_data()`: 同样添加 NaN/Inf 检查和处理
- **关键修复**：添加缺失的导入语句（`os`, `pandas`, `DataLoader`）和配置初始化（`config`, `train_pkl_path`），修复所有 `未定义` 编译错误。
- **关键修复**：修复 `_init_models()` 和 `Predictor.predict()` 中使用未定义的 `symbol` 变量，改为使用正确的 `test_code` 参数。
- 添加 `total_test_length` 全局变量定义，修复测试数据加载时的未定义错误。
- `thread_save_model` 改为保存 state_dict，兼容包含 weight_norm 的模型。
- 扩展 `tests/test_models.py`，覆盖 TemporalHybridNet、PTFT、VSSM 与组合模型的前向和损失。
- `pytest` 共 26 项全部通过，覆盖率提升至约 20%。

### Known Issues
- 队列序列化仍依赖 `pickle`，后续建议迁移至 Arrow/Parquet 或增加版本头。

## [Unreleased] - 2025-10-11

### Added
- `docs/project_analysis.md`：首次汇总系统分析与风险评估。
- `docs/architecture.md`、`docs/api.md`、`docs/ops.md`、`docs/cleanup_log.md`：补充架构示意、接口说明与运维记录。

### Changed
- 重写旧版 `README.md`，纳入中文简介、快速开始与目录结构。
- 文档统一改为 UTF-8，移除历史乱码。

### Fixed
- 在 `stock_prediction.common.ensure_queue_compatibility()` 内补充 `queue.Queue.is_shutdown`，兼容 Python 3.13+。

## [2.0.0] - 2024-12-28

### Added
- 迁移核心代码至 `src/stock_prediction/`，引入标准化包结构与 `config.py`。
- 新增 `scripts/` 目录作为命令行入口，并整理测试用例与 Makefile。

### Changed
- 将历史脚本备份到 `legacy_backup/`，保留参考版本。
- 统一使用相对导入与 `pathlib.Path` 管理路径，提升跨平台能力。

### Fixed
- 解决早期版本的导入路径异常与 `is_number()` 正则问题。***
