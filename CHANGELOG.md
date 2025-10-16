# Changelog

本文件遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与语义化版本（SemVer）。

## [未发布] - 2025-10-20
### Added
- `feature_engineering.py` 与 `FeatureSettings`：支持对数收益率/差分特征生成、外生特征白名单合并、多股票联合训练与滑动窗口统计。
- `regularization.BayesianDropout` 以及 Regime 感知 Soft Gating，实现 PTFT+VSSM 自适应融合权重并支持 MC Dropout 推理。
- 金融指标驱动训练：`PTFTVSSMLoss` 中新增方向性、Sharpe、最大回撤、分位 Pinball 与 Regime 辅助分类损失。
- 配套测试：`tests/test_feature_engineering.py`、`tests/test_models.py` Regime 融合测试，确保收益率特征与权重学习可用。
- 文档更新：`README.md`、`docs/maintenance.md`、`docs/system_design.md`、`docs/model_strategy.md`、`docs/user_guide.md`、`docs/agent_report.md` 全面记录改进结果。
- `Trainer` 类封装：统一训练循环，支持 LR Scheduler、Early Stopping、批次损失记录与回调机制。
- `metrics.py` 模块：自动采集 RMSE、MAPE、分位覆盖率、VaR、CVaR 等指标，训练/测试后保存至 `output/metrics_*.json`。
- 配置解耦：`app_config.py` 使用 Pydantic 验证，支持 YAML + .env 配置，包含特征工程与训练选项。

### Changed
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
