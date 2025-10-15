# Changelog

本文件遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与语义化版本（SemVer）。

## [Unreleased] - 2025-10-15

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
