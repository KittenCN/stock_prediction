# Changelog

本文件遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与语义化版本（SemVer）。

## [Unreleased] - 2025-10-15

### Added
- 实现 ProbTemporalFusionTransformer、VariationalStateSpaceModel 与 PTFTVSSMEnsemble，并接入 --model ptft_vssm 训练入口。
- 新增 PTFTVSSMLoss、	ests/test_models.py 中相关用例，确保前向与损失逻辑可测。
- 更新文档：docs/model_strategy.md、docs/system_design.md、docs/user_guide.md，补充双轨方案实现说明。

- 新增 `TemporalHybridNet`（`--model hybrid`），结合多尺度卷积、双向 GRU、Multi-Head Attention 与统计特征，提升股票多步预测能力。
- 新增 `tests/test_models.py`，覆盖 `TemporalHybridNet` 在单步与多步预测场景下的张量形状。
- 新增 `create_predictor()` 函数，便于外部调用和单元测试创建预测器实例。
- 整合文档为 `docs/system_design.md`、`docs/user_guide.md`、`docs/maintenance.md`，并同步更新 `README.md`、`ASSUMPTIONS.md`。

### Changed
- `scripts/run_all_models.bat` 现包含 `hybrid` 模型，便于横向对比。
- README 重写结构，突出混合模型及统一训练流程。
- **重要改进**：`predict.py` 命令行参数解析移至 `main()` 内部，模块级别使用 `DefaultArgs` 类提供默认配置，彻底解决测试环境中的 `SystemExit: 2` 问题。

### Fixed
- `predict.py` 主逻辑封装为 `main()`，补齐全局变量初始化与 `global` 声明，修复 `UnboundLocalError` / `NameError` 等导入问题。
- **关键修复**：解决模块导入时 `parser.parse_args()` 导致测试失败的问题，现在 `args` 默认为 `DefaultArgs` 实例，仅在 `main()` 运行时解析真实命令行参数。
- `tests/test_models.py` 添加 `sys.path` 动态路径设置，修复 `ModuleNotFoundError: No module named 'stock_prediction'` 导入错误。
- 所有 23 项单元测试现已通过，覆盖率达到 20%。

### Known Issues
- 无重大问题，CLI 与测试环境已完全解耦。

## [Unreleased] - 2025-10-11


- `docs/project_analysis.md`：首次汇总系统分析与风险评估。
- `docs/architecture.md`、`docs/api.md`、`docs/ops.md`、`docs/cleanup_log.md`：补充架构示意、接口说明与运维记录。

### Changed
- 重写旧版 `README.md`，纳入中文简介、快速开始与目录结构。
- 文档统一改为 UTF-8，移除历史乱码。

### Fixed
- 在 `stock_prediction.common.ensure_queue_compatibility()` 内补充 `queue.Queue.is_shutdown`，兼容 Python 3.13+。

## [2.0.0] - 2024-12-28


- 迁移核心代码至 `src/stock_prediction/`，引入标准化包结构与 `config.py`。
- 新增 `scripts/` 目录作为命令行入口，并整理测试用例与 Makefile。

### Changed
- 将历史脚本备份到 `legacy_backup/`，保留参考版本。
- 统一使用相对导入与 `pathlib.Path` 管理路径，提升跨平台能力。

### Fixed
- 解决早期版本的导入路径异常与 `is_number()` 正则问题。
