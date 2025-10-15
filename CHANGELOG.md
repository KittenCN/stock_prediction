# Changelog

本文件遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与语义化版本（SemVer）。

## [Unreleased] - 2025-10-15

### Added
- 新增 `TemporalHybridNet`（`--model hybrid`），结合多尺度卷积、双向 GRU、Multi-Head Attention 与统计特征，提升股票多步预测能力。
- 新增 `tests/test_models.py`，覆盖 `TemporalHybridNet` 在单步与多步预测场景下的张量形状。
- 整合文档为 `docs/system_design.md`、`docs/user_guide.md`、`docs/maintenance.md`，并同步更新 `README.md`、`ASSUMPTIONS.md`。

### Changed
- `scripts/run_all_models.bat` 现包含 `hybrid` 模型，便于横向对比。
- README 重写结构，突出混合模型及统一训练流程。

### Fixed
- `predict.py` 主逻辑封装为 `main()`，补齐全局变量初始化与 `global` 声明，修复 `UnboundLocalError` / `NameError` 等导入问题。

### Known Issues
- `predict.py` 在导入时仍会解析命令行参数，`pytest -q` 等携带额外参数的场景可能触发 `SystemExit: 2`；需进一步解耦 CLI。

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
- 解决早期版本的导入路径异常与 `is_number()` 正则问题。
