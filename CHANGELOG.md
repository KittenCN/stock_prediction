# Changelog

本文件遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与语义化版本（SemVer）。

## [Unreleased] - 2025-10-11

### Added
- `docs/project_analysis.md`：新增系统分析报告，集中记录架构、问题、风险与迭代建议。
- 重写 `docs/architecture.md`、`docs/api.md`、`docs/ops.md`、`docs/cleanup_log.md`，同步现状并提供运行指引。

### Changed
- 重新编写 `README.md`，提供简洁的中文简介、快速开始、目录结构与常见问题。
- 更新文档编码为 UTF-8，移除历史乱码。

### Fixed
- 在 `stock_prediction.common.ensure_queue_compatibility()` 内兜底补齐 `queue.Queue.is_shutdown` 属性，避免 Python 3.13+ 读取旧 pickle 时抛出 `AttributeError`。

### Known Issues
- `src/stock_prediction/predict.py` 在导入时即解析命令行参数，导致 `pytest` 与脚本复用失败。
- `scripts/predict.py` 仍引用不存在的 `stock_prediction.predict.main` 函数。

## [2.0.0] - 2024-12-28

### Added
- 迁移核心代码至 `src/stock_prediction/`，引入标准化包结构与 `config.py` 目录管理。
- 新增 `scripts/` 目录作为命令行入口，初步整理测试用例与 Makefile。

### Changed
- 将历史脚本备份至 `legacy_backup/`，保持向后参考能力。
- 统一使用相对导入/Path 管理路径，便于跨平台运行。

### Fixed
- 解决早期版本的导入路径与 `is_number()` 正则问题。

