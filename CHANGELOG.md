# Changelog

遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与语义化版本（SemVer）。

## [Unreleased] - 2025-10-21
### Added
- 新增 per-symbol 归一化实现：`Stock_Data`、`stock_queue_dataset` 支持 `norm_symbol`，模型保存的 `*_norm_params*.json` 写入 `per_symbol` 映射；升级 `scripts/verify_normalization.py` 支持 `--norm_file` 与 `--ts_code` 参数。
- 文档与报告全面更新为 UTF-8，`docs/diagnosis_hybrid_training_inference_gap.md`、`ASSUMPTIONS.md`、`agent_report.md` 记录新的归一化流程与自检步骤。
### Changed
- `test()` / `predict()` 在加载模型时优先匹配 per-symbol 归一化统计，并在检测到偏差（>1e-3）时输出警告，确保训练/测试/预测使用同一尺度。
### Fixed
- 修复 `--full_train 1` 场景下模型仅保存第一只股票归一化统计导致训练、预测偏差的问题；预测阶段不再出现 139.50 等异常值。

## [2.0.0] - 2024-12-28
### Added
- 迁移核心代码至 `src/stock_prediction/`，统一包结构并引入配置模块 `config.py`。
- 新增 `scripts/` 目录作为命令行入口，补齐测试用例与 Makefile。
### Changed
- 将历史脚本迁移至 `legacy_backup/` 以保留参考版本；统一使用 `pathlib.Path` 管理路径。
### Fixed
- 修复旧版导入路径异常与数值工具函数 `is_number()` 的正则匹配问题。
