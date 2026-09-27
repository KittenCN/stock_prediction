# Changelog

遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与语义化版本（SemVer）。

## [Unreleased]
### Changed (2026-09-27)
- 清理 README 中错位和重复的中英文内容，统一安装、验证与贡献说明。
- CI 不再忽略依赖安装或测试失败；增加 Python 语法检查，并补齐配置模块依赖。
- `make ci` 改为执行真实的语法检查和测试，格式化及风格检查仍待配置。
- 将安全策略模板替换为当前项目实际的支持范围与私下报告流程。
- 记录申请开源维护支持计划前的真实维护状态和后续任务。

### Fixed (2026-09-27)
- 修复预测模块在导入时读取不存在的训练 PKL 的问题；仅在实际需要 symbol embedding 时加载索引。
- 修复特征工程测试引用不存在的外部 CSV，改为测试自身构造可复现的宏观数据。

### Changed (2025-10-21)
### Added
- 新增 per-symbol 归一化实现：Stock_Data、stock_queue_dataset 支持 
orm_symbol，模型保存的 *_norm_params*.json 写入 per_symbol 映射；scripts/verify_normalization.py 支持 --norm_file 与 --ts_code 参数。
- 文档与报告全面更新为 UTF-8，docs/diagnosis_hybrid_training_inference_gap.md、ASSUMPTIONS.md、gent_report.md 记录新的归一化流程与自检步骤。
- 新增 scripts/analyze_predictions.py，统一计算 RMSE/振幅/均值指标并输出诊断 JSON，支持针对 png/test、png/predict 批量巡检。

### Changed
- 	est() / predict() 在加载模型时优先匹配 per-symbol 归一化统计，并在检测到偏差 >1e-3 时输出警告，确保训练/测试/预测使用同一尺度。
- 调整 HybridLoss 默认权重，启用波动/极值/均值/收益四项约束（0.12/0.02/0.05/0.08），并在训练/预测阶段记录 distribution_report。

### Fixed
- 修复 --full_train 1 场景下模型仅保存第一只股票归一化统计导致训练/预测偏差的问题；预测阶段不再出现 139.50 等异常值。
- 修复 Hybrid 预测中 Open 轨道输出常数、振幅塌缩的问题：增加分布告警与校准策略。

## [2.0.0] - 2024-12-28
### Added
- 迁移核心代码至 src/stock_prediction/，统一包结构并引入配置模块 config.py。
- 新增 scripts/ 目录作为命令行入口，补齐测试用例与 Makefile。

### Changed
- 将历史脚本迁移至 legacy_backup/ 以保留参考版本；统一使用 pathlib.Path 管理路径。

### Fixed
- 修复旧版导入路径异常与数值工具函数 is_number() 的正则匹配问题。
