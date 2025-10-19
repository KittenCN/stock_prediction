# 本次自动执行报告 | Automation Execution Report

## 需求摘要 | Requirement Summary
- 背景与目标 | Background & objectives: 修复 Hybrid 模型在 png/test、png/predict CSV 上的振幅塌缩与均值偏移问题，补齐训练/预测诊断能力。
- 核心功能点 | Key features: HybridLoss 权重调优、分布诊断输出、批量分析脚本、新测试用例。

## 关键假设 | Key Assumptions
- （详见 ASSUMPTIONS.md）

## 方案概览 | Solution Overview
- 架构与模块 | Architecture & modules: 更新 HybridLoss、	rain.py、predict.py 协同诊断；新增 diagnostics.py 与脚本 scripts/analyze_predictions.py，配套测试 	est_hybrid_loss_penalties.py。
- 选型与权衡 | Choices & trade-offs: 保留原 MSE 主体，通过新增约束与分布告警降低回归风险，暂未改写数据生成流程以控制改动范围。

## 实现与自测 | Implementation & Self-testing
- 一键命令 | One-liner: conda activate stock_prediction && python scripts/analyze_predictions.py --folder png/test --std-threshold 0.8
- 覆盖率 | Coverage: 新增单测验证 HybridLoss 均值/收益惩罚；其余测试待本地环境恢复后统一执行。
- 主要测试清单 | Major tests: 单元 1 项 / 集成 0 项。
- 构建产物 | Build artefacts: output/metrics_* JSON 结构新增 distribution 报告，可与 scripts/analyze_predictions.py 互通。

## 风险与后续改进 | Risks & Next Steps
- 已知限制 | Known limitations: 暂未重新训练模型验证权重效果；Windows 控制台字符集导致部分中文显示异常，但文件内编码为 UTF-8。
- 建议迭代 | Suggested iterations: 1) 在 CI 补充 make diagnose，固定对 png/test 进行分布巡检；2) 对历史权重执行一键迁移脚本，避免多人环境参数不一致。
