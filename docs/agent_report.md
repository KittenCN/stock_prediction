# 本次自动执行报告 | Automation Execution Report

## 需求摘要 | Requirement Summary
- 背景与目标 | Background & objectives: 按 `docs/ptft_vssm_analysis_20251015.md` 落地 PTFT+VSSM 改进计划，缓解价格序列非平稳性、增强 Regime 感知融合，并引入金融指标驱动的损失约束。
- 核心功能点 | Key features:
  1. 数据侧：实现收益率/差分建模、外生特征合并、多股票联合训练与滑动窗口统计。
  2. 模型侧：根据 VSSM Regime 概率自适应调整融合权重，引入贝叶斯 Dropout 与 L2 正则。
  3. 损失侧：追加方向性惩罚、Sharpe/最大回撤约束以及分位数 Pinball Loss。

## 关键假设 | Key Assumptions
- （详见 `ASSUMPTIONS.md`）| (See ASSUMPTIONS.md)
  - 数据文件包含 `trade_date`（八位字符串）与 `ts_code` 字段。
  - 外生特征以 CSV 形式放置于 `config/external/`，并在 `config/config.yaml` 中显式列入白名单。
  - 训练及测试均在 `conda activate stock_prediction` 环境内执行。

## 方案概览 | Solution Overview
- 架构与模块 | Architecture & modules:
  - 新增 `feature_engineering.py` 与 `FeatureSettings`，在数据加载阶段统一生成对数收益率、差分特征、外生变量与滑动窗口统计。
  - `PTFTVSSMEnsemble` 采用 Regime 感知 Soft Gating：以 VSSM 的 regime 概率、PTFT/VSSM 输出联合作为输入，输出自适应融合权重，并引入贝叶斯 Dropout。
  - `PTFTVSSMLoss` 增加方向性、Sharpe、最大回撤、分位 Pinball 及 Regime 辅助分类损失，形成金融指标导向的多目标训练。
- 选型与权衡 | Choices & trade-offs:
  1. 采用 Pydantic 驱动的配置体系，兼顾灵活性与类型安全。
  2. Regime 融合使用学习式 Soft Gating，避免手工规则的脆弱性；贝叶斯 Dropout 支持推理阶段 MC 采样。
  3. 金融指标损失以 MSE/MSE-like 形式整合，确保与现有训练流程兼容；考虑到训练稳定性，保留 KL 项权重为可调参数。

## 实现与自测 | Implementation & Self-testing
- 一键命令 | One-liner: `make setup && make ci && make run`
- 覆盖率 | Coverage: `pytest -q` 共 28 项通过（新增特征工程与 Regime 融合测试），详细覆盖率报告可执行 `pytest --cov=src --cov-report=xml`。
- 主要测试清单 | Major tests: 单元 28 项 / 集成 1 项 | 28 unit / 1 integration tests
- 构建产物 | Build artefacts:
  - 新模块：`src/stock_prediction/feature_engineering.py`、`src/stock_prediction/regularization.py`、`src/stock_prediction/trainer.py`、`src/stock_prediction/metrics.py`、`src/stock_prediction/app_config.py`
  - 文档更新：`docs/maintenance.md`、`docs/system_design.md`、`docs/model_strategy.md`、`docs/user_guide.md`
  - 配置示例：`config/external/macro_sample.csv`、`config/config.yaml` 中的 `features` 节
  - 指标输出：训练/测试后自动生成 `output/metrics_*.json`，包含 RMSE、MAPE、VaR、CVaR 等

## 风险与后续改进 | Risks & Next Steps
- 已知限制 | Known limitations:
  1. 外生特征示例仅覆盖宏观指标，行业/舆情仍需扩展数据源。
  2. 金融指标损失使用批内统计，对极短序列仍可能不稳定，后续可引入滚动窗口或长周期验证。
  3. 多股票联合训练仍依赖内存加载队列，后续可结合 Arrow/Parquet 或在线特征服务。
- 建议迭代 | Suggested iterations:
  1. 扩充外部信号库（行业指数、新闻情绪、风险因子）并建立特征重要性分析。
  2. 引入回测脚本（如 `make backtest`）校验金融指标损失对策略收益的提升效果。
  3. 考察 Diffusion / Graph 时序模型作为 Regime 识别与情景模拟的补充方案。
