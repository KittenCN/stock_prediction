# 维护与运维记录

## 1. 结构调整与演进
- **主要时间点**：2024-12-28（目录重构）、2025-10-15（模型体系升级）
- **目标**：统一包结构，淘汰遗留脚本，引入多尺度与概率模型，区分训练/推理职责。
- **关键动作**：
  1. 将脚本迁移至 `src/stock_prediction/`，核心逻辑集中在 `common.py`、`train.py`、`predict.py`。
  2. 使用 `config.py` 管理所有目录；`init.py` 统一常量、设备与共享队列。
  3. 2025-10-15 引入 `ProbTemporalFusionTransformer`、`VariationalStateSpaceModel`、`PTFTVSSMEnsemble`，并升级 `TemporalHybridNet`。
  4. 将训练流程拆分至 `train.py`，推理由 `predict.py` 独立承担；`scripts/` 新增 `train.py`，`run_all_models.bat` 改为调用训练脚本。
  5. `thread_save_model` 改为保存 state_dict，避免 weight_norm 深拷贝问题。

## 2. 关键修复与经验
### 2.1 CLI 导入
- 问题：模块级解析命令行导致测试环境触发 `SystemExit: 2`。
- 解决：封装 `main(argv=None)`，默认使用 `DefaultArgs`；推理端提供 `create_predictor()`。

### 2.2 CPU 环境兼容
- 问题：CPU 模式初始化 AMP 产生警告。
- 解决：训练与推理在 CPU 上自动降级，可按需关闭 AMP。

### 2.3 模型保存
- 问题：包含 weight_norm 的模型在深拷贝时阻塞。
- 解决：保存前统一迁移至 CPU，并仅保存 state_dict。

## 3. 自动执行摘要
- **最新结果**：`pytest` 26 项全部通过，覆盖 TemporalHybridNet、PTFT、VSSM、数据管线与配置模块。
- **推荐命令**：`conda run -n stock_prediction pytest -q`
- **主要新增**：
  - `models/ptft.py`、`models/vssm.py`、`models/ptft_vssm.py` 及 `PTFTVSSMLoss`。
  - `train.py` / `predict.py` CLI 拆分，新增 `scripts/train.py`。
  - `tests/test_models.py` 扩展，覆盖新模型前向与损失逻辑。

## 4. 后续建议
1. 引入 `.env` / YAML 配置与 `pydantic` 校验，减少硬编码。
2. 封装训练器（Trainer）、学习率调度与 EarlyStopping 流程。
3. 记录 RMSE、MAPE、分位覆盖率、VaR/CVaR 等指标并接入监控。
4. 探索扩散模型、图结构建模等方案，可参考 `docs/model_strategy.md`。

---
本文件用于持续记录项目结构演进与关键运维经验，保持与仓库最新实现一致。
