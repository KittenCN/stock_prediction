# 维护与运维记录

## 1. 结构调整与演进
- **主要时间点**：2024-12-28（目录重构）、2025-10-15（模型体系升级）
- **目标**：统一包结构，逐步淘汰遗留脚本，同时引入更强的多尺度与概率模型。
- **关键动作**：
  1. 将旧版脚本迁移至 `src/stock_prediction/`，核心逻辑集中在 `common.py`、`predict.py`。
  2. 使用 `config.py` 管理所有目录路径；`init.py` 负责统一的常量、设备与共享队列。
  3. 2025-10-15 引入新的 `ProbTemporalFusionTransformer`、`VariationalStateSpaceModel` 和 `PTFTVSSMEnsemble`，并升级 `TemporalHybridNet`。
  4. `thread_save_model` 改为保存 state_dict，以避免 weight_norm 等模块的深拷贝限制。

## 2. 关键修复与经验
### 2.1 `predict.py` 导入问题
- **问题**：模块级解析命令行导致测试环境触发 `SystemExit: 2`。
- **解决**：封装 `main(argv=None)`，默认使用 `DefaultArgs`，测试可直接导入。
- **提示**：若编写新脚本，请调用 `create_predictor()` 或 `main(custom_args)`。

### 2.2 CPU 环境兼容
- **问题**：CPU 模式使用 AMP 引发警告或错误。
- **解决**：`GradScaler` 在 CPU 环境自动降级，可按需关闭 AMP。

### 2.3 模型保存阻塞
- **问题**：包含 weight_norm 的模型在深拷贝时触发异常。
- **解决**：保存前统一迁移到 CPU 并仅存 state_dict。

## 3. 自动执行摘要
- **最新结果**：`pytest`（26 项）全部通过，覆盖 TemporalHybridNet、PTFT、VSSM、数据管线与配置模块。
- **推荐命令**：`conda run -n stock_prediction pytest -q`
- **主要新增**：
  - `src/stock_prediction/models/ptft.py`：ProbTemporalFusionTransformer，支持变量选择与分位输出。
  - `src/stock_prediction/models/vssm.py`：时间依赖的变分状态空间模型，输出隐藏状态与 regime 概率。
  - `src/stock_prediction/models/ptft_vssm.py`：双轨融合模型与 `PTFTVSSMLoss`。
  - `tests/test_models.py`：覆盖 TemporalHybridNet、PTFT、VSSM 与组合模型的前向及损失逻辑。

## 4. 后续建议
1. **配置管理**：引入 `.env` / YAML 配置与 `pydantic` 校验，减少硬编码。
2. **训练流程**：封装 Trainer，对学习率调度、EarlyStopping 等通用逻辑统一处理。
3. **监控指标**：记录 RMSE、MAPE、分位覆盖率、VaR/CVaR 等指标，配合自动化报警。
4. **模型扩展**：研究扩散式时序生成模型、图结构建模等方案，可参考 `docs/model_strategy.md`。

---
本文件用于持续记录项目结构演进与关键运维经验，保持与仓库最新实现一致。 
