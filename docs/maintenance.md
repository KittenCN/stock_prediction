# 维护与运维记录

## 1. 结构调整与演进
- **关键时间点**：2024-12-28（目录重构）、2025-10-15（模型体系升级）  
- **目标**：统一包结构、淘汰遗留脚本，引入多尺度与概率模型，区分训练/推理职责。  
- **主要动作**：  
  1. 脚本迁移至 `src/stock_prediction/`，核心逻辑集中在 `common.py`、`train.py`、`predict.py`。  
  2. `config.py` 统一目录管理；`init.py` 集中常量、设备与共享队列。  
  3. 引入 ProbTemporalFusionTransformer、Variational SSM、PTFTVSSMEnsemble，并升级 TemporalHybridNet。  
  4. 训练/推理 CLI 完全拆分，新增 `scripts/train.py` 与批处理脚本适配。  
  5. `thread_save_model` 保存 state_dict 并迁移至 CPU，规避 weight_norm 深拷贝问题。  

## 2. 关键修复与经验
### 2.1 CLI 导入
- **问题**：模块级解析命令行导致测试环境触发 `SystemExit: 2`。  
- **处理**：封装 `main(argv=None)` 并提供默认参数对象；推理侧暴露 `create_predictor()` 以便复用。  

### 2.2 CPU 环境兼容
- **问题**：CPU 模式初始化 AMP 报警。  
- **处理**：训练/推理在 CPU 上自动降级，必要时完全禁用 AMP。  

### 2.3 模型保存
- **问题**：包含 weight_norm 的模型在深拷贝时阻塞。  
- **处理**：保存前统一迁移至 CPU，仅保存 state_dict。  

## 3. 自动执行摘要
- **最新结果**：`pytest` 现有 28 项全部通过，新增加的特征工程与 Regime 自适应融合测试运行正常。  
- **推荐命令**：`conda run -n stock_prediction pytest -q`。  
- **主要新增**：落地 PTFT+VSSM 改进计划 Phase A~C，包含收益率建模、Regime 自适应权重、贝叶斯 Dropout 与金融指标驱动损失。  

## 4. 后续建议与路线
1. [已完成] 引入 `.env` / YAML 配置并结合 `pydantic` 校验，减少硬编码。  
2. [已完成] 封装 Trainer、学习率调度与 EarlyStopping，提升训练可重复性。
3. [已完成] 建立 RMSE、MAPE、分位覆盖率、VaR/CVaR 等指标的自动采集与监控。  
4. [进行中] 扩散模型、图结构建模预研：首版 DiffusionForecaster/GraphTemporalModel 已集成（`--model diffusion/graph`），后续持续迭代，详见 `docs/research_diffusion_graph.md`。  
5. [进行中] Hybrid 2.0 总线化重构：统一 PTFT/VSSM/Diffusion/Graph 分支，`HybridLoss` 已引入（MSE+分位+方向+Regime），详见 `docs/hybrid_rearchitecture.md`。  
6. **PTFT+VSSM 改进计划（源自 `docs/ptft_vssm_analysis_20251015.md`）**：  
   - [已完成] 收益率/差分建模以缓解非平稳性。  
   - [已完成] 基于 VSSM Regime 的自适应融合权重。  
   - [已完成] 模型瘦身与正则化（贝叶斯 Dropout、L2、Regime 辅助分类）。  
   - [已完成] 引入宏观、行业、舆情等外生特征，支持多股票联合训练与滑动窗口集成。  
   - [已完成] 在损失中加入方向性、夏普/最大回撤等金融指标约束，充分利用分位输出。  

> 改进路线的详细任务拆解与优先级已同步至 `docs/model_strategy.md`，后续迭代需逐项评审并在本文件记录执行结果。  

---  
本文件持续跟踪结构演进与运维经验，确保仓库实现与文档同步。 

## 5. 预测均线吸附问题 & 优化计划（2025-10-16）

近期多股票联合训练 → 单股票推理的实验中，预测结果出现明显“靠近全局均线”的懒惰解。为避免跨资产尺度污染与信息丢失，制定如下行动计划：

1. **股票级相对归一（已完成，2025-10-16）**  
   - 在 `feature_engineering` 中对每只股票执行局部均值/标准差归一，保留还原所需统计量，并通过 `enable_symbol_normalization` 控制；
   - 兼容仅单股票/缺乏统计数据的情况，自动回退至原始尺度。
2. **引入股票 ID 特征（T+2 周）**  
   - 在数据加载阶段附带 `ts_code`，并在 Hybrid/PTFT/Graph 分支中共享嵌入向量；
   - `FeatureSettings` 增加 `use_symbol_embedding`、`embedding_dim` 配置。
3. **强化损失函数（并行进行）**  
   - 扩展 `HybridLoss`/`PTFTVSSMLoss`，加入预测波动度约束、方向/极值奖励，抑制“均值吸附”行为；
   - 结合当前 `HybridLoss`（MSE + 分位 + 方向 + Regime）进一步评估权重设置。
4. **模型层次化改造（Hybrid 2.1，T+3~4 周）**  
   - 在 Hybrid 总线中实现“分组 → 全局”两级融合：先按行业/市值等标签对分支输出聚合，再通过门控输出；
   - 为 PTFT/VSSM 增加趋势/波动多任务头，提高特征区分度。
5. **分组训练与多任务蒸馏（规划阶段）**  
   - 对行业/风格分组的子集独立训练，再将经验蒸馏到统一的 Hybrid 模型；
   - 设定蒸馏指标（MSE、方向、一致性）并记录在 `docs/hybrid_rearchitecture.md` 执行情况。

执行完成后需同步更新：README、CHANGELOG、以及本节记录的状态。详细任务排期与负责人请见 `docs/model_strategy.md`。
