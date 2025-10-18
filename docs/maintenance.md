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

### 2.4 归一化参数持久化与自动回填（2025-10-18）
- **问题**：PKL 模式训练时，全局 `mean_list/std_list` 可能为空，导致后续推理反归一化失败；历史权重缺失 `*_norm_params*.json`。
- **处理**：
   1) `save_model()` 优先写出稳定副本 `saved_mean_list/saved_std_list`；为空则自动从 `train_pkl_path` 反序列化队列计算均值/方差并写入。
   2) `train.py` 的 `test()` 和 `predict.py` 的 `test()` 在加载权重前会优先读取 `*_norm_params*.json` 并更新全局参数。
   3) 历史权重可用 `scripts/fix_norm_params.py` 一次性修复，后续训练不再需要该脚本。

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

近期多股票联合训练及单股票推理实验中，预测结果出现明显“靠近全局均线”的懒惰解。为避免跨资产尺度污染与信息丢失，制定如下行动计划：

1. **股票级相对归一（已完成 · 2025-10-16）**  
   - 在 `feature_engineering` 中对每只股票执行局部均值/标准差归一，保留还原所需统计量，并通过 `enable_symbol_normalization` 控制；
   - 兼容仅单股票或缺乏统计数据的情况，自动回退至原始尺度。
2. **引入股票 ID 特征（已完成 · 2025-10-17，详见 5.1 节）**  
   - 在数据加载阶段附带 `ts_code`，并在 Hybrid/PTFT/Diffusion/Graph 分支中共享嵌入向量；
   - `FeatureSettings` 增加 `use_symbol_embedding`、`symbol_embedding_dim` 配置。
3. **强化损失函数（已完成 · 2025-10-20，详见 5.2 节）**  
   - 扩展 `HybridLoss`/`PTFTVSSMLoss`，加入预测波动度约束与极值奖励，抑制“均值吸附”行为；
   - 调整各项权重，在保留 MSE/分位/方向/Regime 基线的前提下补充波动指标评估。
4. **模型层次化改造（Hybrid 2.1，T+3~4 周）**  
   - 在 Hybrid 总线中实现“分组→全局”两级融合：先按行业/市值对分支输出聚合，再通过门控输出；
   - 为 PTFT/VSSM 增加趋势/波动多任务头，提高特征区分度。
5. **分组训练与多任务蒸馏（规划阶段）**  
   - 对行业/风格分组的子集独立训练，再将经验蒸馏到统一 Hybrid 模型；
   - 设定蒸馏指标（MSE、方向、一致性）并记录在 `docs/hybrid_rearchitecture.md`。

执行完成后需同步更新 README、CHANGELOG，以及本节记录的状态；详细排期与负责人请参阅 `docs/model_strategy.md`。

### 5.1 股票 ID 嵌入执行总结（2025-10-17）
- ✅ 数据集与特征工程统一输出 `_symbol_index`，训练/推理阶段自动缓存嵌入索引，支持多股票联合训练。
- ✅ TemporalHybridNet、PTFTVSSMEnsemble、DiffusionForecaster、GraphTemporalModel 均已接入可学习的 ts_code 嵌入，支持 CLI `--model hybrid/diffusion/graph/ptft_vssm`。
- ✅ Trainer/预测管线兼容新增符号张量，旧模型保持向后兼容。
- ✅ 相关测试：``tests/test_models.py`` 新增股票嵌入形状与梯度验证用例。

### 5.2 损失函数强化执行总结（2025-10-20）
- ✅ HybridLoss 增加波动度与极值罚项，默认权重可在训练脚本中调整，缓解预测均值吸附。
- ✅ PTFTVSSMLoss 同步加入波动度/极值项，并整理 Sharpe、最大回撤等指标权重，支持更细粒度调参。
- ✅ 训练脚本继续使用统一接口，旧权重配置保持兼容。
- ✅ 	ests/test_models.py 新增相关单元测试，覆盖梯度反传与有限值校验。

### 5.3 扩散/图模型迭代总结（2025-10-20）
- ✅ `DiffusionForecaster` 支持 `schedule`（linear/cosine）与 `context_dim` 条件输入，结合股票嵌入保持接口兼容。
- ✅ `GraphTemporalModel` 新增动态邻接混合（`use_dynamic_adj` / `dynamic_alpha`），可按批次特征自适应调整关联权重。
- ✅ `tests/test_models.py` 补充扩散/图模型的上下文、动态邻接测试，验证梯度反传稳定。
- ✅ 相关文档：`docs/research_diffusion_graph.md` 更新阶段成果与后续计划。

### 5.4 Hybrid 分支门控执行总结（2025-10-20）
- ✅ `branch_config` 支持传入 `{"enabled": bool, "weight": float}`，用于按阶段控制各分支及先验权重。
- ✅ `TemporalHybridNet` 内部新增软门控向量与温度参数，对 Legacy/PTFT/VSSM/Diffusion/Graph/Regime 特征进行自适应加权。
- ✅ `get_last_details()` 暴露 `fusion_gate` 及 logits，便于监控各分支贡献度。
- ✅ 相关单元测试：`tests/test_models.py` 新增 Hybrid 门控测试，覆盖权重归一性与梯度检验。

### 5.5 全量训练接口准备（2025-10-20）
- ✅ 新增 --full_train 参数，开启后数据不再拆分训练/测试集，训练阶段跳过验证流程。
- ✅ 	est() / contrast_lines() 在全量模式下自动跳过，避免无意义的验证开销。
- ✅ Pkl 队列加载逻辑在全量模式下仅注入训练队列，保持下游流程兼容。

