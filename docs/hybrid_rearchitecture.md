# Hybrid 2.0 总线化设计文档

> 目标：在保持 `src/stock_prediction/train.py` / `predict.py` 既有入口不变的前提下，完成 `TemporalHybridNet` 的全面升级，使其成为“多模态特征总线”，能够并行融合 Transformer / Graph / Diffusion / Regime 等核心模型的表征，同时确保训练与推理流程与现有命令行参数完全兼容。

---

## 1. 背景与痛点

| 现状 | 问题 | 影响 |
| ---- | ---- | ---- |
| `TemporalHybridNet` 主要依赖卷积 + Bi-GRU + Attention | 模型结构固化，难以复用 PTFT、VSSM、Diffusion、Graph 等最新成果 | 新特征需要单独重写模型或维护多套代码 |
| 多模型输出依赖 `contrast_lines` 或独立脚本对比 | 缺乏统一的融合框架 | 训练成本高、难以进行多模态协同训练 |
| 外生特征、收益率特征已由 `feature_engineering` 提供 | Hybrid 未充分利用 | 现有模型对宏观/行业/舆情敏感度有限 |

---

## 2. 设计目标

1. **接口兼容**：保持 `--model hybrid` CLI 参数及原有训练/推理逻辑不变，支持 `predict_days`、`trend` 等既有参数。
2. **模块化融合**：将 PTFT、VSSM、DiffusionForecaster、GraphTemporalModel 等输出作为子分支输入，通过共享的融合总线输出预测结果。
3. **可配置性**：通过配置文件/命令行控制各分支启用、权重、loss 项，便于实现多阶段实验。
4. **可扩展性**：为未来新增模型分支预留接口（如 LlamaTime、Temporal Diffusion Transformer 等）。

---

## 3. 总体架构

```
HybridAggregator (新)
├─ BaseEncoder            # Conv/GRU/Attention 保留 + 可选轻量特征提取
├─ BranchManager          # 动态注册各分支（PTFT/VSSM/Diff/Graph/...）
│   ├─ PTFTBranch         # 复用 ProbTemporalFusionTransformer
│   ├─ VSSMBranch         # 复用 VariationalStateSpaceModel（提供 regime 信息）
│   ├─ DiffusionBranch    # 调用 DiffusionForecaster 得到场景向量
│   ├─ GraphBranch        # 调用 GraphTemporalModel 输出多资产表征
│   └─ LegacyBranch       # 保留旧版 convolution+GRU 输出，作为回退
├─ FusionLayer            # Regime-aware gating + Attention 融合
└─ PredictionHead         # 输出预测值 + 可选分位数/方向标签
```

核心元素：
1. **BranchManager**：负责根据配置构建子模型实例，并在 `forward` 时传入统一的 `feature_dict`（包含收益率、外生特征、Regime、Graph adjacency 等）。
2. **FusionLayer**：将各分支输出与 VSSM 提供的 regime 概率组合。可选实现：
   - 线性加权（learnable gate）；
   - 注意力（以 regime/行情特征为 Query）；
   - Mixture-of-Experts（可引入 load balancing loss）。
3. **PredictionHead**：默认输出点预测；当配置启用时输出分位数（借鉴 PTFT）、方向标签（classification head）。

---

## 4. 模块与接口设计

| 模块 | 说明 | 主要接口 |
| ---- | ---- | ---- |
| `HybridAggregator` | 统一调度器，替换原 `TemporalHybridNet` 主体 | `forward(x, extra=None, predict_steps=None)` |
| `HybridConfig` | 模型内部配置，支持分支开关、loss 权重 | 由 `AppConfig` 或 CLI 读取 |
| `BranchManager` | 注册并管理分支模型 | `register_branch(name, factory)`、`forward(feature_dict)` |
| `FeatureHub` | 输入特征组织者，包含 `feature_engineering` 输出、Regime、Graph 结构 | 通过 `common.py` 创建 |
| `FusionLayer` | 自适应融合层 | 支持 `regime_aware_gate`、`attention_fusion` |
| `HybridLoss` | 组合 MSE / 分位 / 方向性 / Regime 等 loss | `HybridLoss(output, targets, aux)` |

**输入兼容性**：
- `x`: `(batch, seq_len, input_dim)`，与现有数据集保持一致。
- `extra`: 字典形式，包含：
  - `feature_engineered`: 经过 `FeatureEngineer` 处理后的 DataFrame/Tensor。
  - `regime_probs`: 来自 VSSM 或历史模型的 regime 概率。
  - `graph_adj`: Graph 模型所需的邻接矩阵（可选）。
  - `scenario_samples`: Diffusion 模型生成的场景（可选）。

---

## 5. 训练与推理策略

### 5.1 训练
- 使用统一的 `HybridLoss`：`loss = mse + alpha*quantile + beta*direction + gamma*regime + delta*graph_consistency`。
- 当只启用部分分支时，自动降级至当前子集的 loss 组合。
- 支持阶段式训练：
  1. 冻结子分支、训练 FusionLayer（快速调权）。
  2. 解冻关键分支进行联合微调。
  3. 视需要引入知识蒸馏（例如以 PTFT 输出为 teacher）。

### 5.2 推理
- 维持 `predict.py` 接口：
  - `predict_days > 0` 输出多步预测，默认使用 FusionLayer 结果。
  - 可选输出：各分支单独预测（用于诊断）、Regime 权重、图注意力。
- 记录 `HybridAggregator` 的 `last_details`，包括分支权重、Regime 辅助指标等，方便分析。

---

## 6. 实施计划

| 阶段 | 时间 | 关键任务 | 产出 |
| ---- | ---- | -------- | ---- |
| Phase 1 | T+1 周 | 重构 `TemporalHybridNet` → `HybridAggregator`，实现 BranchManager/FusionLayer 框架，保持旧行为 | PR：重构代码 + 单元测试 |
| Phase 2 | T+2~3 周 | 集成 PTFT/VSSM/Diffusion/Graph 分支；实现 `HybridLoss`；新增配置 | PR：多分支混合训练版 + CLI 开关 |
| Phase 3 | T+4 周 | 实验验证（与旧版对比），完善文档、指标输出、图可视化 | 评估报告 + 文档更新 |
| Phase 4 | 待定 | 引入知识蒸馏/在线学习等高级功能 | 可选优化 |

---

## 7. 风险与缓解

| 风险 | 说明 | 缓解 |
| ---- | ---- | ---- |
| 参数量膨胀 | 多分支组合可能导致显存/显存超标 | 引入分支开关、逐步激活；必要时使用轻量版（如低维 PTFT） |
| 训练不稳定 | 分支 loss 尺度不同 | 采用自适应权重（如 GradNorm）或分阶段训练 |
| 输入对齐 | 不同分支的输入形状/含义不一致 | 使用 `FeatureHub` 统一预处理，确保数据规范 |
| 推理延迟 | 多分支会增加推理时间 | 支持只启用关键分支、量化或蒸馏 |

---

## 8. 参考实现与文档对齐

- `docs/research_diffusion_graph.md`：扩散与图建模预研路线；Hybrid 2.0 将作为该路线的整合目标。
- `docs/system_design.md`：新增 Hybrid 2.0 架构描述，保持文档同步。
- `docs/model_strategy.md`：更新 Hybrid 相关章节，说明新的融合策略与训练方法。

---

本设计文档作为 Hybrid 2.0 总线化改造的蓝图，后续具体实现与实验结果需要在 `docs/maintenance.md`、`CHANGELOG.md` 中持续跟进。*** End Patch
