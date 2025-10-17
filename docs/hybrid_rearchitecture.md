# Hybrid 2.0 总线化设计文档

> 目标：在保持 `src/stock_prediction/train.py` / `predict.py` 既有入口不变的前提下，完成 `TemporalHybridNet` 的全面升级，使其成为“多模态特征总线”，能够并行融合 Transformer / Graph / Diffusion / Regime 等核心模型的表征，同时确保训练与推理流程与现有命令行参数完全兼容。
>
> **状态更新（2025-10-16）**：Hybrid 2.0 已集成至主干。`TemporalHybridNet` 默认启用 Legacy + PTFT + VSSM + Diffusion + Graph 分支，可通过 `branch_config` 控制分支开关。

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
3. **可配置性**：通过 `branch_config` 控制各分支启用、权重、loss 项，便于实现多阶段实验（默认全开）。
4. **可扩展性**：为未来新增模型分支预留接口（如 LlamaTime、Temporal Diffusion Transformer 等）。

---

## 3. 总体架构
```
TemporalHybridNet (Hybrid Aggregator)
├─ LegacyBranch   # Conv/GRU/Attention，兼容旧版 Hybrid 表现
├─ PTFTBranch     # ProbTemporalFusionTransformer 分位输出
├─ VSSMBranch     # VariationalStateSpaceModel，提供 regime 概率
├─ DiffusionBranch# DiffusionForecaster，提供场景向量
├─ GraphBranch    # GraphTemporalModel，捕捉多资产关联
└─ FusionLayer    # Regime-aware gating + 注意力融合
   └─ PredictionHead (single/multi-step, quantile、方向标签可选)
```

---

## 4. 模块与接口
| 模块 | 说明 | 主要接口 |
| ---- | ---- | ---- |
| `TemporalHybridNet` | 统一调度器，替换原 `TemporalHybridNet` 主体 | `forward(x, predict_days=None)` |
| `branch_config` | 分支开关字典，示例：`{"diffusion": False}` | 构造函数参数 |
| `ProbTemporalFusionTransformer` | 复用现有实现 | 提供分位点、attention 解释 |
| `VariationalStateSpaceModel` | 复用现有实现 | 提供 `regime_probs`、KL loss |
| `DiffusionForecaster` | 新增扩散骨架 | 输出多步预测/场景向量 |
| `GraphTemporalModel` | 新增 GNN 骨架 | 输出多资产关联表征 |
| `FusionLayer` | 将所有分支特征统一融合 | 线性投影 + GELU |
| `get_last_details()` | 记录各分支最近的中间结果 | 调试/解释用 |

---

## 5. 训练与推理策略
### 5.1 训练
- 默认损失：`HybridLoss`（MSE + 分位 Pinball + 方向性 + Regime 对齐）；仍可根据需要叠加 PTFT/VSSM 专属项。
- 支持阶段式训练：
  1. 冻结子分支，仅训练 FusionLayer（快速调权）。
  2. 解冻关键分支联合微调。
  3. 可选知识蒸馏（以 PTFT 输出为 teacher）。

### 5.2 推理
- 与旧版接口一致：
  - `predict_days <= 1` 输出单点预测。
  - `predict_days > 1` 输出多步预测，自动裁剪长度。
- `get_last_details()`：返回 base、ptft、vssm、diffusion、graph、regime 等信息，便于日志与可视化。

---

## 6. 实施进度
| 阶段 | 状态 | 关键成果 |
| ---- | ---- | -------- |
| Phase 1 | ✅ | 重构 `TemporalHybridNet` → Hybrid Aggregator；保持旧接口 |
| Phase 2 | ✅ | 集成 PTFT/VSSM/Diffusion/Graph 分支；默认全开，配置可控 |
| Phase 3 | ◻ | 深度实验与指标评估（待进行）：重点验证相对归一、股票 ID 嵌入、HybridLoss 权重等策略 |
| Phase 4 | ◻ | 深入优化（如分组训练 → 多任务蒸馏、在线学习） |

---

## 7. 风险与缓解
| 风险 | 说明 | 缓解措施 |
| ---- | ---- | ---- |
| 参数量上升 | 多分支组合易导致显存压力 | 通过 `branch_config` 精简分支，或采用轻量化子模型 |
| loss 尺度不一致 | 分支损失难以统一 | 使用 GradNorm 或分阶段训练 |
| 输入对齐问题 | 不同分支对特征要求不同 | `feature_engineering` 输出统一格式，Fusion 前做裁剪 |
| 推理延迟 | 多分支增加延迟 | 支持按需关闭分支或蒸馏到轻量模型 |

---

## 8. 文档与参考
- `docs/research_diffusion_graph.md`：扩散 / 图建模预研路线，Hybrid 2.0 是其中的集成落地。
- `docs/system_design.md`：已更新 Hybrid 架构说明。
- `docs/maintenance.md`：记录 Hybrid 2.0 规划状态。
- 源码入口：`src/stock_prediction/models/temporal_hybrid.py`（`TemporalHybridNet`）、`train.py`、`predict.py`。

---
Hybrid 2.0 的实现为后续多模态实验提供统一入口，仍需结合实际业务需求持续评估分支配置、loss 策略与性能表现。***

### Hybrid 2.0 追加说明（股票嵌入 · 已完成 2025-10-17）
- Legacy 分支前置 `nn.Embedding`，与卷积/GRU/Attention 输入拼接，门控融合时共享符号上下文。
- 各子分支（PTFT/VSSM/Diffusion/Graph）统一接收扩展后的特征维度，避免重复构造嵌入表。
- 嵌入与分支开关通过配置 `features.use_symbol_embedding` 控制，可在文档或 CLI 中动态启停。

### Hybrid 2.0 追加说明（分支门控 · 已完成 2025-10-20）
- `branch_config` 支持 `{"enabled": bool, "weight": float}` 形式，可为 PTFT/VSSM/Diffusion/Graph 分支设置启用与先验权重。
- 新增软门控向量与温度参数，对 Legacy + 各子分支 + Regime 辅助特征进行加权，再送入融合层。
- `get_last_details()` 暴露 `fusion_gate`、`fusion_gate_logits` 等诊断信息，便于调试与可视化。
- 兼容阶段式训练：可通过 `branch_config` 快速冻结或调低分支权重，配合 HybridLoss 调参。
