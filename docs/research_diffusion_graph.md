# 扩散模型与图结构建模迭代记录（2024-2025）

> 目标：在现有 PTFT + VSSM 组合之外，持续评估扩散式时间序列模型与图神经网络（GNN）的可行性，为多资产、跨市场预测提供储备能力。仓库已集成 `DiffusionForecaster` 与 `GraphTemporalModel`（`--model diffusion/graph`），本文件同步记录后续迭代。

---

## 1. 扩散模型（Diffusion Forecaster）

### 1.1 背景
- 代表模型：TimeGrad、DiffWave、Score-based Sequence Model。
- 优势：可生成未来价格分布，支持 VaR/CVaR 等风险评估；能够刻画多峰或尾部风险。

### 1.2 关键步骤
1. **数据准备**：按资产/时间切片构建历史路径样本，保证尺度一致并完成收益率标准化。
2. **基础实现**：参考 TimeGrad，搭建噪声注入 + 反向解码器骨干，先以单资产样本验证训练稳定性。
3. **条件信息**：尝试将 PTFT 的上下文编码作为条件输入，加速扩散模型收敛。
4. **评估指标**：生成样本统计属性（均值、方差、偏度、峰度）、分位覆盖率、尾部风险指标、采样延迟与推理吞吐量。
5. **风险点**：训练成本大、超参敏感；需对噪声调度、beta schedule 做 grid search。

### 1.3 现状更新（2025-10-20 · 第一轮迭代完成）
- `DiffusionForecaster` 新增 `schedule` 参数（支持 `linear` / `cosine`）以及 `context_dim` 条件输入，可在扩散过程中融合外部上下文。
- 保持股票 ID 嵌入一致化，实现与 Hybrid/PTFT 分支同源的符号特征；推理接口向后兼容。
- `tests/test_models.py` 覆盖 `cosine` 调度与上下文条件的形状/梯度测试，确保实现稳定。

### 1.4 后续计划
- 引入可学习噪声调度或 DDIM 采样，探索更高效的生成流程。
- 结合风险指标（VaR/CVaR）建立回测脚本验证业务价值。

---

## 2. 图结构建模（Graph Temporal Forecasting）

### 2.1 背景
- 通过图神经网络捕捉多资产之间的联动关系，如行业共振、指数与成分股的传导。
- 代表模型：GCN + GRU、Temporal Graph Attention、Graph WaveNet。

### 2.2 关键步骤
1. **图构建策略**  
   - 静态图：按行业分类、市值相似度或相关系数构图。  
   - 动态图：基于滚动相关、互信息构建时间变化的邻接矩阵。
2. **模型框架**：GCN/GraphSAGE 编码 + 时序解码（GRU/Transformer），输出多资产同步预测。
3. **特征融合**：利用 `feature_engineering` 生成的收益率、外生变量作为节点特征，可引入行业/宏观指标作为全局上下文。
4. **评估指标**：跨资产 RMSE、方向一致性、图注意力可解释性。
5. **落地考虑**：图规模与计算复杂度、邻接矩阵缓存及增量更新。

### 2.3 现状更新（2025-10-20 · 第一轮迭代完成）
- `GraphTemporalModel` 新增动态邻接矩阵构建逻辑（`use_dynamic_adj` / `dynamic_alpha`），可基于当前批次特征自适应地与可学习邻接混合。
- 股票嵌入保持与其他分支一致，默认仍使用静态参数矩阵以兼容旧配置。
- 新增单元测试覆盖动态邻接路径，验证形状与梯度计算稳定。

### 2.4 后续计划
- 评估不同动态构图策略（滚动相关、行业先验），并与静态图性能对比。
- 探索图注意力权重的可解释性展示，以及与 Hybrid 总线的融合方式。

---

## 3. 集成与基础设施
1. **Pipeline 规划**：`train.py` / `predict.py` 已保留 `--model diffusion` / `--model graph`，后续扩展需保持接口稳定。
2. **实验目录**：`research/` 下预留扩散与图建模实验脚手架，可通过 Makefile 添加如 `make research-diffusion` 等任务。
3. **阶段目标**：  
   - **Phase 1**：单模型验证，关注训练稳定性与基准性能。  
   - **Phase 2**：与 PTFT/VSSM 集成（条件输入或融合层），探索组合优势。  
   - **Phase 3**：结合回测框架评估策略层收益与风险表现。

---

## 4. 参考资料
- TimeGrad: Modeling Conditional Distributions of Future Sequences with Diffusion Models, AAAI 2022.
- Graph WaveNet: Deep Spatial-Temporal Graph Modeling for Complex Urban Systems, IJCAI 2019.
- Wu et al., “Graph Neural Networks in Temporal Domain: A Survey”, 2023.

---
本文件与 `docs/maintenance.md`、`docs/model_strategy.md` 联动更新，如需新增实验或记录结果，请同步补充对应章节。
