# 扩散模型与图结构建模预研路线

> 目标：在现有 PTFT + VSSM 组合之外，评估扩散式时间序列模型与图神经网络（GNN）的可行性，为多资产、跨市场预测提供储备能力。当前仓库已集成首版 `DiffusionForecaster` 与 `GraphTemporalModel`（`--model diffusion/graph`），供后续实验快速迭代。

---

## 1. 扩散模型（Diffusion Forecaster）预研

### 1.1 背景
- 代表模型：TimeGrad、DiffWave、Score-based Sequence Model。
- 优势：可生成未来价格分布，支持 VaR/CVaR 等风险评估；适合刻画多峰或尾部风险。

### 1.2 关键步骤
1. **数据准备**：按资产/时间切片构建历史路径样本，确保尺度一致并完成收益率标准化。
2. **基础实现**：参考 TimeGrad，搭建扩散 + 反向解码器骨干，初期使用单资产实验验证训稳性。
3. **条件信息**：尝试将 PTFT 的上下文编码作为条件输入，加速扩散模型收敛。
4. **评估指标**：
   - 生成样本的统计属性（均值、方差、偏度、峰度）。
   - 分位覆盖率、左尾/右尾风险指标。
   - 采样延迟、推理吞吐量。
5. **风险点**：训练成本大、超参敏感；需对噪声调度、beta schedule 做 grid search。

### 1.3 预期产出
- `research/diffusion/` 实验脚本与配置示例。
- 与当前 PTFT+VSSM 的对比报告（生成质量、风险指标）。
- 适配 `predict.py` 的扩散采样 demo。

---

## 2. 图结构建模（Graph Temporal Forecasting）预研

### 2.1 背景
- 使用图神经网络（GNN）捕捉多资产之间的联动关系，如行业同涨同跌、指数与成分股的传导。
- 代表模型：GCN + GRU、Temporal Graph Attention、Graph WaveNet。

### 2.2 关键步骤
1. **图构建策略**：
   - 静态图：按行业分类、市值相似度或相关系数构图。
   - 动态图：基于滚动相关、互信息构建时间变化的邻接矩阵。
2. **模型框架**：GCN/GraphSAGE 编码 + 时序解码（GRU/Transformer），输出多资产同步预测。
3. **特征融合**：利用现有 `feature_engineering` 生成的收益率、外生变量作为节点特征；可引入行业/宏观指标作为全局上下文。
4. **评估指标**：跨资产 RMSE、方向一致性、图注意力可解释性。
5. **落地考虑**：图规模（资产数）与计算复杂度平衡；需预处理邻接矩阵缓存。

### 2.3 预期产出
- `research/graph_forecast/` 实验工程与 README。
- 不同构图策略的对比实验记录。
- 与 PTFT+VSSM 多资产扩展的性能/可解释性对照。

---

## 3. 集成与下一步
1. **Pipeline 规划**：在 `train.py` / `predict.py` 中保留扩展接口（如 `--model diffusion`、`--model gnn`），确保与现有配置兼容。
2. **基础设施**：
   - 新增 `research/` 目录及 `Makefile` 任务（如 `make research-diffusion`）。
   - 结合 `feature_engineering` 输出多资产数据、保存邻接矩阵缓存。
3. **阶段目标**：
   - **Phase 1**：单模型验证（扩散 vs 图 GNN），关注训练稳定性与基准性能。
   - **Phase 2**：与 PTFT/VSSM 集成（条件输入或融合层），探索组合优势。
   - **Phase 3**：引入回测框架，评估策略层收益/风险表现。

---

## 4. 参考资料
- TimeGrad: Modeling Conditional Distributions of Future Sequences with Diffusion Models, AAAI 2022.
- Graph WaveNet: Deep Spatial-Temporal Graph Modeling for Complex Urban Systems, IJCAI 2019.
- Temporal GNN Survey: Wu et al., "Graph Neural Networks in Temporal Domain: A Survey", 2023.

---
本文件作为扩散模型与图结构建模的预研路线图，后续可在实验推进后同步更新至 `docs/maintenance.md` 与 `docs/model_strategy.md`。*** End Patch

## 股票嵌入实验要点（2025-10-17 · 已完成）
- DiffusionForecaster 与 GraphTemporalModel 均新增 `use_symbol_embedding` 参数，默认关闭，可在 CLI 中通过配置开启。
- 嵌入向量在时间维度上广播并与原始特征拼接，不改变现有推理入口。
- 新增 `tests/test_models.py` 针对 Diffusion/Graph 的符号嵌入形状验证，防止回归。
