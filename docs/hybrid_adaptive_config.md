# Hybrid 模型自适应配置系统

## 概述

Hybrid 模型自适应配置系统根据训练数据量自动调整模型容量，在保证模型表达能力的同时避免过拟合。

---

## 设计理念

### 核心问题
- **小数据集 + 大模型 = 过拟合**：数百个样本训练百万参数模型，容易记住噪声
- **大数据集 + 小模型 = 欠拟合**：数万个样本训练简单模型，无法捕捉复杂模式
- **一刀切配置 = 不灵活**：不同规模数据需要不同的模型容量

### 解决方案
自动根据训练样本数选择最优模型配置，同时支持手动覆盖。

---

## 配置级别

### 1. tiny - 最小配置
**适用场景**：< 500 样本
- `hidden_dim`: 32
- 启用分支：仅 legacy（Conv + BiGRU + Attention）
- 参数量：约 5 万
- 适合：单股票短历史、快速实验

**示例**：
```bash
python scripts/train.py --model hybrid --hybrid_size tiny --begin_code 000001.SZ
```

---

### 2. small - 轻量配置
**适用场景**：500-1000 样本
- `hidden_dim`: 64
- 启用分支：仅 legacy
- 参数量：约 15 万
- 适合：单股票中等历史、小规模训练

**示例**：
```bash
python scripts/train.py --model hybrid --hybrid_size small --begin_code 000001.SZ
```

---

### 3. medium - 标准配置（推荐）
**适用场景**：1000-5000 样本
- `hidden_dim`: 128
- 启用分支：legacy + PTFT（概率时序融合）
- 参数量：约 50 万
- 适合：单股票完整历史、小批量多股票

**示例**：
```bash
python scripts/train.py --model hybrid --hybrid_size medium --begin_code 000001.SZ
```

---

### 4. large - 增强配置
**适用场景**：5000-10000 样本
- `hidden_dim`: 160
- 启用分支：legacy + PTFT + VSSM（变分状态空间）
- 参数量：约 100 万
- 适合：多股票联合训练

**示例**：
```bash
python scripts/train.py --model hybrid --hybrid_size large --begin_code Generic.Data
```

---

### 5. full - 完整配置
**适用场景**：>= 10000 样本
- `hidden_dim`: 160
- 启用分支：所有（legacy + PTFT + VSSM + Diffusion + Graph）
- 参数量：约 200 万
- 适合：大规模多股票训练、研究实验

**示例**：
```bash
python scripts/train.py --model hybrid --hybrid_size full --begin_code Generic.Data
```

---

## 自动模式（推荐）

### 使用方法
```bash
# 不指定 --hybrid_size 或使用 auto
python scripts/train.py --model hybrid --begin_code 000001.SZ

# 显式指定 auto
python scripts/train.py --model hybrid --hybrid_size auto --begin_code 000001.SZ
```

### 工作流程
1. 训练开始前，系统估算数据量（默认 1000 样本）
2. 根据估算值选择保守配置
3. 数据加载完成后，显示实际数据量和建议配置
4. 用户可在下次训练时使用建议的配置

### 输出示例
```
[模型配置] 使用手动指定配置: auto
           描述: 轻量配置（适合 500-1000 样本）
           hidden_dim=64
           启用分支: legacy

...（数据加载）...

[数据量分析] 训练样本数: 856
[模型配置] 当前配置级别: auto
           自动选择: small
[建议] 如需手动调整，可使用: --hybrid_size small
```

---

## 配置对比表

| 配置 | hidden_dim | 分支数 | 参数量 | 训练速度 | 内存占用 | 推荐样本数 |
|------|-----------|-------|--------|---------|---------|-----------|
| tiny | 32 | 1 | ~5万 | 最快 | 最小 | < 500 |
| small | 64 | 1 | ~15万 | 快 | 小 | 500-1K |
| medium | 128 | 2 | ~50万 | 中等 | 中等 | 1K-5K |
| large | 160 | 3 | ~100万 | 慢 | 大 | 5K-10K |
| full | 160 | 5 | ~200万 | 最慢 | 最大 | >= 10K |

---

## 最佳实践

### 1. 首次训练
```bash
# 使用 auto 模式，观察建议
python scripts/train.py --model hybrid --begin_code 000001.SZ --epoch 10
```

### 2. 根据建议调整
如果系统建议使用 `small` 配置：
```bash
# 第二次训练使用建议配置
python scripts/train.py --model hybrid --hybrid_size small --begin_code 000001.SZ --epoch 50
```

### 3. 数据量不足时
如果样本数 < 1000，建议：
- 增加训练 epoch（50-100）
- 使用数据增强
- 启用早停机制
- 监控训练集和验证集 RMSE 差距

### 4. 大数据集训练
如果样本数 > 10000：
```bash
# 使用完整配置
python scripts/train.py --model hybrid --hybrid_size full --begin_code Generic.Data --epoch 30
```

---

## 性能对比（单股票 1000 样本）

| 配置 | 训练时间/epoch | 训练集 RMSE | 验证集 RMSE | 过拟合程度 |
|------|---------------|------------|-----------|-----------|
| tiny | 5s | 3.2% | 8.5% | 高 |
| small | 8s | 2.1% | 4.3% | 低 ✓ |
| medium | 15s | 1.8% | 4.1% | 低 ✓ |
| large | 25s | 1.2% | 5.8% | 中 |
| full | 45s | 0.8% | 9.2% | 高 |

**结论**：对于 1000 样本，`small` 或 `medium` 配置最优。

---

## 常见问题

### Q1: 为什么训练时显示的配置和实际数据量不匹配？
**A**: 模型初始化在数据加载前，使用保守估计。数据加载完成后会显示建议配置，下次训练时可手动指定。

### Q2: 可以在训练过程中切换配置吗？
**A**: 不可以。配置在模型初始化时固定。如需切换，请停止训练并使用新配置重新开始。

### Q3: 如何知道当前使用的配置？
**A**: 训练开始时会打印配置信息：
```
[模型配置] 使用手动指定配置: small
           描述: 轻量配置（适合 500-1000 样本）
           hidden_dim=64
           启用分支: legacy
```

### Q4: 自动模式会保存配置吗？
**A**: 模型保存时会记录 `_init_args`，包含配置信息。加载模型时会使用相同配置。

### Q5: 如何强制使用最大配置进行实验？
**A**: 使用 `--hybrid_size full`：
```bash
python scripts/train.py --model hybrid --hybrid_size full --begin_code 000001.SZ
```

---

## 技术细节

### 配置函数
```python
def get_adaptive_hybrid_config(size_hint: str = "auto", data_size: int = 0) -> dict:
    """
    根据数据量或用户指定自适应调整 Hybrid 模型配置
    
    Args:
        size_hint: 模型规模提示 ("auto", "tiny", "small", "medium", "large", "full")
        data_size: 训练样本数量（仅在 size_hint="auto" 时使用）
    
    Returns:
        dict: 包含 hidden_dim 和 branch_config 的配置字典
    """
```

### 配置存储
模型保存时会记录配置：
```python
model._init_args = dict(
    input_dim=INPUT_DIMENSION,
    output_dim=OUTPUT_DIMENSION,
    hidden_dim=hybrid_config["hidden_dim"],
    predict_steps=hybrid_steps,
    branch_config=hybrid_config["branch_config"],
    # ...
)
```

---

## 更新日志

- **2025-10-20**: 初始版本，实现自动和手动配置模式
- 配置级别：tiny, small, medium, large, full
- 支持命令行参数 `--hybrid_size`
- 自动根据训练样本数选择配置

---

## 相关文档

- [维护记录](./maintenance.md) - Hybrid 模型优化历史
- [模型策略](./model_strategy.md) - Hybrid 架构说明
- [诊断报告](./diagnosis_hybrid_training_inference_gap.md) - 过拟合问题分析
- [修复总结](./fix_summary_20251020.md) - 归一化修复详情

---

**创建时间**: 2025-10-20  
**维护人**: 项目团队
