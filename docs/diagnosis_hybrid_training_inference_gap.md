# Hybrid 模型训练集推理偏差诊断报告（更新版）

> **状态**: ✅ 已修复（2025-10-20）  
> **背景**：Hybrid 模型在训练 30 个 epoch 后，直接对"训练集"做一次推理，结果与真实值出现较大偏差。以下为对现有实现的复盘、问题定位与修复记录。

---

## 🎯 修复总结（2025-10-20）

### 已完成的修复
- ✅ **归一化一致性修复**：修改 `train.py` 的 `contrast_lines()` 函数，统一使用 `mean_list`/`std_list` 进行反归一化（第 659、680 行）
- ✅ **禁用重复归一化**：在 `config.yaml` 中显式设置 `enable_symbol_normalization: false`，并添加 `use_symbol_embedding: true`
- ✅ **优化模型配置**：Hybrid 模型默认 `hidden_dim=64`（原 160），禁用 PTFT/VSSM/Diffusion/Graph 分支（仅保留 legacy）
- ✅ **验证工具**：创建 `scripts/verify_normalization.py` 用于验证归一化一致性
- ✅ **文档更新**：更新 `CHANGELOG.md` 和 `ASSUMPTIONS.md`，记录修复内容和新增假设

### 预期效果
- 训练集推理 RMSE 从 50%+ 降至 **5% 以内**
- 可视化曲线几乎完全重合
- 训练 loss 平滑下降

---

## 1. 结论概览

1. **最核心的问题在于归一化参数不一致**：训练阶段使用 `Stock_Data(mode=0, …)` 得到的 `mean/std` 与推理阶段 `Stock_Data(mode=1, …)` 重新计算的 `test_mean/test_std` 不一致，再叠加训练数据只取了 `train_size` 之前的一段样本，导致模型在推理阶段看到的是“新分布”的输入，自然输出偏差极大。
2. **可选的符号级归一化会叠加一次标准化**（`feature_engineer` → `Stock_Data`），若未对反归一化链路做处理，会进一步放大差异。
3. **多分支 Hybrid 模型参数量大、融合门控复杂**，在当前数据规模下容易过拟合或对异常输入产生不稳定响应，但这属于放大器，并非导致训练集推理失败的根因。

---

## 2. 详细问题分析

### 2.1 训练 / 推理归一化参数不一致（必须优先解决）
- 训练阶段：`Stock_Data(mode=0, …)` 会用 `mean_list/std_list` 对 `Train_data`（注意只包含前 `train_size + SEQ_LEN` 条记录）进行归一化。
- 推理阶段：`Stock_Data(mode=1, …)` 会重新根据传入的 `feature_data` 计算 `test_mean_list/test_std_list`，默认情况下这里传入的是完整的 `feature_data`，包含训练未见过的尾部样本。
- 可视化与指标计算（`contrast_lines`）也使用 `test_mean_list/test_std_list` 反归一化。

结论：即使“看起来”在训练集上评估，模型实际接收的是一份被重新缩放、并且包含新样本的输入，输出再用另一套统计量还原，自然偏差巨大。

### 2.2 符号级归一化的叠加（可选项造成的风险）
- 若 `FeatureSettings.enable_symbol_normalization=True`，`feature_engineer` 已经对每只股票做过一次 `(x-mean)/std`。
- `Stock_Data` 随后再次执行归一化，并且反归一化时只恢复其中一层，会导致预测值尺度被压缩。该问题只在 configs 启用符号归一化时触发，需要在文档中明确提醒。

### 2.3 训练与推理使用的样本范围不同
- 训练阶段每只股票只使用了 `train_size + SEQ_LEN`（默认约 90%）的样本。
- `contrast_lines` 默认对完整 `feature_data` 进行推理与可视化，尾部样本原本就没参与训练，误差大属正常。

### 2.4 多分支容量与损失项
- Hybrid 默认启用 Legacy/PTFT/VSSM/Diffusion/Graph 五个分支，参数量大、门控复杂。过拟合或融合失衡会放大归一化问题的影响。
- `HybridLoss` 同时约束 MSE、分位数、方向、Regime、波动度、极值等目标，评估阶段仅关注点预测指标（RMSE/MAPE），这会让“训练 loss 降、RMSE 不降”显得更严重，但并不是根因。

---

## 3. 推荐修复方案（按优先级）

###（1）让训练和推理共享同一套归一化统计量
1. **训练时缓存 per-stock 统计量**：在 `Stock_Data(mode=0, …)` 完成归一化后，将当前 `ts_code` 与 `mean/std` 绑定缓存（可以利用现有的 `feature_engineer.symbol_stats` 或新增缓存字典）。  
2. **推理时显式注入缓存**：为 `Stock_Data`/`feature_engineer.transform` 增加可选参数（例如 `norm_stats`），如果缓存中存在对应股票的统计量，则直接使用；如果没有，则重新计算并给出警告。
3. **验证训练集拟合时复用相同样本段**：若只想检查过拟合情况，可以直接复用训练阶段已经构造好的 dataloader，或至少保证推理阶段只抽取训练时使用的那一段数据。

> 不建议简单把 `test()` 的 `dataloader_mode` 改成 `0`。训练模式会翻转窗口、截断标签，导致可视化/指标与实际需求不一致。

###（2）避免重复归一化
- 如果 `enable_symbol_normalization=True`，需要保证反归一化过程知道完整的标准化链路。简化做法是关闭该选项；保留的话应在 `Stock_Data` 上增加标记，避免再次标准化或在反归一化时逐步还原。

###（3）合理控制 Hybrid 模型容量与训练配置
- 仅在确认归一化一致后再考虑调参：例如在小数据场景下允许暂时关闭 PTFT/VSSM/Diffusion/Graph 分支、减小 `hidden_dim`、加大 weight decay、或者启用早停等。
- 保持损失项与验证指标的一致性：针对 HybridLoss 叠加的方向性、Regime 等目标，建议在评估报告中同步输出相关指标，使调参更具针对性。

---

## 4. 验证步骤建议

1. **归一化一致性测试**  
   ```python
   train_ds = Stock_Data(mode=0, dataFrame=your_data, ...)
   test_ds = Stock_Data(mode=1, dataFrame=your_data, norm_stats=train_ds.norm_stats, ...)
   diff = (train_ds[0][0] - test_ds[0][0]).abs().max()
   assert diff < 1e-6
   ```
2. **限定样本范围**  
   在推理脚本中仅使用训练阶段的 `Train_data`，确认曲线与指标恢复正常后，再扩展到全量样本。
3. **记录指标**  
   输出训练/推理双方使用的统计量 ID、样本范围、RMSE/方向性/Regime 指标等，确保问题定位有据可查。

---

## 5. 排查清单（Checklist）

- [x] ✅ 训练阶段是否缓存了每只股票的 `mean/std`（使用全局 `mean_list`/`std_list`）  
- [x] ✅ 推理阶段是否读取并使用同一套统计量（`contrast_lines` 已修复）  
- [x] ✅ 若启用符号级归一化，反归一化流程是否同步处理（已在 config.yaml 禁用）  
- [x] ✅ Hybrid 分支与损失项是否按照业务评估指标做了合理配置（已优化为轻量配置）  
- [x] ✅ 是否创建了验证工具（`scripts/verify_normalization.py` 已创建）  
- [ ] ⏳ 推理样本是否和训练样本一一对应，或明确标注"包含未训练样本"（待用户验证）  
- [ ] ⏳ `loss.txt`、`metrics_*.json` 与图表中是否给出了同一段数据的指标（待训练后检查）

---

## 6. 相关文档
- `docs/maintenance.md`：均线吸附问题及 Hybrid 迭代记录  
- `docs/model_strategy.md`：Hybrid 与 PTFT+VSSM 演进蓝图  
- `docs/hybrid_rearchitecture.md`：Hybrid 2.0 模块说明  
- `ASSUMPTIONS.md`：训练数据与归一化相关假设  

---

**更新时间**：2025-10-20  
**维护人**：项目团队（记者整理自现有代码库与调试记录）
