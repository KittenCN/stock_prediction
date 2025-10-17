# 测试模式修复进展报告

## 📊 当前状态

### ✅ 已完成的修复

#### 1. 归一化参数空值问题（已解决）
**问题**：测试模式加载的归一化参数文件中 `mean_list` 和 `std_list` 为空

**根本原因**：
- 归一化参数在每次调用 `normalize_data()` 时都会 `clear()`
- 模型保存时，全局变量可能已被清空
- PKL 模式不重新计算归一化参数

**修复方案**：
1. ✅ 在 `src/stock_prediction/init.py` 中添加稳定副本变量：
   ```python
   saved_mean_list = []
   saved_std_list = []
   ```

2. ✅ 在 `src/stock_prediction/common.py` 的 `normalize_data()` 中保存副本：
   ```python
   if not saved_mean_list or not saved_std_list:
       saved_mean_list = mean_list.copy()
       saved_std_list = std_list.copy()
   ```

3. ✅ 在 `save_model()` 中使用稳定副本

4. ✅ 创建 `scripts/fix_norm_params.py` 工具修复现有空文件

**验证结果**：
```bash
python scripts\fix_norm_params.py
# ✓ 成功更新2个文件，mean_list和std_list各30个值

python scripts\train.py --model Hybrid --mode test --test_code 000019
# ✓ [LOG] Loaded normalization params from ...
# ✓ [LOG] mean_list length: 30, std_list length: 30
# ✓ 不再出现 "[WARN] Index X out of range" 错误
```

#### 2. 模型导入错误（已解决）
**问题**：`cannot import name 'Hybrid' from 'stock_prediction.models'`

**修复**：
```python
# 修改前
from stock_prediction.models import Hybrid
test_model = Hybrid(**model_args)

# 修改后
from stock_prediction.models import TemporalHybridNet
test_model = TemporalHybridNet(**model_args)
```

**文件**：`src/stock_prediction/train.py` 第 224 行

---

### ⚠️ 仍存在的问题

#### 维度不匹配错误
**错误信息**：
```
test error: mat1 and mat2 shapes cannot be multiplied (40x30 and 46x128)
```

**根本原因**：
- **训练时**：启用 symbol embedding，输入维度 = 30 (特征) + 16 (embedding) = 46
- **测试时**：数据加载未提供 `symbol_index`，输入维度 = 30
- **结果**：模型期望46维输入，但只收到30维

**详细分析**：
1. 模型配置（`HYBRID_out4_time5_Model_args.json`）：
   ```json
   {
     "input_dim": 30,
     "use_symbol_embedding": true,
     "symbol_embedding_dim": 16,
     "max_symbols": 4096
   }
   ```

2. 模型实际输入维度：
   ```python
   actual_input_dim = input_dim (30) + symbol_embedding_dim (16) = 46
   ```

3. 测试数据流程：
   ```
   contrast_lines()
   → load_data() / PKL 加载
   → Stock_Data / stock_queue_dataset
   → DataLoader
   → test() 函数
   → 模型前向传播 ❌ 维度不匹配
   ```

**需要修复的位置**：
1. `contrast_lines()` 函数（train.py）
2. `test()` 函数的数据加载逻辑
3. `Stock_Data` 类需要支持测试模式的 symbol_index

---

## 🔧 建议的解决方案

### 方案 1：在测试数据加载时添加 symbol_index（推荐）

**位置**：`src/stock_prediction/train.py` 的 `test()` 函数

**修改点**：
1. 检测模型是否使用 symbol embedding
2. 如果使用，确保测试数据包含 `symbol_index`
3. 从 PKL 文件或 CSV 文件中提取股票代码并映射到 symbol_index

**伪代码**：
```python
def test(dataset, testmodel=None, dataloader_mode=0):
    # ... 现有代码 ...
    
    # 检查模型是否使用 symbol embedding
    use_symbol_embedding = getattr(test_model, 'use_symbol_embedding', False)
    
    if use_symbol_embedding:
        # 确保数据加载器提供 symbol_index
        if dataloader_mode in [0, 2]:
            stock_predict = Stock_Data(
                mode=dataloader_mode,
                dataFrame=dataset,
                label_num=OUTPUT_DIMENSION,
                predict_days=int(args.predict_days),
                trend=int(args.trend),
                enable_symbol_index=True  # 新参数
            )
    # ... 其余代码 ...
```

### 方案 2：在模型加载时检测并调整输入维度

**位置**：`src/stock_prediction/train.py` 的 `test()` 函数

**修改点**：
1. 加载模型后，检测实际数据维度
2. 如果维度不匹配，禁用 symbol embedding 并重新创建模型

**优点**：向后兼容，不需要修改数据加载逻辑
**缺点**：测试时的模型配置与训练时不一致

### 方案 3：重新训练不使用 symbol embedding 的模型

**步骤**：
```bash
# 在配置中禁用 symbol embedding
python scripts\train.py --model Hybrid --epoch 10 --config use_symbol_embedding=false
```

**优点**：简单直接
**缺点**：失去 symbol embedding 的优势

---

## 📝 修改文件清单

### 已修改
1. ✅ `src/stock_prediction/init.py` - 添加稳定副本变量
2. ✅ `src/stock_prediction/common.py` - 保存和使用稳定副本
3. ✅ `src/stock_prediction/train.py` - 修复模型导入错误
4. ✅ `CHANGELOG.md` - 记录修复详情

### 新增
1. ✅ `scripts/fix_norm_params.py` - 归一化参数修复工具
2. ✅ `docs/fix_test_mode_loading.md` - 修复文档

### 待修改（解决维度问题）
1. ⏳ `src/stock_prediction/train.py` - test() 函数
2. ⏳ `src/stock_prediction/common.py` - Stock_Data 类（可选）

---

## 🧪 测试验证

### 单元测试
```bash
python -m pytest tests/ -v
# 结果：46/46 通过 ✅
```

### 归一化参数修复
```bash
python scripts\fix_norm_params.py
# 结果：✓ 成功更新2个文件 ✅
```

### 测试模式
```bash
python scripts\train.py --model Hybrid --mode test --test_code 000019
# 结果：
# ✅ 归一化参数正确加载（长度30）
# ✅ 模型参数正确加载
# ❌ 维度不匹配错误（40x30 vs 46x128）
```

---

##  后续步骤

### 立即行动
1. 决定使用哪个方案解决维度问题
2. 实现选定的方案
3. 运行完整测试验证

### 短期改进
1. 在模型保存时记录是否使用 symbol embedding
2. 在测试加载时自动检测并适配
3. 添加维度匹配验证

### 长期优化
1. 统一训练和测试的数据加载流程
2. 实现配置驱动的特征工程
3. 添加自动化测试覆盖测试模式

---

## 📚 相关文档

- `CHANGELOG.md` - 变更日志
- `agent_report.md` - 详细执行报告
- `docs/fix_test_mode_loading.md` - 修复总结
- `AGENTS.md` - 开发规范

---

**更新时间**：2025-10-20
**状态**：部分完成，归一化参数问题已解决，维度问题待修复
