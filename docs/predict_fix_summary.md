# Predict.py 修复总结

## 修复日期
2025-10-15

## 问题描述

运行 `python scripts/predict.py --model ptft_vssm --test_code 000615` 时出现以下问题：

1. **Pandas FutureWarning**: Series 使用位置索引（`[0]`）已过时
2. **'Stock_Data' object has no attribute 'trend'**: 属性初始化顺序错误
3. **FileNotFoundError**: 模型路径错误（`models/000615/PTFT_VSSM/...` 而不是 `models/GenericData/PTFT_VSSM/...`）

## 修复内容

### 1. 修复 pandas FutureWarning (predict.py)

**位置**: `src/stock_prediction/predict.py` 第 219, 225, 235 行

**修改前**:
```python
if str(item['ts_code'][0]).zfill(6) == str(test_codes[0]):
if data.empty or data['ts_code'][0] == "None":
lastdate = predict_data['Date'][0].strftime('%Y%m%d')
```

**修改后**:
```python
if str(item['ts_code'].iloc[0]).zfill(6) == str(test_codes[0]):
if data.empty or data['ts_code'].iloc[0] == "None":
lastdate = predict_data['Date'].iloc[0].strftime('%Y%m%d')
```

**原因**: pandas 2.x 中，Series 的整数索引将被视为标签而非位置，必须使用 `.iloc[pos]` 进行位置访问。

---

### 2. 修复 Stock_Data 类的 trend 属性错误 (common.py)

**位置**: `src/stock_prediction/common.py` 第 196-208 行

**修改前**:
```python
def __init__(self, mode=0, transform=None, dataFrame=None, label_num=1, predict_days=0, trend=0):
    try:
        assert mode in [0, 1, 2]
        self.mode = mode
        self.predict_days = predict_days
        self.data = self.load_data(dataFrame)
        self.normalize_data()
        self.value, self.label = self.generate_value_label_tensors(label_num)
        self.trend = trend  # ❌ trend 在调用 generate_value_label_tensors 之后赋值
```

**修改后**:
```python
def __init__(self, mode=0, transform=None, dataFrame=None, label_num=1, predict_days=0, trend=0):
    try:
        assert mode in [0, 1, 2]
        self.mode = mode
        self.predict_days = predict_days
        self.trend = trend  # ✅ trend 必须在 generate_value_label_tensors 之前赋值
        self.data = self.load_data(dataFrame)
        self.normalize_data()
        self.value, self.label = self.generate_value_label_tensors(label_num)
```

**原因**: `generate_value_label_tensors` 方法在第 319 行使用了 `self.trend`，因此必须在调用该方法**之前**赋值，否则会抛出 `AttributeError`。

---

### 3. 修复模型路径错误 (predict.py)

**位置**: `src/stock_prediction/predict.py` 第 274 行

**修改前**:
```python
def main(argv=None):
    # ...
    if not args.test_code:
        raise ValueError('test_code 参数不能为空')
    _init_models(args.test_code)  # ❌ 使用 test_code (如 000615) 作为 symbol
    predict([args.test_code])
```

**修改后**:
```python
def main(argv=None):
    # ...
    if not args.test_code:
        raise ValueError('test_code 参数不能为空')
    _init_models(symbol)  # ✅ 使用 symbol (Generic.Data) 作为模型路径
    predict([args.test_code])
```

**原因**: 
- `test_code` 是具体的股票代码（如 `000615`），用于从 pkl 队列中查找数据
- `symbol` 是模型类别（如 `Generic.Data`），用于确定模型保存路径
- 错误的做法导致路径为 `models/000615/PTFT_VSSM/...`
- 正确的路径应该是 `models/GenericData/PTFT_VSSM/...`

---

## 模型路径结构说明

### 正确的路径结构
```
models/
├── GenericData/          # 通用数据集（多股票训练）
│   ├── LSTM/
│   ├── TRANSFORMER/
│   ├── PTFT_VSSM/
│   │   └── PTFT_VSSM_out4_time5_Model.pkl
│   └── ...
└── [特定股票代码]/       # 单股票训练（如果需要）
    └── ...
```

### Config.get_model_path() 方法

**定义位置**: `src/stock_prediction/config.py` 第 66-70 行

```python
def get_model_path(self, model_type, symbol="Generic.Data"):
    """获取特定模型的保存路径"""
    symbol_clean = symbol.replace(".", "")  # Generic.Data -> GenericData
    model_dir = self.models_path / symbol_clean / model_type.upper()
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / model_type.upper()
```

**使用方式**:
- 训练模式 (`train.py`): 使用 `symbol` 变量（从 `common.py` 导入）
- 预测模式 (`predict.py`): 应使用 `symbol` 变量，而不是 `args.test_code`

---

## 验证修复

运行以下命令验证修复：

```bash
# 1. 运行单元测试
pytest tests/ -v

# 2. 测试预测功能
python scripts/predict.py --model ptft_vssm --test_code 000615

# 3. 测试训练功能（确保路径一致）
python scripts/train.py --mode train --model ptft_vssm --epoch 1
```

---

## 相关文件

- `src/stock_prediction/predict.py`: 预测脚本（修复 pandas warning 和模型路径）
- `src/stock_prediction/common.py`: 数据集类（修复 trend 属性初始化顺序）
- `src/stock_prediction/config.py`: 配置管理（模型路径生成）
- `src/stock_prediction/train.py`: 训练脚本（参考正确的 symbol 使用方式）

---

## 注意事项

1. **symbol vs test_code**:
   - `symbol`: 模型分类标识（如 `Generic.Data`），决定模型保存路径
   - `test_code`: 具体股票代码（如 `000615`），用于数据查询

2. **pandas 兼容性**:
   - 始终使用 `.iloc[pos]` 进行位置访问
   - 始终使用 `.loc[label]` 进行标签访问

3. **类属性初始化顺序**:
   - 如果方法中使用了某个属性，该属性必须在调用方法前初始化

---

## 后续优化建议

1. 在 `predict.py` 中添加更详细的错误提示，指导用户检查模型路径
2. 在 `config.py` 中添加 `list_available_models()` 方法，列出所有可用模型
3. 考虑支持单股票专用模型（路径为 `models/{stock_code}/...`）
