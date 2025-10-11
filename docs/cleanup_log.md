# 项目文件清理记录

## 清理日期
2024年12月28日

## 清理操作

### 已移动到 `legacy_backup/` 的文件
以下文件是旧版本的实现，已被新的标准化包结构替代：

1. **common_old.py** (原 common.py)
   - 原大小: 40,285 字节 (1002 行)
   - 最后修改: 2025年7月26日
   - 状态: 包含与新版本相同的函数，但使用硬编码路径

2. **init.py** 
   - 原大小: 3,259 字节 (120 行)
   - 最后修改: 2025年7月19日
   - 状态: 使用硬编码路径系统，已被配置系统替代

3. **utils.py**
   - 原大小: 6,172 字节
   - 最后修改: 2025年10月11日
   - 状态: 早期版本，功能已迁移到 src/stock_prediction/utils.py

4. **getdata.py**
   - 原大小: 11,825 字节 (288 行)
   - 最后修改: 2025年10月11日
   - 状态: 包含导入错误，功能已迁移到 src/stock_prediction/getdata.py

5. **data_preprocess.py**
   - 原大小: 1,625 字节
   - 最后修改: 2025年10月11日
   - 状态: 早期版本，功能已迁移到 src/stock_prediction/data_preprocess.py

6. **predict.py**
   - 原大小: 40,600 字节 (819 行)
   - 最后修改: 2025年10月11日
   - 状态: 包含导入错误，功能已重构到 src/stock_prediction/predict.py

7. **target.py**
   - 最后修改: 旧文件
   - 状态: 功能已迁移到 src/stock_prediction/target.py

8. **test.py**
   - 简单的API测试脚本
   - 状态: 临时测试文件，不影响主要功能

## 当前活跃文件结构

### src/stock_prediction/ (新的标准包)
- `__init__.py`: 包初始化
- `config.py`: 统一配置管理 ✨
- `common.py`: 核心功能 (35,393 字节, 952 行)
- `init.py`: 初始化模块 (2,561 字节, 99 行)
- `utils.py`: 工具函数 (5,269 字节)
- `getdata.py`: 数据获取 (10,991 字节)
- `data_preprocess.py`: 数据预处理 (1,490 字节)
- `predict.py`: 预测逻辑 (10,690 字节)
- `target.py`: 目标处理

### scripts/ (命令行入口)
- `predict.py`: 预测/训练入口 (408 字节)
- `getdata.py`: 数据获取入口 (390 字节)
- `data_preprocess.py`: 数据预处理入口 (401 字节)

## 清理效果

### 优点
1. **消除混淆**: 不再有重复的文件名导致导入错误
2. **标准化结构**: 符合Python包开发最佳实践
3. **清晰入口**: scripts/ 提供清洁的命令行接口
4. **向后兼容**: 保留所有功能，维持API兼容性

### 安全保障
1. **完整备份**: 所有旧文件保存在 legacy_backup/
2. **测试验证**: 所有20个测试用例通过 ✅
3. **功能验证**: scripts入口正常工作 ✅

## 建议

### 可以删除（在确认后）
- `legacy_backup/` 中的文件在确认新系统稳定运行后可以删除
- 一些旧的数据文件夹如果不再使用也可以清理

### 保留原因
- BERT相关文件 (`bert_*.py`) 暂时保留，等待后续整合
- `stock_data_spider.py` 可能包含特殊爬虫逻辑，暂时保留
- 模型和数据目录 (`models/`, `stock_daily/` 等) 包含用户数据，保留