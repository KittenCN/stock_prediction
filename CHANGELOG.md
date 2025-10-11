# Changelog

## [2.0.0] - 2024-12-28

### Major Changes 重大变更
- **完成项目重构**: 迁移到标准化的Python包结构
- **统一配置管理**: 新增 `src/stock_prediction/config.py` 集中管理所有配置
- **清洁命令行接口**: 新增 `scripts/` 目录提供统一的入口脚本
- **全面测试覆盖**: 建立 20+ 测试用例，覆盖单元、集成和烟雾测试

### Added 新增
- `src/stock_prediction/config.py`: 集中化配置管理类
- `src/stock_prediction/predict.py`: 统一的预测逻辑接口  
- `scripts/`: 清洁的命令行入口脚本
- `tests/`: 全面的测试套件（4个测试文件，20个测试用例）
- 增强的 `Makefile` 支持多种任务（Windows用户可用pytest直接运行）

### Changed 变更
- **核心模块迁移**: 所有核心模块移动到 `src/stock_prediction/`
  - `utils.py`: 文件操作和工具函数，更新为相对导入
  - `getdata.py`: 数据获取模块，支持多数据源配置
  - `data_preprocess.py`: 数据预处理，保持队列处理逻辑
  - `common.py`: 核心功能模块，保持现有功能
- **导入系统**: 全面更新为包内相对导入
- **路径管理**: 统一使用配置系统管理所有路径
- `README.md`: 更新项目结构说明和使用指南

### Fixed 修复
- `is_number()` 函数: 修复正则表达式bug，现在正确处理无效数字格式
- 包导入问题: 修复所有模块间的导入依赖
- 路径问题: 统一路径管理，支持自动目录创建

### Migration Guide 迁移指南
- **新的入口点**: 使用 `scripts/predict.py` 替代原来的 `predict.py`
- **配置访问**: 通过 `from stock_prediction.config import Config` 访问配置
- **数据获取**: 使用 `scripts/getdata.py` 或 `python -m stock_prediction.getdata`
- **测试运行**: 使用 `pytest tests/` 运行所有测试

## Previous Versions 历史版本
- 引入标准目录结构（src/scripts/tests/docs/config/CI/Makefile）。
