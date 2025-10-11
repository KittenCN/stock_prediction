# 项目文件清理记录

## 清理日期
2024-12-28

## 清理说明

为推进标准化包结构，旧版脚本已迁移至 `src/stock_prediction/`，原始实现备份至 `legacy_backup/`。核心调整包括：

1. **common_old.py → src/stock_prediction/common.py**  
   - 原文件 40,285 字节，存在硬编码路径与重复函数。  
   - 当前版本集中保留数据集、模型、训练辅助函数。

2. **init.py（旧版） → src/stock_prediction/init.py**  
   - 原文件 3,259 字节，采用硬编码路径；现通过 `config.py` 管理目录。  
   - 新版新增 GPU/CPU 设备检测、全局队列初始化。

3. **utils.py / getdata.py / data_preprocess.py / predict.py / target.py**  
   - 均完成路径修复与模块化迁移。  
   - 旧文件保存在 `legacy_backup/` 以便回滚比对。

4. **test.py 等临时脚本**  
   - 已移入备份目录，不再参与主流程。

## 现有活跃结构

```
src/stock_prediction/
├── __init__.py
├── config.py              # 路径与目录配置
├── init.py                # 全局常量、队列、Torch 设备
├── common.py              # 数据集、模型、训练工具、技术指标辅助
├── getdata.py             # 行情抓取脚本
├── data_preprocess.py     # 数据预处理
├── predict.py             # 训练/测试/预测主流程
├── target.py              # 技术指标函数库
└── utils.py               # 文件/日志相关工具

scripts/
├── predict.py             # 命令行入口（需补充 main() 接口支持）
├── getdata.py
└── data_preprocess.py
```

## 清理收益

- **结构清晰**：避免历史文件与新实现同名导致导入冲突。
- **统一入口**：`scripts/` 目录提供简化的命令行脚本。
- **可追溯性**：`legacy_backup/` 完整保留旧逻辑，便于查阅。

## 后续建议

- 在确认新版流程稳定后，可考虑删除冗余备份文件，或转存至归档仓库。
- 为避免再次出现乱码问题，文档统一保存为 UTF-8 编码。
- 建议在 CI 中增加“文档同步检查”步骤，确保清理记录及时更新。
