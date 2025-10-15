# 维护与运维记录

## 1. 结构调整与清理
- **主要时间点**：2024-12-28。  
- **目标**：迁移旧版脚本到标准化包结构，避免历史文件与新实现冲突。  
- **关键动作**：  
  1. `common_old.py`→`src/stock_prediction/common.py`，集中保留数据集、模型与训练工具。  
  2. `init.py` 路径硬编码改由 `config.py` 管理，并增加 GPU/CPU 设备检测、全局队列初始化。  
  3. `utils.py`、`getdata.py`、`data_preprocess.py`、`predict.py`、`target.py` 统一迁移至 `src/stock_prediction/`。  
  4. 临时脚本与旧实现移入 `legacy_backup/`，以便对比和回滚。  
- **收获**：结构更清晰、入口更统一、旧代码保留在备份目录中；建议在上线稳定后再清理冗余备份。

## 2. 关键修复与经验
### predict.py 导入错误（已解决）
- **问题**：缺少可导出的 `main()` 导致 `ImportError`；局部变量覆盖全局 `device` 引发 `UnboundLocalError`。  
- **修复**：  
  1. 将脚本主体重构为 `main()`，并在文件末尾保留 `if __name__ == "__main__": main()`。  
  2. 为训练、预测相关函数补充必要的 `global` 声明与变量初始化。  
  3. 修正 `test_code`/`test_codes` 命名错误。  
- **验证**：可直接导入 `stock_prediction.predict.main`，命令行帮助正常输出，基础训练流程可用。  
- **遗留风险**：顶层仍会解析 `sys.argv`，需继续推进 CLI 解耦；全局变量过多，需要后续改为类或上下文对象。

### TemporalHybridNet 引入
- **问题背景**：现有模型缺乏多尺度建模能力，多步预测路径不统一。  
- **解决方案**：新增 `TemporalHybridNet`，并在 `predict.py`/`models/__init__.py`/`scripts/run_all_models.bat` 中注册 `--model hybrid`。  
- **测试覆盖**：`tests/test_models.py` 针对单步与多步输出进行维度检测。  
- **后续改进**：考虑引入变量选择网络、多任务头、EarlyStopping 及更多外部特征。

## 3. 自动化执行概览
- **目标**：在最少交互下完成模型优化与文档同步。  
- **执行重点**：  
  - 新增 TemporalHybridNet 模型与单元测试。  
  - 更新批处理脚本 `scripts/run_all_models.bat`。  
  - 重写 README、假设、维护日志等文档。  
- **建议命令**：`conda run -n stock_prediction pytest -q`。  
- **当前状态**：测试通过，最新 CLI 改造及更深层的训练上下文拆分仍在规划中。  
- **后续计划**：继续解耦 `predict.py` CLI，补齐 EarlyStopping、变量选择、外部特征等提升项。

---
本文件统一记录结构清理、故障修复与自动化执行要点，便于后续迭代时参考。*** End Patch
