#
# 本次自动执行报告 | Automation Execution Report

## 需求摘要 | Requirement Summary
- 目标：彻底修复 `--full_train 1` 场景下 Hybrid 模型训练集、测试集、预测阶段归一化不一致的问题。
- 核心痛点：训练末尾误差无法收敛、预测阶段出现 139.50 的异常值、归一化文件无法按股票区分。

## 方案概览 | Solution Overview
- **按股票持久化归一化统计**：`Stock_Data` 与 `stock_queue_dataset` 新增 `norm_symbol` 支持，每次归一化都会向全局 `symbol_norm_map` 写入零填充 6 位 `ts_code` 对应的均值/方差。
- **模型保存写入 per_symbol 映射**：`save_model()` 生成的 `*_norm_params*.json` 包含 `per_symbol` 字段，并记录版本号；推理前由 `_apply_norm_from_params()` 根据目标股票优先载入对应统计。
- **训练/测试/预测共享统计**：`test()`、`predict()` 在加载模型时应用 per-symbol 归一化，并在检测到与当前数据差异大于 1e-3 时输出警告。
- **自检工具升级**：`scripts/verify_normalization.py` 支持 `--stock_code/--ts_code`、`--norm_file` 参数，可同时对原始数据和持久化文件进行一致性核查。
- **文档同步**：`docs/diagnosis_hybrid_training_inference_gap.md`、`ASSUMPTIONS.md`、`CHANGELOG.md`、`agent_report.md` 更新为 UTF-8 并记录新的流程与风险提示。

## 实现与自测 | Implementation & Self-testing
- 代码要点：
  - `src/stock_prediction/common.py`：归一化函数记录 per-symbol 统计，`save_model()` 写入 `per_symbol` 映射。
  - `src/stock_prediction/train.py`：新增 `_apply_norm_from_params`、`_warn_if_norm_mismatch` 辅助函数，`test()`、`predict()`、`contrast_lines()` 全量贯通 `norm_symbol`。
  - `scripts/verify_normalization.py` 重写，支持股票过滤与 `_norm_params` 文件对比。
- 自测命令：
  1. `python scripts/train.py --mode predict --model hybrid --test_code 000019 --full_train 1`
  2. `python scripts/verify_normalization.py --stock_code 000019 --norm_file models/GenericData/HYBRID/HYBRID_out4_time5_norm_params.json`
- 预期结果：预测 CSV 中 `Forecast` 落在 30~35 区间，`verify_normalization` 返回 `Verification passed`。

## 已知限制 | Known Limitations
1. 历史权重若缺少 per-symbol 记录，需要重新保存模型或执行完整训练流程才能获得新的 `*_norm_params*.json`。
2. `symbol_norm_map` 在 `--mode predict` 单独运行前需先加载模型；如果用户手动清空缓存，需重新调用 `_apply_norm_from_params()`。
3. CLI 警告目前仅在 `test`/`predict` 流程输出提示，尚未将异常直接视为错误返回码。

## 建议迭代 | Suggested Iterations
1. 将 per-symbol 归一化加载封装为公共工具，供 `src/stock_prediction/predict.py` 与批量推理脚本共用。
2. 在 `_apply_norm_from_params()` 中校验 `show_list/name_list` 长度与模型配置一致，发现异常立即终止流程。
3. 为历史模型提供一次性迁移脚本（读取旧版文件并补写 `per_symbol` 映射），提升兼容性。

## 风险与下一步 | Risks & Next Steps
- 风险：若用户继续使用旧版 `*_norm_params*.json` 文件，仍会触发尺度错配；需要在发布说明中强调重新训练或执行修复脚本。
- 下一步：将 CLI 警告接入日志模块，并在文档中新增“常见归一化异常及处理”条目，降低后续排障成本。

---
**更新时间**：2025-10-21  
**维护者**：项目团队
