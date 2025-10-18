# Hybrid 模型训练/推理偏差诊断（2025-10-21）

## 1. 现象回顾
- 以 `python scripts/train.py --mode test --model hybrid --full_train 1` 复现，`png/test/000019_HYBRID_test_close_*.csv` 中 RMSE≈1.44、MAPE≈3.97%，训练末尾仍然存在系统性低估。
- 以 `python scripts/train.py --mode predict --model hybrid --test_code 000019 --full_train 1` 复现，`png/predict/000019_HYBRID_predict_close_*.csv` 最后一行出现 139.50 的极端预测。
- 日志显示测试阶段仍然提示 `[WARN] No symbol-specific normalization stats found for 000019; using global statistics.`，说明归一化统计未按股票区分。

## 2. 根因分析
1. **归一化统计仅保存首只股票**  
   原始实现中 `Stock_Data.normalize_data()` 只在第一次调用时写入 `saved_mean_list/saved_std_list`，随后覆盖 `mean_list/std_list`。模型保存时直接写入这份全局数组，导致 `*_norm_params*.json` 中 `mean_list/std_list` 仅对应首只股票。
2. **预测阶段复用全局统计**  
   `test()`、`predict()` 在加载模型后仅将 `mean_list/std_list` 回填，并未按 `ts_code` 精细化处理，导致 000019 按其它股票的尺度进行反归一化，从而出现 139.50 的偏差。
3. **自检工具无法聚焦单只股票**  
   旧版 `scripts/verify_normalization.py` 只能检查训练/测试模式是否一致，无法对比持久化的 `*_norm_params*.json`，难以及时定位 per-symbol 错配。

## 3. 修复方案
1. **数据管道按股票缓存统计**  
   - `Stock_Data`、`stock_queue_dataset` 新增 `norm_symbol` 参数；归一化时会记录零填充 6 位的 `ts_code`（如 `000019`）并写入全局 `symbol_norm_map`。  
   - 训练时始终传入准确的股票编码（PKL 队列同样适用），保证 `symbol_norm_map` 覆盖所有参与训练的股票。
2. **模型保存携带 `per_symbol` 映射**  
   - `save_model()` 输出的 `*_norm_params*.json` 增加 `version=2` 与 `per_symbol` 字段，形如：
     ```json
     {
       "version": 2,
       "mean_list": [...],  // 全局回退
       "std_list": [...],
       "per_symbol": {
         "000019": { "mean_list": [...], "std_list": [...] },
         "...": { ... }
       }
     }
     ```
   - 旧文件仍可读取，但会在缺失 `per_symbol` 时提示警告并回退全局统计。
3. **模型加载优先匹配目标股票**  
   - `train.py/test()`、`predict.py/test()` 引入 `_apply_norm_from_params()`，在加载模型时根据 `per_symbol` 自动选择当前 `ts_code` 对应的均值/方差，并在差异超过 1e-3 时打印 `Normalization mismatch` 提示。  
   - `predict.py` 修复 `_init_models()`，确保 CLI 场景也会依据 `--test_code` 选择正确统计信息。
4. **自检脚本升级**  
   - `scripts/verify_normalization.py` 支持 `--ts_code/--stock_code`、`--norm_file` 参数，可同时比较训练/测试阶段的实际归一化与持久化文件中的 per-symbol 统计，便于回归检查。

## 4. 修复后使用方式
1. 训练：`python scripts/train.py --mode train --model hybrid --epoch 30 --full_train 1`  
   - 训练结束后 `models/GenericData/HYBRID/HYBRID_out4_time5_norm_params.json` 中应包含 `per_symbol.000019`。
2. 预测：`python scripts/train.py --mode predict --model hybrid --test_code 000019 --full_train 1`  
   - 日志应显示 `Using symbol-specific norm stats for 000019`，预测 CSV 中不再出现 139.50 异常值。
3. 自检：  
   ```
   python scripts/verify_normalization.py --ts_code 000019 --norm_file models/GenericData/HYBRID/HYBRID_out4_time5_norm_params.json
   ```
   - 输出需显示 `Verification passed`，并给出持久化文件与训练统计的差值。

## 5. 后续建议
- 为历史权重写一个迁移脚本：读取旧版 `*_norm_params*.json`，按 `stock_daily/*.csv` 重新计算并补写 `per_symbol` 字段，减少重复训练的开销。
- 将 CLI 警告接入统一日志模块，必要时在归一化错配时直接抛出异常，避免用户忽略警示信息。
- 针对多模型批量预测场景，考虑将 `_apply_norm_from_params()` 抽出至公共模块，供其他管线复用。
