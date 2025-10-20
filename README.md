# stock_prediction

A PyTorch-based stock price prediction project covering the full workflow from data collection, preprocessing, model training, to inference. The project focuses on multi-scale hybrid models (TemporalHybridNet) and PTFT + Variational SSM combinations.

## Features
- **Market Data Collection**: Supports tushare, akshare, and yfinance as data sources.
- **Data Preprocessing**: Aggregates daily CSVs into `pkl_handle/train.pkl`, supports repeated loading.
- **Feature Engineering**: Automatically generates log returns, differences, macro/industry/sentiment exogenous variables, supports per-symbol normalization and stock embedding.
- **Model Training**: Unified entry, supports LSTM, Transformer, TemporalHybridNet, PTFT_VSSM, Diffusion, Graph, etc., with built-in trainer and early stopping.
- **Inference**: Consistent feature processing and embedding as in training, outputs prediction charts and metrics.
- **Evaluation Metrics**: Automatically collects RMSE, MAPE, quantile coverage, VaR, CVaR, etc., and saves results to `output/metrics_*.json`.
- **Technical Indicators**: Built-in MACD, KDJ, DMI, ATR, and more.

## Normalization Parameters
- When saving models, writes `*_norm_params*.json` (includes mean_list, std_list, show_list, name_list).
- If global mean/std is missing, automatically computes from `pkl_handle/train.pkl`.
- test()/predict() will load the corresponding norm_params file before loading weights to ensure consistent denormalization.

## Quick Start
```bash
conda activate stock_prediction
pip install -r requirements.txt
python scripts/train.py --mode train --model ptft_vssm --pkl 1 --epoch 2
python scripts/predict.py --model ptft_vssm --test_code 000001
```
For first-time use, run:
```bash
python scripts/getdata.py --api akshare --code 000001.SZ
python scripts/data_preprocess.py --pklname train.pkl
```

## Directory Structure
```
project-root/
├─ src/stock_prediction/      # Core code
├─ scripts/                   # CLI scripts
├─ tests/                     # Test cases
├─ docs/                      # Documentation
├─ models/                    # Trained models
├─ stock_daily/               # Raw market data
├─ pkl_handle/                # Preprocessed queues
└─ CHANGELOG.md               # Change log
```

## Supported Models
| Argument        | Structure                        | Scenario                |
| -------------- | -------------------------------- | ----------------------- |
| lstm           | 3-layer LSTM                     | Baseline                |
| attention_lstm | LSTM + Attention                 | Key time segments       |
| bilstm         | Bidirectional LSTM               | Enhanced context        |
| tcn            | Temporal Convolutional Network   | Local patterns          |
| multibranch    | Dual-branch LSTM                 | Multi-feature families  |
| transformer    | Custom Transformer               | Long sequences          |
| cnnlstm        | CNN + LSTM + Attention           | Multi-step prediction   |
| hybrid         | Conv/GRU + PTFT/VSSM/Diff/Graph  | Multimodal fusion       |
| ptft_vssm      | PTFT + Variational SSM           | Probabilistic forecast  |
| diffusion      | DiffusionForecaster              | Scenario generation     |
| graph          | GraphTemporalModel               | Multi-asset modeling    |

## Common Commands
```bash
python scripts/getdata.py --api akshare --code 000001.SZ
python scripts/data_preprocess.py --pklname train.pkl
python scripts/train.py --mode train --model transformer --epoch 2
python scripts/predict.py --model transformer --test_code 000001 --predict_days 3
pytest -q
```

## Testing & Quality
- pytest covers feature engineering, regime adaptation, stock embedding, etc.
- Run `pytest -q` before commit; use ruff/black/mypy as needed.

## Documentation Index
- docs/system_design.md: Architecture & decisions
- docs/model_strategy.md: Model design & recommendations
- docs/user_guide.md: CLI/module usage
- docs/maintenance.md: Structure changes & fixes

## FAQ
| Issue                        | Solution                                                      |
| ---------------------------- | ------------------------------------------------------------- |
| queue.Queue deserialization  | Python 3.13+ attribute change, fallback provided, re-gen pkl   |
| SystemExit on import         | CLI now uses default args, can be imported directly            |
| AMP warning on CPU           | Auto fallback, can be ignored                                 |
| Model save blocking          | Now saves state_dict and moves to CPU                         |
| Empty norm param file        | Run python scripts/fix_norm_params.py, new models auto-patch   |
| 30 vs 46 dim mismatch        | Symbol index unified, ensure *_Model_args.json is loaded       |

## Contribution Guide
1. Add new models in `src/stock_prediction/models/` and register entry.
2. Update tests and docs accordingly.
3. Follow PEP8 + type hints, run pytest before commit.

## Tips for Viewing Chinese (Windows)
- In cmd, run `chcp 65001` to switch to UTF-8 and avoid garbled text.
- Use VS Code Markdown preview for best results.

---
For more background and future plans, see docs/model_strategy.md and docs/system_design.md.
- docs/user_guide.md：命令行/模块用法├─ stock_daily/              # 原始行情数据

- docs/maintenance.md：结构调整与修复记录├─ pkl_handle/               # 预处理后的队列文件

└─ CHANGELOG.md              # 变更记录

## 常见问题```

| 问题 | 解决方式 |

| ---- | -------- |## 支持模型 (`--model`)

| queue.Queue 反序列化报错 | Python 3.13+ 属性变更，已兜底，必要时重生成 train.pkl || 参数 | 结构概述 | 场景 |

| 导入触发 SystemExit | CLI 已改为默认参数对象，可直接导入 || ---- | -------- | ---- |

| CPU 模式 AMP 提示 | 自动降级，可忽略 || `lstm` | 3 层 LSTM + 全连接 | 基线验证 |

| 模型保存阻塞 | 已改为保存 state_dict 并迁移到 CPU || `attention_lstm` | LSTM + 注意力 | 关注关键时间片段 |

| 归一化参数文件为空 | 对历史模型执行 python scripts/fix_norm_params.py，新模型自动回填 || `bilstm` | 双向 LSTM | 加强上下文建模 |

| 推理时 30 vs 46 维度不匹配 | 已统一注入 symbol 索引，确保 predict/test 读取 *_Model_args.json || `tcn` | 时序卷积网络 | 捕捉局部模式 |

| `multibranch` | 价格/指标双分支 LSTM | 面向多特征族 |

## 贡献指南| `transformer` | 自定义 Transformer 编解码结构 | 长序列建模 |

1. 新增模型请在 src/stock_prediction/models/ 实现，并注册入口。| `cnnlstm` | CNN + LSTM + Attention | 多步预测（需 `predict_days` > 0）|

2. 同步更新测试和文档。| `hybrid` | Hybrid Aggregator（卷积/GRU + PTFT/VSSM/Diffusion/Graph 总线）| 多模态特征融合 |

3. 遵循 PEP8+类型注释，提交前请运行 pytest。| `ptft_vssm` | PTFT + Variational SSM 双轨组合 | 概率预测与风险评估 |

| `diffusion` | DiffusionForecaster（扩散式去噪解码） | 情景生成、尾部风险分析 |

## 查看中文的小贴士（Windows）| `graph` | GraphTemporalModel（自适应图结构） | 多资产关联建模 |

- cmd 下可先执行：`chcp 65001` 切换到 UTF-8，避免乱码。

- 建议在 VS Code 预览 Markdown。批量对比可执行 `scripts\run_all_models.bat`（默认运行训练+测试模式）。



---## 特征工程配置

更多背景与未来计划请见 docs/model_strategy.md、docs/system_design.md。- 所有特征加工策略由 `config/config.yaml` 的 `features` 节定义：

  - `target_mode=log_return`、`return_kind=log`：默认对收盘价生成对数收益率标签。
  - `difference_columns`、`volatility_columns`：控制差分与滑动窗口统计字段。
  - `external_sources`：白名单式引入宏观/行业/舆情 CSV，按 `trade_date` 对齐并支持前向填充。
  - `multi_stock: true`：默认在多股票场景聚合训练样本，自动生成方向标签。
- 若自定义外生特征，保持日期列为八位字符串（如 `20241203`），并放置在 `config/external/` 下即可。

## 常用命令
```bash
# 抓取行情
python scripts/getdata.py --api akshare --code 000001.SZ

# 数据预处理
python scripts/data_preprocess.py --pklname train.pkl

# 训练示例
python scripts/train.py --mode train --model transformer --epoch 2

# 推理示例
python scripts/predict.py --model transformer --test_code 000001 --predict_days 3

# 运行测试
pytest -q

# 查看训练指标（训练后自动生成）
type output\metrics_*.json
```

## 测试与质量
- `pytest`：覆盖特征工程、Regime 自适应以及股票嵌入相关单元测试（Diffusion/Graph/Hybrid/PTFT 等），当前 30+ 项均通过。
- 建议在提交前执行 `pytest -q`，并按需运行 `ruff` / `black` / `mypy`。
- 暴露 `create_predictor()` 帮助脚本或测试快速构造推理器。

## 文档索引
- `docs/system_design.md`：架构拓扑与关键决策
- `docs/model_strategy.md`：模型方案设计与推荐组合
- `docs/user_guide.md`：命令行/模块用法、运维与多模型测试
- `docs/maintenance.md`：结构调整、修复记录与改进建议

## 常见问题
| 问题 | 原因 | 解决方式 |
| ---- | ---- | -------- |
| `queue.Queue` 反序列化报错 | Python 3.13+ 属性变更 | 已在 `common.ensure_queue_compatibility()` 兜底；必要时重新生成 `train.pkl` |
| 导入触发 `SystemExit` | CLI 在导入阶段解析命令行参数 | `predict.py` / `train.py` 已改为默认参数对象，可直接导入 |
| CPU 模式出现 AMP 提示 | GradScaler 默认针对 CUDA | 推理与训练会自动降级，可忽略或关闭 AMP |
| 模型保存阻塞 | weight_norm 与深拷贝冲突 | `thread_save_model` 已改为保存 state_dict 并迁移到 CPU |
| 归一化参数文件为空（旧模型） | 早期版本未保存/计算 mean/std | 对历史模型执行一次 `python scripts\fix_norm_params.py`；新训练的模型在保存时会自动从 PKL 回填并写出 `*_norm_params*.json` |
| 推理时 30 vs 46 维度不匹配 | 启用 symbol embedding 后需要 `_symbol_index` | 训练/测试/推理已统一在数据管线中注入 symbol 索引；确保 predict/test 读取到了 `*_Model_args.json` 以复现训练配置 |

## 贡献指南
1. 新增模型请在 `src/stock_prediction/models/` 中实现，并在训练/推理入口注册。
2. 同步更新测试（`tests/test_models.py`）与文档（尤其是 `docs/model_strategy.md`、`CHANGELOG.md`）。
3. 遵循编码规范（PEP8 + 类型注释），提交前请运行 `pytest`。

## 查看中文的小贴士（Windows）
- 在 cmd 窗口中可先执行：`chcp 65001` 切换到 UTF-8 代码页，避免中文显示为乱码。
- 建议在 VS Code 中查看 Markdown（预览自动使用 UTF-8）。

---
更多背景与未来计划，请参阅 `docs/model_strategy.md` 与 `docs/system_design.md`。
| `cnnlstm` | CNN + LSTM + Attention | 多步预测（需 `predict_days` > 0）|

| `hybrid` | Hybrid Aggregator（卷积/GRU + PTFT/VSSM/Diffusion/Graph 总线）| 多模态特征融合 |
