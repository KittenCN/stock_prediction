# stock_prediction

基于 PyTorch 的股票价格预测实验项目，提供从数据采集、预处理到模型训练与推理的全流程示例。当前重点验证多尺度混合模型（TemporalHybridNet）以及 PTFT + Variational SSM 双轨组合。

## 功能概览
- **行情采集**：封装 `tushare`/`akshare`/`yfinance` 三类数据源。
- **数据预处理**：将日线 CSV 聚合为 `pkl_handle/train.pkl` 队列，支持重复加载。
- **特征工程**：
eature_engineering.py 自动生成对数收益与差分特征，按配置合并宏观、行业、舆情外生变量，支持 per-symbol 归一 (enable_symbol_normalization) 并缓存统计量；启用 use_symbol_embedding 时会输出股票 ID 索引，模型可共享可学习的嵌入向量。
- **模型训练**：`src/stock_prediction/train.py` 提供统一入口，可选择 LSTM、Transformer、TemporalHybridNet、PTFT_VSSM、Diffusion、Graph 等模型，内置 Trainer/LR Scheduler/Early Stopping；Hybrid 默认使用 `HybridLoss`（MSE+分位+方向+Regime）并扩展波动度/极值约束。
- **Hybrid 总线**：--model hybrid 聚合卷积/GRU/Attention 与 PTFT、VSSM、Diffusion、Graph 分支输出，可通过 ranch_config 设置分支先验权重，软门控自动调节贡献度，并在多股票场景下共享股票嵌入。
- **推理预测**：src/stock_prediction/predict.py 负责加载模型权重并输出预测结果，保持与训练阶段一致的特征加工与嵌入策略。
- **评估指标**：`metrics.py` 自动采集 RMSE、MAPE、分位覆盖率、VaR、CVaR 等金融指标，训练/测试后保存至 `output/metrics_*.json`。
- **技术指标库**：`target.py` 内置常见指标（MACD、KDJ、DMI、ATR 等）。

## 快速开始
```bash
conda activate stock_prediction
pip install -r requirements.txt
python scripts/train.py --mode train --model ptft_vssm --pkl 1 --epoch 2
# 推理示例
python scripts/predict.py --model ptft_vssm --test_code 000001
```
> 首次运行请先执行 `python scripts/getdata.py --api akshare --code 000001.SZ` 与 `python scripts/data_preprocess.py --pklname train.pkl` 准备数据。

## 目录结构
```
project-root/
├─ src/stock_prediction/
│  ├─ config.py              # 路径与目录管理
│  ├─ init.py                # 超参数、设备与共享队列
│  ├─ common.py              # 数据集、可视化、模型保存工具
│  ├─ models/                # 模型集合（LSTM/Transformer/TemporalHybridNet/PTFT_VSSM 等）
│  ├─ train.py               # 训练 / 测试主流程
│  ├─ predict.py             # 推理入口
│  ├─ getdata.py             # 行情采集脚本
│  ├─ data_preprocess.py     # 预处理与序列化
│  └─ target.py              # 技术指标函数库
├─ scripts/
│  ├─ train.py               # 命令行训练封装
│  └─ predict.py             # 命令行推理封装
├─ tests/                    # PyTest 用例
├─ docs/                     # 架构、策略、运维文档
├─ models/                   # 训练后的模型权重
├─ stock_daily/              # 原始行情数据
├─ pkl_handle/               # 预处理后的队列文件
└─ CHANGELOG.md              # 变更记录
```

## 支持模型 (`--model`)
| 参数 | 结构概述 | 场景 |
| ---- | -------- | ---- |
| `lstm` | 3 层 LSTM + 全连接 | 基线验证 |
| `attention_lstm` | LSTM + 注意力 | 关注关键时间片段 |
| `bilstm` | 双向 LSTM | 加强上下文建模 |
| `tcn` | 时序卷积网络 | 捕捉局部模式 |
| `multibranch` | 价格/指标双分支 LSTM | 面向多特征族 |
| `transformer` | 自定义 Transformer 编解码结构 | 长序列建模 |
| `cnnlstm` | CNN + LSTM + Attention | 多步预测（需 `predict_days` > 0）|
| `hybrid` | Hybrid Aggregator（卷积/GRU + PTFT/VSSM/Diffusion/Graph 总线）| 多模态特征融合 |
| `ptft_vssm` | PTFT + Variational SSM 双轨组合 | 概率预测与风险评估 |
| `diffusion` | DiffusionForecaster（扩散式去噪解码） | 情景生成、尾部风险分析 |
| `graph` | GraphTemporalModel（自适应图结构） | 多资产关联建模 |

批量对比可执行 `scripts\run_all_models.bat`（默认运行训练+测试模式）。

## 特征工程配置
- 所有特征加工策略由 `config/config.yaml` 的 `features` 节定义：
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
cat output/metrics_*.json
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

## 贡献指南
1. 新增模型请在 `src/stock_prediction/models/` 中实现，并在训练/推理入口注册。
2. 同步更新测试（`tests/test_models.py`）与文档（尤其是 `docs/model_strategy.md`、`CHANGELOG.md`）。
3. 遵循编码规范（PEP8 + 类型注释），提交前请运行 `pytest`。

---
更多背景与未来计划，请参阅 `docs/model_strategy.md` 与 `docs/system_design.md`。
