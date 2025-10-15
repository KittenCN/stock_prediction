# 使用与运维指南

## 1. 命令行速览
| 命令 | 作用 | 备注 |
| ---- | ---- | ---- |
| `python scripts/getdata.py --api akshare --code 000001.SZ` | 抓取指定股票的日线行情，写入 `stock_daily/` | 运行前请执行 `conda activate stock_prediction` 并安装对应数据源依赖 |
| `python scripts/data_preprocess.py --pklname train.pkl` | 聚合 `stock_daily/` 下的 CSV，生成 `pkl_handle/train.pkl` 队列 | 依赖 `stock_prediction.init` 中的共享队列与 `dill` |
| `python scripts/predict.py --mode train --model ptft_vssm --predict_days 3` | 启动 PTFT + VSSM 双轨模型训练 | 其他可选模型：`lstm`、`attention_lstm`、`bilstm`、`tcn`、`multibranch`、`transformer`、`cnnlstm`、`hybrid` |
| `pytest -q` | 运行测试套件 | CLI 已解耦导入副作用，可直接执行 |

## 2. 包内能力
| 模块 | 能力 | 使用方式 |
| ---- | ---- | ---- |
| `stock_prediction.config` | `Config` 类及常用路径 | `from stock_prediction.config import config, train_pkl_path` |
| `stock_prediction.common` | 数据集、绘图、模型保存工具 | 导入前确保 `init.py` 已初始化全局状态 |
| `stock_prediction.models` | 模型集合（TemporalHybridNet、ProbTFT、VSSM、PTFT_VSSM 等） | `from stock_prediction.models import PTFTVSSMEnsemble` |
| `stock_prediction.getdata` | 行情抓取函数 `set_adjust` / `get_stock_list` / `get_stock_data` | 可按需组合 |
| `stock_prediction.data_preprocess` | 批量预处理 `preprocess_data()` | 生成新的序列化队列 |
| `stock_prediction.target` | 技术指标函数库 | 可直接用于特征工程 |

> 规划中的演进：拆分 `common.py`，引入 `.env` + `pydantic` 配置体系；提供 `main(argv=None)`、`run_train(cfg)` 等程序化接口；探索服务化部署。

## 3. 多模型横向测试
1. 确认 `stock_daily/`、`pkl_handle/train.pkl` 等数据准备完毕。
2. 执行批处理脚本：
   ```bat
   scripts\run_all_models.bat
   ```
3. 观察 `output/`、`png/` 目录中的日志与图表。

常用参数说明：
- `--model`：`lstm` / `attention_lstm` / `bilstm` / `tcn` / `multibranch` / `transformer` / `cnnlstm` / `hybrid` / `ptft_vssm`
- `--mode`：`train` / `test` / `predict`
- `--test_code`：测试用股票代码
- `--pkl`：是否使用 `train.pkl`（1 表示使用）
- `--epoch`：训练轮数

建议流程：
1. 以较小 `epoch` 跑通全部模型，确认配置正确。
2. 根据需求调整 `epoch`、`batch_size`、特征组合开展正式对比。
3. 结合业务目标（趋势、价格、波动率等）选择合适结构。
4. 新增模型请在 `src/stock_prediction/models/` 中实现，并在 `predict.py` 注册，即可纳入脚本。

## 4. 运维与环境
### 4.1 基础环境
- 推荐使用 `conda activate stock_prediction`，Python ≥ 3.10。
- 安装依赖：`pip install -r requirements.txt`；如需 CUDA，请额外安装匹配版本的 `torch` / `torchvision`。
- 外部数据源按需安装 `tushare`、`akshare`、`yfinance` 并准备 API Token。

### 4.2 常用操作
| 场景 | 命令 | 说明 |
| ---- | ---- | ---- |
| 初始化环境 | `conda activate stock_prediction`<br>`pip install -r requirements.txt` | 首次或依赖更新时执行 |
| 拉取行情 | `python scripts/getdata.py --api akshare --code 000001.SZ` | 结果写入 `stock_daily/` |
| 聚合数据 | `python scripts/data_preprocess.py --pklname train.pkl` | 生成 `pkl_handle/train.pkl` |
| 训练/预测 | `python scripts/predict.py --mode train --model ptft_vssm` | CLI 支持所有模型模式 |
| 运行测试 | `pytest -q` | 默认使用 `DefaultArgs`，不会触发 `SystemExit` |

### 4.3 目录约定
- 行情数据：`stock_daily/`
- 预处理结果：`pkl_handle/train.pkl`
- 模型权重：`models/<symbol>/<MODEL_TYPE>/`
- 可视化图：`png/`
- 日志：`data/log/`（建议升级为结构化日志）

### 4.4 注意事项
1. **队列序列化**：`train.pkl` 对 Python 版本敏感，升级环境后建议重新生成或使用 `ensure_queue_compatibility()`。
2. **外部参数**：CLI 已解耦，测试环境可直接导入；脚本中建议使用 `create_predictor()`。
3. **API 限流**：`getdata.py` 尚未内建限速和重试，批量下载时需关注速率与异常处理。
4. **设备切换**：`--cpu 1` 会自动降级 AMP；如需完全关闭可在配置中禁用 `GradScaler`。

---
如需更详尽的背景与下一步规划，请参考 `docs/system_design.md` 与 `docs/model_strategy.md`。***
