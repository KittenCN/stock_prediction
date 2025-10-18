# 使用与运维指南

## 1. 命令行速查
| 命令 | 作用 | 备注 |
| ---- | ---- | -------- |
| `python scripts/getdata.py --api akshare --code 000001.SZ` | 抓取指定股票日线数据写入 `stock_daily/` | 运行前请激活 `conda activate stock_prediction` 并确保安装数据源依赖 |
| `python scripts/data_preprocess.py --pklname train.pkl` | 将 `stock_daily/` 下的 CSV 聚合为 `pkl_handle/train.pkl` | 依赖 `stock_prediction.init` 中的共享队列与 `dill` |
| `python scripts/train.py --mode train --model ptft_vssm --predict_days 3` | 训练 PTFT + VSSM 双轨模型 | 其他模型：`lstm`、`attention_lstm`、`bilstm`、`tcn`、`multibranch`、`transformer`、`cnnlstm`、`hybrid` |
| `python scripts/predict.py --model ptft_vssm --test_code 000001` | 推理指定股票的未来走势 | 支持设置 `--predict_days` 输出多日区间 |
| `pytest -q` | 运行全部测试 | CLI 已解耦导入副作用，可直接执行 |
| `cat output/metrics_*.json` | 查看训练/测试后的评估指标 | 包含 RMSE、MAPE、VaR、CVaR 等 |

## 2. 包内复用能力
| 模块 | 能力 | 使用方式 |
| ---- | ---- | ---- |
| `stock_prediction.config` | `Config` 类及常用路径 | `from stock_prediction.config import config, train_pkl_path` |
| `stock_prediction.common` | 数据集、可视化、模型保存工具 | 导入前确保 `init.py` 已初始化全局状态 |
| `stock_prediction.feature_engineering` | 收益率/差分特征、外生变量融合、滑动窗口聚合 | `FeatureEngineer(AppConfig.from_env_and_yaml(...).features)` |
| `stock_prediction.models` | 模型集合（Hybrid Aggregator、ProbTFT、VSSM、Diffusion、Graph 等） | `from stock_prediction.models import TemporalHybridNet` |
| `stock_prediction.getdata` | 行情采集函数 `set_adjust` / `get_stock_list` / `get_stock_data` | 可按需组合 |
| `stock_prediction.data_preprocess` | 批量预处理接口 `preprocess_data()` | 生成新的序列化队列 |
| `stock_prediction.target` | 技术指标函数库 | 可直接用于特征工程 |

> 后续计划：拆分 `common.py`（数据 / 模型 / 可视化）、引入 `.env` + `pydantic` 配置体系，并探索服务化部署。

## 3. 多模型横向测试
1. 确认 `stock_daily/`、`pkl_handle/train.pkl` 等数据已准备完毕。
2. 执行批处理脚本：
   ```bat
   scripts
un_all_models.bat
   ```
3. 观察 `output/`、`png/` 目录中的日志与图表。

常用参数：
- `--model`：`lstm` / `attention_lstm` / `bilstm` / `tcn` / `multibranch` / `transformer` / `cnnlstm` / `hybrid` / `ptft_vssm`
- `--mode`：`train` / `test`
- `--test_code`：测试股票代码
- `--pkl`：是否使用 `train.pkl`（1 表示使用）
- `--epoch`：训练轮数

建议流程：先用较小 `epoch` 跑通所有模型，确认配置无误；再根据需求调整 `epoch`、`batch_size` 与特征集合开展正式对比；结合业务目标（趋势、价格、波动率等）选择最合适的结构；新增模型时，在 `src/stock_prediction/models/` 实现并分别在 `train.py` 与 `predict.py` 注册即可。

## 4. 运维与环境
### 4.1 环境说明
- 推荐环境：`conda activate stock_prediction`，Python ≥ 3.10。
- 依赖安装：`pip install -r requirements.txt`；如需 CUDA，请额外安装匹配版本的 `torch`/`torchvision`。
- 外部数据源：按需安装 `tushare`、`akshare`、`yfinance`，并准备 API Token。

### 4.2 常用操作
| 场景 | 命令 | 说明 |
| ---- | ---- | ---- |
| 初始化环境 | `conda activate stock_prediction`<br>`pip install -r requirements.txt` | 首次或依赖更新时执行 |
| 拉取行情 | `python scripts/getdata.py --api akshare --code 000001.SZ` | 结果写入 `stock_daily/` |
| 聚合数据 | `python scripts/data_preprocess.py --pklname train.pkl` | 生成 `pkl_handle/train.pkl` |
| 训练/测试 | `python scripts/train.py --mode train --model transformer` | CLI 支持所有模型模式 |
| 推理 | `python scripts/predict.py --model transformer --test_code 000001` | 根据需要设置 `--predict_days` |
| 运行测试 | `pytest -q` | 默认使用 `DefaultArgs`，不会触发 `SystemExit` |

### 4.3 目录约定
- 行情数据：`stock_daily/`
- 预处理结果：`pkl_handle/train.pkl`
- 模型权重：`models/<symbol>/<MODEL_TYPE>/`
- 可视化图：`png/`
- 运行日志：`data/log/`（建议升级为结构化日志）

### 4.4 注意事项
1. **队列序列化**：`train.pkl` 对 Python 版本敏感，升级环境后建议重新生成或使用 `ensure_queue_compatibility()`。
2. **脚本导入**：CLI 已解耦，测试环境可直接导入；推荐在脚本中使用 `create_predictor()`。
3. **API 限流**：`getdata.py` 暂未内建限速和重试，批量下载时需关注速率。
4. **设备切换**：`--cpu 1` 会自动降级 AMP；如需完全关闭可在配置中禁用 `GradScaler`。
5. **外生特征文件**：默认读取 `config/external/*.csv`，保持 `trade_date` 为八位字符串（如 `20241203`），缺失值会按配置自动前向填充。需在 `config/config.yaml` 的 `features.external_sources` 指定白名单路径。

### 4.5 归一化参数：保存与加载（重要）
- 训练保存模型时会生成与权重同名的 `*_norm_params*.json`，包含 `mean_list/std_list/show_list/name_list`。
- 若使用 PKL 队列训练且运行态的均值/方差为空，系统会自动从 `pkl_handle/train.pkl` 计算后写入，无需手动脚本。
- 测试与推理在加载权重前，会优先读取对应的 `*_norm_params*.json` 并更新全局参数以保证反归一化正确。
- 历史权重若缺少或为空，可执行一次：
   ```bat
   python scripts\fix_norm_params.py
   ```
   新训练的模型不再需要该脚本。

---
如需更多背景与规划，请参考 `docs/system_design.md` 与 `docs/model_strategy.md`。
