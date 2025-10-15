# 使用与运维指南

## 1. 命令行快捷表
| 命令 | 作用 | 备注 |
| ---- | ---- | ---- |
| `python scripts/getdata.py --api akshare --code 000001.SZ` | 抓取指定股票日线数据，写入 `stock_daily/` | 运行前请执行 `conda activate stock_prediction` 并安装对应数据源依赖 |
| `python scripts/data_preprocess.py --pklname train.pkl` | 聚合 `stock_daily/` 下的 CSV，生成 `pkl_handle/train.pkl` | 依赖 `stock_prediction.init` 中的队列与 `dill` |
| `python scripts/predict.py --mode train --model hybrid --predict_days 3` | 启动 TemporalHybridNet 训练流程 | 其他模型：`lstm`、`attention_lstm`、`bilstm`、`tcn`、`multibranch`、`transformer`、`cnnlstm` |
| `pytest -q` | 运行测试套件 | 当前需先解决 `predict.py` 导入副作用，否则会解析外部参数 |

> 提示：在 CLI 重构完成前，`src/stock_prediction/predict.py` 仍会在导入时解析 `sys.argv`。如需在带额外参数的环境（例如 `pytest -k` 或 `ipython`）中导入，请优先调用 `python -m stock_prediction.predict --help` 清空参数，或等待后续改造。  

## 2. 包内复用接口
| 模块 | 能力 | 使用方式 |
| ---- | ---- | ---- |
| `stock_prediction.config` | `Config` 类、常用路径 | `from stock_prediction.config import config, train_pkl_path` |
| `stock_prediction.common` | 数据集类（`Stock_Data`、`stock_queue_dataset`）、训练辅助函数、绘图工具 | 导入前需确保 `init.py` 已加载全局状态 |
| `stock_prediction.models` | 各类模型（LSTM、Transformer、CNNLSTM、TemporalHybridNet 等） | `from stock_prediction.models import TemporalHybridNet` |
| `stock_prediction.getdata` | 行情抓取函数（`set_adjust`、`get_stock_list`、`get_stock_data`） | 可组合自定义采集流程 |
| `stock_prediction.data_preprocess` | 批量预处理接口 `preprocess_data()` | 生成新的数据队列并序列化到 `pkl_handle/` |
| `stock_prediction.target` | 技术指标纯函数（`MA`、`MACD`、`RSI`、`BOLL` 等） | 不依赖全局状态，可直接调用 |

规划中的演进：  
1. 将 CLI 与程序接口彻底解耦，提供 `main(argv=None)`、`run_train(cfg)` 等函数式入口。  
2. 按职责拆分 `common.py`，将数据、模型、可视化逻辑独立。  
3. 引入 `.env` + `pydantic`/`dynaconf` 做参数校验。  
4. 改造服务化接口，为 REST/gRPC 做准备。

## 3. 多模型横向测试
1. 确认数据已准备完毕（`stock_daily/`、`pkl_handle/train.pkl` 等）。  
2. 执行批处理脚本：
   ```bat
   scripts\run_all_models.bat
   ```
3. 观察日志与输出文件。默认情况下：  
   - 日志、预测曲线输出在 `output/` 及 `png/`。  
   - 各模型采用相同的 `--test_code`、`--epoch`。  

常用参数说明：  
- `--model`：`lstm` / `attention_lstm` / `bilstm` / `tcn` / `multibranch` / `transformer` / `cnnlstm` / `hybrid`。  
- `--mode`：`train` / `test` / `predict`。  
- `--test_code`：待测试的股票代码。  
- `--pkl`：是否使用 `train.pkl`（1 为使用）。  
- `--epoch`：训练轮数。

建议流程：  
1. 先用较小 `epoch` 跑通全部模型，确认无配置错误。  
2. 根据需要调节 `epoch`、`batch_size`、特征集进行正式对比。  
3. 结合业务目标（趋势、价格、波动率等）选择适配结构。  
4. 扩展新模型时，在 `src/stock_prediction/models/` 新增实现，并于 `predict.py` 中注册，即可纳入脚本。

## 4. 运维与环境
### 4.1 基础环境
- 推荐环境：`conda activate stock_prediction`，Python ≥ 3.10。  
- 依赖安装：`pip install -r requirements.txt`。如需 CUDA，请额外安装匹配版本的 `torch`/`torchvision`。  
- 外部数据源：按需安装 `tushare`、`akshare`、`yfinance`，准备 API Token（若必需）。

### 4.2 常用操作
| 场景 | 命令 | 说明 |
| ---- | ---- | ---- |
| 初始化环境 | `conda activate stock_prediction`<br>`pip install -r requirements.txt` | 首次或依赖更新时执行 |
| 抓取行情 | `python scripts/getdata.py --api akshare --code 000001.SZ` | 结果写入 `stock_daily/` |
| 聚合数据 | `python scripts/data_preprocess.py --pklname train.pkl` | 生成 `pkl_handle/train.pkl` |
| 训练或预测 | `python scripts/predict.py --mode train --model transformer` | CLI 需先完成参数解耦改造 |
| 测试 | `pytest -q` | 需规避 `predict.py` 导入副作用 |

### 4.3 目录约定
- 行情原始数据：`stock_daily/`  
- 预处理缓存：`pkl_handle/train.pkl`（内部为 `queue.Queue` 序列化）  
- 模型权重：`models/<symbol>/<MODEL_TYPE>/`  
- 可视化结果：`png/`  
- 运行日志：`data/log/`（需补充结构化格式）

### 4.4 运维注意事项
1. **队列序列化**：`train.pkl` 对 Python 版本敏感，升级环境后建议重新预处理或使用 `ensure_queue_compatibility()`。  
2. **命令行冲突**：在修复 CLI 之前，避免在携带额外参数的进程中直接 `import stock_prediction.predict`。  
3. **API 限流**：`getdata.py` 尚未内建重试/限速，批量下载时需关注速率。  
4. **设备切换**：`--cpu 1` 仅部分生效，后续需完善 AMP/CPU 判定流程。

### 4.5 推荐改进
- 引入结构化日志与监控指标，记录数据抓取、训练耗时与失败率。  
- 将关键脚本纳入 CI，保障依赖与接口变更可见。  
- 使用 `.env` 或 YAML 配置管理 API Token、数据路径等敏感信息，避免直接修改源码。

---
本指南整合了原有的 API、运维与多模型测试文档，按“使用 → 维护”顺序描述项目操作流程。*** End Patch
