# 接口与命令行指南

## 命令行入口

| 命令 | 作用 | 备注 |
| ---- | ---- | ---- |
| `python scripts/getdata.py --api akshare --code 000001.SZ` | 拉取指定股票的日线数据并输出到 `stock_daily/` | 运行前需 `conda activate stock_prediction`，保证安装了对应数据源依赖 |
| `python scripts/data_preprocess.py --pklname train.pkl` | 将 `stock_daily/` 下的 CSV 数据聚合并序列化为 `pkl_handle/train.pkl` | 依赖 `stock_prediction.init.data_queue`；输出为 `queue.Queue` pickle |
| `python scripts/predict.py --mode train --model transformer` | 启动训练/测试/预测主流程 | 需先修复 `stock_prediction.predict` 的 `main()` 接口才能正常运行 |

> 提醒：当前 `src/stock_prediction/predict.py` 在导入阶段会立即解析命令行参数，若进程存在额外参数（如 `pytest -q`），导入会触发 `SystemExit`。重构时应将解析逻辑置于 `main(argv=None)` 函数内。

## 包内可复用能力

| 模块 | 能力 | 使用方式 |
| ---- | ---- | ---- |
| `stock_prediction.config` | `Config` 类及常用路径（`train_pkl_path`, `daily_path` 等） | `from stock_prediction.config import config, train_pkl_path` |
| `stock_prediction.common` | 数据集（`Stock_Data`, `stock_queue_dataset`）、模型定义（LSTM/Transformer/CNNLSTM）、训练辅助函数、绘图工具 | 多数函数依赖 `init.py` 中的全局状态，导入顺序需保持一致 |
| `stock_prediction.getdata` | `set_adjust()`, `get_stock_list()`, `get_stock_data()` 等行情抓取工具 | 可单独调用，实现自定义采集流程 |
| `stock_prediction.data_preprocess` | `preprocess_data(pkl_name="train.pkl")` | 生成新的数据队列并写入 `pkl_handle/` |
| `stock_prediction.target` | 大量技术指标纯函数（`MA`, `MACD`, `RSI`, `BOLL` 等） | 不依赖全局状态，可在特征工程中直接使用 |

## 规划中的 API 演进

1. **CLI 与程序接口解耦**：为 `predict.py` 提供 `main(argv=None)`、`run_train(config)`、`run_predict(config)` 等显式函数。
2. **模块化拆分**：将 `common.py` 拆分为 `data`, `models`, `metrics`, `visualization` 等子模块，减少导入耦合。
3. **配置校验**：结合 `.env` 与 `pydantic`/`dynaconf` 对配置进行类型与范围校验。
4. **服务化准备**：预留 REST/gRPC 层所需的程序化接口，支撑后续自动化部署。
