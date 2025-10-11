# 运维与运行指南

## 环境与依赖

- **推荐环境**：`conda activate stock_prediction`，Python ≥ 3.10。
- **依赖安装**：`pip install -r requirements.txt`（建议在 conda 环境中执行）。
- **GPU 支持**：如需 CUDA，需要额外安装匹配版本的 `torch`/`torchvision`。
- **外部数据源**：按需安装 `tushare`、`akshare`、`yfinance`，并准备对应的 API Token（若适用）。

## 常用命令

| 场景 | 命令 | 说明 |
| ---- | ---- | ---- |
| 初始化环境 | `conda activate stock_prediction`<br>`pip install -r requirements.txt` | 仅首次/更新依赖时执行 |
| 拉取行情数据 | `python scripts/getdata.py --api akshare --code 000001.SZ` | 写入 `stock_daily/000001.SZ.csv` |
| 聚合训练数据 | `python scripts/data_preprocess.py --pklname train.pkl` | 生成 `pkl_handle/train.pkl` |
| 启动训练/预测 | `python scripts/predict.py --mode train --model transformer` | 当前脚本需在修复 CLI 接口后使用 |
| 运行测试 | `pytest -q` | 目前会因 `predict.py` 导入副作用失败，需先重构 |

## 日志与数据定位

- **行情原始数据**：`stock_daily/`
- **预处理缓存**：`pkl_handle/train.pkl`（内部为 `queue.Queue` 序列化，对 Python 版本敏感）
- **模型输出**：`models/<symbol>/<MODEL_TYPE>/`
- **可视化结果**：`png/`
- **运行日志**：`data/log/`（由 `utils.py` 写入，格式未统一）

## 运维注意事项

1. **队列序列化风险**：`train.pkl` 受 Python 版本影响较大，升级后建议重新执行数据预处理或引入兼容函数 `ensure_queue_compatibility()`。
2. **命令行参数冲突**：在修复 `predict.py` 之前，避免在带额外参数的进程（如 `pytest -q`、`ipython`）中直接 `import stock_prediction.predict`。
3. **API 限流**：`getdata.py` 尚未内建重试/限速策略，批量下载时需关注数据源限制。
4. **GPU/CPU 切换**：`--cpu 1` 仅部分生效，`GradScaler('cuda')` 仍会在 CPU 环境初始化，务必确认设备配置。

## 建议的运维迭代

- 引入结构化日志与监控指标，至少记录数据抓取/训练耗时与失败率。
- 将关键路径脚本接入 CI，保障依赖与接口变更可见。
- 通过 `.env` 或 YAML 配置开放 API Token、数据路径等运维项，避免修改源码。
