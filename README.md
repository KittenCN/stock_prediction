# stock_prediction

基于 PyTorch 的股票预测原型项目，支持从多数据源抓取日线行情、构建序列数据集并训练 LSTM/Transformer/CNN-LSTM 等模型。代码目前以研究性质为主，尚未完全产品化。

## 功能清单

- 多数据源行情抓取：`tushare` / `akshare` / `yfinance`。
- 数据预处理：将日线 CSV 聚合到 `pkl_handle/train.pkl`（队列序列化）。
- 模型训练与推理：在单机环境下训练并生成预测结果、损失曲线图。
- 技术指标库：内置常见技术指标函数，可用于扩展特征。

## 快速开始（≤3 条命令）

```bash
conda activate stock_prediction
pip install -r requirements.txt
python scripts/getdata.py --api akshare --code 000001.SZ
```

随后可执行 `python scripts/data_preprocess.py --pklname train.pkl` 生成训练数据，再调用 `python scripts/predict.py --mode train --model transformer` 启动训练（需要先修复入口函数，详见文档）。

## 目录结构

```
project-root/
├─ src/stock_prediction/        # 核心包
│  ├─ config.py                 # 目录配置
│  ├─ init.py                   # 全局常量与设备/队列
│  ├─ common.py                 # 数据集、模型、训练工具
│  ├─ predict.py                # 训练/测试/预测主流程
│  ├─ getdata.py                # 行情抓取
│  ├─ data_preprocess.py        # 预处理
│  ├─ target.py                 # 技术指标
│  └─ utils.py                  # 日志与工具
├─ scripts/                     # 命令行入口（依赖 src）
├─ tests/                       # 现有测试用例（需修复导入副作用）
├─ docs/                        # 文档（架构、API、运维、分析等）
├─ stock_daily/                 # 行情 CSV 输出
├─ pkl_handle/                  # 预处理结果（`train.pkl`）
├─ models/                      # 模型保存目录
├─ Makefile                     # 常用命令集合
├─ requirements.txt             # 依赖清单
├─ ASSUMPTIONS.md               # 假设记录
├─ CHANGELOG.md                 # 变更记录
└─ agent_report.md              # 自动执行报告
```

## 核心流程概览

1. `scripts/getdata.py`：按股票代码下载行情至 `stock_daily/`。
2. `scripts/data_preprocess.py`：读取 CSV、填充缺失值、写入 `queue.Queue` 并序列化到 `pkl_handle/train.pkl`。
3. `scripts/predict.py`：解析命令行参数，加载 `train.pkl` 队列，构造 `torch.utils.data.Dataset`，执行训练/测试/预测。
4. `common.py`：提供数据集类、模型定义、优化器/调度器初始化、评估绘图、技术指标等工具函数。

详细设计、风险与改进建议请参考 `docs/project_analysis.md` 与 `docs/architecture.md`。

## 配置说明

- 默认路径通过 `stock_prediction.config.Config` 管理，首次运行会自动创建 `stock_daily/`、`pkl_handle/`、`models/` 等目录。
- 目前尚未集成 `.env`/YAML 配置，若需定制参数可直接修改 `init.py` 或在命令行中传入（如 `--mode`、`--model`、`--predict_days`）。
- 建议在 `conda activate stock_prediction` 环境中运行，以保持依赖一致性。

## 常见问题

| 问题 | 原因/定位 | 解决方式 |
| ---- | -------- | -------- |
| `AttributeError: 'Queue' object has no attribute 'is_shutdown'` | Python 3.13+ 在反序列化旧版 `queue.Queue` 时缺少新属性 | 已在 `common.py` 添加 `ensure_queue_compatibility()`，若旧 pickle 仍出错，请重新执行数据预处理 |
| `SystemExit: unrecognized arguments: -q` 在 `pytest` 中出现 | `src/stock_prediction/predict.py` 导入就解析命令行参数 | 将参数解析包装进 `main(argv=None)` 是后续计划，目前避免在带额外参数的进程中直接导入 |
| `scripts/predict.py` 找不到 `main` | 脚本引用 `stock_prediction.predict.main`，但当前模块未导出 | 需补充 `main()` 封装函数或调整脚本调用方式 |
| CPU 环境使用 `--cpu 1` 仍报错 | `GradScaler('cuda')` 在 CPU 环境初始化 | 待修复：根据设备类型条件化构建 AMP |

更多运维注意事项见 `docs/ops.md`。

## 测试与质量

- 现有测试位于 `tests/` 目录，但在修复 CLI 副作用前运行 `pytest` 会失败。
- Makefile 中的 `test`/`ci` 命令依赖 `python scripts/predict.py`，建议在修复入口前暂时直接调用包内函数或手动执行测试。
- 推荐扩展 `ruff`、`black`、`mypy` 等静态检查工具，并在 CI 中执行。

## 后续路线

- **短期**：修复命令行入口、拆分全局状态、替换队列序列化格式。
- **中期**：引入配置化训练流程、完善日志与监控、补齐 CI。
- **长期**：模块化模型管理、提供 REST/gRPC 服务接口、整合情绪分析等多模态信号。

## 文档索引

- `docs/project_analysis.md`：系统分析与风险评估
- `docs/architecture.md`：架构与模块说明
- `docs/api.md`：命令行与包接口指南
- `docs/ops.md`：运维与运行须知
- `docs/cleanup_log.md`：历史清理记录

欢迎在使用过程中记录新的假设与问题，更新 `ASSUMPTIONS.md`、`CHANGELOG.md` 与 `agent_report.md`。
