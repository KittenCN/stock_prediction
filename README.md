# stock_prediction

基于 PyTorch 的股票价格预测实验项目，提供从数据采集、预处理到模型训练、评估的完整流程。当前重点验证多尺度混合模型（TemporalHybridNet）以及 PTFT + Variational SSM 双轨组合等先进结构，强调可复现性与可维护性。

## 功能总览
- **行情采集**：封装 `tushare` / `akshare` / `yfinance` 三类接口。
- **数据预处理**：将日线 CSV 聚合为 `pkl_handle/train.pkl` 队列，可重复加载。
- **模型训练**：统一 CLI 入口，支持 LSTM、Transformer、TemporalHybridNet、PTFT_VSSM 等模型。
- **技术指标库**：`target.py` 内置常用指标（MACD、KDJ、BOLL、DMI、ATR …）。
- **可视化/持久化**：自动生成损失曲线与预测对比图，并按模型类型落盘权重。

## 快速开始
```bash
conda activate stock_prediction
pip install -r requirements.txt
python scripts/predict.py --mode train --model ptft_vssm --pkl 1 --epoch 2
```
> 初次运行请先执行 `python scripts/getdata.py --api akshare --code 000001.SZ` 与 `python scripts/data_preprocess.py --pklname train.pkl` 准备数据。

## 目录结构
```
project-root/
├─ src/stock_prediction/
│  ├─ config.py              # 路径与目录管理
│  ├─ init.py                # 超参数、设备与共享队列
│  ├─ common.py              # 数据集、可视化与模型保存工具
│  ├─ models/                # 模型实现（LSTM/Transformer/TemporalHybridNet/PTFT_VSSM 等）
│  ├─ predict.py             # CLI 主流程与训练循环
│  ├─ getdata.py             # 行情采集脚本
│  ├─ data_preprocess.py     # 预处理与序列化
│  └─ target.py              # 技术指标函数库
├─ scripts/                  # 命令行封装脚本
├─ tests/                    # PyTest 用例
├─ docs/                     # 架构、策略、操作文档
├─ models/                   # 训练后的模型权重
├─ stock_daily/              # 原始行情数据
├─ pkl_handle/               # 预处理后的队列文件
├─ Makefile                  # 常用任务入口
├─ requirements.txt          # 依赖清单
├─ ASSUMPTIONS.md            # 假设与约束
├─ CHANGELOG.md              # 变更记录
└─ agent_report.md           # 自动执行摘要
```

## 支持模型
| `--model` 参数 | 结构概述 | 适用场景 |
| -------------- | -------- | -------- |
| `lstm` | 3 层 LSTM + 全连接 | 基线模型，快速验证 |
| `attention_lstm` | LSTM + 注意力 | 强调关键时间片段 |
| `bilstm` | 双向 LSTM | 强化上下文感知 |
| `tcn` | 时序卷积网络 | 捕捉局部模式 |
| `multibranch` | 价格 / 指标双分支 LSTM | 面向多特征族的融合 |
| `transformer` | 自定义 Transformer 编解码结构 | 长序列建模 |
| `cnnlstm` | CNN + LSTM + Attention | 多步预测（需 `predict_days` > 0） |
| `hybrid` | TemporalHybridNet：多尺度卷积 + Bi-GRU + 多头注意力 + 统计特征 | 多尺度回归、长/短期混合 |
| `ptft_vssm` | PTFT + Variational SSM 双轨组合，输出分位预测与状态概率 | 需要概率输出、风险评估的场景 |

批量对比可执行 `scripts\run_all_models.bat`，脚本已覆盖所有模型。

## 常用命令
```bash
# 抓取行情
python scripts/getdata.py --api akshare --code 000001.SZ

# 数据预处理
python scripts/data_preprocess.py --pklname train.pkl

# 训练（示例：PTFT + VSSM）
python scripts/predict.py --mode train --model ptft_vssm --epoch 2 --pkl 1 --predict_days 3

# 运行测试
pytest -q
```

## 测试与质量
- `pytest`：26 项测试全部通过，覆盖 TemporalHybridNet、PTFT、VSSM 及数据流程。
- 推荐提交前运行 `pytest -q`，必要时配合 `ruff` / `black` / `mypy`。
- 暴露 `create_predictor()`，便于单元测试或外部脚本直接构造预测器。

## 文档索引
- `docs/system_design.md`：架构拓扑、系统分析与关键决策。
- `docs/model_strategy.md`：零基模型策略设计与推荐组合。
- `docs/user_guide.md`：命令行/模块用法、运维要点、多模型测试流程。
- `docs/maintenance.md`：结构清理、关键修复与自动化执行记录。
- `ASSUMPTIONS.md`：默认假设与约束条件。
- `CHANGELOG.md`：按 SemVer 记录项目变更。

## 常见问题
| 问题 | 原因 | 解决方案 |
| ---- | ---- | -------- |
| `queue.Queue` 反序列化报错 | Python 3.13+ 属性变更 | `common.ensure_queue_compatibility` 已兜底，必要时重新生成 `train.pkl` |
| 测试时出现 `SystemExit: 2` | CLI 在导入时解析参数 | 已改为默认参数对象，可直接导入；测试无需特殊处理 |
| CPU 模式提示 AMP | GradScaler 默认面向 CUDA | 已自动降级为 CPU 路径，可忽略警告或关闭 AMP |
| 模型保存阻塞 | `thread_save_model` 深拷贝失败 | 现已改为保存 state_dict 并迁移到 CPU |

## 贡献指引
1. 新增模型请在 `src/stock_prediction/models/` 中实现，并在 `predict.py` 注册入口。
2. 同步更新测试（`tests/test_models.py`）与文档（尤其是 `docs/model_strategy.md`、`CHANGELOG.md`）。
3. 遵循编码规范（PEP8 + 类型标注），提交前执行 `pytest`。

---
更多背景与未来规划，详见 `docs/model_strategy.md` 与 `docs/system_design.md`。欢迎贡献。 
