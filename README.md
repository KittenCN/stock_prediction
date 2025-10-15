# stock_prediction

基于 PyTorch 的股票预测研究型项目，支持从多数据源抓取日线行情、构建序列数据集，并在统一入口下训练多种神经网络（含最新的多尺度混合模型 `hybrid`）。仓库对标研究原型，强调可复现与可维护性。

## 功能清单
- 多数据源行情抓取：内置 `tushare`、`akshare`、`yfinance` 三类接口
- 队列化预处理：将日线 CSV 聚合为 `pkl_handle/train.pkl`，便于批量训练
- 模型管理：集成 LSTM、Transformer、CNNLSTM、TemporalHybridNet 等结构
- 技术指标库：`target.py` 提供主流技术指标计算，可按需扩展特征
- 绘图与持久化：自动输出损失曲线、预测对比图及模型权重

## 快速开始（3 条命令）
```bash
conda activate stock_prediction
pip install -r requirements.txt
python scripts/predict.py --mode train --model hybrid --epoch 2 --pkl 1 --predict_days 3
```
> 若尚未准备数据，可先运行 `python scripts/getdata.py --api akshare --code 000001.SZ` 与 `python scripts/data_preprocess.py --pklname train.pkl`。

## 目录结构
```
project-root/
├─ src/stock_prediction/
│  ├─ config.py              # 路径与目录管理
│  ├─ init.py                # 全局超参数与共享队列
│  ├─ common.py              # 数据集定义、通用工具
│  ├─ models/                # 模型集合（含 TemporalHybridNet）
│  ├─ predict.py             # 训练 / 测试 / 预测主入口
│  ├─ getdata.py             # 行情抓取脚本
│  ├─ data_preprocess.py     # 预处理与序列化
│  └─ target.py              # 技术指标函数库
├─ scripts/                  # 命令行封装
├─ tests/                    # PyTest 用例
├─ docs/                     # 架构 / API / 运维 / 决策等文档
├─ models/                   # 训练后的模型权重
├─ stock_daily/              # 原始行情数据
├─ pkl_handle/               # 预处理后的序列化数据
├─ Makefile                  # 常用命令集合
├─ requirements.txt          # 依赖清单
├─ ASSUMPTIONS.md            # 假设与约束
├─ CHANGELOG.md              # 变更记录
└─ agent_report.md           # 自动执行报告
```

## 核心流程
1. `scripts/getdata.py`：抓取指定股票的日线行情，写入 `stock_daily/`
2. `scripts/data_preprocess.py`：加载 CSV、补充技术指标、序列化至 `pkl_handle/train.pkl`
3. `scripts/predict.py`：构建数据集、选择模型（含 `hybrid`）、执行训练或预测
4. `common.py`：提供数据加载、特征加工、模型保存、可视化等辅助能力

## 模型能力一览
| 参数取值 (`--model`) | 结构概述 | 适用场景 |
| ------------------- | -------- | -------- |
| `lstm` | 3 层 LSTM + 全连接 | 基线模型，快速验证 |
| `attention_lstm` | LSTM + 简单注意力 | 突出关键时间片段 |
| `bilstm` | 双向 LSTM | 强化上下文信息 |
| `tcn` | 时序卷积网络 | 捕捉局部模式 |
| `multibranch` | 价格 / 指标双分支 LSTM | 区分特征族处理 |
| `transformer` | 自定义 Transformer 编码器/解码器 | 长序列建模 |
| `cnnlstm` | 卷积 + LSTM + Attention | 多步预测（需 `predict_days` > 0） |
| `hybrid` | 新增 TemporalHybridNet：多尺度卷积 + Bi-GRU + Multi-Head Attention + 统计特征融合 | 多尺度回归、长/短期混合预测 |

批量比较可运行 `scripts/run_all_models.bat`，该脚本已加入 `hybrid` 模型。

## 配置说明
- **推荐环境**：在 `conda activate stock_prediction` 环境中执行，确保依赖一致
- **路径管理**：所有路径由 `stock_prediction.config.Config` 自动创建，可通过 `config.get_model_path()` 获取保存目录
- **参数调整**：通过命令行参数调整训练流程，如 `--mode`, `--model`, `--predict_days`, `--epoch`
- **模块导入**：✅ 已修复导入问题，现在可以安全地 `from stock_prediction.predict import main, create_predictor`
- **测试支持**：`create_predictor()` 函数可用于单元测试和外部调用

## 常见问题
| 问题 | 原因 | 解决方式 |
| ---- | ---- | -------- |
| `AttributeError: 'Queue' object has no attribute 'is_shutdown'` | Python 3.13+ 反序列化旧 `queue.Queue` | 已在 `common.ensure_queue_compatibility` 兜底，必要时重新生成 `train.pkl` |
| `ModuleNotFoundError: No module named 'stock_prediction'` | 测试环境路径问题 | ✅ 已修复，测试文件已添加动态路径设置 |
| `SystemExit: 2` 测试失败 | 模块导入时解析命令行参数 | ✅ 已修复，现在使用 `DefaultArgs` 类提供默认配置 |
| `--cpu 1` 仍报错 | `GradScaler` 仅支持 CUDA | 暂未在 CPU 分支禁用混合精度，训练 CPU 模式时请关闭 AMP |

## 测试与质量
### 运行测试
```bash
# 安装测试依赖
pip install pytest pytest-cov

# 运行所有测试
python -m pytest tests\ -v

# 运行测试并生成覆盖率报告
python -m pytest tests\ --cov=src --cov-report=term --cov-report=xml -v

# 运行特定测试文件
python -m pytest tests\test_models.py -v
```

### 测试状态
- ✅ **23 项测试全部通过**（100% 通过率）
- 📊 **覆盖率**：20%（目标：≥80%）
- 🎯 **测试套件**：
  - 配置测试：4 项
  - 数据处理测试：5 项
  - 导入测试：2 项
  - 技术指标测试：4 项
  - 模型测试：2 项
  - 集成测试：6 项

### 质量保证
- 新增 `tests/test_models.py`，覆盖 `TemporalHybridNet` 单步与多步输出
- 建议执行 `pytest`、`ruff`、`black`、`mypy`（严格模式）后再提交
- `scripts/run_all_models.bat` 可用于手动回归多个模型
- 所有模块导入问题已修复，支持测试驱动开发(TDD)

## 文档索引
- `docs/system_design.md`：架构拓扑、系统分析与核心设计决策
- `docs/user_guide.md`：命令行/模块用法、运维要点与多模型测试流程
- `docs/maintenance.md`：结构清理、关键修复与自动化执行摘要
- `ASSUMPTIONS.md`：默认假设与约束条件
- `CHANGELOG.md`：遵循 SemVer 的变更记录

欢迎在开发过程中同步更新 `ASSUMPTIONS.md`、`CHANGELOG.md` 与 `agent_report.md`，保持项目可追溯性。
