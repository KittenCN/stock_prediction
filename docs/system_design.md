# 系统设计与架构总览

## 1. 架构拓扑
项目采用“命令行脚本 + 核心包”分层方式，整体数据流如下：

```mermaid
flowchart LR
    subgraph CLI
        G[getdata.py]
        P[data_preprocess.py]
        R[predict.py]
    end

    G -->|抓取 CSV| Daily[stock_daily/]
    P -->|序列化队列| PKL[pkl_handle/train.pkl]
    R -->|mode=train/test/predict| Core(predict.py 主流程)

    subgraph Package[src/stock_prediction]
        Core --> Common[common.py\n数据集与训练工具]
        Core --> Models[models/\n模型集合]
| models/ptft.py 等 | PTFT、V-SSM 与双轨融合实现 | 新增 ProbTemporalFusionTransformer / VariationalStateSpaceModel / PTFTVSSMEnsemble |
        Core --> Target[target.py\n指标库]
        Core --> Init[init.py\n全局常量与队列]
        Core --> Config[config.py\n路径配置]
        Common --> Target
        Models --> Common
    end

    Common --> Torch[PyTorch]
    G --> ExternalAPI[(tushare / akshare / yfinance)]
```

## 2. 系统分析
### 2.1 模块职责
| 模块 | 主要职责 | 备注 |
| ---- | -------- | ---- |
| `config.py` | 统一管理目录路径并自动创建输出目录 | 建议引入 `.env`/配置校验 |
| `init.py` | 训练参数、设备、共享 `queue.Queue` 等全局状态 | 全局变量较多，影响测试 |
| `common.py` | 数据集、模型工具、技术指标、绘图与模型保存 | 职责繁杂，应拆分子模块 |
| `models/` | 独立存放模型结构（LSTM、Transformer、TemporalHybridNet 等） | 与 `common.py` 逐步解耦 |
| `predict.py` | 训练 / 测试 / 预测主流程及 CLI 参数解析 | 顶层仍会解析命令行参数 |
| `getdata.py` | 通过 `tushare` / `akshare` / `yfinance` 抓取行情 | 缺少速率限制与健壮日志 |
| `data_preprocess.py` | 读取 CSV、填补指标并序列化至 `train.pkl` | 对 Python 版本高度敏感 |
| `target.py` | 技术指标函数库 | 可作为特征工程基础 |
| `utils.py` | 文件、日志等通用函数 | 日志尚未结构化 |

### 2.2 关键问题与建议
| 优先级 | 问题 | 建议方案 | 影响范围 |
| ------ | ---- | -------- | -------- |
| 高 | `predict.py` 模块导入即解析 `sys.argv`，阻断 `pytest`/脚本复用 | 将 `parse_args()` 收纳至 `main(argv=None)`，顶层只保留 `if __name__ == "__main__"` | 测试、自动化、服务化 |
| 高 | 命令行脚本期望 `predict.main`，但历史版本未导出 | 暴露 `main()` 或改写脚本调用方式 | CLI、Makefile |
| 高 | 训练流程大量依赖全局状态（模型/优化器/队列） | 拆分为 `Trainer`/`DataModule` 等对象，显式传参 | 可维护性、并行训练 |
| 中 | 数据缓存依赖 `queue.Queue` + `dill`，对 Python 版本脆弱 | 迁移至 Arrow/Parquet 等自描述格式或增加版本头 | 迁移成本、跨环境稳定性 |
| 中 | CPU 模式仍初始化 `GradScaler("cuda")` | 按设备类型初始化 AMP，CPU 直接禁用 | CPU 训练稳定性 |
| 中 | `common.py` 体积过大，混杂多种职责 | 按数据、模型、可视化等拆分子文件 | 代码阅读、测试 |
| 中 | 行情抓取缺少限速、重试与配置化 | 增加 retry/backoff、统一日志与 `.env` | 数据稳定性 |
| 低 | 文档与事实存在偏差 | 建立“变更即更新”流程（PR 模板、pre-commit） | 知识传递 |
| 低 | BERT/NLP 代码与主流程耦合 | 独立子包或延迟加载 | 依赖体积 |

### 2.3 后续路线
1. **基础重构**：修复 CLI 导入副作用，拆分全局状态，引入配置系统。  
2. **数据与训练**：替换队列序列化、封装训练器、加入 EarlyStopping/Checkpoint。  
3. **运维自动化**：补齐结构化日志、CI 流程与容器化环境。  
4. **功能扩展**：提供服务化接口，引入情绪/行业指数等多模态特征。

## 3. 核心设计决策
### 3.1 背景
- 目标：为股票预测流水线提供统一、可扩展的多尺度模型，实现单步与多步预测一致。  
- 现状：已有 LSTM/Attention-LSTM/BiLSTM/TCN/MultiBranch/Transformer/CNNLSTM，但缺乏统一的多尺度处理能力。

### 3.2 诊断结论
- 特征处理割裂：`add_target` 计算大量指标，却缺乏特征选择与加权策略。  
- 多尺度能力不足：现有模型只能覆盖单尺度时序信息。  
- 多步预测路径不一致：仅 CNNLSTM 支持 `predict_days`。  
- 注意力设计薄弱：缺少门控残差，不利于高噪声序列。

### 3.3 决策摘要
- 引入 `TemporalHybridNet`，结合多尺度卷积、双向 GRU、Multi-Head Attention 与门控残差。  
- 在模型内部融合窗口均值、标准差、末值等统计特征，弥补数据归一化不足。  
- 保持现有 CLI 习惯，新增 `--model hybrid` 选项。

### 3.4 行动清单
- [x] 建立设计记录并汇总诊断结论。  
- [x] 新增 `temporal_hybrid.py` 并接入模型入口。  
- [x] 更新 `predict.py`、`models/__init__.py`、`scripts/run_all_models.bat` 支持 `hybrid`。  
- [x] 编写 `tests/test_models.py` 覆盖单步与多步前向。  
- [x] 同步 README、CHANGELOG、ASSUMPTIONS 等文档。

### 3.5 风险与缓解
- `TemporalHybridNet` 结构复杂，易过拟合 → 默认启用 LayerNorm、Dropout，并通过单元测试把关形状。  
- 多步输出 reshape 容易出错 → 在模型内部统一处理，并用测试覆盖不同 `predict_days`。

## 4. 模型与优化策略
### 4.1 现有模型族
- **循环类**：LSTM、Attention-LSTM、BiLSTM（擅长短期依赖）。  
- **卷积类**：TCN、CNNLSTM（侧重局部模式）。  
- **注意力类**：Transformer（长序列建模）。  
- **多分支**：MultiBranch（区分价格/指标通道）。  
- **混合**：TemporalHybridNet（多尺度 + 注意力 + 统计特征）。

### 4.2 TemporalHybridNet 结构
1. **多尺度卷积分支**：使用不同核大小和 dilation 的深度可分离卷积提取多尺度特征。  
2. **双向 GRU**：整合前后序列信息。  
3. **Multi-Head Attention**：重点关注关键时间片段。  
4. **统计特征融合**：窗口均值、标准差、末值与序列编码拼接。  
5. **多步输出**：统一生成 `(batch, steps, output_dim)`，兼容单步预测。

### 4.3 演进建议
1. 引入 Variable Selection Network 等特征选择机制。  
2. 为混合模型增加多任务头（趋势、波动率等）。  
3. 补充 EarlyStopping、学习率调度、自动化超参搜索。  
4. 融合行业指数、新闻情绪等外部信号构建多模态分支。

---
以上内容吸收原有的架构图、系统分析、决策记录与模型优化文档，形成按“设计/实现/演进”分类的统一说明。
*** End Patch