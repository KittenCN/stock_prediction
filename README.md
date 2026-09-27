# stock_prediction

基于 PyTorch 的股票时间序列预测实验项目，包含行情获取、特征处理、训练、推理和模型评估。代码支持 LSTM、Transformer、Hybrid、PTFT/VSSM、Diffusion 和 Graph 等实验模型。预测输出仅供研究，不构成投资建议；模型效果需要在独立样本和真实交易成本下验证。

## 快速开始

推荐 Python 3.10，并使用独立的 `stock_prediction` conda 环境或等效虚拟环境。完整模型测试需要安装 PyTorch；CPU 环境也可运行单元测试。

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q
```

训练前先准备行情数据；示例命令会访问外部数据源：

```bash
python scripts/getdata.py --api akshare --code 000001.SZ
python scripts/data_preprocess.py --pklname train.pkl
python scripts/train.py --mode train --model transformer --epoch 2
python scripts/predict.py --model transformer --test_code 000001 --predict_days 3
```

数据接口可能因上游服务变化而失效。研究样例需要自行检查数据授权、复权方式和训练/测试时间边界。

## 功能与目录

| 路径 | 用途 |
| --- | --- |
| `src/stock_prediction/` | 数据处理、模型、训练与预测逻辑 |
| `scripts/` | 抓取、预处理、训练、推理命令入口 |
| `tests/` | 单元与集成测试 |
| `config/config.yaml` | 特征和路径等设置 |
| `docs/` | 使用、设计与维护说明 |

`--model` 支持 `lstm`、`attention_lstm`、`bilstm`、`tcn`、`multibranch`、`transformer`、`cnnlstm`、`hybrid`、`ptft_vssm`、`diffusion`、`graph`。部分模型对数据、配置或机器资源有额外要求，详见 [`docs/user_guide.md`](docs/user_guide.md) 与 [`docs/model_strategy.md`](docs/model_strategy.md)。

模型保存时会写入归一化参数文件 `*_norm_params*.json`；预测时应同时使用对应的模型权重和配置，以免特征维度或反归一化结果不匹配。

## 配置与验证

配置示例见 [`config/config.yaml`](config/config.yaml) 和 [`.env.example`](.env.example)。查看命令参数可运行 `python scripts/train.py --help` 或 `python scripts/predict.py --help`。

```bash
make test       # 本地运行测试
make ci         # 本地质量入口
```

GitHub Actions 会安装依赖并执行测试。CI 结果应与实际命令的退出状态一致；历史运行记录中的绿色状态不应被视为测试通过的证据，详见 [`docs/oss_maintenance_2026-09.md`](docs/oss_maintenance_2026-09.md)。

## 常见问题

| 现象 | 建议 |
| --- | --- |
| 导入模块时报依赖缺失 | 在项目环境内重新安装 `requirements.txt` 并确认正在使用该环境的 Python |
| 旧模型没有归一化参数 | 查看 `scripts/fix_norm_params.py`，并在副本上验证历史模型的参数匹配情况 |
| 推理维度与权重不匹配 | 检查训练时保存的 `*_Model_args.json` 与模型、特征配置是否对应 |
| Windows 中文乱码 | 在 cmd 执行 `chcp 65001`，或使用 UTF-8 终端 |

## 贡献与安全

修改模型时请同步补充测试，并在 PR 中写明复现步骤和测试结果。架构与维护背景见 [`docs/system_design.md`](docs/system_design.md)、[`docs/maintenance.md`](docs/maintenance.md)。安全问题请按 [`SECURITY.md`](SECURITY.md) 私下报告。项目采用 [GPL-3.0](LICENSE) 许可证。
