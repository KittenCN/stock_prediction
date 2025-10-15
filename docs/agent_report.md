# 自动执行报告（摘要）

## 最新迭代
- **模型升级**：实现 ProbTemporalFusionTransformer、强化版 Variational State Space Model（时间依赖先验 + 变分推断）以及 PTFTVSSMEnsemble；同时改进 TemporalHybridNet 的 Gated Residual Network。
- **训练入口**：`predict.py` 新增 `--model ptft_vssm`，默认使用 `PTFTVSSMLoss`（MSE + KL）。
- **测试状态**：`pytest` 共 26 项全部通过，覆盖 TemporalHybridNet、PTFT、VSSM 及数据/训练流程。

## 推荐命令
```bash
conda run -n stock_prediction pytest -q
```

## 参考文档
- `docs/system_design.md`
- `docs/model_strategy.md`
- `docs/user_guide.md`
- `docs/maintenance.md`

> 本报告仅保留高层要点，详细实现与实验记录请参见上述文档及 `CHANGELOG.md`。***
