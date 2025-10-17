import sys
from pathlib import Path

# 添加 src 目录至 Python 路径
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch
from stock_prediction.models import (
    TemporalHybridNet,
    ProbTemporalFusionTransformer,
    VariationalStateSpaceModel,
    PTFTVSSMEnsemble,
    PTFTVSSMLoss,
    DiffusionForecaster,
    GraphTemporalModel,
    HybridLoss,
)


def test_temporal_hybrid_single_step_shape():
    model = TemporalHybridNet(input_dim=30, output_dim=4, predict_steps=1)
    x = torch.randn(2, 5, 30)
    out = model(x)
    assert out.shape == (2, 4)


def test_temporal_hybrid_multi_step_shape():
    model = TemporalHybridNet(input_dim=30, output_dim=4, predict_steps=3)
    x = torch.randn(2, 6, 30)

    out_full = model(x)
    assert out_full.shape == (2, 3, 4)

    out_partial = model(x, predict_days=2)
    assert out_partial.shape == (2, 2, 4)


def test_temporal_hybrid_symbol_embedding():
    model = TemporalHybridNet(
        input_dim=30,
        output_dim=4,
        predict_steps=2,
        use_symbol_embedding=True,
        symbol_embedding_dim=8,
        max_symbols=32,
    )
    x = torch.randn(3, 6, 30)
    symbols = torch.tensor([0, 1, 5])
    out = model(x, predict_days=2, symbol_index=symbols)
    assert out.shape == (3, 2, 4)


def test_temporal_hybrid_branch_gating():
    model = TemporalHybridNet(
        input_dim=30,
        output_dim=4,
        predict_steps=2,
        branch_config={"ptft": {"enabled": True, "weight": 0.5}, "diffusion": False},
    )
    x = torch.randn(2, 6, 30)
    out = model(x, predict_days=2)
    assert out.shape == (2, 2, 4)
    details = model.get_last_details()
    assert "fusion_gate" in details
    weights = details["fusion_gate"]
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights.sum(dim=1)), atol=1e-5)


def test_prob_tft_quantiles():
    model = ProbTemporalFusionTransformer(input_dim=30, output_dim=4, predict_steps=2)
    x = torch.randn(3, 7, 30)
    result = model(x)
    assert "point" in result
    assert result["point"].shape == (3, 2, 4)
    assert set(result["quantiles"].keys()) >= {"0.10", "0.50", "0.90"}
    assert result["quantiles"]["0.10"].shape == (3, 2, 4)


def test_vssm_outputs_and_kl():
    model = VariationalStateSpaceModel(input_dim=30, output_dim=4, predict_steps=2)
    x = torch.randn(3, 6, 30)
    result = model(x)
    assert result["prediction"].shape == (3, 2, 4)
    assert result["regime_probs"].shape == (3, 2, model.regime_classes)
    assert torch.is_tensor(result["kl"])
    assert result["kl"] >= 0


def test_ptft_vssm_ensemble_and_loss():
    model = PTFTVSSMEnsemble(input_dim=30, output_dim=4, predict_steps=2)
    criterion = PTFTVSSMLoss(model, mse_weight=1.0, kl_weight=1e-2)
    x = torch.randn(2, 6, 30)
    target = torch.randn(2, 2, 4)
    output = model(x, predict_steps=2)
    assert output.shape == (2, 2, 4)
    loss = criterion(output, target)
    assert loss.shape == ()
    details = model.get_last_details()
    assert "ptft" in details and "vssm" in details
    assert "fusion_weights" in details

    fusion_weights = details["fusion_weights"]
    assert fusion_weights.shape == (2, 2, 4, 2)
    weight_sum = fusion_weights.sum(dim=-1)
    assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-5)

    loss.backward()
    grad_exists = any(param.grad is not None for param in model.parameters())
    assert grad_exists, "梯度应能回传至模型参数"


def test_ptft_vssm_symbol_embedding():
    model = PTFTVSSMEnsemble(
        input_dim=30,
        output_dim=4,
        predict_steps=2,
        use_symbol_embedding=True,
        symbol_embedding_dim=8,
        max_symbols=32,
    )
    x = torch.randn(2, 6, 30)
    symbols = torch.tensor([0, 3])
    output = model(x, predict_steps=2, symbol_index=symbols)
    assert output.shape == (2, 2, 4)


def test_ptft_vssm_mc_dropout_toggle():
    model = PTFTVSSMEnsemble(input_dim=30, output_dim=4, predict_steps=1, mc_dropout=False)
    x = torch.randn(1, 5, 30)
    model.eval()
    out1 = model(x)
    assert model.fusion_dropout.mc_dropout is False
    model.set_mc_dropout(True)
    assert model.fusion_dropout.mc_dropout is True
    out2 = model(x)
    # 启用 MC Dropout 后，两次推理结果应存在差异
    assert not torch.allclose(out1, out2)


def test_ptft_vssm_loss_volatility_extreme_terms():
    model = PTFTVSSMEnsemble(input_dim=30, output_dim=4, predict_steps=2)
    criterion = PTFTVSSMLoss(
        model,
        mse_weight=1.0,
        kl_weight=0.0,
        direction_weight=0.0,
        sharpe_weight=0.0,
        max_drawdown_weight=0.0,
        regime_weight=0.0,
        quantile_weight=0.0,
        l2_weight=0.0,
        volatility_weight=0.1,
        extreme_weight=0.1,
    )
    x = torch.randn(2, 6, 30)
    target = torch.randn(2, 2, 4)
    prediction = model(x, predict_steps=2)
    loss = criterion(prediction, target)
    assert torch.isfinite(loss)
    loss.backward()
    grad_exists = any(param.grad is not None for param in model.parameters())
    assert grad_exists


def test_diffusion_forecaster_shapes():
    model = DiffusionForecaster(input_dim=30, output_dim=4, predict_steps=3)
    x = torch.randn(2, 6, 30)
    out_full = model(x)
    assert out_full.shape == (2, 3, 4)
    out_single = model(x, predict_steps=1)
    assert out_single.shape == (2, 4)


def test_diffusion_forecaster_symbol_embedding():
    model = DiffusionForecaster(
        input_dim=30,
        output_dim=4,
        predict_steps=2,
        use_symbol_embedding=True,
        symbol_embedding_dim=8,
        max_symbols=64,
    )
    x = torch.randn(2, 5, 30)
    symbols = torch.tensor([3, 7])
    out = model(x, symbol_index=symbols)
    assert out.shape == (2, 2, 4)


def test_diffusion_forecaster_cosine_context():
    model = DiffusionForecaster(
        input_dim=30,
        output_dim=4,
        predict_steps=2,
        schedule="cosine",
        context_dim=8,
    )
    x = torch.randn(2, 6, 30)
    context = torch.randn(2, 6, 8)
    out = model(x, context=context)
    assert out.shape == (2, 2, 4)


def test_diffusion_forecaster_learnable_schedule_ddim():
    model = DiffusionForecaster(
        input_dim=30,
        output_dim=4,
        predict_steps=1,
        learnable_schedule=True,
        use_ddim=True,
        ddim_eta=0.5,
    )
    assert isinstance(model.beta_schedule, torch.nn.Parameter)
    x = torch.randn(3, 5, 30)
    out = model(x)
    assert out.shape == (3, 4)
    loss = out.mean()
    loss.backward()
    assert model.beta_schedule.grad is not None


def test_graph_temporal_model_shapes():
    model = GraphTemporalModel(input_dim=30, output_dim=4, predict_steps=2)
    x = torch.randn(2, 5, 30)
    out = model(x)
    assert out.shape == (2, 2, 4)
    out_single = model(x, predict_steps=1)
    assert out_single.shape == (2, 4)


def test_graph_temporal_model_symbol_embedding():
    model = GraphTemporalModel(
        input_dim=30,
        output_dim=4,
        predict_steps=1,
        use_symbol_embedding=True,
        symbol_embedding_dim=8,
        max_symbols=32,
    )
    x = torch.randn(2, 5, 30)
    symbols = torch.tensor([2, 15])
    out = model(x, symbol_index=symbols)
    assert out.shape == (2, 4)


def test_graph_temporal_model_dynamic_adj():
    model = GraphTemporalModel(
        input_dim=30,
        output_dim=4,
        predict_steps=2,
        use_dynamic_adj=True,
        dynamic_alpha=0.6,
    )
    x = torch.randn(2, 6, 30)
    out = model(x)
    assert out.shape == (2, 2, 4)


def test_hybrid_loss_forward():
    model = TemporalHybridNet(input_dim=30, output_dim=4, predict_steps=2)
    criterion = HybridLoss(model, mse_weight=1.0, quantile_weight=0.0, direction_weight=0.1, regime_weight=0.0)
    x = torch.randn(3, 6, 30)
    target = torch.randn(3, 2, 4)
    prediction = model(x, predict_days=2)
    loss = criterion(prediction, target)
    assert loss.shape == () and torch.isfinite(loss)


def test_hybrid_loss_volatility_extreme_terms():
    model = TemporalHybridNet(input_dim=30, output_dim=4, predict_steps=3)
    criterion = HybridLoss(
        model,
        mse_weight=1.0,
        quantile_weight=0.0,
        direction_weight=0.0,
        regime_weight=0.0,
        volatility_weight=0.1,
        extreme_weight=0.1,
    )
    x = torch.randn(2, 7, 30)
    target = torch.randn(2, 3, 4)
    prediction = model(x, predict_days=3)
    loss = criterion(prediction, target)
    assert torch.isfinite(loss)
    loss.backward()
    grad_exists = any(param.grad is not None for param in model.parameters())
    assert grad_exists
