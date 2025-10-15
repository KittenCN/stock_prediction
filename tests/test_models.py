import sys
from pathlib import Path

# 添加 src 目录到 Python 路径
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
