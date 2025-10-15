import torch

from stock_prediction.models import TemporalHybridNet


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
