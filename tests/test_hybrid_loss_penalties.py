import torch
import torch.nn as nn

from stock_prediction.models.hybrid_loss import HybridLoss


class _DummyModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def get_last_details(self):
        return {}


def test_hybrid_loss_mean_penalty_reacts_to_bias() -> None:
    loss_fn = HybridLoss(
        _DummyModel(),
        mse_weight=0.0,
        quantile_weight=0.0,
        direction_weight=0.0,
        regime_weight=0.0,
        volatility_weight=0.0,
        extreme_weight=0.0,
        mean_weight=1.0,
        return_weight=0.0,
    )
    prediction = torch.zeros((2, 3, 4))
    target = torch.ones((2, 3, 4))
    loss = loss_fn(prediction, target)
    assert loss.item() > 0.0


def test_hybrid_loss_return_penalty_prefers_matching_trends() -> None:
    loss_fn = HybridLoss(
        _DummyModel(),
        mse_weight=0.0,
        quantile_weight=0.0,
        direction_weight=0.0,
        regime_weight=0.0,
        volatility_weight=0.0,
        extreme_weight=0.0,
        mean_weight=0.0,
        return_weight=1.0,
    )
    target = torch.tensor([[[0.0], [0.5], [1.0], [1.5]]])
    dull_prediction = torch.zeros_like(target)
    matching_prediction = target.clone()

    dull_loss = loss_fn(dull_prediction, target).item()
    matching_loss = loss_fn(matching_prediction, target).item()
    assert dull_loss > matching_loss


def test_hybrid_loss_volatility_penalty_handles_single_step() -> None:
    loss_fn = HybridLoss(
        _DummyModel(),
        mse_weight=0.0,
        quantile_weight=0.0,
        direction_weight=0.0,
        regime_weight=0.0,
        volatility_weight=1.0,
        extreme_weight=0.0,
        mean_weight=0.0,
        return_weight=0.0,
    )
    # Single-step outputs (steps=1). Prediction is constant, target varies.
    prediction = torch.zeros((8, 1, 2))
    target = torch.linspace(0.0, 1.0, steps=16, dtype=torch.float32).view(8, 1, 2)
    loss = loss_fn(prediction, target)
    assert loss.item() > 0.0
