from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridLoss(nn.Module):
    """组合损失：在保留 MSE 的基础上，融合分位、方向性与 Regime 约束。"""

    def __init__(
        self,
        model: nn.Module,
        mse_weight: float = 1.0,
        quantile_weight: float = 0.1,
        direction_weight: float = 0.05,
        regime_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.model = model
        self.mse_weight = mse_weight
        self.quantile_weight = quantile_weight
        self.direction_weight = direction_weight
        self.regime_weight = regime_weight
        self.mse = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = self.mse_weight * self.mse(prediction, target)

        details = {}
        if hasattr(self.model, "get_last_details"):
            details = self.model.get_last_details() or {}

        if self.quantile_weight > 0:
            quantile_loss = self._quantile_loss(details, target, prediction.device)
            total = total + self.quantile_weight * quantile_loss

        if self.direction_weight > 0:
            direction_loss = self._direction_loss(prediction, target)
            total = total + self.direction_weight * direction_loss

        if self.regime_weight > 0:
            regime_loss = self._regime_alignment_loss(details, target, prediction.device)
            if regime_loss is not None:
                total = total + self.regime_weight * regime_loss

        return total

    def _quantile_loss(self, details: Dict[str, torch.Tensor], target: torch.Tensor, device: torch.device) -> torch.Tensor:
        ptft_details: Optional[Dict[str, torch.Tensor]] = details.get("ptft_details")
        if not ptft_details:
            return torch.zeros((), device=device)

        quantiles = ptft_details.get("quantiles")
        if not quantiles:
            return torch.zeros((), device=device)

        target_expanded = target
        if target_expanded.dim() == 2:
            target_expanded = target_expanded.unsqueeze(1)

        losses = []
        for key, value in quantiles.items():
            if not isinstance(value, torch.Tensor):
                continue
            tensor = value
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(1)
            q_value = float(key) if isinstance(key, str) else key
            if not 0 < q_value < 1:
                continue
            error = target_expanded - tensor.to(target_expanded.device)
            pinball = torch.maximum(q_value * error, (q_value - 1) * error)
            losses.append(pinball.mean())

        if not losses:
            return torch.zeros((), device=device)
        stacked = torch.stack(losses)
        return stacked.mean()

    def _direction_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dir = torch.tanh(prediction)
        true_dir = torch.tanh(target)
        return F.mse_loss(pred_dir, true_dir)

    def _regime_alignment_loss(
        self,
        details: Dict[str, torch.Tensor],
        target: torch.Tensor,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        regime_probs = details.get("vssm_regime")
        if regime_probs is None:
            return None

        probs = regime_probs.to(device)
        if probs.dim() == 2:
            probs = probs.unsqueeze(1)

        steps = probs.size(1)
        classes = probs.size(-1)
        regime_scale = torch.linspace(-1.0, 1.0, classes, device=device)
        regime_value = (probs * regime_scale).sum(dim=-1)

        if target.dim() == 2:
            target_view = target.unsqueeze(1)
        else:
            target_view = target
        target_view = target_view[:, :steps, :]
        target_return = target_view.mean(dim=-1)
        target_value = torch.tanh(target_return)

        return F.mse_loss(regime_value, target_value)


__all__ = ["HybridLoss"]
