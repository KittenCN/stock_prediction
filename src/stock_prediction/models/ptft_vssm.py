from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ptft import ProbTemporalFusionTransformer
from .regularization import BayesianDropout
from .vssm import VariationalStateSpaceModel


class PTFTVSSMEnsemble(nn.Module):
    """PTFT + V-SSM 双轨模型，提供 Regime 感知的自适应融合。"""

    _global_last_details: Dict[str, torch.Tensor] | None = None

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        state_dim: int = 48,
        regime_classes: int = 4,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        predict_steps: int = 1,
        dropout: float = 0.1,
        ensemble_dropout: float = 0.1,
        fusion_hidden_dim: int = 96,
        mc_dropout: bool = False,
        use_symbol_embedding: bool = False,
        symbol_embedding_dim: int = 16,
        max_symbols: int = 4096,
    ) -> None:
        super().__init__()
        self.predict_steps = max(1, int(predict_steps))
        self.output_dim = output_dim
        self.regime_classes = regime_classes
        self.use_symbol_embedding = bool(use_symbol_embedding)
        self.symbol_embedding_dim = int(symbol_embedding_dim) if self.use_symbol_embedding else 0
        self.max_symbols = max(1, int(max_symbols))
        self.symbol_embedding = (
            nn.Embedding(self.max_symbols, self.symbol_embedding_dim)
            if self.use_symbol_embedding and self.symbol_embedding_dim > 0
            else None
        )
        self.model_input_dim = input_dim + self.symbol_embedding_dim

        self.ptft = ProbTemporalFusionTransformer(
            input_dim=self.model_input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            predict_steps=self.predict_steps,
            quantiles=quantiles,
        )
        self.vssm = VariationalStateSpaceModel(
            input_dim=self.model_input_dim,
            output_dim=output_dim,
            state_dim=state_dim,
            regime_classes=regime_classes,
            predict_steps=self.predict_steps,
        )

        self.gate_network = nn.Sequential(
            nn.Linear(output_dim * 2 + regime_classes, fusion_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_hidden_dim),
            nn.Linear(fusion_hidden_dim, output_dim * 2),
        )
        self.fusion_dropout = BayesianDropout(p=ensemble_dropout, mc_dropout=mc_dropout)
        self.last_details: Dict[str, torch.Tensor] = {}

    def set_mc_dropout(self, enabled: bool) -> None:
        """在推理阶段启用/关闭贝叶斯 Dropout。"""
        self.fusion_dropout.set_mc_dropout(enabled)

    def forward(
        self,
        x: torch.Tensor,
        predict_steps: Optional[int] = None,
        symbol_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        steps = self.predict_steps if predict_steps is None else max(1, int(predict_steps))
        x_augmented, symbol_embed = self._augment_with_symbol(x, symbol_index)
        ptft_result = self.ptft(x_augmented, steps)
        vssm_result = self.vssm(x_augmented, steps)

        ptft_point = self._ensure_step_dim(ptft_result["point"], steps)
        vssm_point = self._ensure_step_dim(vssm_result["prediction"], steps)
        regime_probs = self._ensure_regime_dim(vssm_result["regime_probs"], steps)

        fusion_input = torch.cat([ptft_point, vssm_point, regime_probs], dim=-1)
        gate_logits = self.gate_network(fusion_input)
        gate_logits = gate_logits.view(gate_logits.size(0), steps, self.output_dim, 2)
        fusion_weights = torch.softmax(gate_logits, dim=-1)

        fused = fusion_weights[..., 0] * ptft_point + fusion_weights[..., 1] * vssm_point
        fused = self.fusion_dropout(fused)
        output = fused.squeeze(1) if steps == 1 else fused

        self.last_details = {
            "ptft": ptft_result,
            "vssm": vssm_result,
            "steps": steps,
            "fusion_weights": fusion_weights,
            "gate_logits": gate_logits,
            "fused": fused,
        }
        if symbol_embed is not None:
            self.last_details["symbol_embed"] = symbol_embed.detach()
        PTFTVSSMEnsemble._global_last_details = self.last_details
        return output

    def get_last_details(self) -> Dict[str, torch.Tensor]:
        if self.last_details:
            return self.last_details
        return PTFTVSSMEnsemble._global_last_details or {}

    @staticmethod
    def _ensure_step_dim(tensor: torch.Tensor, steps: int) -> torch.Tensor:
        if tensor.dim() == 2:
            return tensor.unsqueeze(1) if steps >= 1 else tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _ensure_regime_dim(tensor: torch.Tensor, steps: int) -> torch.Tensor:
        if tensor.dim() == 2:
            return tensor.unsqueeze(1) if steps >= 1 else tensor.unsqueeze(0)
        return tensor

    def _augment_with_symbol(
        self, x: torch.Tensor, symbol_index: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.symbol_embedding is None or symbol_index is None:
            return x, None
        if symbol_index.dim() > 1:
            symbol_index = symbol_index.reshape(symbol_index.size(0), -1)[:, 0]
        symbol_index = symbol_index.to(dtype=torch.long, device=x.device)
        symbol_index = torch.clamp(symbol_index, min=0, max=self.max_symbols - 1)
        embed = self.symbol_embedding(symbol_index)
        expanded = embed.unsqueeze(1).expand(-1, x.size(1), -1)
        augmented = torch.cat([x, expanded], dim=-1)
        return augmented, embed


class PTFTVSSMLoss(nn.Module):
    """综合 MSE、KL、方向性、Sharpe/回撤等指标的损失函数。"""

    def __init__(
        self,
        model: PTFTVSSMEnsemble,
        mse_weight: float = 1.0,
        kl_weight: float = 1e-3,
        direction_weight: float = 5e-2,
        sharpe_weight: float = 1e-2,
        max_drawdown_weight: float = 1e-2,
        regime_weight: float = 5e-2,
        quantile_weight: float = 5e-2,
        l2_weight: float = 1e-6,
        volatility_weight: float = 2e-2,
        extreme_weight: float = 2e-2,
    ) -> None:
        super().__init__()
        self.model = model
        self.mse_weight = mse_weight
        self.kl_weight = kl_weight
        self.direction_weight = direction_weight
        self.sharpe_weight = sharpe_weight
        self.max_drawdown_weight = max_drawdown_weight
        self.regime_weight = regime_weight
        self.quantile_weight = quantile_weight
        self.l2_weight = l2_weight
        self.volatility_weight = volatility_weight
        self.extreme_weight = extreme_weight
        self.mse = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(prediction.device)
        prediction_view = self._ensure_three_dim(prediction)
        target_view = self._ensure_three_dim(target)

        base_loss = self.mse(prediction, target)
        total = self.mse_weight * base_loss

        details = self.model.get_last_details()
        if details:
            vssm_details = details.get("vssm", {})
            kl = vssm_details.get("kl", 0.0)
            if isinstance(kl, torch.Tensor):
                total = total + self.kl_weight * kl

            if self.quantile_weight > 0:
                quantile_loss = self._quantile_loss(details, target_view)
                total = total + self.quantile_weight * quantile_loss

            if self.regime_weight > 0 and self.model.predict_steps >= 1:
                regime_loss = self._regime_auxiliary_loss(details, target_view)
                total = total + self.regime_weight * regime_loss

        if self.direction_weight > 0:
            direction_loss = F.binary_cross_entropy_with_logits(
                prediction_view,
                (target_view > 0).float()
            )
            total = total + self.direction_weight * direction_loss

        flat_pred = prediction_view.reshape(prediction_view.size(0), -1)
        flat_target = target_view.reshape(target_view.size(0), -1)

        if self.sharpe_weight > 0:
            sharpe_loss = F.mse_loss(self._sharpe(flat_pred), self._sharpe(flat_target))
            total = total + self.sharpe_weight * sharpe_loss

        if self.max_drawdown_weight > 0:
            mdd_loss = F.mse_loss(self._max_drawdown(flat_pred), self._max_drawdown(flat_target))
            total = total + self.max_drawdown_weight * mdd_loss

        if self.volatility_weight > 0:
            volatility_loss = self._volatility_penalty(prediction_view, target_view)
            total = total + self.volatility_weight * volatility_loss

        if self.extreme_weight > 0:
            extreme_loss = self._extreme_penalty(prediction_view, target_view)
            total = total + self.extreme_weight * extreme_loss

        if self.l2_weight > 0:
            l2_penalty = sum(torch.sum(param ** 2) for param in self.model.parameters())
            total = total + self.l2_weight * l2_penalty

        return total

    def _quantile_loss(self, details: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        ptft_details = details.get("ptft", {})
        quantiles = ptft_details.get("quantiles", {})
        if not quantiles:
            return torch.zeros(1, device=target.device, dtype=target.dtype)[0]

        target_view = self._ensure_three_dim(target)
        total_loss = torch.zeros(1, device=target.device, dtype=target.dtype)[0]
        count = 0
        for alias, quantile_pred in quantiles.items():
            try:
                q = float(alias)
            except (TypeError, ValueError):
                continue
            pred_view = self._ensure_three_dim(quantile_pred).to(target.device)
            diff = target_view - pred_view
            pinball = torch.maximum(q * diff, (q - 1) * diff)
            total_loss = total_loss + pinball.mean()
            count += 1
        if count == 0:
            return torch.zeros(1, device=target.device, dtype=target.dtype)[0]
        return total_loss / count

    def _regime_auxiliary_loss(self, details: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        vssm_details = details.get("vssm", {})
        regime_probs = vssm_details.get("regime_probs")
        if regime_probs is None:
            return torch.zeros(1, device=target.device, dtype=target.dtype)[0]

        regime_view = self._ensure_three_dim(regime_probs).to(target.device)
        steps = regime_view.size(1)
        returns = target.mean(dim=-1)
        flat_returns = returns.reshape(-1)
        num_classes = regime_view.size(-1)

        if flat_returns.numel() < num_classes:
            thresholds = torch.linspace(flat_returns.min(), flat_returns.max(), num_classes - 1, device=target.device)
        else:
            quantiles = torch.linspace(0, 1, num_classes + 1, device=target.device)[1:-1]
            thresholds = torch.quantile(flat_returns, quantiles)

        regime_target = torch.bucketize(returns, thresholds)
        regime_target = regime_target.clamp_(0, num_classes - 1)
        log_probs = torch.log(regime_view + 1e-8)
        loss = F.nll_loss(log_probs.view(-1, num_classes), regime_target.view(-1).long())
        return loss

    def _ensure_three_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 2:
            return tensor.unsqueeze(1)
        return tensor

    @staticmethod
    def _sharpe(returns: torch.Tensor) -> torch.Tensor:
        mean = returns.mean(dim=-1)
        std = returns.std(dim=-1) + 1e-6
        return mean / std

    @staticmethod
    def _max_drawdown(returns: torch.Tensor) -> torch.Tensor:
        cumulative = torch.cumsum(returns, dim=-1)
        running_max, _ = torch.cummax(cumulative, dim=-1)
        drawdown = running_max - cumulative
        max_dd, _ = torch.max(drawdown, dim=-1)
        return max_dd

    @staticmethod
    def _volatility_penalty(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_std = pred.std(dim=1, unbiased=False)
        target_std = target.std(dim=1, unbiased=False)
        return F.mse_loss(pred_std, target_std)

    @staticmethod
    def _extreme_penalty(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_max = pred.amax(dim=1)
        target_max = target.amax(dim=1)
        pred_min = pred.amin(dim=1)
        target_min = target.amin(dim=1)
        max_loss = F.mse_loss(pred_max, target_max)
        min_loss = F.mse_loss(pred_min, target_min)
        return max_loss + min_loss
