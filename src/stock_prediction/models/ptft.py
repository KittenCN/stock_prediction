import math
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualNetwork(nn.Module):
    
    """Layer-normalised residual block with contextual gating, mirroring TFT.
    """


    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        context_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim) if context_dim else None
        self.activation = nn.GELU()
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        self.skip_proj = nn.Linear(input_dim, input_dim) if input_dim != hidden_dim else None
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        residual = self.skip_proj(x) if self.skip_proj is not None else x
        x_norm = self.layer_norm(x)
        h = self.input_proj(x_norm)
        if context is not None and self.context_proj is not None:
            context = self.context_proj(context)
            h = h + context
        h = self.activation(h)
        h = self.hidden_proj(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.output_proj(h)
        h = self.dropout(h)
        gate = self.gate(h)
        out = gate * h + residual
        return out


class FeatureSelection(nn.Module):
    
    """Learnable feature gating network used within PTFT.
    """


    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )
        self.projection = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, input_dim)
        gates = self.gate_network(x)
        selected = x * gates
        embedded = self.projection(selected)
        return embedded, gates


class ProbTemporalFusionTransformer(nn.Module):
    
    """Probabilistic Temporal Fusion Transformer implementation.
    Provides quantile forecasts with variable selection and attention.
    """


    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 160,
        num_heads: int = 4,
        dropout: float = 0.1,
        predict_steps: int = 1,
        quantiles: Iterable[float] = (0.1, 0.5, 0.9),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.predict_steps = max(1, int(predict_steps))
        self.quantiles = tuple(sorted(set(float(q) for q in quantiles)))
        if 0.5 not in self.quantiles:
            self.quantiles += (0.5,)
            self.quantiles = tuple(sorted(set(self.quantiles)))
        self.median_key = min(self.quantiles, key=lambda q: abs(q - 0.5))
        self._quantile_key_map = {
            self._format_quantile_key(q): f"{q:.2f}" for q in self.quantiles
        }

        self.feature_selector = FeatureSelection(input_dim, hidden_dim)
        self.encoder_grn = GatedResidualNetwork(hidden_dim, hidden_dim, dropout=dropout)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.post_attn_grn = GatedResidualNetwork(hidden_dim, hidden_dim, dropout=dropout)

        self.future_proj = nn.Linear(hidden_dim, hidden_dim * self.predict_steps)
        self.context_norm = nn.LayerNorm(hidden_dim)

        self.quantile_layers = nn.ModuleDict(
            {k: nn.Linear(hidden_dim, output_dim) for k in self._quantile_key_map}
        )
        self.dropout = nn.Dropout(dropout)
        self.last_details: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, predict_steps: int | None = None) -> Dict[str, torch.Tensor]:
        
        """Forward pass producing point forecasts and attention diagnostics.

        Args:
            x: Input tensor shaped (batch, sequence, input_dim).
            predict_steps: Optional override for forecast horizon.

        Returns:
            Dictionary containing point forecasts and auxiliary details.
        """

        steps = self.predict_steps if predict_steps is None else max(1, int(predict_steps))
        features, gates = self.feature_selector(x)
        features = self.encoder_grn(features)
        enc_out, _ = self.encoder(features)
        attn_out, attn_weights = self.attention(enc_out, enc_out, enc_out, need_weights=True)
        fused = self.post_attn_grn(attn_out + enc_out)
        context = self.context_norm(fused[:, -1, :])

        future_base = self.future_proj(context).view(x.size(0), steps, self.hidden_dim)
        future_base = torch.tanh(future_base)
        future_base = self.dropout(future_base)

        quantile_outputs = {}
        for key, layer in self.quantile_layers.items():
            alias = self._quantile_key_map[key]
            quantile_outputs[alias] = layer(future_base)

        median = quantile_outputs[f"{self.median_key:.2f}"]
        if steps == 1:
            median = median.squeeze(1)
            for key in quantile_outputs:
                quantile_outputs[key] = quantile_outputs[key].squeeze(1)

        self.last_details = {
            "feature_gates": gates.mean(dim=1),
            "attention": attn_weights.detach(),
            "quantiles": quantile_outputs,
            "steps": steps,
        }

        return {
            "point": median,
            "quantiles": quantile_outputs,
            "feature_gates": gates,
            "attention": attn_weights,
        }

    def get_last_details(self) -> Dict[str, torch.Tensor]:
        return self.last_details

    @staticmethod
    def _format_quantile_key(q: float) -> str:
        return f"q{int(round(q * 100)):03d}"
