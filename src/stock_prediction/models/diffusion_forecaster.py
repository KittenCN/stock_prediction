from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class DiffusionForecaster(nn.Module):
    """轻量级扩散式时间序列预测模型。

    该实现并非完整的扩散采样流程，而是模拟噪声注入与去噪迭代，
    以便在现有训练/推理接口下验证扩散模型的潜力。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        timesteps: int = 6,
        predict_steps: int = 1,
        use_symbol_embedding: bool = False,
        symbol_embedding_dim: int = 16,
        max_symbols: int = 4096,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.timesteps = max(1, timesteps)
        self.predict_steps = max(1, predict_steps)
        self.use_symbol_embedding = bool(use_symbol_embedding)
        self.symbol_embedding_dim = int(symbol_embedding_dim) if self.use_symbol_embedding else 0
        self.max_symbols = max(1, int(max_symbols))
        self.symbol_embedding = (
            nn.Embedding(self.max_symbols, self.symbol_embedding_dim)
            if self.use_symbol_embedding and self.symbol_embedding_dim > 0
            else None
        )
        self.model_input_dim = self.input_dim + self.symbol_embedding_dim

        self.encoder = nn.GRU(self.model_input_dim, hidden_dim, batch_first=True)
        self.denoiser = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Linear(hidden_dim, output_dim * self.predict_steps)

        beta = torch.linspace(1e-4, 2e-2, steps=self.timesteps)
        self.register_buffer("beta_schedule", beta)

    def forward(
        self,
        x: torch.Tensor,
        predict_steps: Optional[int] = None,
        symbol_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        steps = max(1, predict_steps) if predict_steps is not None else self.predict_steps
        x_augmented, _ = self._augment_with_symbol(x, symbol_index)
        enc_out, hidden = self.encoder(x_augmented)
        h = hidden[-1]

        for beta in self.beta_schedule:
            noise = torch.randn_like(h) * beta
            h = h + noise
            h = self.denoiser(h)

        out = self.head(h)
        out = out.view(x.size(0), self.predict_steps, self.output_dim)
        if steps != self.predict_steps:
            out = out[:, :steps, :]
        if steps == 1:
            out = out.squeeze(1)
        return out

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


__all__ = ["DiffusionForecaster"]
