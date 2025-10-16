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
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.timesteps = max(1, timesteps)
        self.predict_steps = max(1, predict_steps)

        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
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

    def forward(self, x: torch.Tensor, predict_steps: Optional[int] = None) -> torch.Tensor:
        steps = max(1, predict_steps) if predict_steps is not None else self.predict_steps
        enc_out, hidden = self.encoder(x)
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


__all__ = ["DiffusionForecaster"]
