from __future__ import annotations

import math
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
        schedule: str = "linear",
        context_dim: int = 0,
        context_dropout: float = 0.1,
        learnable_schedule: bool = False,
        use_ddim: bool = False,
        ddim_eta: float = 0.0,
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
        self.schedule = schedule.lower()
        self.context_dim = max(0, int(context_dim))
        self.learnable_schedule = bool(learnable_schedule)
        self.use_ddim = bool(use_ddim)
        self.ddim_eta = float(ddim_eta)
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
        self.context_proj = None
        if self.context_dim > 0:
            self.context_proj = nn.Sequential(
                nn.Linear(self.context_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(context_dropout),
            )

        beta = self._build_beta_schedule(self.schedule, self.timesteps)
        if self.learnable_schedule:
            self.beta_schedule = nn.Parameter(beta)
        else:
            self.register_buffer("beta_schedule", beta)

    def forward(
        self,
        x: torch.Tensor,
        predict_steps: Optional[int] = None,
        symbol_index: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        steps = max(1, predict_steps) if predict_steps is not None else self.predict_steps
        x_augmented, _ = self._augment_with_symbol(x, symbol_index)
        _, hidden = self.encoder(x_augmented)
        h = hidden[-1]

        if context is not None:
            context_vec = context
            if context_vec.dim() == 3:
                context_vec = context_vec.mean(dim=1)
            if self.context_proj is not None:
                context_vec = self.context_proj(context_vec)
            else:
                context_vec = context_vec.to(h.device)
            h = h + context_vec
        betas = self.beta_schedule
        if not torch.is_tensor(betas):
            betas = torch.tensor(betas, device=h.device)
        else:
            betas = betas.to(h.device)
        for beta in betas:
            beta = beta.clamp(min=1e-6)
            if self.use_ddim:
                noise_scale = self.ddim_eta * beta
            else:
                noise_scale = beta
            noise = torch.randn_like(h) * noise_scale
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

    @staticmethod
    def _build_beta_schedule(kind: str, timesteps: int) -> torch.Tensor:
        kind = (kind or "linear").lower()
        if kind == "cosine":
            timesteps = max(1, timesteps)
            t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float32)
            s = 0.008
            alphas_cumprod = torch.cos(((t / timesteps + s) / (1 + s)) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            alphas = torch.clamp(alphas_cumprod[1:] / alphas_cumprod[:-1], min=1e-4, max=0.999)
            beta = 1 - alphas
            return beta
        else:
            return torch.linspace(1e-4, 2e-2, steps=timesteps, dtype=torch.float32)


__all__ = ["DiffusionForecaster"]
