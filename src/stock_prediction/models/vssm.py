from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalStateSpaceModel(nn.Module):
    """
    变分状态空间模型（简化版）。
    使用 RNN 编码器获得潜变量的均值与方差，通过重参数化采样未来状态，再预测未来价格。
    额外输出市场状态概率与 KL 项供训练与解释。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        state_dim: int = 64,
        regime_classes: int = 4,
        predict_steps: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.state_dim = state_dim
        self.regime_classes = regime_classes
        self.predict_steps = max(1, int(predict_steps))

        self.encoder = nn.GRU(input_dim, state_dim, batch_first=True)
        self.latent_mean = nn.Linear(state_dim, state_dim)
        self.latent_logvar = nn.Linear(state_dim, state_dim)

        self.future_proj = nn.Linear(state_dim, state_dim * self.predict_steps)
        self.decoder = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.GELU(),
        )
        self.obs_head = nn.Linear(state_dim, output_dim)
        self.regime_head = nn.Linear(state_dim, regime_classes)

        self.last_details: Dict[str, torch.Tensor] = {}

    def _sample(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x: torch.Tensor, predict_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        steps = self.predict_steps if predict_steps is None else max(1, int(predict_steps))
        enc_out, _ = self.encoder(x)
        latent_mean = self.latent_mean(enc_out)
        latent_logvar = torch.clamp(self.latent_logvar(enc_out), min=-10.0, max=10.0)
        latent_sample = self._sample(latent_mean, latent_logvar)

        last_state = latent_sample[:, -1, :]  # (batch, state_dim)
        future_states = self.future_proj(last_state).view(x.size(0), steps, self.state_dim)
        future_states = self.decoder(future_states)
        obs = self.obs_head(future_states)  # (batch, steps, output_dim)
        regime_logits = self.regime_head(future_states)  # (batch, steps, regime_classes)
        regime_probs = torch.softmax(regime_logits, dim=-1)

        if steps == 1:
            obs = obs.squeeze(1)
            regime_probs = regime_probs.squeeze(1)

        kl = 0.5 * torch.mean(
            torch.exp(latent_logvar) + latent_mean.pow(2) - 1.0 - latent_logvar
        )

        self.last_details = {
            "latent_mean": latent_mean.detach(),
            "latent_logvar": latent_logvar.detach(),
            "regime_probs": regime_probs.detach(),
            "kl": kl.detach(),
            "steps": steps,
        }

        return {
            "prediction": obs,
            "regime_probs": regime_probs,
            "kl": kl,
        }

    def get_last_details(self) -> Dict[str, torch.Tensor]:
        return self.last_details
