from typing import Dict, Optional

import torch
import torch.nn as nn


def _kl_divergence_gaussian(
    mean_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mean_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    """KL divergence between diagonal Gaussian distributions."""
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = logvar_p - logvar_q + (var_q + (mean_q - mean_p) ** 2) / (var_p + 1e-8) - 1.0
    return 0.5 * torch.sum(kl, dim=-1)


class VariationalStateSpaceModel(nn.Module):
    """Variational state-space model capturing latent temporal regimes."""


    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        state_dim: int = 64,
        regime_classes: int = 4,
        predict_steps: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.state_dim = state_dim
        self.regime_classes = regime_classes
        self.predict_steps = max(1, int(predict_steps))

        self.encoder = nn.GRU(input_dim, state_dim, batch_first=True, bidirectional=True)
        self.posterior_proj = nn.Linear(state_dim * 2, state_dim * 2)

        self.prior_cell = nn.GRUCell(state_dim, state_dim)
        self.prior_proj = nn.Linear(state_dim, state_dim * 2)

        self.emission = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, output_dim),
        )
        self.regime_head = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.GELU(),
            nn.Linear(state_dim, regime_classes),
        )

        self.last_details: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, predict_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        device = x.device
        batch, seq_len, _ = x.shape
        steps = self.predict_steps if predict_steps is None else max(1, int(predict_steps))

        enc_out, _ = self.encoder(x)
        posterior_params = self.posterior_proj(enc_out)
        post_mean, post_logvar = torch.chunk(posterior_params, 2, dim=-1)
        post_logvar = torch.clamp(post_logvar, min=-10.0, max=5.0)

        z_samples = []
        kl_terms = []
        prior_hidden = torch.zeros(batch, self.state_dim, device=device)
        prior_mean = torch.zeros(batch, self.state_dim, device=device)
        prior_logvar = torch.zeros(batch, self.state_dim, device=device)

        for t in range(seq_len):
            eps = torch.randn_like(post_mean[:, t, :])
            std_q = torch.exp(0.5 * post_logvar[:, t, :])
            z_t = post_mean[:, t, :] + std_q * eps
            z_samples.append(z_t)

            if t > 0:
                prior_hidden = self.prior_cell(z_samples[-2], prior_hidden)
                prior_params = self.prior_proj(prior_hidden)
                prior_mean, prior_logvar = torch.chunk(prior_params, 2, dim=-1)
                prior_logvar = torch.clamp(prior_logvar, min=-6.0, max=4.0)
            else:
                prior_mean = torch.zeros_like(post_mean[:, t, :])
                prior_logvar = torch.zeros_like(post_mean[:, t, :])

            kl_t = _kl_divergence_gaussian(
                mean_q=post_mean[:, t, :],
                logvar_q=post_logvar[:, t, :],
                mean_p=prior_mean,
                logvar_p=prior_logvar,
            )
            kl_terms.append(kl_t)

        z_samples_tensor = torch.stack(z_samples, dim=1)
        kl = torch.mean(torch.stack(kl_terms, dim=1))

        state = z_samples[-1]
        prior_state = prior_hidden
        future_states = []
        future_regimes = []
        for _ in range(steps):
            prior_state = self.prior_cell(state, prior_state)
            prior_params = self.prior_proj(prior_state)
            state_mean, state_logvar = torch.chunk(prior_params, 2, dim=-1)
            state_logvar = torch.clamp(state_logvar, min=-6.0, max=4.0)
            state = state_mean
            future_states.append(state)
            future_regimes.append(self.regime_head(state))

        future_states = torch.stack(future_states, dim=1)
        obs = self.emission(future_states)
        regime_logits = torch.stack(future_regimes, dim=1)
        regime_probs = torch.softmax(regime_logits, dim=-1)

        if steps == 1:
            obs = obs.squeeze(1)
            regime_probs = regime_probs.squeeze(1)

        self.last_details = {
            "posterior_mean": post_mean.detach(),
            "posterior_logvar": post_logvar.detach(),
            "sampled_z": z_samples_tensor.detach(),
            "kl_terms": torch.stack(kl_terms, dim=1).detach(),
            "kl": kl.detach(),
            "regime_probs": regime_probs.detach(),
            "regime_logits": regime_logits.detach(),
        }

        return {
            "prediction": obs,
            "regime_probs": regime_probs,
            "regime_logits": regime_logits,
            "kl": kl,
        }

    def get_last_details(self) -> Dict[str, torch.Tensor]:
        return self.last_details
