from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_forecaster import DiffusionForecaster
from .graph_temporal import GraphTemporalModel
from .ptft import ProbTemporalFusionTransformer
from .vssm import VariationalStateSpaceModel


class _DepthwiseConvBlock(nn.Module):
    """深度可分离卷积分支，保持 legacy 模型的多尺度卷积能力。"""

    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hidden_dim,
            bias=False,
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class TemporalHybridNet(nn.Module):
    """Hybrid 2.0：在原卷积/GRU/Attention 基础上，集成 PTFT/VSSM/Diffusion/Graph 等分支。"""

    DEFAULT_BRANCHES: Dict[str, bool] = {
        "legacy": True,
        "ptft": True,
        "vssm": True,
        "diffusion": True,
        "graph": True,
    }

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 160,
        attn_heads: int = 4,
        conv_kernel_sizes: tuple[int, ...] = (3, 5, 7),
        conv_dilations: tuple[int, ...] = (1, 2, 3),
        dropout: float = 0.2,
        predict_steps: int = 1,
        branch_config: Optional[Dict[str, bool]] = None,
    ) -> None:
        super().__init__()
        if len(conv_kernel_sizes) != len(conv_dilations):
            raise ValueError("conv_kernel_sizes and conv_dilations must have same length")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.predict_steps = max(1, int(predict_steps))
        self.branch_switches = {**self.DEFAULT_BRANCHES, **(branch_config or {})}

        # ---------------- Legacy branch (Conv + BiGRU + Attention) ----------------
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.conv_blocks = nn.ModuleList(
            [_DepthwiseConvBlock(hidden_dim, ks, dil, dropout) for ks, dil in zip(conv_kernel_sizes, conv_dilations)]
        )
        self.conv_merge = nn.Linear(hidden_dim * (len(self.conv_blocks) + 1), hidden_dim)
        self.bigru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(hidden_dim * 2, num_heads=attn_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim * 2)
        self.stat_proj = nn.Sequential(nn.Linear(input_dim * 3, hidden_dim * 2), nn.GELU())
        self.base_proj = nn.Sequential(nn.Linear(hidden_dim * 6, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout))

        # ---------------- Auxiliary branches ----------------
        branch_feature_dim = hidden_dim * 2
        self.branch_dim = branch_feature_dim
        self.branch_projs = nn.ModuleDict()
        self.branches: Dict[str, nn.Module] = {}
        self.regime_proj: Optional[nn.Sequential] = None

        if self.branch_switches.get("ptft", False):
            self.branches["ptft"] = ProbTemporalFusionTransformer(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=max(96, hidden_dim),
                predict_steps=self.predict_steps,
                quantiles=(0.1, 0.5, 0.9),
            )
            self.branch_projs["ptft"] = nn.Sequential(
                nn.Linear(output_dim * self.predict_steps, branch_feature_dim),
                nn.GELU(),
            )

        if self.branch_switches.get("vssm", False):
            self.branches["vssm"] = VariationalStateSpaceModel(
                input_dim=input_dim,
                output_dim=output_dim,
                state_dim=max(48, hidden_dim // 3),
                regime_classes=4,
                predict_steps=self.predict_steps,
            )
            self.branch_projs["vssm"] = nn.Sequential(
                nn.Linear(output_dim * self.predict_steps, branch_feature_dim),
                nn.GELU(),
            )
            self.regime_proj = nn.Sequential(
                nn.Linear(4, branch_feature_dim),
                nn.GELU(),
            )

        if self.branch_switches.get("diffusion", False):
            self.branches["diffusion"] = DiffusionForecaster(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=max(96, hidden_dim),
                predict_steps=self.predict_steps,
            )
            self.branch_projs["diffusion"] = nn.Sequential(
                nn.Linear(output_dim * self.predict_steps, branch_feature_dim),
                nn.GELU(),
            )

        if self.branch_switches.get("graph", False):
            self.branches["graph"] = GraphTemporalModel(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=max(96, hidden_dim),
                predict_steps=self.predict_steps,
            )
            self.branch_projs["graph"] = nn.Sequential(
                nn.Linear(output_dim * self.predict_steps, branch_feature_dim),
                nn.GELU(),
            )

        total_branches = 1 + len(self.branch_projs) + (1 if self.regime_proj is not None else 0)
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.branch_dim * total_branches, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.single_step_head = nn.Linear(hidden_dim * 2, output_dim)
        self.multi_step_head = nn.Linear(hidden_dim * 2, output_dim * self.predict_steps)

        self.last_attention_weights: Optional[torch.Tensor] = None
        self.last_details: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, predict_days: Optional[int] = None) -> torch.Tensor:
        steps = self._normalize_predict_steps(predict_days)
        x = x.to(torch.float32)

        base_feature = self._legacy_branch(x)
        branch_features = [base_feature]
        self.last_details = {"base": base_feature.detach()}

        regime_feature: Optional[torch.Tensor] = None

        branch_call_steps = self.predict_steps
        for name, branch in self.branches.items():
            branch_output = branch(x, branch_call_steps)
            if isinstance(branch_output, dict):
                if "prediction" in branch_output:
                    tensor = branch_output["prediction"]
                else:
                    tensor = branch_output.get("point")
                if tensor is None:
                    continue
                if name == "vssm":
                    regime_probs = branch_output.get("regime_probs")
                    if regime_probs is not None and self.regime_proj is not None:
                        if regime_probs.dim() == 2:
                            regime_feature = regime_probs
                        else:
                            regime_feature = regime_probs.mean(dim=1)
                    if regime_probs is not None:
                        self.last_details["vssm_regime"] = regime_probs.detach()
                self.last_details[name] = tensor.detach()
            else:
                tensor = branch_output
                self.last_details[name] = branch_output.detach()

            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(1)
            tensor = tensor[:, : self.predict_steps, ...]
            flat = tensor.reshape(tensor.size(0), -1)
            branch_features.append(self.branch_projs[name](flat))

        if regime_feature is not None and self.regime_proj is not None:
            if regime_feature.dim() > 2:
                regime_feature = regime_feature.mean(dim=1)
            branch_features.append(self.regime_proj(regime_feature))

        fused = torch.cat(branch_features, dim=-1)
        fused = self.fusion_proj(fused)

        if steps == 1:
            return self.single_step_head(fused)

        multi_step = self.multi_step_head(fused)
        if steps != self.predict_steps:
            multi_step = multi_step[:, : self.output_dim * steps]
        return multi_step.view(-1, steps, self.output_dim)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _legacy_branch(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.input_norm(x)
        base = self.input_proj(norm_x)
        conv_feats = [base]
        for block in self.conv_blocks:
            conv_feats.append(block(base))
        multi_scale = torch.cat(conv_feats, dim=-1)
        multi_scale = self.conv_merge(multi_scale)

        rnn_out, _ = self.bigru(multi_scale)
        attn_out, attn_weights = self.attn(rnn_out, rnn_out, rnn_out, need_weights=True)
        attn_out = self.attn_dropout(attn_out)
        fused_seq = self.attn_norm(rnn_out + attn_out)
        self.last_attention_weights = attn_weights

        last_state = fused_seq[:, -1, :]
        mean_state = fused_seq.mean(dim=1)
        window_mean = x.mean(dim=1)
        window_std = x.std(dim=1, unbiased=False)
        window_last = x[:, -1, :]
        stats = torch.cat([window_mean, window_std, window_last], dim=-1)
        stats = self.stat_proj(stats)

        fusion = torch.cat([last_state, mean_state, stats], dim=-1)
        return self.base_proj(fusion)

    def _normalize_predict_steps(self, predict_steps: Optional[int]) -> int:
        if predict_steps is None or predict_steps <= 0:
            return self.predict_steps
        return int(predict_steps)

    def get_last_details(self) -> Dict[str, torch.Tensor]:
        return self.last_details
