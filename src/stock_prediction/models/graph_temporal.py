from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphTemporalModel(nn.Module):
    """简化版图结构时间序列模型。

    - 将输入特征视作图节点，学习对称邻接矩阵；
    - 使用线性信息聚合 + GRU 提取时间依赖；
    - 输出与传统模型兼容的预测张量。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        predict_steps: int = 1,
        dropout: float = 0.1,
        use_symbol_embedding: bool = False,
        symbol_embedding_dim: int = 16,
        max_symbols: int = 4096,
        use_dynamic_adj: bool = False,
        dynamic_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
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
        self.use_dynamic_adj = bool(use_dynamic_adj)
        self.dynamic_alpha = float(dynamic_alpha)

        self.adj_param = nn.Parameter(torch.randn(self.model_input_dim, self.model_input_dim) * 0.01)
        self.node_proj = nn.Linear(self.model_input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, output_dim * self.predict_steps)

    def forward(
        self,
        x: torch.Tensor,
        predict_steps: Optional[int] = None,
        symbol_index: Optional[torch.Tensor] = None,
        adj_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        steps = max(1, predict_steps) if predict_steps is not None else self.predict_steps

        x_augmented, _ = self._augment_with_symbol(x, symbol_index)

        adj = self._build_adjacency()
        if self.use_dynamic_adj:
            dynamic_adj = self._build_dynamic_adjacency(x_augmented)
            if dynamic_adj is not None:
                adj = (1 - self.dynamic_alpha) * adj + self.dynamic_alpha * dynamic_adj
        if adj_override is not None:
            adj = adj_override.to(adj.device)
        graph_feature = torch.matmul(x_augmented, adj)
        projected = self.node_proj(graph_feature)

        enc_out, hidden = self.gru(projected)
        h = hidden[-1]
        h = self.dropout(h)
        out = self.head(h).view(x.size(0), self.predict_steps, self.output_dim)
        if steps != self.predict_steps:
            out = out[:, :steps, :]
        if steps == 1:
            out = out.squeeze(1)
        return out

    def _build_adjacency(self) -> torch.Tensor:
        sym = (self.adj_param + self.adj_param.t()) / 2
        mask = torch.eye(self.model_input_dim, device=sym.device)
        adj = F.softmax(sym, dim=-1)
        adj = adj * (1 - mask) + mask
        return adj

    def _build_dynamic_adjacency(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if x.dim() != 3 or x.size(0) == 0:
            return None
        device = x.device
        mean_feat = x.mean(dim=1)  # (batch, nodes)
        norm = mean_feat / (mean_feat.norm(dim=-1, keepdim=True) + 1e-6)
        dynamic = torch.matmul(norm.transpose(0, 1), norm) / norm.size(0)
        dynamic = torch.clamp(dynamic, min=-1.0, max=1.0)
        dynamic = (dynamic + 1.0) / 2.0
        dynamic = dynamic.softmax(dim=-1)
        return dynamic.to(device)

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


__all__ = ["GraphTemporalModel"]
