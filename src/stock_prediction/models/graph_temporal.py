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
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.predict_steps = max(1, predict_steps)

        self.adj_param = nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)
        self.node_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, output_dim * self.predict_steps)

    def forward(self, x: torch.Tensor, predict_steps: Optional[int] = None) -> torch.Tensor:
        steps = max(1, predict_steps) if predict_steps is not None else self.predict_steps

        adj = self._build_adjacency()
        graph_feature = torch.matmul(x, adj)
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
        mask = torch.eye(self.input_dim, device=sym.device)
        adj = F.softmax(sym, dim=-1)
        adj = adj * (1 - mask) + mask
        return adj


__all__ = ["GraphTemporalModel"]
