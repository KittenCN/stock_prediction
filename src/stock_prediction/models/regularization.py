"""Regularisation utilities including Bayesian Dropout with可配置 MC 推理支持。"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class BayesianDropout(nn.Dropout):
    """Dropout 变体，可在 eval 阶段维持随机失活以近似贝叶斯不确定性。"""

    def __init__(self, p: float = 0.5, mc_dropout: bool = False) -> None:
        super().__init__(p=p, inplace=False)
        self.mc_dropout = mc_dropout

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        training = self.training or self.mc_dropout
        return F.dropout(input, self.p, training=training, inplace=self.inplace)

    def set_mc_dropout(self, enabled: bool) -> None:
        """启用或关闭 MC Dropout。"""
        self.mc_dropout = bool(enabled)


__all__ = ["BayesianDropout"]
