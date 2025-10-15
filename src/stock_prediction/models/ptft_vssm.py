from typing import Dict, Optional

import torch
import torch.nn as nn

from .ptft import ProbTemporalFusionTransformer
from .vssm import VariationalStateSpaceModel


class PTFTVSSMEnsemble(nn.Module):
    """
    主推方案：PTFT + V-SSM 双轨融合模型。
    - PTFT 提供分位预测与特征注意力信息
    - V-SSM 输出状态空间预测与市场状态概率
    - 通过融合层综合两者结果，产出最终价格预测
    """

    _global_last_details: Dict[str, torch.Tensor] | None = None

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 160,
        state_dim: int = 64,
        regime_classes: int = 4,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        predict_steps: int = 1,
        dropout: float = 0.1,
        ensemble_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.predict_steps = max(1, int(predict_steps))
        self.ptft = ProbTemporalFusionTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            predict_steps=self.predict_steps,
            quantiles=quantiles,
        )
        self.vssm = VariationalStateSpaceModel(
            input_dim=input_dim,
            output_dim=output_dim,
            state_dim=state_dim,
            regime_classes=regime_classes,
            predict_steps=self.predict_steps,
        )
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim * 2),
            nn.GELU(),
            nn.Dropout(ensemble_dropout),
            nn.Linear(output_dim * 2, output_dim),
        )
        self.last_details: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, predict_steps: Optional[int] = None) -> torch.Tensor:
        steps = self.predict_steps if predict_steps is None else max(1, int(predict_steps))
        ptft_result = self.ptft(x, steps)
        vssm_result = self.vssm(x, steps)

        ptft_point = ptft_result["point"]
        vssm_point = vssm_result["prediction"]
        fused_input = torch.cat([ptft_point, vssm_point], dim=-1)
        fused = self.fusion(fused_input)

        self.last_details = {
            "ptft": self.ptft.get_last_details(),
            "vssm": self.vssm.get_last_details(),
            "steps": steps,
            "fused": fused.detach(),
        }
        PTFTVSSMEnsemble._global_last_details = self.last_details
        return fused

    def get_last_details(self) -> Dict[str, torch.Tensor]:
        if self.last_details:
            return self.last_details
        return PTFTVSSMEnsemble._global_last_details or {}


class PTFTVSSMLoss(nn.Module):
    """
    MSE + KL 复合损失，用于训练双轨组合模型。
    """

    def __init__(self, model: PTFTVSSMEnsemble, mse_weight: float = 1.0, kl_weight: float = 1e-3) -> None:
        super().__init__()
        self.model = model
        self.mse_weight = mse_weight
        self.kl_weight = kl_weight
        self.mse = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base_loss = self.mse(prediction, target)
        extra = 0.0
        details = self.model.get_last_details()
        if details:
            vssm_details = details.get("vssm", {})
            kl = vssm_details.get("kl", 0.0)
            if isinstance(kl, torch.Tensor):
                extra = kl
        return self.mse_weight * base_loss + self.kl_weight * extra
