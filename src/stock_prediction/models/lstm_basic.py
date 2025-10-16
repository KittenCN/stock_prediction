import torch
import torch.nn as nn

from stock_prediction.init import OUTPUT_DIMENSION


class LSTM(nn.Module):
    
    """Baseline multi-layer LSTM extracted from the legacy implementation.
    """


    def __init__(self, input_dim: int, output_dim: int = OUTPUT_DIMENSION) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=3,
            batch_first=True,
            dropout=0.5,
        )
        self.linear1 = nn.Linear(128, 16)
        self.linear2 = nn.Linear(16, output_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor, _tgt: torch.Tensor, predict_days: int = 0) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.activation(self.linear1(out))
        out = self.linear2(out)
        if predict_days > 0:
            out = out.unsqueeze(1)
        return out
