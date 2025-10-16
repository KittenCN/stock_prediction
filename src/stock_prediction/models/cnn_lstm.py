import torch
import torch.nn as nn
import torch.nn.functional as F

from stock_prediction.init import OUTPUT_DIMENSION


class Attention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, outputs: torch.Tensor) -> torch.Tensor:
        weights = self.softmax(self.attn(outputs))
        return torch.bmm(weights.transpose(1, 2), outputs).squeeze(1)


class CNNLSTM(nn.Module):
    
    """CNN + LSTM + attention hybrid rewritten from the historical implementation.
    """


    def __init__(self, input_dim: int, num_classes: int = 2, predict_days: int = 1, dropout_rate: float = 0.5) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=3, batch_first=True)
        self.attention = Attention(256)
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.predict_days = predict_days

    def forward(self, x_3d: torch.Tensor, _tgt: torch.Tensor, _predict_days: int = 1) -> torch.Tensor:
        self.lstm.flatten_parameters()
        batch_size, _, _ = x_3d.size()
        placeholder = torch.zeros(batch_size, 1, OUTPUT_DIMENSION, device=x_3d.device)
        for idx in range(batch_size):
            placeholder[idx] = x_3d[idx][-1][:OUTPUT_DIMENSION]
        x_3d = x_3d.transpose(1, 2)
        conv = F.relu(self.conv1(x_3d))
        conv = F.relu(self.conv2(conv))
        r_in = conv.transpose(1, 2)
        r_out, _ = self.lstm(r_in)
        attn_out = self.attention(r_out)
        x = F.relu(self.fc1(attn_out))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x.view(batch_size, self.predict_days, -1)
