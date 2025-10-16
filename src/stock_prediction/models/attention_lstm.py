import torch
import torch.nn as nn

class AttentionLSTM(nn.Module):
    
    """LSTM plus attention architecture for time-series forecasting.

    Input shape: (batch, seq_len, input_dim)
    Output shape: (batch, output_dim)
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.attn = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden_dim)
        out = self.fc(context)  # (batch, output_dim)
        return out
