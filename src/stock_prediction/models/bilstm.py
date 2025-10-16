import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    
    """Bidirectional LSTM for extracting temporal dependencies.

    Input shape: (batch, seq_len, input_dim)
    Output shape: (batch, output_dim)
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.bilstm(x)  # (batch, seq_len, hidden_dim*2)
        out = out[:, -1, :]
        out = self.fc(out)  # (batch, output_dim)
        return out
