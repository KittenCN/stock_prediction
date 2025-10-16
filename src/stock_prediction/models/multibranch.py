import torch
import torch.nn as nn

class MultiBranchNet(nn.Module):
    
    """Dual-branch architecture combining price and technical indicator features.

    price_x shape: (batch, seq_len, price_dim)
    tech_x shape: (batch, seq_len, tech_dim)
    Output shape: (batch, output_dim)
    """

    def __init__(self, price_dim, tech_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.price_lstm = nn.LSTM(price_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.tech_lstm = nn.LSTM(tech_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
    def forward(self, price_x, tech_x):
        price_out, _ = self.price_lstm(price_x)
        tech_out, _ = self.tech_lstm(tech_x)

        price_feat = price_out[:, -1, :]
        tech_feat = tech_out[:, -1, :]
        feat = torch.cat([price_feat, tech_feat], dim=-1)
        out = self.fc(feat)
        return out
