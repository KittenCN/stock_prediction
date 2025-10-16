import torch
import torch.nn as nn

from stock_prediction.init import OUTPUT_DIMENSION


def generate_attention_mask(input_data: torch.Tensor) -> torch.Tensor:
    mask = (input_data != -0.0).any(dim=-1)
    mask = mask.to(torch.float32)
    mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask.T


class TransformerEncoderLayerWithNorm(nn.TransformerEncoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = "relu", norm=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        if norm is not None:
            self.norm1 = norm
            self.norm2 = norm


class TransformerDecoderLayerWithNorm(nn.TransformerDecoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = "relu", norm=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        if norm is not None:
            self.norm1 = norm
            self.norm2 = norm
            self.norm3 = norm


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerModel(nn.Module):
    """Classic Transformer model for price forecasting, extracted from the legacy implementation."""



    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        output_dim: int = OUTPUT_DIMENSION,
        max_len: int = 5000,
        mode: int = 0,
    ) -> None:
        super().__init__()
        assert d_model % 2 == 0, "d_model must be a multiple of 2"
        self.embedding = MLP(input_dim, d_model // 2, d_model)
        self.positional_encoding: torch.Tensor | None = None
        dropout = 0.5 if mode == 0 else 0.0

        encoder_layer = TransformerEncoderLayerWithNorm(
            d_model,
            nhead,
            dim_feedforward,
            dropout=dropout,
            norm=nn.LayerNorm(d_model),
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = TransformerDecoderLayerWithNorm(
            d_model,
            nhead,
            dim_feedforward,
            dropout=dropout,
            norm=nn.LayerNorm(d_model),
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.target_embedding = nn.Linear(output_dim, d_model)
        self.pooling = nn.AdaptiveAvgPool1d
        self.fc = nn.Linear(d_model, output_dim)
        self.d_model = d_model
        self.output_dim = output_dim
        self.max_len = max_len
        self._initialize_weights()

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, predict_days: int = 0) -> torch.Tensor:
        src = src.permute(1, 0, 2)
        attention_mask = generate_attention_mask(src)

        src_embedding = self.embedding(src)
        seq_len, batch_size, _ = src_embedding.size()

        if self.positional_encoding is None or self.positional_encoding.size(0) < seq_len:
            self.positional_encoding = self.generate_positional_encoding(seq_len, self.d_model).to(src.device)

        positions = torch.arange(seq_len, device=src.device).unsqueeze(1).expand(seq_len, batch_size)
        src_encoded = src_embedding + self.positional_encoding[positions]
        memory = self.transformer_encoder(src_encoded, src_key_padding_mask=attention_mask)

        if predict_days <= 0:
            tgt = tgt.unsqueeze(0)
        else:
            tgt = tgt.permute(1, 0, 2)

        tgt_embedding = self.target_embedding(tgt)
        tgt_len = tgt_embedding.size(0)
        tgt_positions = torch.arange(tgt_len, device=tgt.device).unsqueeze(1).expand(tgt_len, batch_size)
        tgt_encoded = tgt_embedding + self.positional_encoding[tgt_positions]

        output = self.transformer_decoder(tgt_encoded, memory)
        output = output.permute(1, 2, 0)

        if predict_days <= 0:
            pooled = self.pooling(1)(output).squeeze(2)
            return self.fc(pooled)

        pooled = self.pooling(predict_days)(output)
        return self.fc(pooled.permute(0, 2, 1))

    def generate_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
