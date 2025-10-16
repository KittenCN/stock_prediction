import torch
import torch.nn as nn


class _DepthwiseConvBlock(nn.Module):
    """
    \u591a\u5c3a\u5ea6\u4e00\u7ef4\u5377\u79ef\u6a21\u5757\uff0c\u4f7f\u7528\u6df1\u5ea6\u53ef\u5206\u79bb\u5377\u79ef\u63d0\u53d6\u5c40\u90e8\u5f62\u6001\u3002
    \u8f93\u5165: (batch, seq_len, hidden_dim)
    \u8f93\u51fa: (batch, seq_len, hidden_dim)
    """

    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hidden_dim,
            bias=False,
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # \u5148\u5c06\u65f6\u95f4\u7ef4\u4e0e\u901a\u9053\u7ef4\u4e92\u6362\uff0c\u4f7f\u5377\u79ef\u6cbf\u65f6\u95f4\u8f74\u6ed1\u52a8
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class TemporalHybridNet(nn.Module):
    """
    \u591a\u5c3a\u5ea6\u65f6\u5e8f\u6df7\u5408\u7f51\u7edc (TemporalHybridNet)\uff0c\u7ed3\u5408\u5377\u79ef\u3001\u53cc\u5411 GRU \u4e0e\u591a\u5934\u6ce8\u610f\u529b\u3002
    \u76ee\u6807\u662f\u9488\u5bf9\u80a1\u7968\u9ad8\u566a\u58f0\u3001\u591a\u5c3a\u5ea6\u7279\u5f81\u63d0\u4f9b\u7a33\u5065\u9884\u6d4b\u80fd\u529b\u3002
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 160,
        attn_heads: int = 4,
        conv_kernel_sizes: tuple[int, ...] = (3, 5, 7),
        conv_dilations: tuple[int, ...] = (1, 2, 3),
        dropout: float = 0.2,
        predict_steps: int = 1,
    ):
        super().__init__()
        assert len(conv_kernel_sizes) == len(conv_dilations), "kernel sizes and dilations must have matching length"

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.predict_steps = max(1, int(predict_steps))

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.conv_blocks = nn.ModuleList(
            [
                _DepthwiseConvBlock(hidden_dim, ks, dil, dropout)
                for ks, dil in zip(conv_kernel_sizes, conv_dilations)
            ]
        )
        self.conv_merge = nn.Linear(hidden_dim * (len(self.conv_blocks) + 1), hidden_dim)

        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim * 2)

        # Temporal feature transformation: rolling mean, standard deviation, tail values
        self.stat_proj = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim * 2),
            nn.GELU(),
        )

        fusion_dim = hidden_dim * 6
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.single_step_head = nn.Linear(hidden_dim * 2, output_dim)
        self.multi_step_head = nn.Linear(hidden_dim * 2, output_dim * self.predict_steps)

        self.last_attention_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, predict_days: int | None = None) -> torch.Tensor:
        """
        :param x: Input tensor shaped (batch, seq_len, input_dim)
        :param predict_days: Optional override for forecast horizon
        :return: Tuple of point forecasts (batch, output_dim) and multi-step forecasts (batch, steps, output_dim)
        """
        steps = self._normalize_predict_steps(predict_days)

        x = x.to(dtype=torch.float32)
        norm_x = self.input_norm(x)
        base = self.input_proj(norm_x)

        conv_feats = [base]
        for block in self.conv_blocks:
            conv_feats.append(block(base))
        multi_scale = torch.cat(conv_feats, dim=-1)
        multi_scale = self.conv_merge(multi_scale)

        rnn_out, _ = self.bigru(multi_scale)

        attn_out, attn_weights = self.attn(rnn_out, rnn_out, rnn_out, need_weights=True)
        attn_out = self.attn_dropout(attn_out)
        fused_seq = self.attn_norm(rnn_out + attn_out)
        self.last_attention_weights = attn_weights

        last_state = fused_seq[:, -1, :]
        mean_state = fused_seq.mean(dim=1)

        window_mean = x.mean(dim=1)
        window_std = x.std(dim=1, unbiased=False)
        window_last = x[:, -1, :]
        stats = torch.cat([window_mean, window_std, window_last], dim=-1)
        stats = self.stat_proj(stats)

        fusion = torch.cat([last_state, mean_state, stats], dim=-1)
        fusion = self.fusion_proj(fusion)

        if steps == 1:
            return self.single_step_head(fusion)

        multi_step = self.multi_step_head(fusion)
        if steps != self.predict_steps:
            max_len = self.output_dim * steps
            multi_step = multi_step[:, :max_len]
        multi_step = multi_step.view(-1, steps, self.output_dim)
        return multi_step

    def _normalize_predict_steps(self, predict_steps: int | None) -> int:
        if predict_steps is None or predict_steps <= 0:
            return self.predict_steps
        return int(predict_steps)
