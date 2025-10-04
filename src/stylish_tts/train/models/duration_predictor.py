import math
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from einops import rearrange
from .common import LinearNorm
from .text_encoder import TextEncoder
from .text_style_encoder import TextStyleEncoder
from .ada_norm import AdaptiveLayerNorm
from ..utils import sequence_mask
from torch.nn.utils.parametrizations import weight_norm
from .text_encoder import MultiHeadAttention


class DurationPredictor(torch.nn.Module):
    def __init__(
        self, style_dim, inter_dim, text_config, style_config, duration_config
    ):
        super().__init__()
        self.text_encoder = TextEncoder(inter_dim=inter_dim, config=text_config)
        self.bert_encoder = torch.nn.Linear(768, inter_dim)
        self.style_encoder = TextStyleEncoder(
            inter_dim,
            style_dim,
            config=style_config,
        )
        self.conv_next = torch.nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=inter_dim,
                    intermediate_dim=inter_dim * 4,
                    style_dim=style_dim,
                )
                for _ in range(duration_config.n_layer)
            ]
        )
        self.dropout = torch.nn.Dropout1d(duration_config.last_dropout)
        self.duration_proj = LinearNorm(inter_dim, duration_config.duration_classes)
        # TODO: Remove this vestigial layer
        self.coefficient_proj = LinearNorm(inter_dim, 5)

        cross_channels = inter_dim
        self.query_norm = AdaptiveLayerNorm(style_dim, cross_channels)
        self.key_norm = AdaptiveLayerNorm(style_dim, cross_channels)
        self.cross_attention = MultiHeadAttention(
            channels=cross_channels,
            out_channels=cross_channels,
            n_heads=8,
            p_dropout=0.5,
        )
        self.cross_window = 5

        self.cross_post = torch.nn.Sequential(
            weight_norm(
                torch.nn.Conv1d(
                    cross_channels,
                    cross_channels,
                    kernel_size=5,
                    padding=2,
                    groups=cross_channels,
                )
            ),
            torch.nn.SiLU(),
            weight_norm(torch.nn.Conv1d(cross_channels, cross_channels, kernel_size=1)),
        )

    def compute_cross(self, text_encoding, style, text_mask):
        query = text_encoding
        query = self.query_norm(query, style).transpose(1, 2)
        key = text_encoding
        key = self.key_norm(key, style).transpose(1, 2)

        attention_mask = text_mask.unsqueeze(2) * text_mask.unsqueeze(-1)
        attention = self.cross_attention(query, key, attn_mask=attention_mask)
        attention = self.cross_post(attention)
        return (attention + text_encoding.transpose(1, 2)) / math.sqrt(2.0)

    def forward(self, texts, text_lengths):
        encoding, _, _ = self.text_encoder(texts, text_lengths)
        style = self.style_encoder(encoding, text_lengths)
        # bert_encoding = self.bert_encoder(bert_in)
        encoding = rearrange(encoding, "b t c -> b c t")
        mask = torch.unsqueeze(sequence_mask(text_lengths, encoding.size(1)), 1).to(
            encoding.dtype
        )
        prosody = self.compute_cross(encoding, style, mask)
        for block in self.conv_next:
            prosody = block(prosody, style)
            prosody = prosody * mask
            prosody = self.dropout(prosody)
        prosody = rearrange(prosody, "b c t -> b t c")
        duration = self.duration_proj(prosody)
        rest = torch.abs(duration)[:, :, 1:]
        duration = torch.cat([duration[:, :, :1], rest], dim=2)
        duration = torch.cumsum(duration, dim=2)
        duration = -torch.abs(duration)
        return duration


class ConvNeXtBlock(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        style_dim,
    ):
        super().__init__()
        self.dwconv = torch.nn.Conv1d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv

        self.norm = AdaptiveLayerNorm(style_dim, dim, eps=1e-6)
        self.pwconv1 = torch.nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = torch.nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = torch.nn.Linear(intermediate_dim, dim)

    def forward(self, x, style):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x, style)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class GRN(torch.nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = torch.nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
