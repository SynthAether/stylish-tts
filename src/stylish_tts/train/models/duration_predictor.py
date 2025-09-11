import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from einops import rearrange
from .common import LinearNorm
from .text_encoder import TextEncoder
from .text_style_encoder import TextStyleEncoder
from .prosody_encoder import ProsodyEncoder
from .hybrid_dropout import HybridDropout
from ..utils import length_to_mask


class DurationPredictor(torch.nn.Module):
    def __init__(
        self, style_dim, inter_dim, text_config, style_config, duration_config
    ):
        super().__init__()
        self.text_encoder = TextEncoder(inter_dim=inter_dim, config=text_config)
        self.style_encoder = TextStyleEncoder(
            inter_dim,
            style_dim,
            config=style_config,
        )
        self.prosody_encoder = ProsodyEncoder(
            sty_dim=style_dim,
            d_model=inter_dim,
            nlayers=duration_config.n_layer,
            dropout=duration_config.dropout,
        )
        # self.dropout = torch.nn.Dropout(duration_config.last_dropout)
        self.dropout = HybridDropout(p=duration_config.last_dropout, beta=0.2)
        self.duration_proj = LinearNorm(
            inter_dim + style_dim, duration_config.duration_classes
        )

    def forward(self, texts, text_lengths):
        encoding, _, _ = self.text_encoder(texts, text_lengths)
        style = self.style_encoder(encoding, text_lengths)
        prosody = self.prosody_encoder(encoding, style, text_lengths)
        mask = ~length_to_mask(text_lengths, prosody.shape[1]).unsqueeze(2)
        prosody = prosody * mask
        if self.training:
            # batch_size = texts.shape[0]
            prosody = pack_padded_sequence(
                prosody, text_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            # prosody = rearrange(prosody, "b k c -> (b k) c")
            prosody = PackedSequence(
                data=self.dropout(prosody.data),
                batch_sizes=prosody.batch_sizes,
                sorted_indices=prosody.sorted_indices,
                unsorted_indices=prosody.unsorted_indices,
            )
            # prosody = rearrange(prosody, "(b k) c -> b k c", b=batch_size)
            prosody, _ = pad_packed_sequence(prosody, batch_first=True)
        # prosody = self.dropout(prosody)
        duration = self.duration_proj(prosody)
        return duration
