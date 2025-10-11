import torch
from torch.nn.utils.parametrizations import weight_norm
from .ada_norm import AdaptiveDecoderBlock


class Decoder(torch.nn.Module):
    def __init__(
        self,
        *,
        dim_in,
        style_dim,
        dim_out,
        hidden_dim,
        residual_dim,
    ):
        super().__init__()

        self.encode = AdaptiveDecoderBlock(
            dim_in=dim_in + 3,
            dim_out=hidden_dim,
            style_dim=style_dim,
        )

        self.decode = torch.nn.ModuleList(
            [
                AdaptiveDecoderBlock(
                    dim_in=hidden_dim + 3 + residual_dim,
                    dim_out=hidden_dim,
                    style_dim=style_dim,
                )
                for _ in range(4)
            ]
        )

        self.F0_conv = weight_norm(
            torch.nn.Conv1d(1, 1, kernel_size=3, stride=1, groups=1, padding=1)
        )

        self.N_conv = weight_norm(
            torch.nn.Conv1d(1, 1, kernel_size=3, stride=1, groups=1, padding=1)
        )

        self.voiced_conv = weight_norm(
            torch.nn.Conv1d(1, 1, kernel_size=3, stride=1, groups=1, padding=1)
        )

        self.asr_res = torch.nn.Sequential(
            weight_norm(torch.nn.Conv1d(dim_in, residual_dim, kernel_size=1))
        )

    def forward(self, asr, F0_curve, N, s, voiced):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N = self.N_conv(N.unsqueeze(1))
        voiced = self.voiced_conv(voiced.unsqueeze(1))

        x = torch.cat([asr, F0, N, voiced], axis=1)
        x = self.encode(x, s)

        asr_res = self.asr_res(asr)

        for block in self.decode:
            x = torch.cat([x, asr_res, F0, N, voiced], axis=1)
            x = block(x, s)

        return x, F0_curve
