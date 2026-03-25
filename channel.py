import torch
import torch.nn as nn
import numpy as np


class Channel(nn.Module):
    """
    JSCC channel model, support Rayleigh and AWGN
    """

    def __init__(self, channel_type='awgn'):
        super().__init__()
        self.channel_type = channel_type.lower()

    def generate_noise(self, shape, std, device):
        noise_real = torch.randn(shape, device=device) * (std / np.sqrt(2))
        noise_imag = torch.randn(shape, device=device) * (std / np.sqrt(2))
        return torch.complex(noise_real, noise_imag).detach()

    def rayleigh_fading(self, shape, device):
        h = torch.sqrt(torch.randn(shape, device=device) ** 2 + torch.randn(shape, device=device) ** 2) / np.sqrt(2)
        return h.detach()

    def forward(self, x, snr, target_power=None):

        B, T, E = x.shape
        N = E // 2

        if E % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, got {E}")

        # transform complex number: [B, T, N]
        real_part = x[..., :N]
        imag_part = x[..., N:]
        x_complex = torch.complex(real_part, imag_part)

        # compute original power
        pwr_orig = torch.mean(torch.abs(x_complex) ** 2, dim=-1, keepdim=True)

        if target_power is None:
            target_power = torch.ones((B, T, 1), device=x.device)  # 单位功率
        elif isinstance(target_power, (int, float)):
            target_power = torch.full((B, T, 1), target_power, device=x.device)
        else:
            target_power = target_power.view(B, T, 1)

        scale_in = torch.sqrt(target_power / pwr_orig)
        x_norm = x_complex * scale_in

        snr_linear = 10 ** (snr / 10.0)
        noise_std = torch.sqrt(target_power / (2 * snr_linear)).clamp(min=1e-8)

        if self.channel_type == 'none':
            y_complex = x_norm

        elif self.channel_type == 'rayleigh':
            h = self.rayleigh_fading(x_norm.shape, x_norm.device)
            noise = self.generate_noise(x_norm.shape, noise_std, x_norm.device)
            y_complex = x_norm * h + noise

        else:
            noise = self.generate_noise(x_norm.shape, noise_std, x_norm.device)
            y_complex = x_norm + noise

        # return to real number
        y_real = torch.real(y_complex)  # [B, T, N]
        y_imag = torch.imag(y_complex)  # [B, T, N]

        y = torch.cat([y_real, y_imag], dim=-1) * torch.sqrt(pwr_orig)

        return noise_std, y
