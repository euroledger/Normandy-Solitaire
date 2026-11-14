
import math

import torch
from torch import nn
from torch.nn import functional as F


class NoisyLinear(nn.Module):
    # Factorised Gaussian NoisyNet linear layer.

    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma0 = sigma0

        # Trainable mean parameters.
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.bias_mu = nn.Parameter(
            torch.empty(out_features)
        )

        # Trainable noise-scale parameters.
        self.weight_sigma = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.bias_sigma = nn.Parameter(
            torch.empty(out_features)
        )

        # Noise buffers are not trainable.
        self.register_buffer(
            "weight_epsilon",
            torch.zeros(out_features, in_features),
            persistent=False,
        )
        self.register_buffer(
            "bias_epsilon",
            torch.zeros(out_features),
            persistent=False,
        )

        self.noise_enabled = True

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)

        nn.init.uniform_(
            self.weight_mu, -bound, bound
        )
        nn.init.uniform_(
            self.bias_mu, -bound, bound
        )

        nn.init.constant_(
            self.weight_sigma,
            self.sigma0 / math.sqrt(self.in_features),
        )
        nn.init.constant_(
            self.bias_sigma,
            self.sigma0 / math.sqrt(self.out_features),
        )

    @staticmethod
    def scale_noise(size, device, dtype):
        noise = torch.randn(
            size, device=device, dtype=dtype
        )

        return noise.sign() * noise.abs().sqrt()

    @torch.no_grad()
    def reset_noise(self):
        # Sample independent input and output noise.
        epsilon_in = self.scale_noise(
            self.in_features,
            self.weight_mu.device,
            self.weight_mu.dtype,
        )

        epsilon_out = self.scale_noise(
            self.out_features,
            self.weight_mu.device,
            self.weight_mu.dtype,
        )

        # Factorised outer product.
        self.weight_epsilon.copy_(
            epsilon_out.outer(epsilon_in)
        )
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.noise_enabled:
            weight = (
                self.weight_mu
                + self.weight_sigma * self.weight_epsilon
            )
            bias = (
                self.bias_mu
                + self.bias_sigma * self.bias_epsilon
            )
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def enable_noise(self):
        self.noise_enabled = True

    def disable_noise(self):
        self.noise_enabled = False
