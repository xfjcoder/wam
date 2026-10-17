"""Pixel-rendering mini-plan, step 6: "Stage 1 for pixels" -- does the pixel encoder's latent
(wam/pixel_encoder.py's ConvAutoencoder, step 5) support PREDICTION, not just reconstruction?

`LatentDynamics` is the same idea as wam/model.py's `WorldModel` (predict a DELTA, not the
raw next value; built-in input/output normalization as buffers), just over the encoder's
32-dim latent instead of the 33-dim physical state. The encoder stays FROZEN here -- this
step only asks whether a small predictor can move around in that already-learned latent
space, not whether jointly training an encoder+dynamics model end-to-end would do better
(a bigger, separate question for later if this works at all).
"""
import torch
import torch.nn as nn


class LatentDynamics(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 128, n_hidden: int = 2):
        super().__init__()
        layers = []
        in_dim = latent_dim + action_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, latent_dim)]
        self.net = nn.Sequential(*layers)

        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_std", torch.ones(latent_dim))
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))
        self.register_buffer("delta_mean", torch.zeros(latent_dim))
        self.register_buffer("delta_std", torch.ones(latent_dim))

    def fit_normalization(self, latents: torch.Tensor, actions: torch.Tensor, deltas: torch.Tensor):
        self.latent_mean.copy_(latents.mean(0))
        self.latent_std.copy_(latents.std(0).clamp_min(1e-6))
        self.action_mean.copy_(actions.mean(0))
        self.action_std.copy_(actions.std(0).clamp_min(1e-6))
        self.delta_mean.copy_(deltas.mean(0))
        self.delta_std.copy_(deltas.std(0).clamp_min(1e-6))

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns the predicted NEXT latent (not the delta) -- callers never need to
        remember to un-normalize and add it back themselves."""
        latent_n = (latent - self.latent_mean) / self.latent_std
        action_n = (action - self.action_mean) / self.action_std
        delta_n = self.net(torch.cat([latent_n, action_n], dim=-1))
        delta = delta_n * self.delta_std + self.delta_mean
        return latent + delta
