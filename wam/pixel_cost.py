"""Pixel-rendering mini-plan, step 15: a small head that reads a planning-relevant cost
DIRECTLY off a latent, so CEM-style scoring never has to decode a predicted latent back to a
full depth image. The dynamics rollout itself (wam/pixel_dynamics.py's LatentDynamics) is
already cheap -- a 32-dim latent is smaller than the state-based WorldModel's 33-dim state, so
the per-candidate MLP math costs about the same either way. The expensive step would only be
decoding every candidate's every predicted step back to pixels/depth to compute something
interpretable from it; this head removes that step from the loop entirely.

Same predict-a-scalar-with-built-in-normalization pattern as wam/model.py's WorldModel and
wam/pixel_dynamics.py's LatentDynamics, just mapping latent -> a single number instead of
latent -> next latent.
"""
import torch
import torch.nn as nn


class LatentCostHead(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 64, n_hidden: int = 2):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_std", torch.ones(latent_dim))
        self.register_buffer("target_mean", torch.zeros(1))
        self.register_buffer("target_std", torch.ones(1))

    def fit_normalization(self, latents: torch.Tensor, targets: torch.Tensor):
        self.latent_mean.copy_(latents.mean(0))
        self.latent_std.copy_(latents.std(0).clamp_min(1e-6))
        self.target_mean.copy_(targets.mean(0))
        self.target_std.copy_(targets.std(0).clamp_min(1e-6))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Returns the predicted cost in its original (physical) units -- callers never need
        to remember to un-normalize themselves."""
        latent_n = (latent - self.latent_mean) / self.latent_std
        pred_n = self.net(latent_n)
        return pred_n * self.target_std + self.target_mean
