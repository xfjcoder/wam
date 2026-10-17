"""Pixel-rendering mini-plan, step 6: "Stage 1 for pixels" -- does the pixel encoder's latent
(wam/pixel_encoder.py's ConvAutoencoder, step 5) support PREDICTION, not just reconstruction?

`LatentDynamics` is the same idea as wam/model.py's `WorldModel` (predict a DELTA, not the
raw next value; built-in input/output normalization as buffers), just over the encoder's
32-dim latent instead of the 33-dim physical state. The encoder stays FROZEN here -- this
step only asks whether a small predictor can move around in that already-learned latent
space, not whether jointly training an encoder+dynamics model end-to-end would do better
(a bigger, separate question for later if this works at all).
"""
import numpy as np
import torch
import torch.nn as nn


def build_window_starts(episode_ids: np.ndarray, horizon: int) -> np.ndarray:
    """Start indices `i` such that frames i, i+1, ..., i+horizon all belong to the same
    episode -- i.e. an H-step pixel rollout from `i` (predicting frames i+1..i+horizon) never
    needs a frame from a different episode.

    NOTE this is a frame-ADJACENCY check, unlike train_world_model.py's build_window_starts
    over STATE data: there, each transition already carries its own next_state, so a
    horizon-H state rollout from `i` only needs actions/next_states at i..i+H-1 within the
    segment (no requirement that row i+1 itself be "the next frame"). The pixel dataset has
    no such per-row next_frame field -- frame[i+1] IS frame[i]'s next frame only because
    collect_data.py / collect_wall_approach_demo.py capture one frame per transition,
    back-to-back within an episode -- hence the stricter off-by-one here."""
    starts = []
    n = len(episode_ids)
    i = 0
    while i < n:
        j = i
        while j < n and episode_ids[j] == episode_ids[i]:
            j += 1
        last_valid_start = j - 1 - horizon
        if last_valid_start >= i:
            starts.extend(range(i, last_valid_start + 1))
        i = j
    return np.array(starts, dtype=np.int64)


def consecutive_pairs(episode_ids: np.ndarray) -> np.ndarray:
    """Indices i such that (i, i+1) are consecutive transitions of the SAME episode --
    i.e. frame[i+1] really is frame[i]'s next frame, not some other episode's first frame.
    The horizon=1 case of build_window_starts above, used by every pixel-dynamics training
    script for its one-step loss."""
    return build_window_starts(episode_ids, 1)


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
