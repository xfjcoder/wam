"""A small MLP dynamics model: predicts the next 33-dim state given the current state and
the 3-dim velocity-command action (see wam/state.py). Predicts a DELTA (next - current),
not the raw next state directly -- standard practice for dynamics models, since most state
components (joint angles, base height, etc.) change only a little per 50ms macro-step, so a
delta is a much easier regression target than the raw value.

Input/output normalization is built into the model itself (as buffers, so it saves/loads
with the checkpoint automatically) rather than handled ad hoc by the caller -- one less
thing a training or eval script can get subtly wrong by normalizing differently.
"""
import torch
import torch.nn as nn


class WorldModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256, n_hidden: int = 3):
        super().__init__()
        layers = []
        in_dim = state_dim + action_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, state_dim)]
        self.net = nn.Sequential(*layers)

        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))
        self.register_buffer("delta_mean", torch.zeros(state_dim))
        self.register_buffer("delta_std", torch.ones(state_dim))

    def fit_normalization(self, states: torch.Tensor, actions: torch.Tensor, deltas: torch.Tensor):
        self.state_mean.copy_(states.mean(0))
        self.state_std.copy_(states.std(0).clamp_min(1e-6))
        self.action_mean.copy_(actions.mean(0))
        self.action_std.copy_(actions.std(0).clamp_min(1e-6))
        self.delta_mean.copy_(deltas.mean(0))
        self.delta_std.copy_(deltas.std(0).clamp_min(1e-6))

    def _normalized_input(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        s_n = (state - self.state_mean) / self.state_std
        a_n = (action - self.action_mean) / self.action_std
        return torch.cat([s_n, a_n], dim=-1)

    def predict_delta_normalized(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Normalized delta -- what the network actually outputs; used directly in the
        training loss (comparing against the normalized target delta) rather than
        unnormalizing first and losing the benefit of a well-scaled loss."""
        return self.net(self._normalized_input(state, action))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns the predicted NEXT state (state + unnormalized predicted delta) -- what a
        caller doing rollouts actually wants."""
        delta_n = self.predict_delta_normalized(state, action)
        delta = delta_n * self.delta_std + self.delta_mean
        return state + delta
