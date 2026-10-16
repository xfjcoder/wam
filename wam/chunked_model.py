"""Stage 6: a history-conditioned, action-chunked world model -- predicts a whole CHUNK
of H future states in ONE non-autoregressive forward pass, given a short window of past
(state, action) history plus the H future actions, instead of calling a single-step model
(wam/model.py's WorldModel, used by every earlier stage) H times autoregressively.

Borrows two ideas central to modern "action chunking" robot-learning architectures
(Physical Intelligence's pi0 and similar -- though those are POLICY models, mapping an
observation to an action chunk, not dynamics models; this applies the same two ideas to
DYNAMICS prediction instead):
  1. Condition on a short HISTORY of past (state, action) pairs, not just the
     instantaneous state -- lets the model implicitly infer context the instantaneous
     33-dim state doesn't carry (e.g. roughly where in the gait cycle it currently is).
  2. Predict a whole CHUNK of future outcomes jointly from a chunk of future actions,
     rather than one step at a time -- replaces H sequential autoregressive calls (each
     re-injecting the model's own, possibly-imperfect prediction as the next input) with
     ONE batched call.

Token sequence fed to a small causal transformer, length `history_len + 1 + chunk_len`:
  - `history_len` tokens, each a PAST (state, action) pair (type 0).
  - 1 token: the CURRENT state the chunk is rooted at, no action yet (type 1) -- the
    pivot every future prediction's delta is measured relative to (see forward()).
  - `chunk_len` tokens, each a FUTURE action alone, no state (type 2) -- these are what
    the planner/caller is asking "what happens if I do THIS".
A causal mask means each future-action token can attend to all history/current tokens
and every EARLIER future-action token, but not later ones -- the predicted state at
chunk position h is a function of actions 0..h, never a later action in the same chunk
(matches physical causality: what happens at step h can't depend on an action not taken
yet), the same discipline an autoregressive model gets "for free" from its own structure.
"""
import torch
import torch.nn as nn


class ChunkedWorldModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, history_len: int, chunk_len: int,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 2, d_ff: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_len = history_len
        self.chunk_len = chunk_len
        total_len = history_len + 1 + chunk_len

        self.history_embed = nn.Linear(state_dim + action_dim, d_model)
        self.current_embed = nn.Linear(state_dim, d_model)
        self.future_embed = nn.Linear(action_dim, d_model)
        self.type_embed = nn.Embedding(3, d_model)      # 0=history, 1=current, 2=future
        self.pos_embed = nn.Embedding(total_len, d_model)

        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True,
                                            activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.out_head = nn.Linear(d_model, state_dim)

        mask = torch.triu(torch.full((total_len, total_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask)
        type_ids = torch.cat([torch.zeros(history_len, dtype=torch.long),
                               torch.ones(1, dtype=torch.long),
                               torch.full((chunk_len,), 2, dtype=torch.long)])
        self.register_buffer("type_ids", type_ids)
        self.register_buffer("positions", torch.arange(total_len))

        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))
        self.register_buffer("chunk_delta_mean", torch.zeros(state_dim))
        self.register_buffer("chunk_delta_std", torch.ones(state_dim))

    def fit_normalization(self, hist_states, hist_actions, current_state, future_actions,
                           future_states):
        all_states = torch.cat([hist_states.reshape(-1, self.state_dim),
                                 current_state.reshape(-1, self.state_dim)], dim=0)
        all_actions = torch.cat([hist_actions.reshape(-1, self.action_dim),
                                  future_actions.reshape(-1, self.action_dim)], dim=0)
        chunk_delta = future_states - current_state.unsqueeze(1)
        self.state_mean.copy_(all_states.mean(0))
        self.state_std.copy_(all_states.std(0).clamp_min(1e-6))
        self.action_mean.copy_(all_actions.mean(0))
        self.action_std.copy_(all_actions.std(0).clamp_min(1e-6))
        self.chunk_delta_mean.copy_(chunk_delta.reshape(-1, self.state_dim).mean(0))
        self.chunk_delta_std.copy_(chunk_delta.reshape(-1, self.state_dim).std(0).clamp_min(1e-6))

    def _n_state(self, s):
        return (s - self.state_mean) / self.state_std

    def _n_action(self, a):
        return (a - self.action_mean) / self.action_std

    def predict_chunk_normalized(self, hist_states, hist_actions, current_state, future_actions):
        """hist_states: (B,K,S), hist_actions: (B,K,A), current_state: (B,S),
        future_actions: (B,H,A) -> normalized chunk deltas (B,H,S), each relative to
        current_state (NOT to the previous predicted step -- there is no previous
        predicted step, this is computed in one shot)."""
        B = hist_states.shape[0]
        hist_in = torch.cat([self._n_state(hist_states), self._n_action(hist_actions)], dim=-1)
        hist_tok = self.history_embed(hist_in)                                  # (B,K,d)
        cur_tok = self.current_embed(self._n_state(current_state)).unsqueeze(1)   # (B,1,d)
        fut_tok = self.future_embed(self._n_action(future_actions))              # (B,H,d)

        tokens = torch.cat([hist_tok, cur_tok, fut_tok], dim=1)                  # (B,K+1+H,d)
        tokens = tokens + self.pos_embed(self.positions).unsqueeze(0) \
                         + self.type_embed(self.type_ids).unsqueeze(0)

        out = self.encoder(tokens, mask=self.causal_mask)
        fut_out = out[:, self.history_len + 1:, :]                               # (B,H,d)
        return self.out_head(fut_out)

    def forward(self, hist_states, hist_actions, current_state, future_actions):
        """Returns the H predicted FUTURE states (B, H, state_dim) -- what a caller doing
        a chunked rollout actually wants."""
        delta_n = self.predict_chunk_normalized(hist_states, hist_actions, current_state,
                                                 future_actions)
        delta = delta_n * self.chunk_delta_std + self.chunk_delta_mean
        return current_state.unsqueeze(1) + delta
