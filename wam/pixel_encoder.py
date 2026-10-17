"""Pixel-rendering mini-plan, step 5: the first actual check that the pixel stream
collect_data.py's --record-pixels now produces contains learnable structure at all -- a
small convolutional autoencoder (encode RGB frame -> low-dim latent -> decode back to RGB),
trained purely on reconstruction loss. This is deliberately NOT a dynamics model or a planner
input yet: it's the minimum useful test of "is there a compact representation of this
specific camera's view that a network can actually learn", before committing to anything
that predicts FORWARD in time from it.

Small by design (64x64 input, a handful of conv layers, latent_dim default 32) -- this
dataset is a few thousand frames from a handful of episodes, not worth a large model, and a
small model's reconstruction quality is a more honest test of "is the signal learnable" than
a large model's (which could memorize instead).
"""
import numpy as np
import torch
import torch.nn as nn


def to_tensor_01(pixels_u8: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float32 in [0, 1]."""
    t = torch.as_tensor(pixels_u8, dtype=torch.float32) / 255.0
    return t.permute(0, 3, 1, 2).contiguous()


def depth_to_tensor_01(depth_f32: np.ndarray, clip: float = 3.0) -> torch.Tensor:
    """(N, H, W) float32 metric depth (meters) -> (N, 1, H, W) float32 in [0, 1].

    MuJoCo's depth renderer returns real metric distance along the camera ray, not a
    normalized z-buffer -- pixels with no geometry in range (sky/open space) read back as
    the far-plane distance, which can be two orders of magnitude larger than anything in the
    scene (confirmed directly: the flat-ground dataset has a max of 200m against a p99 of
    3.0m). Clipping to `clip` meters before normalizing keeps those outliers from swamping
    the loss -- the same role check_onboard_camera.py's [0, 1.2] display clip served for
    human viewing, applied here for training."""
    t = torch.as_tensor(depth_f32, dtype=torch.float32).clamp(max=clip) / clip
    return t.unsqueeze(1).contiguous()


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 32, resolution: int = 64, channels: int = 3):
        super().__init__()
        assert resolution % 16 == 0, "encoder downsamples by 16x (4 stride-2 conv layers)"
        self.resolution = resolution
        self.latent_dim = latent_dim
        self.channels = channels
        bottleneck = resolution // 16

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 4, stride=2, padding=1), nn.ReLU(),  # res/2
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),        # res/4
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),       # res/8
            nn.Conv2d(128, 128, 4, stride=2, padding=1), nn.ReLU(),      # res/16
        )
        self.to_latent = nn.Linear(128 * bottleneck * bottleneck, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 128 * bottleneck * bottleneck)
        self._bottleneck = bottleneck

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, channels, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def encode(self, rgb_01: torch.Tensor) -> torch.Tensor:
        """rgb_01: (B, channels, H, W) float in [0, 1]."""
        feat = self.encoder_conv(rgb_01)
        return self.to_latent(feat.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        b = self._bottleneck
        feat = self.from_latent(z).view(-1, 128, b, b)
        return self.decoder_conv(feat)

    def forward(self, rgb_01: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(rgb_01))
