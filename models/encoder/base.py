"""Common interface for image encoders that preserve spatial information."""

from abc import ABC, abstractmethod

import torch
from torch import nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def spatial_pair(value):
    pair = (value, value) if isinstance(value, int) else tuple(value)
    if len(pair) != 2 or any(int(v) != v or v <= 0 for v in pair):
        raise ValueError(f"Expected a positive (height, width), got {value}")
    return tuple(int(v) for v in pair)


class SpatialEncoder(nn.Module, ABC):
    """Map floating RGB images [B, 3, H, W] in [-1, 1] to [B, N, D].

    Tokens are ordered row by row, with N = grid_height * grid_width.
    CLS/register tokens and global pooling are excluded. Metadata is available
    at construction time, without running a sample image through the backbone.
    Subclasses own preprocessing and implement ``forward_tokens``.
    """

    latent_ndim = 2

    def __init__(self, name, emb_dim, grid_size, input_size, mean, std,
                 interpolation=InterpolationMode.BICUBIC):
        super().__init__()
        self.name = name
        self.emb_dim = int(emb_dim)
        if self.emb_dim <= 0:
            raise ValueError("emb_dim must be positive")
        self.grid_size = spatial_pair(grid_size)
        self.input_size = spatial_pair(input_size)
        self.interpolation = interpolation
        self.register_buffer("image_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    @property
    def num_patches(self):
        return self.grid_size[0] * self.grid_size[1]

    def preprocess(self, images):
        # The dataset range is also used for reconstruction targets and plots.
        # Convert it here so every backbone receives its own normalization.
        images = (images + 1.0) / 2.0
        images = TF.resize(images, list(self.input_size),
                           interpolation=self.interpolation, antialias=True)
        return (images - self.image_mean.to(images)) / self.image_std.to(images)

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
            raise ValueError("Expected floating RGB images [B, 3, H, W] in [-1, 1]")
        tokens = self.forward_tokens(self.preprocess(images))
        expected = (images.shape[0], self.num_patches, self.emb_dim)
        if not isinstance(tokens, torch.Tensor) or tuple(tokens.shape) != expected:
            raise ValueError(f"{self.name} must return spatial tokens with shape {expected}")
        return tokens

    @abstractmethod
    def forward_tokens(self, images):
        """Return spatial tokens from images already preprocessed by this encoder."""


def encoder_metadata(encoder):
    """Read metadata through a distributed wrapper without bypassing its forward."""
    while hasattr(encoder, "module"):
        encoder = encoder.module
    if not isinstance(encoder, SpatialEncoder) or not hasattr(encoder, "grid_size"):
        raise TypeError("Visual encoders must implement SpatialEncoder; legacy pooled "
                        "encoders/checkpoints must be migrated before use")
    return encoder
