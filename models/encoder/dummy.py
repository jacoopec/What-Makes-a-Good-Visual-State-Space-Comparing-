from torch import nn

from models.encoder.base import SpatialEncoder, spatial_pair


class DummySpatialEncoder(SpatialEncoder):
    """Small trainable patch projection for the existing dummy configuration."""

    def __init__(self, emb_dim=7, input_size=224, patch_size=16):
        size, patch = spatial_pair(input_size), spatial_pair(patch_size)
        if any(s % p for s, p in zip(size, patch)):
            raise ValueError("input_size must be divisible by patch_size")
        super().__init__("dummy", emb_dim, tuple(s // p for s, p in zip(size, patch)),
                         size, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.projection = nn.Conv2d(3, emb_dim, kernel_size=patch, stride=patch)

    def forward_tokens(self, images):
        return self.projection(images).flatten(2).transpose(1, 2)
