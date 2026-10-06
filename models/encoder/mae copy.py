"""Spatial patch-token adapter for Hugging Face ViT-MAE."""

import torch

from models.encoder.base import SpatialEncoder, spatial_pair


class MAEEncoder(SpatialEncoder):
    def __init__(self, model_name="facebook/vit-mae-base", cache_dir=None,
                 local_files_only=False):
        try:
            from transformers import ViTMAEConfig, ViTMAEModel
        except ImportError as exc:
            raise ImportError(
                "MAEEncoder requires transformers>=4.51.3; "
                "install it with `pip install 'transformers>=4.51.3'`"
            ) from exc

        config = ViTMAEConfig.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=local_files_only
        )
        config.mask_ratio = 0.0
        size = spatial_pair(config.image_size)
        patch = spatial_pair(config.patch_size)
        if any(s % p for s, p in zip(size, patch)):
            raise ValueError("MAE image_size must be divisible by patch_size")

        grid = tuple(s // p for s, p in zip(size, patch))
        super().__init__(
            model_name, config.hidden_size, grid, size,
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )
        self.base_model = ViTMAEModel.from_pretrained(
            model_name,
            config=config,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.patch_size = patch

    def forward_tokens(self, images):
        batch_size = images.shape[0]
        noise = torch.arange(self.num_patches, device=images.device)
        noise = noise.unsqueeze(0).expand(batch_size, -1)
        outputs = self.base_model(
            pixel_values=images,
            noise=noise,
            interpolate_pos_encoding=False,
            return_dict=True,
        )
        return outputs.last_hidden_state[:, 1:, :]
