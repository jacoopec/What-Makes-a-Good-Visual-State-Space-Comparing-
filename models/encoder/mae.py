import torch

from transformers import ViTMAEConfig, ViTMAEModel

from models.encoder.base import SpatialEncoder, spatial_pair


class MAEEncoder(SpatialEncoder):
    def __init__(
        self,
        model_name="facebook/vit-mae-base",
        cache_dir="/homes/jpecchini/.cache/huggingface",
        local_files_only=True,
    ):
        config = ViTMAEConfig.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        # Keep every patch
        config.mask_ratio = 0.0

        size = spatial_pair(config.image_size)
        patch = spatial_pair(config.patch_size)
        grid = tuple(s // p for s, p in zip(size, patch))
        super().__init__(model_name, config.hidden_size, grid, size,
                         (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.base_model = ViTMAEModel.from_pretrained(
            model_name,
            config=config,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        self.patch_size = config.patch_size

        # Freeze MAE
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

    def forward_tokens(self, images):
        batch_size = images.shape[0]

        # IMPORTANT:
        # mask_ratio=0 still invokes random_masking().
        # Ordered noise keeps all patches in spatial order.
        noise = torch.arange(
            self.num_patches,
            device=images.device,
            dtype=torch.float32,
        )

        noise = noise.unsqueeze(0).expand(
            batch_size, -1
        )

        outputs = self.base_model(
            pixel_values=images,
            noise=noise,
            interpolate_pos_encoding=False,
            return_dict=True,
        )

        # Exclude CLS, leaving spatial patches in row-major order.
        return outputs.last_hidden_state[:, 1:, :]
