from transformers import SiglipVisionModel

from models.encoder.base import SpatialEncoder, spatial_pair


class SigLIPEncoder(SpatialEncoder):
    def __init__(
        self,
        model_name="google/siglip-base-patch16-224",
        cache_dir="/homes/jpecchini/.cache/huggingface",
        local_files_only=True,
    ):
        base_model = SiglipVisionModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        config = base_model.config
        size = spatial_pair(config.image_size)
        patch = spatial_pair(config.patch_size)
        grid = tuple(s // p for s, p in zip(size, patch))
        super().__init__(model_name, config.hidden_size, grid, size,
                         (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.base_model = base_model
        self.patch_size = config.patch_size

        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

    def forward_tokens(self, images):
        outputs = self.base_model(
            pixel_values=images,
            return_dict=True,
        )

        # SigLIP has no CLS token; every output token represents a spatial patch.
        return outputs.last_hidden_state
