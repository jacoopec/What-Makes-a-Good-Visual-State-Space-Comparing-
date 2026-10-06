"""Spatial patch-token adapter for Hugging Face I-JEPA."""

from models.encoder.base import SpatialEncoder, spatial_pair


class IJEPAEncoder(SpatialEncoder):
    def __init__(self, model_name="facebook/ijepa_vith14_1k", cache_dir=None,
                 local_files_only=False):
        try:
            from transformers import IJepaModel
        except ImportError as exc:
            raise ImportError(
                "IJEPAEncoder requires transformers>=4.51.3; "
                "install it with `pip install 'transformers>=4.51.3'`"
            ) from exc

        model = IJepaModel.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=local_files_only
        )
        config = model.config
        size = spatial_pair(config.image_size)
        patch = spatial_pair(config.patch_size)
        if any(s % p for s, p in zip(size, patch)):
            raise ValueError("I-JEPA image_size must be divisible by patch_size")

        grid = tuple(s // p for s, p in zip(size, patch))
        super().__init__(
            model_name, config.hidden_size, grid, size,
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        )
        self.base_model = model
        self.patch_size = patch

    def forward_tokens(self, images):
        outputs = self.base_model(
            pixel_values=images,
            interpolate_pos_encoding=False,
            return_dict=True,
        )
        return outputs.last_hidden_state
