"""Spatial token adapter for the optional OpenAI CLIP ViT package."""

import torch

from models.encoder.base import SpatialEncoder


class CLIPVisionEncoder(SpatialEncoder):
    def __init__(self, name="ViT-B/16", return_projected_tokens=True,
                 layer=None, download_root=None):
        try:
            import DINOWM_COD.dino_wm2.dino_wm.models.encoder.clip as clip
        except ImportError as exc:
            raise ImportError("CLIPVisionEncoder requires OpenAI CLIP: "
                              "pip install git+https://github.com/openai/CLIP.git") from exc
        model, _ = clip.load(name, device="cpu", jit=False, download_root=download_root)
        visual = model.visual.float()
        if not hasattr(visual, "transformer"):
            raise ValueError("CLIPVisionEncoder requires a ViT backbone, e.g. ViT-B/16")
        depth = len(visual.transformer.resblocks)
        if layer is not None and not 0 <= layer < depth:
            raise ValueError(f"layer must be a zero-based block index in [0, {depth})")
        size = int(visual.input_resolution)
        patch = tuple(visual.conv1.kernel_size)
        grid = tuple(size // p for p in patch)
        dim = visual.conv1.out_channels
        if return_projected_tokens and visual.proj is not None:
            dim = visual.proj.shape[1]
        super().__init__(name, dim, grid, size,
                         (0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711))
        self.visual = visual
        self.patch_size = patch
        self.return_projected_tokens = return_projected_tokens
        self.layer = layer

    def forward_tokens(self, images):
        visual = self.visual
        tokens = visual.conv1(images.to(visual.conv1.weight.dtype))
        tokens = tokens.flatten(2).transpose(1, 2)
        cls = visual.class_embedding.to(tokens).view(1, 1, -1)
        tokens = torch.cat([cls.expand(tokens.shape[0], -1, -1), tokens], dim=1)
        tokens = visual.ln_pre(tokens + visual.positional_embedding.to(tokens))
        tokens = tokens.transpose(0, 1)
        if self.layer is None:
            tokens = visual.transformer(tokens)
        else:
            for index, block in enumerate(visual.transformer.resblocks):
                tokens = block(tokens)
                if index == self.layer:
                    break
        # Apply CLIP's final normalization/projection independently to patches.
        tokens = visual.ln_post(tokens.transpose(0, 1)[:, 1:, :])
        if self.return_projected_tokens and visual.proj is not None:
            tokens = tokens @ visual.proj
        return tokens
