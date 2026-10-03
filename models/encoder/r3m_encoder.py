from torch import nn

from models.encoder.base import SpatialEncoder, spatial_pair


class R3MEncoder(SpatialEncoder):
    """Expose the pretrained R3M convolutional map before global pooling."""

    def __init__(self, modelid="resnet18", input_size=224):
        # Keep R3M's optional loading dependencies local to this adapter.
        from models.encoder.r3m import load_r3m

        model = load_r3m(modelid)
        size = spatial_pair(input_size)
        super().__init__(f"r3m_{modelid}", model.outdim,
                         tuple((s + 31) // 32 for s in size), size,
                         (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.backbone = nn.Sequential(*list(model.convnet.children())[:-2])

    def forward_tokens(self, images):
        return self.backbone(images).flatten(2).transpose(1, 2)
