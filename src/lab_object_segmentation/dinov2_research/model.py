"""One-shot segmentation model: frozen DINOv2 + fusion + decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DINOv2Backbone, SIZE_TO_DIM
from .fusion import FUSION_REGISTRY


class SegmentationDecoder(nn.Module):
    """Lightweight CNN decoder: C_in → num_classes."""

    def __init__(self, in_channels: int, mid_channels: int = 256, num_classes: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, num_classes, 1),
        )

    def forward(self, x):
        return self.net(x)


class OneShotModel(nn.Module):
    """Complete one-shot segmentation model.

    Supports two modes:
    - **cached**: receives pre-extracted patch tokens from DataLoader
    - **live**: receives raw images and runs DINOv2 backbone on the fly
    """

    def __init__(
        self,
        backbone: DINOv2Backbone | None,
        fusion: nn.Module,
        decoder: SegmentationDecoder,
        target_size: int = 322,
    ):
        super().__init__()
        self.backbone = backbone          # None when using cached features
        self.fusion = fusion
        self.decoder = decoder
        self.target_size = target_size

    def _to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        return DINOv2Backbone.tokens_to_spatial(tokens)

    def forward(self, batch: dict) -> torch.Tensor:
        """Return logits [B, 1, H_target, W_target]."""
        use_cached = "support_feat" in batch

        if use_cached:
            s_spatial = self._to_spatial(batch["support_feat"])
            q_spatial = self._to_spatial(batch["query_feat"])
        else:
            with torch.no_grad():
                s_tokens = self.backbone.extract_patch_tokens(batch["support_img"])[-1]
                q_tokens = self.backbone.extract_patch_tokens(batch["query_img"])[-1]
            s_spatial = self._to_spatial(s_tokens)
            q_spatial = self._to_spatial(q_tokens)

        support_mask = batch["support_mask"]

        # Fusion
        fused = self.fusion(q_spatial, s_spatial, support_mask)  # [B, C_out, H, W]

        # Decode
        logits = self.decoder(fused)  # [B, 1, H_feat, W_feat]

        # Upsample to target mask resolution
        logits = F.interpolate(
            logits,
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )
        return logits


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------
def build_model(
    backbone_size: str = "s",
    fusion_type: str = "film",
    decoder_channels: int = 256,
    num_classes: int = 1,
    img_size: int = 322,
    device: str | torch.device = "cpu",
    use_cached: bool = True,
) -> OneShotModel:
    """Build a complete model from config parameters."""
    embed_dim = SIZE_TO_DIM[backbone_size]

    # Backbone (only needed when NOT using cached features)
    backbone = None
    if not use_cached:
        backbone = DINOv2Backbone(size=backbone_size, device=device)

    # Fusion
    fusion_cls = FUSION_REGISTRY[fusion_type]
    fusion = fusion_cls(embed_dim)

    # Decoder
    decoder = SegmentationDecoder(
        in_channels=fusion.out_channels,
        mid_channels=decoder_channels,
        num_classes=num_classes,
    )

    model = OneShotModel(
        backbone=backbone,
        fusion=fusion,
        decoder=decoder,
        target_size=img_size,
    )
    return model.to(device)
