"""One-shot segmentation model: frozen DINOv2 + fusion + decoder."""

from __future__ import annotations

from contextlib import nullcontext

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
        backbone_frozen: bool = True,
    ):
        super().__init__()
        self.backbone = backbone          # None when using cached features
        self.fusion = fusion
        self.decoder = decoder
        self.target_size = target_size
        self.backbone_frozen = backbone_frozen

    def _to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 3:
            return DINOv2Backbone.tokens_to_spatial(tokens)
        if tokens.ndim == 4:
            batch, supports, num_patches, channels = tokens.shape
            spatial = DINOv2Backbone.tokens_to_spatial(tokens.reshape(batch * supports, num_patches, channels))
            _, spatial_channels, height, width = spatial.shape
            return spatial.reshape(batch, supports, spatial_channels, height, width)
        raise ValueError(f"Expected token tensor with 3 or 4 dims, got shape={tuple(tokens.shape)}")

    def _extract_spatial_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.backbone is None:
            raise RuntimeError("Backbone is required for live-image mode.")

        if images.ndim == 4:
            tokens = self.backbone.extract_patch_tokens(images)[-1]
            return self._to_spatial(tokens)

        if images.ndim == 5:
            batch, supports, channels, height, width = images.shape
            flat_images = images.reshape(batch * supports, channels, height, width)
            tokens = self.backbone.extract_patch_tokens(flat_images)[-1]
            spatial = self._to_spatial(tokens)
            _, spatial_channels, feat_h, feat_w = spatial.shape
            return spatial.reshape(batch, supports, spatial_channels, feat_h, feat_w)

        raise ValueError(f"Expected image tensor with 4 or 5 dims, got shape={tuple(images.shape)}")

    def forward(self, batch: dict) -> torch.Tensor:
        """Return logits [B, 1, H_target, W_target]."""
        use_cached = "support_feat" in batch

        if use_cached:
            s_spatial = self._to_spatial(batch["support_feat"])
            q_spatial = self._to_spatial(batch["query_feat"])
        else:
            grad_context = torch.no_grad if self.backbone_frozen else nullcontext
            with grad_context():
                s_spatial = self._extract_spatial_features(batch["support_img"])
                q_spatial = self._extract_spatial_features(batch["query_img"])

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
    backbone_frozen: bool = True,
) -> OneShotModel:
    """Build a complete model from config parameters."""
    embed_dim = SIZE_TO_DIM[backbone_size]

    # Backbone (only needed when NOT using cached features)
    backbone = None
    if not use_cached:
        backbone = DINOv2Backbone(size=backbone_size, device=device, frozen=backbone_frozen)

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
        backbone_frozen=backbone_frozen,
    )
    return model.to(device)
