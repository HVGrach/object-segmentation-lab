"""Support→Query fusion methods for one-shot segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _BaseFusion(nn.Module):
    """Common helpers shared by all fusion methods."""

    out_channels: int  # set by subclass

    @staticmethod
    def _ensure_support_dims(
        support_feat: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if support_feat.ndim == 4:
            support_feat = support_feat.unsqueeze(1)
        elif support_feat.ndim != 5:
            raise ValueError(f"Expected support_feat to have 4 or 5 dims, got shape={tuple(support_feat.shape)}")

        if support_mask.ndim == 4:
            support_mask = support_mask.unsqueeze(1)
        elif support_mask.ndim != 5:
            raise ValueError(f"Expected support_mask to have 4 or 5 dims, got shape={tuple(support_mask.shape)}")

        return support_feat, support_mask

    @staticmethod
    def masked_avg_pool(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked average pooling → prototype vector.

        Parameters
        ----------
        feat : [B, C, H, W]
        mask : [B, 1, H_mask, W_mask]

        Returns
        -------
        [B, C]
        """
        feat, mask = _BaseFusion._ensure_support_dims(feat, mask)
        B, S, C, H, W = feat.shape
        mask = F.interpolate(mask.reshape(B * S, 1, mask.shape[-2], mask.shape[-1]), (H, W), mode="nearest")
        mask = mask.reshape(B, S, 1, H, W)

        numerator = (feat * mask).sum(dim=(3, 4))
        denominator = mask.sum(dim=(3, 4)).clamp(min=1e-6)
        per_support_proto = numerator / denominator

        valid_support = (mask.sum(dim=(3, 4)) > 0).float()
        support_weights = valid_support / valid_support.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (per_support_proto * support_weights).sum(dim=1)


# ------------------------------------------------------------------
# 1. Prototype + cosine similarity (simplest baseline)
# ------------------------------------------------------------------
class PrototypeCosine(_BaseFusion):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.out_channels = embed_dim

    def forward(self, query_feat, support_feat, support_mask):
        proto = self.masked_avg_pool(support_feat, support_mask)        # [B, C]
        p = F.normalize(proto, dim=1)[..., None, None]                 # [B, C, 1, 1]
        q = F.normalize(query_feat, dim=1)                             # [B, C, H, W]
        sim = (q * p).sum(dim=1, keepdim=True).clamp(min=0.0)         # [B, 1, H, W]
        return sim * query_feat                                        # [B, C, H, W]


# ------------------------------------------------------------------
# 2. Concatenation + 1×1 reduction
# ------------------------------------------------------------------
class ConcatFusion(_BaseFusion):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(2 * embed_dim, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.out_channels = embed_dim

    def forward(self, query_feat, support_feat, support_mask):
        proto = self.masked_avg_pool(support_feat, support_mask)
        p_exp = proto[..., None, None].expand_as(query_feat)
        return self.reduce(torch.cat([query_feat, p_exp], dim=1))


# ------------------------------------------------------------------
# 3. FiLM  — Feature-wise Linear Modulation (scale + shift)
# ------------------------------------------------------------------
class FiLMFusion(_BaseFusion):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.fc_gamma = nn.Linear(embed_dim, embed_dim)
        self.fc_beta = nn.Linear(embed_dim, embed_dim)
        self.out_channels = embed_dim

    def forward(self, query_feat, support_feat, support_mask):
        proto = self.masked_avg_pool(support_feat, support_mask)
        gamma = 1.0 + self.fc_gamma(proto)[..., None, None]
        beta = self.fc_beta(proto)[..., None, None]
        return gamma * query_feat + beta


# ------------------------------------------------------------------
# 4. 4D Correlation (dense patch matching)
# ------------------------------------------------------------------
class Correlation4D(_BaseFusion):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.corr_conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.out_channels = 128

    def forward(self, query_feat, support_feat, support_mask):
        support_feat, support_mask = self._ensure_support_dims(support_feat, support_mask)

        # Keep every support location available for matching, then retain the best match.
        B, S, C, Hs, Ws = support_feat.shape
        mask = F.interpolate(
            support_mask.reshape(B * S, 1, support_mask.shape[-2], support_mask.shape[-1]),
            (Hs, Ws),
            mode="nearest",
        ).reshape(B, S, 1, Hs, Ws)
        masked_support = support_feat * mask

        _, _, Hq, Wq = query_feat.shape
        q = F.normalize(query_feat.flatten(2), dim=1)                               # [B, C, Hq*Wq]
        s = F.normalize(masked_support.permute(0, 2, 1, 3, 4).reshape(B, C, -1), dim=1)  # [B, C, S*Hs*Ws]
        support_valid = mask.reshape(B, -1) > 0

        corr = torch.einsum("bcm,bcn->bmn", q, s)
        corr = corr.masked_fill(~support_valid.unsqueeze(1), -1.0)
        corr = corr.amax(dim=-1)
        corr = torch.where(torch.isfinite(corr), corr, torch.zeros_like(corr))
        corr = corr.reshape(B, 1, Hq, Wq)
        return self.corr_conv(corr)


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------
FUSION_REGISTRY: dict[str, type[_BaseFusion]] = {
    "prototype_cosine": PrototypeCosine,
    "concat": ConcatFusion,
    "film": FiLMFusion,
    "correlation_4d": Correlation4D,
}
