"""Support→Query fusion methods for one-shot segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _BaseFusion(nn.Module):
    """Common helpers shared by all fusion methods."""

    out_channels: int  # set by subclass

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
        B, C, H, W = feat.shape
        m = F.interpolate(mask, (H, W), mode="nearest")
        numerator = (feat * m).sum(dim=(2, 3))
        denominator = m.sum(dim=(2, 3)).clamp(min=1e-6)
        return numerator / denominator


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
        sim = (q * p).sum(dim=1, keepdim=True)                         # [B, 1, H, W]
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
        gamma = self.fc_gamma(proto)[..., None, None]
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
        # Mask out background in support
        B, C, Hs, Ws = support_feat.shape
        m = F.interpolate(support_mask, (Hs, Ws), mode="nearest")
        masked_support = support_feat * m

        _, _, Hq, Wq = query_feat.shape
        q = F.normalize(query_feat.flatten(2), dim=1)       # [B, C, Hq*Wq]
        s = F.normalize(masked_support.flatten(2), dim=1)    # [B, C, Hs*Ws]

        # Correlation: mean over support positions → [B, Hq*Wq]
        corr = torch.einsum("bcm,bcn->bmn", q, s).mean(dim=-1)
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
