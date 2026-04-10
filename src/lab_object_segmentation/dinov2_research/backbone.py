"""DINOv2 backbone wrapper with MPS safety."""

from __future__ import annotations

import torch
import torch.nn as nn

SIZE_TO_DIM = {"s": 384, "b": 768, "l": 1024}
SIZE_TO_HUB = {
    "s": "dinov2_vits14",
    "b": "dinov2_vitb14",
    "l": "dinov2_vitl14",
}
SIZE_TO_TIMM = {
    "s": "vit_small_patch14_dinov2.lvd142m",
    "b": "vit_base_patch14_dinov2.lvd142m",
    "l": "vit_large_patch14_dinov2.lvd142m",
}
PATCH_SIZE = 14


class DINOv2Backbone(nn.Module):
    """Frozen DINOv2 feature extractor.

    Loads via torch.hub with timm fallback.  Works on MPS/CUDA/CPU.
    """

    def __init__(self, size: str = "s", device: str | torch.device = "cpu", frozen: bool = True):
        super().__init__()
        self.size = size
        self.embed_dim = SIZE_TO_DIM[size]
        self.patch_size = PATCH_SIZE
        self.device = torch.device(device)
        self.frozen = frozen

        self.model = self._load_model()
        self.model.eval()
        if self.frozen:
            for p in self.model.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    def _load_model(self) -> nn.Module:
        # Try torch.hub first (clean API with get_intermediate_layers)
        try:
            model = torch.hub.load(
                "facebookresearch/dinov2",
                SIZE_TO_HUB[self.size],
                pretrained=True,
            )
            model = model.to(self.device)
            print(f"[backbone] loaded DINOv2-{self.size.upper()} via torch.hub")
            return model
        except Exception as e:
            print(f"[backbone] torch.hub failed ({e}), trying timm ...")

        # Fallback: timm
        import timm

        model = timm.create_model(SIZE_TO_TIMM[self.size], pretrained=True)
        model = model.to(self.device)
        print(f"[backbone] loaded DINOv2-{self.size.upper()} via timm")
        return model

    # ------------------------------------------------------------------
    def extract_patch_tokens(self, x: torch.Tensor, n_last_layers: int = 1) -> list[torch.Tensor]:
        """Return patch tokens from last *n_last_layers* layers.

        Parameters
        ----------
        x : [B, 3, H, W]  —  preprocessed images (ImageNet‑normalised).
        n_last_layers : how many final transformer layers to return.

        Returns
        -------
        list of tensors, each [B, N_patches, C].
        """
        if hasattr(self.model, "get_intermediate_layers"):
            # torch.hub DINOv2
            feats = self.model.get_intermediate_layers(
                x, n=n_last_layers, return_class_token=False
            )
            return list(feats)
        else:
            # timm fallback: just return final features
            feats = self.model.forward_features(x)
            if feats.ndim == 3:
                # [B, N+1, C] — strip cls token
                feats = feats[:, 1:, :]
            return [feats]

    # ------------------------------------------------------------------
    @staticmethod
    def tokens_to_spatial(tokens: torch.Tensor) -> torch.Tensor:
        """Reshape [B, N, C] patch tokens → [B, C, H, W] feature map."""
        B, N, C = tokens.shape
        H = W = int(N**0.5)
        assert H * W == N, f"Non-square patch grid: N={N}"
        return tokens.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def grid_size(img_size: int) -> int:
        """Number of patches along one dimension."""
        return img_size // PATCH_SIZE
