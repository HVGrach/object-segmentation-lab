"""
Faithful ConvNeXt inference utilities derived from solution_convnext_fixed.ipynb.

The original training notebook uses:
- RGBDUPerNet with ConvNeXtV2-Base backbone
- RGB + cached depth input resized to 420x420
- logit averaging across folds/checkpoints
- notebook TTA bundle (D4 + scale + color)
- binary JSON-mask submission format: ImageId,mask
"""
from __future__ import annotations

import csv
import json
import zlib
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


NOTEBOOK_IMAGE_SIZE = 420
NOTEBOOK_THRESHOLD = 0.5
TTA_MODE_CHOICES = ("geometric", "scale_color", "full14")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEPTH_MEAN = 0.5
DEPTH_STD = 0.5
_WARNED_DEPTH_PATHS: set[str] = set()


def get_device() -> torch.device:
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if device.type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.float16)
    return nullcontext()


def adaptive_avg_pool2d_mps_safe(x: torch.Tensor, output_size: int | tuple[int, int]) -> torch.Tensor:
    """
    MPS cannot handle some non-divisible adaptive pooling shapes used by the PPM.
    Fall back to CPU only for those exact cases while keeping the rest on-device.
    """
    if isinstance(output_size, int):
        target_h = target_w = output_size
    else:
        target_h, target_w = output_size

    if x.device.type == "mps":
        height, width = x.shape[-2:]
        if height % target_h != 0 or width % target_w != 0:
            return F.adaptive_avg_pool2d(x.cpu(), output_size).to(x.device)
    return F.adaptive_avg_pool2d(x, output_size)


def _gn(channels: int, groups: int = 32) -> nn.GroupNorm:
    """Notebook-style GroupNorm with automatic group reduction."""
    g = min(groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


class ConvNeXtV2InflatedStem(nn.Module):
    """
    Inflate ConvNeXt stem to accept 4-channel RGBD input.

    The depth channel is initialized from the mean RGB stem weights. During
    inference these values are overwritten by the checkpoint, but keeping the
    original notebook construction makes strict loading possible.
    """

    def __init__(self, original_stem_conv: nn.Conv2d):
        super().__init__()
        old_w = original_stem_conv.weight.data
        c_out, _, k_h, k_w = old_w.shape
        new_conv = nn.Conv2d(
            4,
            c_out,
            k_h,
            stride=original_stem_conv.stride,
            padding=original_stem_conv.padding,
            bias=original_stem_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = old_w
            new_conv.weight[:, 3:] = old_w.mean(dim=1, keepdim=True)
            if original_stem_conv.bias is not None:
                new_conv.bias.copy_(original_stem_conv.bias)
        self.conv = new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class RGBDUPerNet(nn.Module):
    """
    Notebook-faithful ConvNeXtV2-Base + UPerNet-style decoder for RGBD.

    This matches the architecture used in solution_convnext_fixed.ipynb:
    full timm backbone, inflated stem, PPM + FPN decoder, ECA attention,
    depth-gating, and a final 1x1 classifier.
    """

    FEAT_CHANNELS = [128, 256, 512, 1024]
    PPM_POOLS = [1, 2, 3, 6]
    FPN_CHANNELS = 256

    def __init__(
        self,
        unfreeze_last_n: int = 2,
        gradient_checkpointing: bool = False,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            "convnextv2_base.fcmae_ft_in22k_in1k_384",
            pretrained=pretrained_backbone,
        )
        self.backbone.reset_classifier(0, global_pool="")

        orig_stem_conv = self.backbone.stem[0]
        inflated = ConvNeXtV2InflatedStem(orig_stem_conv)
        self.backbone.stem[0] = inflated.conv

        # Preserve notebook structure even though it is irrelevant for inference.
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.stem.parameters():
            p.requires_grad = True
        n_stages = len(self.backbone.stages)
        for stage_idx in range(max(0, n_stages - unfreeze_last_n), n_stages):
            for p in self.backbone.stages[stage_idx].parameters():
                p.requires_grad = True

        if gradient_checkpointing:
            self.backbone.set_grad_checkpointing(True)

        c = self.FPN_CHANNELS

        self.ppm_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    nn.Conv2d(1024, c, 1, bias=False),
                    _gn(c),
                    nn.ReLU(inplace=True),
                )
                for pool_size in self.PPM_POOLS
            ]
        )
        self.ppm_fuse = nn.Sequential(
            nn.Conv2d(1024 + len(self.PPM_POOLS) * c, c, 1, bias=False),
            _gn(c),
            nn.ReLU(inplace=True),
        )

        self.lateral = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, c, 1, bias=False),
                    _gn(c),
                    nn.ReLU(inplace=True),
                )
                for channels in self.FEAT_CHANNELS[:3]
            ]
        )

        self.fpn_out = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, c, 3, padding=1, bias=False),
                    _gn(c),
                    nn.ReLU(inplace=True),
                )
                for _ in self.FEAT_CHANNELS
            ]
        )

        self.fuse_head = nn.Sequential(
            nn.Conv2d(c * 4, c, 3, padding=1, bias=False),
            _gn(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            _gn(c),
            nn.ReLU(inplace=True),
        )

        self.eca_pool = nn.AdaptiveAvgPool2d(1)
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
        self.eca_sigmoid = nn.Sigmoid()

        self.depth_gate = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            _gn(c),
            nn.Sigmoid(),
        )

        self.classifier = nn.Conv2d(c, 1, 1)

    def _ppm(self, x: torch.Tensor) -> torch.Tensor:
        parts = [x]
        target_size = x.shape[-2:]
        for branch in self.ppm_branches:
            pooled = adaptive_avg_pool2d_mps_safe(x, branch[0].output_size)
            pooled = branch[1:](pooled)
            pooled = F.interpolate(pooled, size=target_size, mode="bilinear", align_corners=False)
            parts.append(pooled)
        return self.ppm_fuse(torch.cat(parts, dim=1))

    def _fpn(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        f3_ppm = self._ppm(feats[3])
        p3 = self.fpn_out[3](f3_ppm)
        p2 = self.fpn_out[2](
            self.lateral[2](feats[2])
            + F.interpolate(f3_ppm, size=feats[2].shape[-2:], mode="bilinear", align_corners=False)
        )
        p1 = self.fpn_out[1](
            self.lateral[1](feats[1])
            + F.interpolate(p2, size=feats[1].shape[-2:], mode="bilinear", align_corners=False)
        )
        p0 = self.fpn_out[0](
            self.lateral[0](feats[0])
            + F.interpolate(p1, size=feats[0].shape[-2:], mode="bilinear", align_corners=False)
        )
        return [p0, p1, p2, p3]

    def _forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.backbone.stem(pixel_values)
        feats: list[torch.Tensor] = []
        for stage in self.backbone.stages:
            x = stage(x)
            feats.append(x)

        fpn = self._fpn(feats)
        reference_size = fpn[0].shape[-2:]
        parts = [fpn[0]]
        for feat in fpn[1:]:
            parts.append(F.interpolate(feat, size=reference_size, mode="bilinear", align_corners=False))
        fused = self.fuse_head(torch.cat(parts, dim=1))

        eca_w = self.eca_pool(fused)
        eca_w = eca_w.squeeze(-1).transpose(-1, -2)
        eca_w = self.eca_conv(eca_w)
        eca_w = self.eca_sigmoid(eca_w).transpose(-1, -2).unsqueeze(-1)
        fused = fused * eca_w

        fused = fused * self.depth_gate(fused)
        return fused

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._forward_features(pixel_values))

    def forward_fullres(self, pixel_values: torch.Tensor) -> torch.Tensor:
        height, width = pixel_values.shape[-2:]
        logits = self.forward(pixel_values)
        return F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)

    def get_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._forward_features(pixel_values)


def load_convnext_model(
    checkpoint_path: str | Path,
    device: torch.device,
    gradient_checkpointing: bool = False,
) -> RGBDUPerNet:
    """Load the exact notebook architecture and strict-load the checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = RGBDUPerNet(
        unfreeze_last_n=2,
        gradient_checkpointing=gradient_checkpointing,
        pretrained_backbone=False,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    print(
        f"[ConvNeXt] Loaded from {Path(checkpoint_path).name}: "
        f"val_iou={ckpt.get('val_iou', '?')}, epoch={ckpt.get('epoch', '?')}, fold={ckpt.get('fold', '?')}"
    )
    return model


def _warn_depth_fallback(image_path: Path, depth_path: Path, exc: Exception) -> None:
    depth_key = str(depth_path)
    if depth_key in _WARNED_DEPTH_PATHS:
        return
    _WARNED_DEPTH_PATHS.add(depth_key)
    print(
        "[Depth] WARNING: failed to load "
        f"{depth_path} for image {image_path.name}: {exc}. "
        f"Using fill depth value {DEPTH_MEAN:.3f}."
    )


def load_depth_array(
    image_path: Path,
    depth_root: Path,
    fallback_shape: tuple[int, int],
) -> np.ndarray:
    """Load raw cached depth or return neutral depth if the cache is absent/corrupt."""
    depth_path = depth_root / f"{image_path.stem}.npz"
    if not depth_path.exists():
        return np.full(fallback_shape, DEPTH_MEAN, dtype=np.float32)

    try:
        with np.load(depth_path) as depth_data:
            depth = depth_data["depth"].astype(np.float32)
    except (OSError, ValueError, KeyError, EOFError, BadZipFile, zlib.error) as exc:
        _warn_depth_fallback(image_path, depth_path, exc)
        return np.full(fallback_shape, DEPTH_MEAN, dtype=np.float32)

    if depth.ndim != 2 or depth.size == 0:
        exc = ValueError(f"expected 2D non-empty depth array, got shape {depth.shape}")
        _warn_depth_fallback(image_path, depth_path, exc)
        return np.full(fallback_shape, DEPTH_MEAN, dtype=np.float32)

    return depth


def load_rgbd_with_shape(image_path: Path, depth_root: Path, image_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Notebook-faithful RGBD preprocessing with original image size metadata."""
    with Image.open(image_path) as img_pil:
        img_pil = img_pil.convert("RGB")
        orig_w, orig_h = img_pil.size
        img_resized = img_pil.resize((image_size, image_size), Image.BILINEAR)
        img_np = np.array(img_resized, dtype=np.uint8)

    depth = load_depth_array(image_path, depth_root, fallback_shape=(orig_h, orig_w))
    if depth.shape != (image_size, image_size):
        depth = np.array(
            Image.fromarray(depth).resize((image_size, image_size), Image.BILINEAR),
            dtype=np.float32,
        )

    img_f = img_np.astype(np.float32) / 255.0
    img_f = (img_f - IMAGENET_MEAN) / IMAGENET_STD
    depth_norm = (depth - DEPTH_MEAN) / DEPTH_STD

    rgbd = np.concatenate([img_f, depth_norm[..., None]], axis=-1)
    tensor = torch.from_numpy(np.ascontiguousarray(rgbd.transpose(2, 0, 1))).float()
    return tensor, (orig_h, orig_w)


def load_rgbd(image_path: Path, depth_root: Path, image_size: int) -> torch.Tensor:
    rgbd, _ = load_rgbd_with_shape(image_path, depth_root, image_size)
    return rgbd


class ConvNeXtInferenceDataset(Dataset):
    def __init__(self, image_paths: list[Path], depth_root: Path, image_size: int):
        self.image_paths = image_paths
        self.depth_root = depth_root
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        rgbd, orig_hw = load_rgbd_with_shape(image_path, self.depth_root, self.image_size)
        return rgbd, image_path.name, orig_hw


def convnext_collate_fn(batch):
    rgbd_list, filenames, orig_hws = zip(*batch)
    return torch.stack(list(rgbd_list)), list(filenames), list(orig_hws)


def _scale_fwd_075(x: torch.Tensor) -> torch.Tensor:
    _, _, height, width = x.shape
    h75, w75 = int(height * 0.75), int(width * 0.75)
    small = F.interpolate(x, size=(h75, w75), mode="bilinear", align_corners=False)
    pad_h, pad_w = height - h75, width - w75
    half_h_top = pad_h // 2
    half_h_bot = pad_h - half_h_top
    half_w_left = pad_w // 2
    half_w_right = pad_w - half_w_left
    return F.pad(small, (half_w_left, half_w_right, half_h_top, half_h_bot), mode="reflect")


def _scale_inv_075(logit: torch.Tensor) -> torch.Tensor:
    _, _, height, width = logit.shape
    h75, w75 = int(height * 0.75), int(width * 0.75)
    pad_h, pad_w = height - h75, width - w75
    y0, x0 = pad_h // 2, pad_w // 2
    cropped = logit[:, :, y0 : y0 + h75, x0 : x0 + w75]
    return F.interpolate(cropped, size=(height, width), mode="bilinear", align_corners=False)


def _scale_fwd_125(x: torch.Tensor) -> torch.Tensor:
    _, _, height, width = x.shape
    h125, w125 = int(height * 1.25), int(width * 1.25)
    big = F.interpolate(x, size=(h125, w125), mode="bilinear", align_corners=False)
    y0, x0 = (h125 - height) // 2, (w125 - width) // 2
    return big[:, :, y0 : y0 + height, x0 : x0 + width]


def _scale_inv_125(logit: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = logit.shape
    h125, w125 = int(height * 1.25), int(width * 1.25)
    canvas = torch.zeros(batch, channels, h125, w125, device=logit.device, dtype=logit.dtype)
    y0, x0 = (h125 - height) // 2, (w125 - width) // 2
    canvas[:, :, y0 : y0 + height, x0 : x0 + width] = logit
    return F.interpolate(canvas, size=(height, width), mode="bilinear", align_corners=False)


def _color_brightness(x: torch.Tensor, delta: float) -> torch.Tensor:
    out = x.clone()
    out[:, :3] = out[:, :3] + delta
    return out


def _color_contrast(x: torch.Tensor, factor: float) -> torch.Tensor:
    out = x.clone()
    rgb = out[:, :3]
    mean = rgb.mean(dim=(-2, -1), keepdim=True)
    out[:, :3] = (rgb - mean) * factor + mean
    return out


_id = lambda x: x


def _compose_tta_op(
    op_a: tuple,
    op_b: tuple,
    name: str | None = None,
) -> tuple:
    fwd_a, inv_a, name_a = op_a
    fwd_b, inv_b, name_b = op_b

    def composed_fwd(x: torch.Tensor) -> torch.Tensor:
        return fwd_b(fwd_a(x))

    def composed_inv(x: torch.Tensor) -> torch.Tensor:
        return inv_a(inv_b(x))

    return composed_fwd, composed_inv, name or f"{name_a}_{name_b}"


TTA_OPS_FULL14 = [
    (_id, _id, "e"),
    (lambda x: x.rot90(1, [-2, -1]), lambda x: x.rot90(-1, [-2, -1]), "r1"),
    (lambda x: x.rot90(2, [-2, -1]), lambda x: x.rot90(-2, [-2, -1]), "r2"),
    (lambda x: x.rot90(3, [-2, -1]), lambda x: x.rot90(-3, [-2, -1]), "r3"),
    (lambda x: x.flip(-1), lambda x: x.flip(-1), "f"),
    (
        lambda x: x.flip(-1).rot90(1, [-2, -1]),
        lambda x: x.rot90(-1, [-2, -1]).flip(-1),
        "fr1",
    ),
    (
        lambda x: x.flip(-1).rot90(2, [-2, -1]),
        lambda x: x.rot90(-2, [-2, -1]).flip(-1),
        "fr2",
    ),
    (
        lambda x: x.flip(-1).rot90(3, [-2, -1]),
        lambda x: x.rot90(-3, [-2, -1]).flip(-1),
        "fr3",
    ),
    (_scale_fwd_075, _scale_inv_075, "scale_0.75"),
    (_scale_fwd_125, _scale_inv_125, "scale_1.25"),
    (lambda x: _color_brightness(x, 0.2), _id, "brightness+0.2"),
    (lambda x: _color_brightness(x, -0.2), _id, "brightness-0.2"),
    (lambda x: _color_contrast(x, 1.2), _id, "contrast+1.2"),
    (lambda x: _color_contrast(x, 0.8), _id, "contrast+0.8"),
]

assert len(TTA_OPS_FULL14) == 14

_FLIP_OP = TTA_OPS_FULL14[4]
_SCALE_COLOR_BASE_OPS = [TTA_OPS_FULL14[0], *TTA_OPS_FULL14[8:]]
TTA_OPS_SCALE_COLOR_FLIP = [_SCALE_COLOR_BASE_OPS[0], _FLIP_OP]
for _op in _SCALE_COLOR_BASE_OPS[1:]:
    TTA_OPS_SCALE_COLOR_FLIP.append(_op)
    TTA_OPS_SCALE_COLOR_FLIP.append(_compose_tta_op(_op, _FLIP_OP, name=f"{_op[2]}_flip"))

assert len(TTA_OPS_SCALE_COLOR_FLIP) == 14

TTA_OPS_BY_MODE = {
    "geometric": TTA_OPS_FULL14[:8],
    "scale_color": TTA_OPS_SCALE_COLOR_FLIP,
    "full14": TTA_OPS_FULL14,
}


def resolve_tta_ops(tta_mode: str):
    if tta_mode not in TTA_OPS_BY_MODE:
        raise ValueError(f"Unsupported ConvNeXt TTA mode: {tta_mode}")
    return TTA_OPS_BY_MODE[tta_mode]


@torch.no_grad()
def predict_with_tta(
    model: RGBDUPerNet,
    rgbd: torch.Tensor,
    tta_mode: str,
    device: torch.device,
    use_amp: bool = True,
) -> torch.Tensor:
    """Notebook-style TTA prediction returning averaged logits at input resolution."""
    tta_ops = resolve_tta_ops(tta_mode)
    accumulated = None
    for fwd, inv, _ in tta_ops:
        augmented = fwd(rgbd)
        with autocast_context(device, enabled=use_amp):
            logits = model.forward_fullres(augmented)
        logits_inv = inv(logits)
        accumulated = logits_inv if accumulated is None else accumulated + logits_inv
    return accumulated / len(tta_ops)


@torch.no_grad()
def predict_geometric_tta(
    model: RGBDUPerNet,
    rgbd: torch.Tensor,
    image_size: int,
    device: torch.device,
    use_amp: bool = True,
) -> torch.Tensor:
    batch = rgbd.unsqueeze(0) if rgbd.ndim == 3 else rgbd
    probs = torch.sigmoid(predict_with_tta(model, batch.to(device), "geometric", device, use_amp=use_amp))
    return probs.squeeze(0).squeeze(0).detach().cpu()


@torch.no_grad()
def predict_scale_color_tta(
    model: RGBDUPerNet,
    rgbd: torch.Tensor,
    image_size: int,
    device: torch.device,
    use_amp: bool = True,
) -> torch.Tensor:
    batch = rgbd.unsqueeze(0) if rgbd.ndim == 3 else rgbd
    probs = torch.sigmoid(predict_with_tta(model, batch.to(device), "scale_color", device, use_amp=use_amp))
    return probs.squeeze(0).squeeze(0).detach().cpu()


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    filled = ((mask > 0).astype(np.uint8) * 255).copy()
    contours, _ = cv2.findContours(filled, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return (filled > 127).astype(np.uint8)


def postprocess_mask(
    mask: np.ndarray,
    min_component_area: int = 0,
    fill_holes: bool = True,
) -> np.ndarray:
    if fill_holes:
        mask = fill_mask_holes(mask)
    if min_component_area > 0:
        labeled, n_labels = ndimage.label(mask)
        cleaned = np.zeros_like(mask, dtype=np.uint8)
        for label_idx in range(1, n_labels + 1):
            component = labeled == label_idx
            if int(component.sum()) >= min_component_area:
                cleaned[component] = 1
        mask = cleaned
    return mask.astype(np.uint8)


def rle_encode(mask: np.ndarray) -> str:
    flat = mask.T.flatten()
    padded = np.concatenate([[0], flat, [0]])
    runs = np.where(padded[1:] != padded[:-1])[0]
    if runs.size == 0:
        return "1 0"
    runs[0::2] += 1
    runs[1::2] -= runs[0::2]
    return " ".join(map(str, runs))


def serialize_mask(mask2d: np.ndarray) -> str:
    return json.dumps(mask2d.astype(np.uint8).tolist(), separators=(",", ":"))


def submission_stem(base_name: str, limit: int = 0) -> str:
    return f"{base_name}_limit{limit}" if limit > 0 else base_name


def write_submission_csv(sub_path: Path, rows: list[dict[str, str]]) -> None:
    with open(sub_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ImageId", "mask"])
        for row in rows:
            writer.writerow([row["ImageId"], row["mask"]])


@dataclass
class InferenceConfig:
    checkpoint_paths: list[str]
    test_dir: str
    depth_root: str
    output_dir: str
    image_size: int = NOTEBOOK_IMAGE_SIZE
    threshold: float = NOTEBOOK_THRESHOLD
    tta_mode: str = "full14"
    min_component_area: int = 0
    fill_holes: bool = True
    batch_size: int = 8
    use_amp: bool = True
    export_original_size: bool = True
    limit: int = 0


def _submission_name(cfg: InferenceConfig) -> str:
    base = f"submission_cnxt_{cfg.tta_mode}"
    if cfg.image_size != NOTEBOOK_IMAGE_SIZE:
        base += f"_sz{cfg.image_size}"
    base += f"_thr{int(round(cfg.threshold * 100))}"
    return submission_stem(base, limit=cfg.limit)


def run_convnext_inference(cfg: InferenceConfig) -> Path:
    """Run faithful ConvNeXt inference on the test set and build a submission CSV."""
    if cfg.tta_mode not in TTA_MODE_CHOICES:
        raise ValueError(f"Unsupported ConvNeXt TTA mode: {cfg.tta_mode}")

    device = get_device()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [load_convnext_model(cp, device) for cp in cfg.checkpoint_paths]
    test_dir = Path(cfg.test_dir)
    depth_root = Path(cfg.depth_root)
    image_paths = sorted(test_dir.glob("*.jpg"))
    if cfg.limit > 0:
        image_paths = image_paths[: cfg.limit]

    dataset = ConvNeXtInferenceDataset(image_paths, depth_root=depth_root, image_size=cfg.image_size)
    loader = DataLoader(
        dataset,
        batch_size=max(1, cfg.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=convnext_collate_fn,
    )

    print(
        f"[Inference] {len(image_paths)} images, {len(models)} models, "
        f"TTA={cfg.tta_mode}, thr={cfg.threshold}, size={cfg.image_size}, batch={cfg.batch_size}"
    )

    rows: list[dict[str, str]] = []
    for rgbd_batch, filenames, orig_hws in tqdm(loader, desc=f"ConvNeXt {cfg.tta_mode}"):
        rgbd_batch = rgbd_batch.to(device)
        logit_accum = None
        for model in models:
            batch_logits = predict_with_tta(model, rgbd_batch, cfg.tta_mode, device, use_amp=cfg.use_amp)
            batch_logits = batch_logits.detach().cpu()
            logit_accum = batch_logits if logit_accum is None else logit_accum + batch_logits

        probs = torch.sigmoid(logit_accum / len(models))
        for batch_idx, image_id in enumerate(filenames):
            prob = probs[batch_idx, 0]
            orig_h, orig_w = orig_hws[batch_idx]
            if cfg.export_original_size and tuple(prob.shape) != (orig_h, orig_w):
                prob = F.interpolate(
                    prob.unsqueeze(0).unsqueeze(0),
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)

            mask = (prob.numpy() > cfg.threshold).astype(np.uint8)
            mask = postprocess_mask(mask, min_component_area=cfg.min_component_area, fill_holes=cfg.fill_holes)
            rows.append({"ImageId": image_id, "mask": serialize_mask(mask)})

    sub_name = _submission_name(cfg)
    sub_path = out_dir / f"{sub_name}.csv"
    write_submission_csv(sub_path, rows)

    cfg_path = out_dir / f"{sub_name}_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoints": cfg.checkpoint_paths,
                "tta_mode": cfg.tta_mode,
                "image_size": cfg.image_size,
                "threshold": cfg.threshold,
                "min_component_area": cfg.min_component_area,
                "fill_holes": cfg.fill_holes,
                "batch_size": cfg.batch_size,
                "use_amp": cfg.use_amp,
                "export_original_size": cfg.export_original_size,
                "n_images": len(rows),
                "n_models": len(models),
            },
            f,
            indent=2,
        )

    print(f"[Inference] Saved: {sub_path}")
    return sub_path


def export_probabilities(cfg: InferenceConfig) -> Path:
    """Export averaged ConvNeXt probability maps at model resolution for downstream experiments."""
    if cfg.tta_mode not in TTA_MODE_CHOICES:
        raise ValueError(f"Unsupported ConvNeXt TTA mode: {cfg.tta_mode}")

    device = get_device()
    out_dir = Path(cfg.output_dir) / f"probs_cnxt_{cfg.tta_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [load_convnext_model(cp, device) for cp in cfg.checkpoint_paths]
    test_dir = Path(cfg.test_dir)
    depth_root = Path(cfg.depth_root)
    image_paths = sorted(test_dir.glob("*.jpg"))
    if cfg.limit > 0:
        image_paths = image_paths[: cfg.limit]

    dataset = ConvNeXtInferenceDataset(image_paths, depth_root=depth_root, image_size=cfg.image_size)
    loader = DataLoader(
        dataset,
        batch_size=max(1, cfg.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=convnext_collate_fn,
    )

    file_index = 0
    for rgbd_batch, filenames, _ in tqdm(loader, desc=f"Export probs ({cfg.tta_mode})"):
        rgbd_batch = rgbd_batch.to(device)
        logit_accum = None
        for model in models:
            batch_logits = predict_with_tta(model, rgbd_batch, cfg.tta_mode, device, use_amp=cfg.use_amp)
            batch_logits = batch_logits.detach().cpu()
            logit_accum = batch_logits if logit_accum is None else logit_accum + batch_logits

        probs = torch.sigmoid(logit_accum / len(models)).numpy().astype(np.float16)
        for batch_idx, image_id in enumerate(filenames):
            np.savez_compressed(out_dir / f"{Path(image_id).stem}.npz", prob=probs[batch_idx, 0])
            file_index += 1

    print(f"[Export] Saved {file_index} prob maps to {out_dir}")
    return out_dir
