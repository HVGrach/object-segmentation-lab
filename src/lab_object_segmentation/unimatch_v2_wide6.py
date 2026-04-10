from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import GroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel

from lab_object_segmentation.common.paths import LAB1_DATASET_ROOT, LAB3_DATASET_ROOT, RUNS_ROOT
from lab_object_segmentation.segformer_semisup.utils import (
    binary_dice_score,
    infer_camera_group,
    tune_binary_threshold,
)
from lab_object_segmentation.supervised_v4.ensemble import aggregate_probability_maps, resolve_member_specs
from lab_object_segmentation.supervised_v4.predict_ensemble import load_member, predict_member_probability
from lab_object_segmentation.supervised_v4.train import TrainConfig as SupervisedTrainConfig
from lab_object_segmentation.supervised_v4.train import postprocess_binary_mask


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def get_grad_scaler(device: torch.device):
    return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")


def empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        return
    if device.type == "mps" and hasattr(torch, "mps"):
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_image_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_mask_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def collect_image_paths(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted([path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS])


def collect_labeled_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    image_map = {path.stem: path for path in collect_image_paths(images_dir)}
    pairs: list[tuple[Path, Path]] = []
    for mask_path in sorted(masks_dir.rglob("*")):
        if not mask_path.is_file() or mask_path.suffix.lower() not in IMAGE_EXTS:
            continue
        image_path = image_map.get(mask_path.stem)
        if image_path is not None:
            pairs.append((image_path, mask_path))
    if not pairs:
        raise RuntimeError("No labeled pairs found.")
    return pairs


def split_grouped(samples: list[tuple[Path, Path]], n_splits: int, fold_index: int) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    groups = [infer_camera_group(image_path) for image_path, _ in samples]
    splitter = GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(np.arange(len(samples)), groups=groups))
    if fold_index < 0 or fold_index >= len(splits):
        raise ValueError(f"fold_index={fold_index} out of range for {len(splits)} splits")
    train_idx, val_idx = splits[fold_index]
    train_samples = [samples[idx] for idx in train_idx]
    val_samples = [samples[idx] for idx in val_idx]
    return train_samples, val_samples


@dataclass
class RunConfig:
    run_name: str
    source_notebook: Path

    train_img_dir: Path = LAB3_DATASET_ROOT / "train" / "images"
    train_mask_dir: Path = LAB3_DATASET_ROOT / "train" / "masks"
    unlabeled_dirs: list[Path] = field(
        default_factory=lambda: [
            LAB3_DATASET_ROOT / "unlabeled" / "images",
            LAB1_DATASET_ROOT / "train" / "train",
            LAB1_DATASET_ROOT / "test_images" / "test_images",
        ]
    )

    fold_index: int = 0
    n_splits: int = 5
    image_size: int = 392
    batch_size_labeled: int = 4
    batch_size_unlabeled: int = 4
    num_workers: int = 0
    pin_memory: bool = False
    seed: int = 42
    enable_gradient_checkpointing: bool = False

    backbone: str = "facebook/dinov2-base"
    freeze_backbone: bool = True
    unfreeze_last_n_blocks: int = 2
    decoder_embed_dim: int = 256
    decoder_dropout: float = 0.1

    phase1_epochs: int = 12
    phase1_lr: float = 3e-4
    phase1_lr_backbone: float = 1e-5
    phase1_weight_decay: float = 0.01

    phase2_epochs: int = 18
    phase2_lr: float = 1e-4
    warmup_epochs: int = 6
    warmup_ratio: float = 0.05
    unsup_weight: float = 0.5
    swd_weight: float = 0.05
    swd_projections: int = 64
    complementary_dropout_p: float = 0.5
    conf_threshold_start: float = 0.93
    conf_threshold_end: float = 0.88
    conf_warmup_epochs: int = 12

    phase3_min_improvement: float = 0.003
    phase3_epochs: int = 8
    phase3_lr_scale: float = 0.3
    phase3_min_samples: int = 128

    loss_ce_weight: float = 0.5
    loss_dice_weight: float = 0.5
    grad_clip: float = 1.0
    threshold_grid: list[float] = field(
        default_factory=lambda: [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    )

    teacher_preset: str = "wide6"
    teacher_aggregation: str = "weighted_mean"
    teacher_threshold: float = 0.55
    teacher_member_threshold: float = 0.5
    teacher_tta_ops: str = "flips"
    teacher_tta_scales: list[float] = field(default_factory=lambda: [1.0])
    teacher_postprocess_min_component_area: int = 128
    teacher_postprocess_fill_holes: bool = True
    teacher_save_all: bool = False
    teacher_accept_min_reliable_ratio: float = 0.90
    teacher_accept_min_mean_confidence: float = 0.88
    teacher_accept_min_fg_reliable_ratio: float = 0.70
    teacher_min_object_ratio: float = 0.005
    teacher_max_object_ratio: float = 0.90
    teacher_min_samples: int = 256
    teacher_limit: int | None = None
    teacher_overwrite_cache: bool = False
    teacher_enable_sahi: bool = False
    teacher_sahi_tile_size: int = 320
    teacher_sahi_overlap: float = 0.25

    preview_limit: int = 12
    smoke: bool = False

    @property
    def run_dir(self) -> Path:
        return RUNS_ROOT / "unimatch_v2_wide6" / self.run_name

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def metrics_dir(self) -> Path:
        return self.run_dir / "metrics"

    @property
    def teacher_dir(self) -> Path:
        return self.run_dir / "teacher_cache"

    @property
    def preview_dir(self) -> Path:
        return self.run_dir / "preview"


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Run UniMatchV2 overnight experiment with wide6 teacher.")
    parser.add_argument("--run-name", type=str, default=f"night_{timestamp()}")
    parser.add_argument(
        "--source-notebook",
        type=Path,
        default=Path("/Users/fgrach/Downloads/Telegram Desktop/solution_final.ipynb"),
    )
    parser.add_argument("--image-size", type=int, default=392)
    parser.add_argument("--phase1-epochs", type=int, default=12)
    parser.add_argument("--phase2-epochs", type=int, default=18)
    parser.add_argument("--phase3-epochs", type=int, default=8)
    parser.add_argument("--batch-size-labeled", type=int, default=4)
    parser.add_argument("--batch-size-unlabeled", type=int, default=4)
    parser.add_argument("--teacher-threshold", type=float, default=0.55)
    parser.add_argument("--teacher-limit", type=int, default=None)
    parser.add_argument("--teacher-enable-sahi", action="store_true")
    parser.add_argument("--teacher-overwrite-cache", action="store_true")
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = RunConfig(
        run_name=args.run_name,
        source_notebook=args.source_notebook,
        image_size=args.image_size,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase3_epochs=args.phase3_epochs,
        batch_size_labeled=args.batch_size_labeled,
        batch_size_unlabeled=args.batch_size_unlabeled,
        teacher_threshold=args.teacher_threshold,
        teacher_limit=args.teacher_limit,
        teacher_enable_sahi=args.teacher_enable_sahi,
        teacher_overwrite_cache=args.teacher_overwrite_cache,
        fold_index=args.fold_index,
        smoke=args.smoke,
    )
    if cfg.smoke:
        cfg.phase1_epochs = 1
        cfg.phase2_epochs = 1
        cfg.phase3_epochs = 1
        cfg.batch_size_labeled = 2
        cfg.batch_size_unlabeled = 2
        cfg.teacher_limit = 8
        cfg.preview_limit = 3
        cfg.teacher_min_samples = 4
    return cfg


def config_payload(cfg: RunConfig) -> dict:
    payload = asdict(cfg)
    payload["run_dir"] = str(cfg.run_dir)
    payload["ckpt_dir"] = str(cfg.ckpt_dir)
    payload["metrics_dir"] = str(cfg.metrics_dir)
    payload["teacher_dir"] = str(cfg.teacher_dir)
    payload["preview_dir"] = str(cfg.preview_dir)
    return payload


def build_labeled_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.25),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.90, 1.10),
                translate_percent=(0.0, 0.05),
                rotate=(-20, 20),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_REFLECT_101,
                fill=0,
                fill_mask=0,
                p=0.4,
            ),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=0,
                fill_mask=0,
                p=0.2,
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def build_val_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def build_unlabeled_geom_transform(image_size: int) -> A.Compose:
    crop_size = max(224, min(image_size, int(round(image_size * 0.875))))
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.25),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.0, 0.05),
                rotate=(-20, 20),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_REFLECT_101,
                fill=0,
                fill_mask=0,
                p=0.5,
            ),
            A.RandomCrop(crop_size, crop_size, p=0.4),
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
        ],
        additional_targets={"confidence": "mask"},
    )


def build_unlabeled_photo_transform() -> A.Compose:
    return A.Compose(
        [
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08, p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(std_range=(0.02, 0.10), p=0.25),
            A.CoarseDropout(
                num_holes_range=(1, 10),
                hole_height_range=(8, 40),
                hole_width_range=(8, 40),
                fill=0,
                p=0.2,
            ),
        ]
    )


class LabeledDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, Path]], transform: A.Compose | None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.samples[idx]
        image = read_image_rgb(image_path)
        mask = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        return image, mask.long()


@dataclass
class TeacherPseudoEntry:
    image_path: Path
    mask_path: Path
    confidence_path: Path
    reliable_ratio: float
    mean_confidence: float
    fg_reliable_ratio: float
    area_ratio: float
    selection_score: float
    accepted: bool
    accept_reason: str


class TeacherPseudoDataset(Dataset):
    def __init__(
        self,
        entries: list[TeacherPseudoEntry],
        geom_transform: A.Compose,
        photo_transform: A.Compose,
        normalize_transform: A.Compose,
    ):
        self.entries = entries
        self.geom_transform = geom_transform
        self.photo_transform = photo_transform
        self.normalize_transform = normalize_transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        image = read_image_rgb(entry.image_path)
        mask = (np.array(Image.open(entry.mask_path).convert("L")) > 127).astype(np.uint8)
        confidence = np.load(entry.confidence_path).astype(np.float32)

        aug1 = self.geom_transform(image=image, mask=mask, confidence=confidence)
        aug2 = self.geom_transform(image=image, mask=mask, confidence=confidence)

        img1 = self.photo_transform(image=aug1["image"])["image"]
        img2 = self.photo_transform(image=aug2["image"])["image"]

        img1 = self.normalize_transform(image=img1)["image"]
        img2 = self.normalize_transform(image=img2)["image"]
        pseudo1 = torch.from_numpy(aug1["mask"].astype(np.int64))
        pseudo2 = torch.from_numpy(aug2["mask"].astype(np.int64))
        conf1 = torch.from_numpy(aug1["confidence"].astype(np.float32))
        conf2 = torch.from_numpy(aug2["confidence"].astype(np.float32))

        return img1, img2, pseudo1, pseudo2, conf1, conf2


class SegFormerHead(nn.Module):
    def __init__(self, in_channels: list[int], embed_dim: int = 256, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.linears = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channel, embed_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(embed_dim),
                    nn.ReLU(inplace=True),
                )
                for channel in in_channels
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor], drop_mask: torch.Tensor | None = None) -> torch.Tensor:
        target_h, target_w = features[0].shape[-2:]
        projected = []
        for feature, linear in zip(features, self.linears):
            feat = linear(feature)
            feat = F.interpolate(feat, size=(target_h, target_w), mode="bilinear", align_corners=False)
            projected.append(feat)
        fused = self.fuse(torch.cat(projected, dim=1))
        if drop_mask is not None:
            if drop_mask.shape[-2:] != fused.shape[-2:]:
                drop_mask = F.interpolate(drop_mask.float(), size=fused.shape[-2:], mode="nearest").bool()
            fused = fused * drop_mask.float()
        fused = self.dropout(fused)
        return self.classifier(fused)


class DINOv2Encoder(nn.Module):
    def __init__(self, model_name: str, freeze: bool, unfreeze_last: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.config = self.backbone.config
        self.hidden_dim = int(self.config.hidden_size)

        encoder_layers = list(getattr(self.backbone.encoder, "layer", []))
        n_layers = len(encoder_layers)
        if n_layers == 0:
            raise RuntimeError("Could not locate encoder layers in DINOv2 backbone.")

        if freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            for layer in encoder_layers[max(0, n_layers - unfreeze_last) :]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
            for attr_name in ("layernorm", "norm"):
                layer_norm = getattr(self.backbone, attr_name, None)
                if layer_norm is not None:
                    for parameter in layer_norm.parameters():
                        parameter.requires_grad = True

        self.extract_layers = [
            max(0, n_layers // 4 - 1),
            max(0, n_layers // 2 - 1),
            max(0, (3 * n_layers) // 4 - 1),
            n_layers - 1,
        ]

    def _tokens_to_map(self, tokens: torch.Tensor, patch_size: int, image_size: int) -> torch.Tensor:
        batch_size, num_tokens, channels = tokens.shape
        spatial = image_size // patch_size
        if num_tokens == spatial * spatial + 1:
            tokens = tokens[:, 1:]
        tokens = tokens[:, : spatial * spatial]
        return tokens.transpose(1, 2).reshape(batch_size, channels, spatial, spatial).contiguous()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = self.backbone(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        patch_size = int(getattr(self.config, "patch_size", 14))
        image_size = int(x.shape[-1])

        features = []
        for idx in self.extract_layers:
            feature_map = self._tokens_to_map(hidden_states[idx + 1], patch_size=patch_size, image_size=image_size)
            features.append(feature_map)
        return features


class SegModel(nn.Module):
    def __init__(self, cfg: RunConfig):
        super().__init__()
        self.encoder = DINOv2Encoder(
            model_name=cfg.backbone,
            freeze=cfg.freeze_backbone,
            unfreeze_last=cfg.unfreeze_last_n_blocks,
        )
        hidden_dim = self.encoder.hidden_dim
        self.decoder = SegFormerHead(
            in_channels=[hidden_dim] * 4,
            embed_dim=cfg.decoder_embed_dim,
            num_classes=2,
            dropout=cfg.decoder_dropout,
        )

    def forward(self, x: torch.Tensor, drop_mask: torch.Tensor | None = None) -> torch.Tensor:
        features = self.encoder(x)
        logits = self.decoder(features, drop_mask=drop_mask)
        logits = F.interpolate(logits, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)[-1]


def enable_gradient_checkpointing(model: SegModel) -> None:
    try:
        encoder = model.encoder.backbone.encoder
        if hasattr(encoder, "gradient_checkpointing"):
            encoder.gradient_checkpointing = True
        elif hasattr(model.encoder.backbone, "gradient_checkpointing_enable"):
            model.encoder.backbone.gradient_checkpointing_enable()
    except Exception:
        return


def dice_loss(pred_prob: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    intersection = (pred_prob * target).sum(dim=(1, 2))
    denom = pred_prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return (1.0 - (2.0 * intersection + smooth) / (denom + smooth)).mean()


def supervised_loss(pred_logits: torch.Tensor, target: torch.Tensor, ce_w: float, dice_w: float) -> torch.Tensor:
    ce = F.cross_entropy(pred_logits, target)
    prob = pred_logits.softmax(dim=1)[:, 1]
    dice = dice_loss(prob, target.float())
    return ce_w * ce + dice_w * dice


def gaussian_swd_loss(feats1: torch.Tensor, feats2: torch.Tensor, n_projections: int) -> torch.Tensor:
    _, channels, _, _ = feats1.shape
    f1 = F.normalize(feats1.flatten(2).permute(0, 2, 1).reshape(-1, channels), dim=-1)
    f2 = F.normalize(feats2.flatten(2).permute(0, 2, 1).reshape(-1, channels), dim=-1)
    projections = F.normalize(torch.randn(n_projections, channels, device=feats1.device), dim=-1)
    p1 = f1 @ projections.T
    p2 = f2 @ projections.T
    loss_align = ((p1.mean(dim=0) - p2.mean(dim=0)) ** 2).mean()
    loss_uniform = -p1.var(dim=0).clamp(min=1e-8).log().mean() - p2.var(dim=0).clamp(min=1e-8).log().mean()
    return loss_align + 0.5 * loss_uniform


class AdaptiveThreshold:
    def __init__(self, start: float, end: float, warmup_epochs: int):
        self.start = float(start)
        self.end = float(end)
        self.warmup_epochs = int(max(1, warmup_epochs))

    def get(self, epoch: int) -> float:
        progress = min(max(epoch - 1, 0) / self.warmup_epochs, 1.0)
        return float(self.start - (self.start - self.end) * progress)


def cutmix_unlabeled(
    imgs_strong1: torch.Tensor,
    imgs_strong2: torch.Tensor,
    pseudo_labels: torch.Tensor,
    confidence_masks: torch.Tensor,
    p: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if random.random() > p:
        return imgs_strong2, pseudo_labels, confidence_masks

    batch_size, _, height, width = imgs_strong1.shape
    lam = np.random.beta(1.0, 1.0)
    center_x, center_y = np.random.randint(width), np.random.randint(height)
    cut_width = int(width * np.sqrt(1.0 - lam))
    cut_height = int(height * np.sqrt(1.0 - lam))
    x1 = max(center_x - cut_width // 2, 0)
    x2 = min(center_x + cut_width // 2, width)
    y1 = max(center_y - cut_height // 2, 0)
    y2 = min(center_y + cut_height // 2, height)

    mixed_images = imgs_strong2.clone()
    mixed_images[:, :, y1:y2, x1:x2] = imgs_strong1.roll(1, dims=0)[:, :, y1:y2, x1:x2]

    mixed_labels = pseudo_labels.clone()
    mixed_labels[:, y1:y2, x1:x2] = pseudo_labels.roll(1, dims=0)[:, y1:y2, x1:x2]

    mixed_conf = confidence_masks.clone()
    mixed_conf[:, y1:y2, x1:x2] = confidence_masks.roll(1, dims=0)[:, y1:y2, x1:x2]
    return mixed_images, mixed_labels, mixed_conf


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model: SegModel, loader: DataLoader, threshold: float, device: torch.device) -> tuple[float, float]:
    autocast_context = get_autocast_context(device)
    model.eval()
    ious: list[float] = []
    dices: list[float] = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            with autocast_context:
                logits = model(images)
            probs = logits.softmax(dim=1)[:, 1].detach().cpu().numpy()
            targets = masks.detach().cpu().numpy()
            for prob_map, target_mask in zip(probs, targets):
                pred_mask = (prob_map >= float(threshold)).astype(np.uint8)
                intersection = int((pred_mask & target_mask).sum())
                union = int((pred_mask | target_mask).sum())
                ious.append(float(intersection / (union + 1e-8)))
                denom = int(pred_mask.sum() + target_mask.sum())
                dices.append(float((2.0 * intersection) / (denom + 1e-8)))
    return float(np.mean(ious)), float(np.mean(dices))


def tune_threshold(model: SegModel, loader: DataLoader, device: torch.device, grid: list[float]) -> dict:
    autocast_context = get_autocast_context(device)
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            with autocast_context:
                logits = model(images)
            probabilities.extend(logits.softmax(dim=1)[:, 1].detach().cpu().numpy())
            targets.extend(masks.numpy())
    return tune_binary_threshold(probabilities, targets, threshold_grid=grid)


def tune_threshold_for_checkpoint(
    model: SegModel,
    checkpoint_path: Path,
    loader: DataLoader,
    device: torch.device,
    grid: list[float],
) -> dict:
    load_model_state(model, checkpoint_path, device)
    return tune_threshold(model, loader, device=device, grid=grid)


def load_model_state(model: nn.Module, checkpoint_path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint


def save_model_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_phase1(
    model: SegModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: RunConfig,
    device: torch.device,
) -> tuple[list[dict], Path]:
    backbone_params = [parameter for parameter in model.encoder.backbone.parameters() if parameter.requires_grad]
    decoder_params = list(model.decoder.parameters())
    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": cfg.phase1_lr_backbone},
            {"params": decoder_params, "lr": cfg.phase1_lr},
        ],
        weight_decay=cfg.phase1_weight_decay,
    )
    total_steps = cfg.phase1_epochs * max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_steps=int(total_steps * cfg.warmup_ratio))
    scaler = get_grad_scaler(device)
    autocast_context = get_autocast_context(device)

    best_miou = -1.0
    best_ckpt = cfg.ckpt_dir / "phase1_best.pt"
    history: list[dict] = []

    for epoch in range(1, cfg.phase1_epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context:
                logits = model(images)
                loss = supervised_loss(logits, masks, cfg.loss_ce_weight, cfg.loss_dice_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(1, len(train_loader))
        val_miou, val_dice = evaluate(model, val_loader, threshold=0.5, device=device)
        elapsed = time.time() - start_time
        row = {
            "epoch": epoch,
            "loss": avg_loss,
            "val_miou": val_miou,
            "val_dice": val_dice,
            "elapsed_sec": elapsed,
        }
        history.append(row)
        if val_miou > best_miou:
            best_miou = val_miou
            save_model_checkpoint(
                best_ckpt,
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_miou": val_miou,
                    "val_dice": val_dice,
                    "config": config_payload(cfg),
                },
            )

        print(
            f"[P1 E{epoch:03d}/{cfg.phase1_epochs}] "
            f"loss={avg_loss:.4f} val_mIoU={val_miou:.4f} val_dice={val_dice:.4f} best={best_miou:.4f} "
            f"time={elapsed:.0f}s"
        )
        empty_device_cache(device)

    save_json(cfg.metrics_dir / "phase1_history.json", history)
    return history, best_ckpt


def predict_member_probability_tiled(
    member,
    image_rgb: np.ndarray,
    device: torch.device,
    tile_size: int,
    overlap: float,
) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    if max(height, width) <= tile_size:
        return predict_member_probability(member, image_rgb, device, use_tta=True)

    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    accum = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    for top in range(0, height, stride):
        for left in range(0, width, stride):
            bottom = min(height, top + tile_size)
            right = min(width, left + tile_size)
            top = max(0, bottom - tile_size)
            left = max(0, right - tile_size)
            crop = image_rgb[top:bottom, left:right]
            prob = predict_member_probability(member, crop, device, use_tta=True)
            crop_h, crop_w = prob.shape
            weight = np.ones((crop_h, crop_w), dtype=np.float32)
            accum[top:bottom, left:right] += prob * weight
            weights[top:bottom, left:right] += weight
    return accum / np.clip(weights, a_min=1e-6, a_max=None)


def build_teacher_members(cfg: RunConfig, device: torch.device) -> list:
    members = []
    for run_name, weight in resolve_member_specs(preset=cfg.teacher_preset):
        member = load_member(RUNS_ROOT / "supervised_v4", run_name, weight, device)
        member.config.tta_scales = [float(scale) for scale in cfg.teacher_tta_scales]
        member.config.tta_ops = str(cfg.teacher_tta_ops)
        member.config.postprocess_enabled = True
        member.config.postprocess_min_component_area = int(cfg.teacher_postprocess_min_component_area)
        member.config.postprocess_fill_holes = bool(cfg.teacher_postprocess_fill_holes)
        members.append(member)
    return members


def predict_teacher_probability(members: list, image_rgb: np.ndarray, cfg: RunConfig, device: torch.device) -> np.ndarray:
    probability_maps = []
    for member in members:
        if cfg.teacher_enable_sahi and max(image_rgb.shape[:2]) > cfg.teacher_sahi_tile_size:
            probability_map = predict_member_probability_tiled(
                member,
                image_rgb=image_rgb,
                device=device,
                tile_size=cfg.teacher_sahi_tile_size,
                overlap=cfg.teacher_sahi_overlap,
            )
        else:
            probability_map = predict_member_probability(member, image_rgb, device, use_tta=True)
        probability_maps.append(probability_map)
    return aggregate_probability_maps(
        probability_maps=probability_maps,
        weights=[member.weight for member in members],
        aggregation=cfg.teacher_aggregation,
        member_threshold=cfg.teacher_member_threshold,
    )


def compute_teacher_stats(mask: np.ndarray, confidence: np.ndarray) -> dict:
    reliable = (confidence >= 0.90).astype(np.uint8)
    fg_mask = mask.astype(bool)
    bg_mask = ~fg_mask
    reliable_ratio = float(reliable.mean())
    mean_confidence = float(confidence.mean())
    fg_reliable_ratio = float(reliable[fg_mask].mean()) if fg_mask.any() else 1.0
    bg_reliable_ratio = float(reliable[bg_mask].mean()) if bg_mask.any() else 1.0
    area_ratio = float(mask.mean())
    selection_score = 0.50 * reliable_ratio + 0.35 * mean_confidence + 0.15 * fg_reliable_ratio
    return {
        "reliable_ratio": reliable_ratio,
        "mean_confidence": mean_confidence,
        "fg_reliable_ratio": fg_reliable_ratio,
        "bg_reliable_ratio": bg_reliable_ratio,
        "area_ratio": area_ratio,
        "selection_score": selection_score,
    }


def accept_teacher_sample(stats: dict, cfg: RunConfig) -> bool:
    return bool(
        stats["reliable_ratio"] >= cfg.teacher_accept_min_reliable_ratio
        and stats["mean_confidence"] >= cfg.teacher_accept_min_mean_confidence
        and stats["fg_reliable_ratio"] >= cfg.teacher_accept_min_fg_reliable_ratio
        and cfg.teacher_min_object_ratio <= stats["area_ratio"] <= cfg.teacher_max_object_ratio
    )


def build_teacher_cache(cfg: RunConfig, unlabeled_paths: list[Path], device: torch.device) -> list[TeacherPseudoEntry]:
    summary_path = cfg.teacher_dir / "summary.json"
    if summary_path.exists() and not cfg.teacher_overwrite_cache:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        entries = [
            TeacherPseudoEntry(
                image_path=Path(row["image_path"]),
                mask_path=Path(row["mask_path"]),
                confidence_path=Path(row["confidence_path"]),
                reliable_ratio=float(row["reliable_ratio"]),
                mean_confidence=float(row["mean_confidence"]),
                fg_reliable_ratio=float(row["fg_reliable_ratio"]),
                area_ratio=float(row["area_ratio"]),
                selection_score=float(row["selection_score"]),
                accepted=bool(row["accepted"]),
                accept_reason=str(row["accept_reason"]),
            )
            for row in payload["rows"]
        ]
        print(f"[teacher] Reusing cached teacher labels: {len(entries)} rows from {summary_path}")
        return entries

    cfg.teacher_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = cfg.teacher_dir / "masks"
    confidence_dir = cfg.teacher_dir / "confidence"
    if cfg.teacher_overwrite_cache and cfg.teacher_dir.exists():
        shutil.rmtree(cfg.teacher_dir)
        cfg.teacher_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    confidence_dir.mkdir(parents=True, exist_ok=True)

    members = build_teacher_members(cfg, device)
    postprocess_cfg = copy.deepcopy(members[0].config)
    rows: list[TeacherPseudoEntry] = []
    print(f"[teacher] Building teacher cache for {len(unlabeled_paths)} images | preset={cfg.teacher_preset} | tta={cfg.teacher_tta_ops}/{cfg.teacher_tta_scales}")

    for image_path in tqdm(unlabeled_paths, desc="Teacher pseudo-labels"):
        image_rgb = read_image_rgb(image_path)
        probability_map = predict_teacher_probability(members, image_rgb=image_rgb, cfg=cfg, device=device)
        confidence = np.maximum(probability_map, 1.0 - probability_map).astype(np.float32)
        mask = (probability_map >= float(cfg.teacher_threshold)).astype(np.uint8)
        mask = postprocess_binary_mask(mask, postprocess_cfg)
        stats = compute_teacher_stats(mask, confidence)
        accepted = accept_teacher_sample(stats, cfg)
        accept_reason = "strict" if accepted else "rejected"

        mask_path = mask_dir / f"{image_path.stem}.png"
        confidence_path = confidence_dir / f"{image_path.stem}.npy"
        if accepted or cfg.teacher_save_all:
            save_mask_png(mask_path, mask)
            np.save(confidence_path, confidence.astype(np.float16))

        rows.append(
            TeacherPseudoEntry(
                image_path=image_path,
                mask_path=mask_path,
                confidence_path=confidence_path,
                reliable_ratio=stats["reliable_ratio"],
                mean_confidence=stats["mean_confidence"],
                fg_reliable_ratio=stats["fg_reliable_ratio"],
                area_ratio=stats["area_ratio"],
                selection_score=stats["selection_score"],
                accepted=accepted,
                accept_reason=accept_reason,
            )
        )

    accepted_rows = [row for row in rows if row.accepted]
    if len(accepted_rows) < cfg.teacher_min_samples:
        fallback_rows = sorted(rows, key=lambda row: row.selection_score, reverse=True)[: min(cfg.teacher_min_samples, len(rows))]
        for row in fallback_rows:
            if not row.mask_path.exists():
                image_rgb = read_image_rgb(row.image_path)
                probability_map = predict_teacher_probability(members, image_rgb=image_rgb, cfg=cfg, device=device)
                confidence = np.maximum(probability_map, 1.0 - probability_map).astype(np.float32)
                mask = (probability_map >= float(cfg.teacher_threshold)).astype(np.uint8)
                mask = postprocess_binary_mask(mask, postprocess_cfg)
                save_mask_png(row.mask_path, mask)
                np.save(row.confidence_path, confidence.astype(np.float16))
            row.accepted = True
            if row.accept_reason == "rejected":
                row.accept_reason = "fallback_top_score"
        accepted_rows = [row for row in rows if row.accepted]

    payload = {
        "summary": {
            "teacher_preset": cfg.teacher_preset,
            "teacher_threshold": cfg.teacher_threshold,
            "teacher_tta_ops": cfg.teacher_tta_ops,
            "teacher_tta_scales": cfg.teacher_tta_scales,
            "teacher_enable_sahi": cfg.teacher_enable_sahi,
            "num_total": len(rows),
            "num_accepted": len(accepted_rows),
            "accepted_ratio": len(accepted_rows) / max(1, len(rows)),
        },
        "rows": [
            {
                "image_path": str(row.image_path),
                "mask_path": str(row.mask_path),
                "confidence_path": str(row.confidence_path),
                "reliable_ratio": row.reliable_ratio,
                "mean_confidence": row.mean_confidence,
                "fg_reliable_ratio": row.fg_reliable_ratio,
                "area_ratio": row.area_ratio,
                "selection_score": row.selection_score,
                "accepted": row.accepted,
                "accept_reason": row.accept_reason,
            }
            for row in rows
        ],
    }
    save_json(summary_path, payload)
    return rows


def build_unlabeled_loader(entries: list[TeacherPseudoEntry], cfg: RunConfig) -> DataLoader:
    accepted_entries = [entry for entry in entries if entry.accepted]
    if not accepted_entries:
        raise RuntimeError("Teacher cache produced zero accepted entries.")

    geom_transform = build_unlabeled_geom_transform(cfg.image_size)
    photo_transform = build_unlabeled_photo_transform()
    normalize_transform = build_val_transform(cfg.image_size)
    dataset = TeacherPseudoDataset(
        entries=accepted_entries,
        geom_transform=geom_transform,
        photo_transform=photo_transform,
        normalize_transform=normalize_transform,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size_unlabeled,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )


def train_step_unimatch(
    model: SegModel,
    labeled_batch,
    unlabeled_batch,
    optimizer: torch.optim.Optimizer,
    scaler,
    epoch: int,
    cfg: RunConfig,
    adaptive_threshold: AdaptiveThreshold,
    device: torch.device,
) -> dict:
    autocast_context = get_autocast_context(device)
    imgs_l, masks_l = labeled_batch
    imgs_s1, imgs_s2, pseudo1, pseudo2, conf1, conf2 = unlabeled_batch

    imgs_l = imgs_l.to(device)
    masks_l = masks_l.to(device)
    imgs_s1 = imgs_s1.to(device)
    imgs_s2 = imgs_s2.to(device)
    pseudo1 = pseudo1.to(device)
    pseudo2 = pseudo2.to(device)
    conf1 = conf1.to(device)
    conf2 = conf2.to(device)

    optimizer.zero_grad(set_to_none=True)

    with autocast_context:
        pred_l = model(imgs_l)
        loss_sup = supervised_loss(pred_l, masks_l, cfg.loss_ce_weight, cfg.loss_dice_weight)

    conf_threshold = adaptive_threshold.get(epoch)
    conf_mask1 = conf1 >= conf_threshold
    conf_mask2 = conf2 >= conf_threshold
    acceptance = float(torch.cat([conf_mask1.float().reshape(conf_mask1.size(0), -1), conf_mask2.float().reshape(conf_mask2.size(0), -1)], dim=1).mean().item())

    imgs_s2, pseudo2, conf2 = cutmix_unlabeled(imgs_s1, imgs_s2, pseudo2, conf_mask2.float(), p=0.5)
    conf_mask2 = conf2 >= 0.5

    batch_size = imgs_s1.shape[0]
    drop_fwd = torch.bernoulli(
        torch.full(
            (batch_size, 1, max(1, imgs_s1.shape[-2] // 14), max(1, imgs_s1.shape[-1] // 14)),
            1.0 - cfg.complementary_dropout_p,
            device=device,
        )
    ).bool()
    drop_fwd_up = F.interpolate(drop_fwd.float(), size=(imgs_s1.shape[-2], imgs_s1.shape[-1]), mode="nearest").bool()

    with autocast_context:
        pred_s1 = model(imgs_s1, drop_mask=drop_fwd_up)
        pred_s2 = model(imgs_s2, drop_mask=~drop_fwd_up)
        feats_s1 = model.get_features(imgs_s1)
        feats_s2 = model.get_features(imgs_s2)

    with autocast_context:
        weight1 = conf_mask1.float()
        weight2 = conf_mask2.float()
        denom1 = weight1.sum().clamp_min(1.0)
        denom2 = weight2.sum().clamp_min(1.0)
        loss_u1 = (F.cross_entropy(pred_s1, pseudo1, reduction="none") * weight1).sum() / denom1
        loss_u2 = (F.cross_entropy(pred_s2, pseudo2, reduction="none") * weight2).sum() / denom2
        loss_u = 0.5 * (loss_u1 + loss_u2)
        loss_swd = gaussian_swd_loss(feats_s1, feats_s2, cfg.swd_projections)
        warmup = min(1.0, epoch / max(1, cfg.warmup_epochs))
        loss = loss_sup + cfg.unsup_weight * warmup * loss_u + cfg.swd_weight * loss_swd

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    scaler.step(optimizer)
    scaler.update()

    return {
        "loss": float(loss.item()),
        "loss_sup": float(loss_sup.item()),
        "loss_u": float(loss_u.item()),
        "loss_swd": float(loss_swd.item()),
        "acceptance": acceptance,
        "conf_threshold": conf_threshold,
    }


def train_phase2(
    model: SegModel,
    train_loader: DataLoader,
    unlabeled_loader: DataLoader,
    val_loader: DataLoader,
    phase1_ckpt: Path,
    cfg: RunConfig,
    device: torch.device,
    eval_threshold: float,
) -> tuple[list[dict], Path]:
    load_model_state(model, phase1_ckpt, device)

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg.phase2_lr,
        weight_decay=cfg.phase1_weight_decay,
    )
    total_steps = cfg.phase2_epochs * max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_steps=int(total_steps * cfg.warmup_ratio))
    scaler = get_grad_scaler(device)
    adaptive_threshold = AdaptiveThreshold(cfg.conf_threshold_start, cfg.conf_threshold_end, cfg.conf_warmup_epochs)

    best_miou = -1.0
    best_ckpt = cfg.ckpt_dir / "phase2_best.pt"
    history: list[dict] = []
    unlabeled_iter = iter(unlabeled_loader)

    for epoch in range(1, cfg.phase2_epochs + 1):
        model.train()
        start_time = time.time()
        epoch_stats = {"loss": 0.0, "loss_sup": 0.0, "loss_u": 0.0, "loss_swd": 0.0, "acceptance": 0.0}
        steps = 0
        for labeled_batch in train_loader:
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            stats = train_step_unimatch(
                model=model,
                labeled_batch=labeled_batch,
                unlabeled_batch=unlabeled_batch,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                cfg=cfg,
                adaptive_threshold=adaptive_threshold,
                device=device,
            )
            scheduler.step()
            steps += 1
            for key in epoch_stats:
                epoch_stats[key] += stats[key]

        for key in epoch_stats:
            epoch_stats[key] /= max(1, steps)
        val_miou, val_dice = evaluate(model, val_loader, threshold=eval_threshold, device=device)
        elapsed = time.time() - start_time
        row = {
            "epoch": epoch,
            **epoch_stats,
            "conf_threshold": adaptive_threshold.get(epoch),
            "val_miou": val_miou,
            "val_dice": val_dice,
            "elapsed_sec": elapsed,
        }
        history.append(row)
        if val_miou > best_miou:
            best_miou = val_miou
            save_model_checkpoint(
                best_ckpt,
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_miou": val_miou,
                    "val_dice": val_dice,
                    "config": config_payload(cfg),
                },
            )

        print(
            f"[P2 E{epoch:03d}/{cfg.phase2_epochs}] "
            f"loss={epoch_stats['loss']:.4f} sup={epoch_stats['loss_sup']:.4f} "
            f"u={epoch_stats['loss_u']:.4f} swd={epoch_stats['loss_swd']:.4f} "
            f"accept={epoch_stats['acceptance']:.3f} thr={row['conf_threshold']:.3f} "
            f"val_mIoU={val_miou:.4f} val_dice={val_dice:.4f} best={best_miou:.4f} "
            f"time={elapsed:.0f}s"
        )
        empty_device_cache(device)

    save_json(cfg.metrics_dir / "phase2_history.json", history)
    return history, best_ckpt


def train_phase3(
    model: SegModel,
    labeled_samples: list[tuple[Path, Path]],
    pseudo_entries: list[TeacherPseudoEntry],
    val_loader: DataLoader,
    phase2_ckpt: Path,
    cfg: RunConfig,
    device: torch.device,
    eval_threshold: float,
) -> tuple[list[dict], Path]:
    load_model_state(model, phase2_ckpt, device)
    combined_samples = labeled_samples + [(entry.image_path, entry.mask_path) for entry in pseudo_entries if entry.accepted]
    train_loader = DataLoader(
        LabeledDataset(combined_samples, build_labeled_transform(cfg.image_size)),
        batch_size=cfg.batch_size_labeled,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg.phase2_lr * cfg.phase3_lr_scale,
        weight_decay=cfg.phase1_weight_decay,
    )
    scaler = get_grad_scaler(device)
    autocast_context = get_autocast_context(device)
    best_miou = -1.0
    best_ckpt = cfg.ckpt_dir / "phase3_best.pt"
    history: list[dict] = []

    for epoch in range(1, cfg.phase3_epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context:
                logits = model(images)
                loss = supervised_loss(logits, masks, cfg.loss_ce_weight, cfg.loss_dice_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(1, len(train_loader))
        val_miou, val_dice = evaluate(model, val_loader, threshold=eval_threshold, device=device)
        elapsed = time.time() - start_time
        history.append(
            {
                "epoch": epoch,
                "loss": avg_loss,
                "val_miou": val_miou,
                "val_dice": val_dice,
                "elapsed_sec": elapsed,
            }
        )
        if val_miou > best_miou:
            best_miou = val_miou
            save_model_checkpoint(
                best_ckpt,
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_miou": val_miou,
                    "val_dice": val_dice,
                    "config": config_payload(cfg),
                },
            )

        print(
            f"[P3 E{epoch:03d}/{cfg.phase3_epochs}] "
            f"loss={avg_loss:.4f} val_mIoU={val_miou:.4f} val_dice={val_dice:.4f} best={best_miou:.4f} "
            f"time={elapsed:.0f}s"
        )
        empty_device_cache(device)

    save_json(cfg.metrics_dir / "phase3_history.json", history)
    return history, best_ckpt


def export_preview_cards(
    model: SegModel,
    checkpoint_path: Path,
    samples: list[tuple[Path, Path]],
    cfg: RunConfig,
    device: torch.device,
    threshold: float,
) -> None:
    if not samples:
        return
    load_model_state(model, checkpoint_path, device)
    model.eval()
    transform = build_val_transform(cfg.image_size)
    autocast_context = get_autocast_context(device)
    cfg.preview_dir.mkdir(parents=True, exist_ok=True)

    for idx, (image_path, mask_path) in enumerate(samples[: cfg.preview_limit]):
        image_rgb = read_image_rgb(image_path)
        target_mask = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)
        tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            with autocast_context:
                logits = model(tensor)
        prob = logits.softmax(dim=1)[:, 1].squeeze(0).detach().cpu().numpy()
        pred = (prob >= float(threshold)).astype(np.uint8)
        pred = cv2.resize(pred.astype(np.uint8), (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        axes[0].imshow(image_rgb)
        axes[0].set_title("image")
        axes[1].imshow(target_mask, cmap="gray")
        axes[1].set_title("gt")
        axes[2].imshow(pred, cmap="gray")
        axes[2].set_title("pred")
        for axis in axes:
            axis.axis("off")
        fig.suptitle(f"{image_path.name} | dice={binary_dice_score(pred, target_mask):.4f}")
        fig.tight_layout()
        fig.savefig(cfg.preview_dir / f"preview_{idx:02d}_{image_path.stem}.png", dpi=150)
        plt.close(fig)


def main() -> int:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = get_device()
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)
    cfg.preview_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg.run_dir / "config.json", config_payload(cfg))

    if cfg.source_notebook.exists():
        shutil.copy2(cfg.source_notebook, cfg.run_dir / cfg.source_notebook.name)

    print(f"Run dir  : {cfg.run_dir}")
    print(f"Device   : {device}")
    print(f"Notebook : {cfg.source_notebook}")
    print(f"Teacher  : {cfg.teacher_preset} | TTA={cfg.teacher_tta_ops}/{cfg.teacher_tta_scales} | SAHI={cfg.teacher_enable_sahi}")

    labeled_samples = collect_labeled_pairs(cfg.train_img_dir, cfg.train_mask_dir)
    train_samples, val_samples = split_grouped(labeled_samples, n_splits=cfg.n_splits, fold_index=cfg.fold_index)
    if cfg.smoke:
        train_samples = train_samples[:24]
        val_samples = val_samples[:8]
    unlabeled_paths: list[Path] = []
    for directory in cfg.unlabeled_dirs:
        unlabeled_paths.extend(collect_image_paths(directory))
    unlabeled_paths = sorted(dict.fromkeys(unlabeled_paths))
    if cfg.teacher_limit is not None:
        unlabeled_paths = unlabeled_paths[: cfg.teacher_limit]

    print(f"Labeled   : train={len(train_samples)} val={len(val_samples)}")
    print(f"Unlabeled : {len(unlabeled_paths)}")

    train_loader = DataLoader(
        LabeledDataset(train_samples, build_labeled_transform(cfg.image_size)),
        batch_size=cfg.batch_size_labeled,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        LabeledDataset(val_samples, build_val_transform(cfg.image_size)),
        batch_size=max(1, cfg.batch_size_labeled),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

    model = SegModel(cfg).to(device)
    if cfg.enable_gradient_checkpointing:
        enable_gradient_checkpointing(model)

    phase1_history, phase1_ckpt = train_phase1(model, train_loader, val_loader, cfg, device)
    phase1_threshold = tune_threshold_for_checkpoint(
        model,
        checkpoint_path=phase1_ckpt,
        loader=val_loader,
        device=device,
        grid=cfg.threshold_grid,
    )
    save_json(cfg.metrics_dir / "phase1_threshold.json", phase1_threshold)

    teacher_entries = build_teacher_cache(cfg, unlabeled_paths, device)
    teacher_summary = json.loads((cfg.teacher_dir / "summary.json").read_text(encoding="utf-8"))
    save_json(cfg.metrics_dir / "teacher_summary.json", teacher_summary)

    if not any(entry.accepted for entry in teacher_entries):
        raise RuntimeError("Teacher cache produced no accepted pseudo labels.")

    unlabeled_loader = build_unlabeled_loader(teacher_entries, cfg)
    phase2_history, phase2_ckpt = train_phase2(
        model=model,
        train_loader=train_loader,
        unlabeled_loader=unlabeled_loader,
        val_loader=val_loader,
        phase1_ckpt=phase1_ckpt,
        cfg=cfg,
        device=device,
        eval_threshold=float(phase1_threshold["best_threshold"]),
    )
    phase2_threshold = tune_threshold_for_checkpoint(
        model,
        checkpoint_path=phase2_ckpt,
        loader=val_loader,
        device=device,
        grid=cfg.threshold_grid,
    )
    save_json(cfg.metrics_dir / "phase2_threshold.json", phase2_threshold)

    final_ckpt = phase2_ckpt
    final_threshold = phase2_threshold
    phase3_history: list[dict] = []
    phase2_gain = float(phase2_threshold["best_dice"] - phase1_threshold["best_dice"])
    accepted_entries = [entry for entry in teacher_entries if entry.accepted]
    if phase2_gain >= cfg.phase3_min_improvement and len(accepted_entries) >= cfg.phase3_min_samples:
        phase3_history, phase3_ckpt = train_phase3(
            model=model,
            labeled_samples=train_samples,
            pseudo_entries=accepted_entries,
            val_loader=val_loader,
            phase2_ckpt=phase2_ckpt,
            cfg=cfg,
            device=device,
            eval_threshold=float(phase2_threshold["best_threshold"]),
        )
        phase3_threshold = tune_threshold_for_checkpoint(
            model,
            checkpoint_path=phase3_ckpt,
            loader=val_loader,
            device=device,
            grid=cfg.threshold_grid,
        )
        save_json(cfg.metrics_dir / "phase3_threshold.json", phase3_threshold)
        final_ckpt = phase3_ckpt
        final_threshold = phase3_threshold

    export_preview_cards(
        model=model,
        checkpoint_path=final_ckpt,
        samples=val_samples,
        cfg=cfg,
        device=device,
        threshold=float(final_threshold["best_threshold"]),
    )

    summary = {
        "device": str(device),
        "run_dir": str(cfg.run_dir),
        "phase1_ckpt": str(phase1_ckpt),
        "phase2_ckpt": str(phase2_ckpt),
        "final_ckpt": str(final_ckpt),
        "phase1_best_dice": float(phase1_threshold["best_dice"]),
        "phase1_best_threshold": float(phase1_threshold["best_threshold"]),
        "phase2_best_dice": float(phase2_threshold["best_dice"]),
        "phase2_best_threshold": float(phase2_threshold["best_threshold"]),
        "phase2_gain_dice": phase2_gain,
        "teacher_accepted": int(sum(entry.accepted for entry in teacher_entries)),
        "teacher_total": int(len(teacher_entries)),
        "phase3_ran": bool(phase3_history),
        "preview_dir": str(cfg.preview_dir),
        "metrics_dir": str(cfg.metrics_dir),
    }
    if phase3_history:
        phase3_threshold = json.loads((cfg.metrics_dir / "phase3_threshold.json").read_text(encoding="utf-8"))
        summary["phase3_best_dice"] = float(phase3_threshold["best_dice"])
        summary["phase3_best_threshold"] = float(phase3_threshold["best_threshold"])

    save_json(cfg.run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
