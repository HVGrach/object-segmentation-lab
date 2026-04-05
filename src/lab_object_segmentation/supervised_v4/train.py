#!/usr/bin/env python3
"""Standalone supervised SegFormer-B2 training script (V4).

This script implements the agreed supervised-only pipeline:
- SegFormer-B2 encoder
- multi-scale concat decoder with Lite-ASPP + lightweight attention
- auxiliary boundary head
- grouped-by-camera fold split
- EMA model selection
- scheduled heavy augmentations with Copy-Paste

Run from the project root, for example:

python scripts/train_supervised_v4.py \
    --run-name segformer_b2_fold1_320_moderate_ls003_ema \
    --fold 1 \
    --image-size 320 \
    --epochs 60 \
    --aug moderate \
    --label-smoothing 0.03 \
    --physical-batch-size 8
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
import warnings
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import SegformerModel

from lab_object_segmentation.common.paths import LAB3_DATASET_ROOT, PROJECT_ROOT, RUNS_ROOT
from lab_object_segmentation.segformer_semisup.utils import (
    binary_dice_score,
    infer_camera_group,
    tune_binary_threshold,
)


warnings.filterwarnings("ignore")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
RESIZE_INTERPOLATION_CHOICES = ("nearest", "linear", "cubic", "area", "lanczos4")
TTA_OP_CHOICES = ("flips", "d4")


@dataclass
class TrainConfig:
    run_name: str
    fold: int
    image_size: int
    epochs: int
    aug: str
    label_smoothing: float
    physical_batch_size: int

    effective_batch_size: int = 16
    backbone: str = "nvidia/mit-b2"
    base_lr: float = 6e-5
    llrd_factor: float = 0.7
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    grad_clip_norm: float = 1.0
    boundary_loss_weight: float = 0.1
    focal_gamma: float = 2.0
    focal_alpha_pos: float = 0.75
    mask_loss: str = "dice_focal"
    resize_interpolation: str = "linear"
    ema_decay: float = 0.999
    save_every_n_epochs: int = 5
    eval_every_n_epochs: int = 2
    num_workers: int = 0
    pin_memory: bool = False
    seed: int = 42
    selection_metric: str = "dice_tuned"
    threshold_grid: list[float] = field(
        default_factory=lambda: [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    )
    image_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    image_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    copy_paste_p: float = 0.30
    copy_paste_two_objects_p: float = 0.30
    copy_paste_scale_range: tuple[float, float] = (0.6, 1.4)
    boundary_kernel_size: int = 3
    tta_scales: list[float] = field(default_factory=lambda: [0.75, 1.0, 1.25])
    tta_ops: str = "flips"
    postprocess_enabled: bool = True
    postprocess_min_component_area: int = 128
    postprocess_fill_holes: bool = True
    resume: bool = False
    finetune_from: str = ""
    finetune_lr_scale: float = 0.3
    final_tta: bool = True

    @property
    def accumulation_steps(self) -> int:
        return max(1, self.effective_batch_size // max(1, self.physical_batch_size))

    @property
    def run_dir(self) -> Path:
        return RUNS_ROOT / "supervised_v4" / self.run_name


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train SegFormer-B2 supervised V4 pipeline.")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--aug", choices=["geom", "moderate", "heavy"], required=True)
    parser.add_argument("--label-smoothing", type=float, required=True)
    parser.add_argument("--mask-loss", choices=["dice_focal", "lovasz_focal"], default="dice_focal")
    parser.add_argument(
        "--resize-interpolation",
        choices=RESIZE_INTERPOLATION_CHOICES,
        default="linear",
        help="Interpolation for image resize in train/val transforms.",
    )
    parser.add_argument("--physical-batch-size", type=int, default=8)
    parser.add_argument("--effective-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every-n-epochs", type=int, default=2, help="Evaluate every N epochs (set to 1 for short finetune runs)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest_model.pth in run_dir")
    parser.add_argument("--finetune-from", type=str, default="", help="Path to checkpoint .pth to load model+ema weights from (fresh optimizer/scheduler)")
    parser.add_argument("--finetune-lr-scale", type=float, default=0.3, help="Scale base_lr by this factor for fine-tuning (default 0.3 -> ~1.8e-5)")
    parser.add_argument("--tta-scales", type=float, nargs="+", default=[0.75, 1.0, 1.25], help="Multi-scale TTA scales for final evaluation")
    parser.add_argument(
        "--tta-ops",
        choices=TTA_OP_CHOICES,
        default="flips",
        help="Test-time augmentation transform family: flips or D4-style rotations+flips.",
    )
    parser.add_argument("--disable-postprocess", action="store_true", help="Disable mask post-processing during evaluation")
    parser.add_argument("--postprocess-min-component-area", type=int, default=128, help="Remove connected components smaller than this area")
    parser.add_argument("--disable-postprocess-fill-holes", action="store_true", help="Disable filling holes inside predicted masks")
    args = parser.parse_args()

    if args.fold not in {0, 1, 2}:
        raise ValueError(f"Unsupported fold index: {args.fold}")
    if args.physical_batch_size <= 0:
        raise ValueError("physical-batch-size must be positive.")
    if args.effective_batch_size < args.physical_batch_size:
        raise ValueError("effective-batch-size must be >= physical-batch-size.")
    if args.effective_batch_size % args.physical_batch_size != 0:
        raise ValueError("effective-batch-size must be divisible by physical-batch-size.")

    return TrainConfig(
        run_name=args.run_name,
        fold=args.fold,
        image_size=args.image_size,
        epochs=args.epochs,
        aug=args.aug,
        label_smoothing=args.label_smoothing,
        mask_loss=args.mask_loss,
        resize_interpolation=args.resize_interpolation,
        physical_batch_size=args.physical_batch_size,
        effective_batch_size=args.effective_batch_size,
        seed=args.seed,
        eval_every_n_epochs=args.eval_every_n_epochs,
        tta_scales=[float(scale) for scale in args.tta_scales],
        tta_ops=args.tta_ops,
        postprocess_enabled=not args.disable_postprocess,
        postprocess_min_component_area=args.postprocess_min_component_area,
        postprocess_fill_holes=not args.disable_postprocess_fill_holes,
        resume=args.resume,
        finetune_from=args.finetune_from,
        finetune_lr_scale=args.finetune_lr_scale,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def update_ema(teacher_model: nn.Module, student_model: nn.Module, decay: float = 0.999) -> None:
    for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
        t_param.data = decay * t_param.data + (1.0 - decay) * s_param.data
    for t_buffer, s_buffer in zip(teacher_model.buffers(), student_model.buffers()):
        t_buffer.data.copy_(s_buffer.data)


def build_scheduler(optimizer, total_epochs: int, warmup_epochs: int, min_lr: float):
    if total_epochs <= 1:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (
                0.01 + 0.99 * ((epoch + 1) / max(1, warmup_epochs))
                if epoch < warmup_epochs
                else 1.0
            ),
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs - warmup_epochs),
            eta_min=min_lr,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs),
        eta_min=min_lr,
    )


def empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        return
    if device.type == "mps" and hasattr(torch, "mps"):
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def collect_image_paths(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted([p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def collect_labeled_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    image_map = {p.stem: p for p in collect_image_paths(images_dir)}
    samples: list[tuple[Path, Path]] = []
    missing_images: list[str] = []
    for mask_path in sorted(masks_dir.rglob("*")):
        if not mask_path.is_file() or mask_path.suffix.lower() not in MASK_EXTS:
            continue
        image_path = image_map.get(mask_path.stem)
        if image_path is None:
            missing_images.append(mask_path.name)
            continue
        samples.append((image_path, mask_path))
    if missing_images:
        print(f"[WARN] masks without matching images: {len(missing_images)}")
    if not samples:
        raise RuntimeError("No labeled image/mask pairs were found.")
    return samples


def load_val_sample_ids(project_root: Path, fold: int) -> set[str]:
    candidate_paths = sorted((RUNS_ROOT / "advanced_baseline" / "runs").glob(f"*/fold_{fold}/val_sample_ids.json"))
    if candidate_paths:
        payload = json.loads(candidate_paths[0].read_text(encoding="utf-8"))
        sample_ids = payload["sample_ids"] if isinstance(payload, dict) else payload
        return set(sample_ids)

    folds_summary_path = RUNS_ROOT / "advanced_baseline" / "folds_summary.json"
    if not folds_summary_path.exists():
        raise FileNotFoundError("Could not find val_sample_ids.json or folds_summary.json for fold split.")

    folds_summary = json.loads(folds_summary_path.read_text(encoding="utf-8"))
    fold_payload = folds_summary["folds"][fold]
    val_preview = fold_payload.get("val_groups_preview", [])
    val_count = int(fold_payload.get("val_group_count", len(val_preview)))
    if len(val_preview) == val_count and val_preview:
        all_samples = collect_labeled_pairs(
            LAB3_DATASET_ROOT / "train" / "images",
            LAB3_DATASET_ROOT / "train" / "masks",
        )
        return {image_path.name for image_path, _ in all_samples if infer_camera_group(image_path) in set(val_preview)}

    raise RuntimeError("Could not recover a complete grouped split for the requested fold.")


def split_samples_by_fold(samples: list[tuple[Path, Path]], val_sample_ids: set[str]) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    train_samples = [sample for sample in samples if sample[0].name not in val_sample_ids]
    val_samples = [sample for sample in samples if sample[0].name in val_sample_ids]
    if not train_samples or not val_samples:
        raise RuntimeError("Fold split produced an empty train or validation set.")
    return train_samples, val_samples


def generate_boundary_mask(binary_mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    binary_mask = (binary_mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(binary_mask, kernel)
    eroded = cv2.erode(binary_mask, kernel)
    return (dilated - eroded > 0).astype(np.float32)


def resolve_resize_interpolation(name: str) -> int:
    interpolation_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos4": cv2.INTER_LANCZOS4,
    }
    try:
        return interpolation_map[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported resize interpolation: {name}") from exc


def build_geom_transform(image_size: int, mean: list[float], std: list[float], resize_interpolation: str) -> A.Compose:
    image_resize_interpolation = resolve_resize_interpolation(resize_interpolation)
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.25),
            A.RandomRotate90(p=0.5),
            A.Resize(image_size, image_size, interpolation=image_resize_interpolation, mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def build_moderate_transform(image_size: int, mean: list[float], std: list[float], resize_interpolation: str) -> A.Compose:
    image_resize_interpolation = resolve_resize_interpolation(resize_interpolation)
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.25),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.0, 0.05),
                rotate=(-20, 20),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_REFLECT_101,
                fill=0,
                fill_mask=0,
                p=0.4,
            ),
            A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.06, p=0.5),
            A.GaussNoise(std_range=(0.04, 0.12), p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.15),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=0,
                fill_mask=0,
                p=0.15,
            ),
            A.Resize(image_size, image_size, interpolation=image_resize_interpolation, mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def build_heavy_transform(image_size: int, mean: list[float], std: list[float], resize_interpolation: str) -> A.Compose:
    image_resize_interpolation = resolve_resize_interpolation(resize_interpolation)
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(0.0, 0.10),
                rotate=(-45, 45),
                shear=(-10, 10),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_REFLECT_101,
                fill=0,
                fill_mask=0,
                p=0.6,
            ),
            A.OneOf(
                [
                    A.GridDistortion(num_steps=5, distort_limit=0.3),
                    A.ElasticTransform(),
                    A.Perspective(),
                ],
                p=0.35,
            ),
            A.OneOf(
                [
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    A.RandomGamma(),
                    A.CLAHE(),
                ],
                p=0.6,
            ),
            A.OneOf(
                [
                    A.ToGray(),
                    A.HueSaturationValue(),
                    A.RGBShift(),
                    A.ChannelShuffle(),
                ],
                p=0.45,
            ),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.04, 0.12)),
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=(3, 7)),
                    A.MedianBlur(blur_limit=(3, 7)),
                ],
                p=0.35,
            ),
            A.OneOf(
                [
                    A.ImageCompression(quality_range=(50, 95)),
                    A.Posterize(),
                    A.Solarize(),
                ],
                p=0.20,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 12),
                hole_height_range=(8, 40),
                hole_width_range=(8, 40),
                fill=0,
                fill_mask=0,
                p=0.30,
            ),
            A.Resize(image_size, image_size, interpolation=image_resize_interpolation, mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def build_val_transform(image_size: int, mean: list[float], std: list[float], resize_interpolation: str = "linear") -> A.Compose:
    image_resize_interpolation = resolve_resize_interpolation(resize_interpolation)
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=image_resize_interpolation, mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


class TrainSegmentationDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, Path]],
        image_size: int,
        geom_transform: A.Compose,
        moderate_transform: A.Compose,
        heavy_transform: A.Compose,
        aug_mode: str,
        copy_paste_p: float,
        copy_paste_two_objects_p: float,
        copy_paste_scale_range: tuple[float, float],
        boundary_kernel_size: int,
    ):
        self.samples = samples
        self.image_size = image_size
        self.geom_transform = geom_transform
        self.moderate_transform = moderate_transform
        self.heavy_transform = heavy_transform
        self.aug_mode = aug_mode
        self.copy_paste_p = copy_paste_p
        self.copy_paste_two_objects_p = copy_paste_two_objects_p
        self.copy_paste_scale_range = copy_paste_scale_range
        self.boundary_kernel_size = boundary_kernel_size
        self.phase = "geom" if aug_mode == "geom" else "moderate"
        self.transforms_by_phase = {
            "geom": self.geom_transform,
            "moderate": self.moderate_transform,
            "heavy": self.heavy_transform,
        }

        self.camera_to_indices: dict[str, list[int]] = {}
        for idx, (image_path, _) in enumerate(samples):
            self.camera_to_indices.setdefault(infer_camera_group(image_path), []).append(idx)
        self.cameras = sorted(self.camera_to_indices)

    def __len__(self) -> int:
        return len(self.samples)

    def set_phase(self, phase: str) -> None:
        if phase not in self.transforms_by_phase:
            raise ValueError(f"Unsupported phase: {phase}")
        self.phase = phase

    def _read_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _read_mask(self, path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {path}")
        return (mask > 127).astype(np.uint8)

    def _load_raw_sample(self, idx: int) -> tuple[np.ndarray, np.ndarray, str]:
        image_path, mask_path = self.samples[idx]
        image = self._read_image(image_path)
        mask = self._read_mask(mask_path)
        camera_group = infer_camera_group(image_path)
        return image, mask, camera_group

    def _sample_donor_index(self, host_camera: str) -> int:
        candidate_cameras = [camera for camera in self.cameras if camera != host_camera and self.camera_to_indices[camera]]
        if candidate_cameras:
            donor_camera = random.choice(candidate_cameras)
            return random.choice(self.camera_to_indices[donor_camera])
        return random.randrange(len(self.samples))

    def _paste_single_object(self, host_image: np.ndarray, host_mask: np.ndarray, donor_idx: int) -> tuple[np.ndarray, np.ndarray]:
        donor_image, donor_mask, _ = self._load_raw_sample(donor_idx)
        ys, xs = np.where(donor_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return host_image, host_mask

        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        patch_image = donor_image[y0:y1, x0:x1]
        patch_mask = donor_mask[y0:y1, x0:x1]

        if patch_image.size == 0 or patch_mask.sum() == 0:
            return host_image, host_mask

        scale = random.uniform(*self.copy_paste_scale_range)
        patch_h, patch_w = patch_mask.shape
        new_w = max(1, int(round(patch_w * scale)))
        new_h = max(1, int(round(patch_h * scale)))
        patch_image = cv2.resize(patch_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        patch_mask = cv2.resize(patch_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        patch_mask = (patch_mask > 0).astype(np.uint8)
        if patch_mask.sum() == 0:
            return host_image, host_mask

        host_h, host_w = host_mask.shape
        max_out_x = int(round(new_w * 0.2))
        max_out_y = int(round(new_h * 0.2))
        left = random.randint(-max_out_x, max(host_w - new_w + max_out_x, -max_out_x))
        top = random.randint(-max_out_y, max(host_h - new_h + max_out_y, -max_out_y))

        dst_x0 = max(0, left)
        dst_y0 = max(0, top)
        dst_x1 = min(host_w, left + new_w)
        dst_y1 = min(host_h, top + new_h)
        if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
            return host_image, host_mask

        src_x0 = dst_x0 - left
        src_y0 = dst_y0 - top
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        patch_image_crop = patch_image[src_y0:src_y1, src_x0:src_x1]
        patch_mask_crop = patch_mask[src_y0:src_y1, src_x0:src_x1].astype(bool)
        if not patch_mask_crop.any():
            return host_image, host_mask

        region_image = host_image[dst_y0:dst_y1, dst_x0:dst_x1]
        region_mask = host_mask[dst_y0:dst_y1, dst_x0:dst_x1]
        region_image[patch_mask_crop] = patch_image_crop[patch_mask_crop]
        region_mask = np.logical_or(region_mask.astype(bool), patch_mask_crop).astype(np.uint8)
        host_image[dst_y0:dst_y1, dst_x0:dst_x1] = region_image
        host_mask[dst_y0:dst_y1, dst_x0:dst_x1] = region_mask
        return host_image, host_mask

    def _maybe_copy_paste(self, image: np.ndarray, mask: np.ndarray, host_camera: str) -> tuple[np.ndarray, np.ndarray]:
        if self.aug_mode != "heavy" or self.phase != "heavy":
            return image, mask
        if random.random() >= self.copy_paste_p:
            return image, mask

        image = image.copy()
        mask = mask.copy()
        n_objects = 2 if random.random() < self.copy_paste_two_objects_p else 1
        for _ in range(n_objects):
            donor_idx = self._sample_donor_index(host_camera)
            image, mask = self._paste_single_object(image, mask, donor_idx)
        return image, mask

    def __getitem__(self, idx: int) -> dict:
        image, mask, host_camera = self._load_raw_sample(idx)
        image, mask = self._maybe_copy_paste(image, mask, host_camera)

        transform = self.transforms_by_phase[self.phase]
        transformed = transform(image=image, mask=mask)

        image_t = transformed["image"].float()
        mask_t = transformed["mask"]
        if torch.is_tensor(mask_t):
            mask_t = mask_t.float()
        else:
            mask_t = torch.from_numpy(mask_t).float()
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)
        mask_t = (mask_t > 0.5).float()

        boundary_np = generate_boundary_mask(mask_t.squeeze(0).cpu().numpy().astype(np.uint8), self.boundary_kernel_size)
        boundary_t = torch.from_numpy(boundary_np).unsqueeze(0).float()

        image_path, _ = self.samples[idx]
        return {
            "image": image_t,
            "mask": mask_t,
            "boundary": boundary_t,
            "path": str(image_path),
            "camera_group": host_camera,
        }


class ValSegmentationDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, Path]],
        transform: A.Compose,
        boundary_kernel_size: int,
    ):
        self.samples = samples
        self.transform = transform
        self.boundary_kernel_size = boundary_kernel_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        image_path, mask_path = self.samples[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        mask = (mask > 127).astype(np.uint8)

        transformed = self.transform(image=image, mask=mask)
        image_t = transformed["image"].float()
        mask_t = transformed["mask"]
        if torch.is_tensor(mask_t):
            mask_t = mask_t.float()
        else:
            mask_t = torch.from_numpy(mask_t).float()
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)
        mask_t = (mask_t > 0.5).float()

        boundary_np = generate_boundary_mask(mask_t.squeeze(0).cpu().numpy().astype(np.uint8), self.boundary_kernel_size)
        boundary_t = torch.from_numpy(boundary_np).unsqueeze(0).float()

        return {
            "image": image_t,
            "mask": mask_t,
            "boundary": boundary_t,
            "path": str(image_path),
            "camera_group": infer_camera_group(image_path),
        }


class ConvGNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, dilation: int = 1, groups: int = 32):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ECAAttention(nn.Module):
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.conv(pooled)
        weights = self.sigmoid(weights).transpose(-1, -2).unsqueeze(-1)
        return x * weights


class SpatialGate(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.sigmoid(self.conv(x))
        return x * weights


class LiteASPP(nn.Module):
    def __init__(self, in_channels: int = 1024, out_channels: int = 256):
        super().__init__()
        self.branch1 = ConvGNAct(in_channels, out_channels, kernel_size=1, groups=32)
        self.branch6 = ConvGNAct(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, groups=32)
        self.branch12 = ConvGNAct(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, groups=32)
        self.reduce = ConvGNAct(out_channels * 3, out_channels, kernel_size=1, groups=32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.cat([self.branch1(x), self.branch6(x), self.branch12(x)], dim=1)
        return self.reduce(features)


class SegFormerV4Decoder(nn.Module):
    def __init__(self, hidden_sizes: list[int], proj_channels: int = 256):
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_size, proj_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(32, proj_channels),
                    nn.GELU(),
                )
                for hidden_size in hidden_sizes
            ]
        )
        self.aspp = LiteASPP(in_channels=proj_channels * len(hidden_sizes), out_channels=proj_channels)
        self.eca = ECAAttention(kernel_size=5)
        self.spatial_gate = SpatialGate(proj_channels)
        self.refine = nn.Sequential(
            nn.Conv2d(proj_channels, proj_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, proj_channels),
            nn.GELU(),
            nn.Dropout2d(0.1),
        )
        self.classifier = nn.Conv2d(proj_channels, 1, kernel_size=1)

    def forward(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        target_size = hidden_states[0].shape[-2:]
        projected = []
        for feature_map, projection in zip(hidden_states, self.projections):
            feat = projection(feature_map)
            feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            projected.append(feat)
        fused = torch.cat(projected, dim=1)
        fused = self.aspp(fused)
        fused = self.eca(fused)
        fused = self.spatial_gate(fused)
        fused = self.refine(fused)
        return self.classifier(fused)


class BoundaryHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SegFormerV4(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        self.encoder = SegformerModel.from_pretrained(backbone_name)
        hidden_sizes = list(self.encoder.config.hidden_sizes)
        self.decoder = SegFormerV4Decoder(hidden_sizes=hidden_sizes)
        self.boundary_head = BoundaryHead(in_channels=hidden_sizes[0])

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        seg_logits = self.decoder(hidden_states)
        boundary_logits = self.boundary_head(hidden_states[0])
        return seg_logits, boundary_logits


def upsample_logits(logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(2, 3))
    cardinality = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice.mean()


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.numel() == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted.float())
    return torch.dot(F.relu(errors_sorted), grad)


def lovasz_hinge(logits: torch.Tensor, labels: torch.Tensor, per_image: bool = True) -> torch.Tensor:
    if per_image:
        losses = []
        for logit, label in zip(logits, labels):
            losses.append(lovasz_hinge_flat(logit.reshape(-1), label.reshape(-1)))
        return torch.stack(losses).mean()
    return lovasz_hinge_flat(logits.reshape(-1), labels.reshape(-1))


def binary_focal_loss_with_logits_ls(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha_pos: float = 0.75,
    eps: float = 0.03,
) -> torch.Tensor:
    target_smooth = target * (1.0 - eps) + 0.5 * eps
    bce = F.binary_cross_entropy_with_logits(logits, target_smooth, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * target_smooth + (1.0 - probs) * (1.0 - target_smooth)
    focal_weight = (1.0 - p_t).pow(gamma)
    alpha_t = alpha_pos * target_smooth + (1.0 - alpha_pos) * (1.0 - target_smooth)
    return (alpha_t * focal_weight * bce).mean()


def compute_mask_loss(logits: torch.Tensor, target: torch.Tensor, config: TrainConfig) -> torch.Tensor:
    focal = binary_focal_loss_with_logits_ls(
        logits,
        target,
        gamma=config.focal_gamma,
        alpha_pos=config.focal_alpha_pos,
        eps=config.label_smoothing,
    )
    if config.mask_loss == "lovasz_focal":
        lovasz = lovasz_hinge(logits, target, per_image=True)
        return 0.5 * lovasz + 0.5 * focal
    dice = soft_dice_loss(logits, target)
    return 0.5 * dice + 0.5 * focal


def compute_boundary_loss(boundary_logits: torch.Tensor, boundary_target: torch.Tensor) -> torch.Tensor:
    pos_pixels = boundary_target.sum()
    neg_pixels = boundary_target.numel() - pos_pixels
    pos_weight = (neg_pixels / (pos_pixels + 1e-6)).detach()
    return F.binary_cross_entropy_with_logits(boundary_logits, boundary_target, pos_weight=pos_weight)


def compute_total_loss(
    seg_logits: torch.Tensor,
    boundary_logits: torch.Tensor,
    mask_target: torch.Tensor,
    boundary_target: torch.Tensor,
    config: TrainConfig,
) -> torch.Tensor:
    mask_loss = compute_mask_loss(seg_logits, mask_target, config)
    boundary_loss = compute_boundary_loss(boundary_logits, boundary_target)
    return mask_loss + config.boundary_loss_weight * boundary_loss


def pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float((pred == target).mean())


def mean_iou(pred: np.ndarray, target: np.ndarray, num_classes: int = 2) -> float:
    ious = []
    for cls_idx in range(num_classes):
        pred_mask = pred == cls_idx
        target_mask = target == cls_idx
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        if union == 0:
            ious.append(1.0)
        else:
            ious.append(intersection / union)
    return float(np.mean(ious))


def boundary_f1(pred_boundary: torch.Tensor, target_boundary: torch.Tensor, threshold: float = 0.5) -> float:
    pred_binary = (torch.sigmoid(pred_boundary).detach().cpu().numpy() > threshold).astype(np.uint8)
    target_binary = (target_boundary.detach().cpu().numpy() > 0.5).astype(np.uint8)
    return float(f1_score(target_binary.reshape(-1), pred_binary.reshape(-1), zero_division=0))


def checkpoint_selection_score(metrics: dict, metric_name: str = "dice_tuned") -> float:
    if metric_name == "dice_tuned":
        return float(metrics.get("dice_tuned", metrics.get("dice", metrics.get("mIoU", -1.0))))
    if metric_name == "dice":
        return float(metrics.get("dice", metrics.get("mIoU", -1.0)))
    return float(metrics.get("mIoU", -1.0))


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def hflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=(-1,))


def vflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=(-2,))


def hvflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=(-2, -1))


def rot90(x: torch.Tensor) -> torch.Tensor:
    return torch.rot90(x, 1, dims=(-2, -1))


def rot270(x: torch.Tensor) -> torch.Tensor:
    return torch.rot90(x, 3, dims=(-2, -1))


def get_tta_sequences(tta_ops: str) -> list[tuple[str, ...]]:
    if tta_ops == "flips":
        return [(), ("hflip",), ("vflip",), ("hflip", "vflip")]
    if tta_ops == "d4":
        return [
            (),
            ("hflip",),
            ("vflip",),
            ("hflip", "vflip"),
            ("rot90",),
            ("rot270",),
            ("rot90", "hflip"),
            ("rot90", "vflip"),
        ]
    raise ValueError(f"Unsupported TTA transform family: {tta_ops}")


def apply_tta_sequence(x: torch.Tensor, ops: tuple[str, ...]) -> torch.Tensor:
    for op in ops:
        if op == "hflip":
            x = hflip(x)
        elif op == "vflip":
            x = vflip(x)
        elif op == "rot90":
            x = rot90(x)
        elif op == "rot270":
            x = rot270(x)
        else:
            raise ValueError(f"Unsupported TTA operation: {op}")
    return x


def invert_tta_sequence(x: torch.Tensor, ops: tuple[str, ...]) -> torch.Tensor:
    for op in reversed(ops):
        if op == "hflip":
            x = hflip(x)
        elif op == "vflip":
            x = vflip(x)
        elif op == "rot90":
            x = rot270(x)
        elif op == "rot270":
            x = rot90(x)
        else:
            raise ValueError(f"Unsupported TTA operation: {op}")
    return x


def resize_for_tta(images: torch.Tensor, scale: float) -> torch.Tensor:
    if math.isclose(scale, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        return images
    height, width = images.shape[-2:]
    scaled_h = max(32, int(round(height * scale)))
    scaled_w = max(32, int(round(width * scale)))
    return F.interpolate(images, size=(scaled_h, scaled_w), mode="bilinear", align_corners=False)


def remove_small_connected_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return (mask > 0).astype(np.uint8)
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    filtered = np.zeros_like(mask_u8)
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= min_area:
            filtered[labels == label_idx] = 1
    return filtered


def fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    inv_mask = 1 - mask_u8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv_mask, connectivity=8)
    height, width = mask_u8.shape
    filled = mask_u8.copy()
    for label_idx in range(1, num_labels):
        left = int(stats[label_idx, cv2.CC_STAT_LEFT])
        top = int(stats[label_idx, cv2.CC_STAT_TOP])
        comp_w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        touches_border = left == 0 or top == 0 or (left + comp_w) == width or (top + comp_h) == height
        if not touches_border:
            filled[labels == label_idx] = 1
    return filled


def postprocess_binary_mask(mask: np.ndarray, config: TrainConfig) -> np.ndarray:
    processed = remove_small_connected_components(mask, config.postprocess_min_component_area)
    if config.postprocess_fill_holes:
        processed = fill_binary_holes(processed)
    return processed.astype(np.uint8)


def tune_threshold_with_postprocess(
    probabilities: list[np.ndarray],
    targets: list[np.ndarray],
    threshold_grid: list[float],
    config: TrainConfig,
) -> dict:
    best_threshold = float(threshold_grid[0])
    best_dice = -1.0
    for threshold in threshold_grid:
        dices = []
        for sample_prob, sample_target in zip(probabilities, targets):
            pred_mask = (sample_prob > threshold).astype(np.uint8)
            pred_mask = postprocess_binary_mask(pred_mask, config)
            dices.append(binary_dice_score(pred_mask, sample_target))
        score = float(np.mean(dices)) if dices else 0.0
        if score > best_dice:
            best_dice = score
            best_threshold = float(threshold)
    return {"best_threshold": best_threshold, "best_dice": best_dice}


@torch.no_grad()
def predict_with_tta(model: nn.Module, images: torch.Tensor, target_size: tuple[int, int], config: TrainConfig) -> torch.Tensor:
    preds = []
    for scale in config.tta_scales:
        scaled_images = resize_for_tta(images, float(scale))
        for ops in get_tta_sequences(config.tta_ops):
            aug_images = apply_tta_sequence(scaled_images, ops)
            seg_logits, _ = model(aug_images)
            seg_logits = upsample_logits(seg_logits, target_size)
            probs = torch.sigmoid(seg_logits)
            preds.append(invert_tta_sequence(probs, ops))
    return torch.stack(preds, dim=0).mean(dim=0)


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, config: TrainConfig, use_tta: bool = False) -> dict:
    model.eval()
    losses: list[float] = []
    pixel_accs: list[float] = []
    mious: list[float] = []
    dices: list[float] = []
    boundary_scores: list[float] = []
    positive_probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        masks = batch["mask"].to(device, non_blocking=device.type == "cuda")
        boundaries = batch["boundary"].to(device, non_blocking=device.type == "cuda")

        seg_logits_low, boundary_logits_low = model(images)
        seg_logits = upsample_logits(seg_logits_low, masks.shape[-2:])
        boundary_logits = upsample_logits(boundary_logits_low, boundaries.shape[-2:])
        loss = compute_total_loss(seg_logits, boundary_logits, masks, boundaries, config)

        if use_tta:
            probs = predict_with_tta(model, images, masks.shape[-2:], config)
        else:
            probs = torch.sigmoid(seg_logits)

        prob_np = probs.squeeze(1).detach().cpu().numpy().astype(np.float32)
        target_np = masks.squeeze(1).detach().cpu().numpy().astype(np.uint8)

        losses.append(float(loss.item()))
        boundary_scores.append(boundary_f1(boundary_logits, boundaries))

        for sample_prob, sample_target in zip(prob_np, target_np):
            positive_probabilities.append(sample_prob)
            targets.append(sample_target)
            sample_pred = (sample_prob > 0.5).astype(np.uint8)
            if config.postprocess_enabled:
                sample_pred = postprocess_binary_mask(sample_pred, config)
            dices.append(binary_dice_score(sample_pred, sample_target))
            pixel_accs.append(pixel_accuracy(sample_pred, sample_target))
            mious.append(mean_iou(sample_pred, sample_target, num_classes=2))

    if config.postprocess_enabled:
        threshold_metrics = tune_threshold_with_postprocess(
            probabilities=positive_probabilities,
            targets=targets,
            threshold_grid=[float(x) for x in config.threshold_grid],
            config=config,
        )
    else:
        threshold_metrics = tune_binary_threshold(
            probabilities=positive_probabilities,
            targets=targets,
            threshold_grid=[float(x) for x in config.threshold_grid],
            ignore_index=255,
        )

    return {
        "mIoU": float(np.mean(mious)) if mious else 0.0,
        "dice": float(np.mean(dices)) if dices else 0.0,
        "dice_tuned": float(threshold_metrics["best_dice"]),
        "best_threshold": float(threshold_metrics["best_threshold"]),
        "pixel_acc": float(np.mean(pixel_accs)) if pixel_accs else 0.0,
        "boundary_f1": float(np.mean(boundary_scores)) if boundary_scores else 0.0,
        "loss": float(np.mean(losses)) if losses else 0.0,
        "tta": bool(use_tta),
    }


def get_epoch_phase(config: TrainConfig, epoch_idx: int) -> str:
    if config.aug == "geom":
        return "geom"
    if config.aug == "moderate":
        return "moderate"
    # Schedule: first 12.5% moderate warmup, middle 62.5% heavy, final 25% moderate cooldown
    # For 80 epochs: 1-10 moderate, 11-60 heavy, 61-80 moderate
    total = config.epochs
    warmup_end = max(1, round(total * 0.125))
    cooldown_start = max(warmup_end + 1, total - max(1, round(total * 0.25)))
    epoch_num = epoch_idx + 1
    if epoch_num <= warmup_end:
        return "moderate"
    if epoch_num <= cooldown_start:
        return "heavy"
    return "moderate"


def build_optimizer(model: SegFormerV4, config: TrainConfig) -> AdamW:
    groups: dict[tuple[str, str], list[torch.nn.Parameter]] = {
        ("encoder_stage1", "decay"): [],
        ("encoder_stage1", "no_decay"): [],
        ("encoder_stage2", "decay"): [],
        ("encoder_stage2", "no_decay"): [],
        ("encoder_stage3", "decay"): [],
        ("encoder_stage3", "no_decay"): [],
        ("encoder_stage4", "decay"): [],
        ("encoder_stage4", "no_decay"): [],
        ("decoder", "decay"): [],
        ("decoder", "no_decay"): [],
    }

    stage_scale = {
        "encoder_stage1": config.llrd_factor**3,
        "encoder_stage2": config.llrd_factor**2,
        "encoder_stage3": config.llrd_factor,
        "encoder_stage4": 1.0,
    }

    def is_no_decay(name: str, param: torch.nn.Parameter) -> bool:
        name_l = name.lower()
        return param.ndim <= 1 or "bias" in name_l or "norm" in name_l or "ln" in name_l or "layernorm" in name_l

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        bucket_type = "no_decay" if is_no_decay(name, param) else "decay"
        if name.startswith("encoder.encoder.patch_embeddings.0") or name.startswith("encoder.encoder.block.0") or name.startswith("encoder.encoder.layer_norm.0"):
            groups[("encoder_stage1", bucket_type)].append(param)
        elif name.startswith("encoder.encoder.patch_embeddings.1") or name.startswith("encoder.encoder.block.1") or name.startswith("encoder.encoder.layer_norm.1"):
            groups[("encoder_stage2", bucket_type)].append(param)
        elif name.startswith("encoder.encoder.patch_embeddings.2") or name.startswith("encoder.encoder.block.2") or name.startswith("encoder.encoder.layer_norm.2"):
            groups[("encoder_stage3", bucket_type)].append(param)
        elif name.startswith("encoder.encoder.patch_embeddings.3") or name.startswith("encoder.encoder.block.3") or name.startswith("encoder.encoder.layer_norm.3"):
            groups[("encoder_stage4", bucket_type)].append(param)
        elif name.startswith("decoder.") or name.startswith("boundary_head."):
            groups[("decoder", bucket_type)].append(param)
        else:
            groups[("decoder", bucket_type)].append(param)

    param_groups = []
    for (group_name, bucket_type), params in groups.items():
        if not params:
            continue
        lr_scale = stage_scale.get(group_name, 1.0)
        param_groups.append(
            {
                "params": params,
                "lr": config.base_lr * lr_scale,
                "weight_decay": config.weight_decay if bucket_type == "decay" else 0.0,
                "group_name": f"{group_name}_{bucket_type}",
            }
        )

    return AdamW(param_groups)


def save_checkpoint(
    config: TrainConfig,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: AdamW,
    scheduler,
    epoch: int,
    metrics: dict,
    is_best: bool,
    best_score: float = -1.0,
) -> None:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "metrics": metrics,
        "best_score": best_score,
        "config": asdict(config),
        "model_state_dict": model.state_dict(),
        "ema_model_state_dict": ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(payload, config.run_dir / "latest_model.pth")
    if is_best:
        torch.save(payload, config.run_dir / "best_model.pth")
    if (epoch + 1) % config.save_every_n_epochs == 0:
        torch.save(payload, config.run_dir / f"checkpoint_epoch_{epoch + 1}.pth")


def train_one_epoch(
    model: nn.Module,
    ema_model: nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    scaler,
    device: torch.device,
    config: TrainConfig,
    epoch_idx: int,
) -> dict:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    loss_meter = 0.0
    batch_count = 0
    optimizer_steps = 0

    progress = tqdm(dataloader, desc=f"Train Epoch {epoch_idx + 1}", leave=False)
    for step_idx, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        masks = batch["mask"].to(device, non_blocking=device.type == "cuda")
        boundaries = batch["boundary"].to(device, non_blocking=device.type == "cuda")

        with get_autocast_context(device):
            seg_logits_low, boundary_logits_low = model(images)
            seg_logits = upsample_logits(seg_logits_low, masks.shape[-2:])
            boundary_logits = upsample_logits(boundary_logits_low, boundaries.shape[-2:])
            total_loss = compute_total_loss(seg_logits, boundary_logits, masks, boundaries, config)
            loss = total_loss / config.accumulation_steps

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        batch_count += 1
        loss_meter += float(total_loss.item())

        should_step = (step_idx + 1) % config.accumulation_steps == 0 or (step_idx + 1) == len(dataloader)
        if should_step:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_ema(ema_model, model, decay=config.ema_decay)
            optimizer_steps += 1

        progress.set_postfix(loss=f"{total_loss.item():.4f}")

    return {
        "train_loss": loss_meter / max(1, batch_count),
        "optimizer_steps": optimizer_steps,
    }


def build_dataloaders(train_samples: list[tuple[Path, Path]], val_samples: list[tuple[Path, Path]], config: TrainConfig, device: torch.device) -> tuple[TrainSegmentationDataset, DataLoader, DataLoader]:
    geom_transform = build_geom_transform(config.image_size, config.image_mean, config.image_std, config.resize_interpolation)
    moderate_transform = build_moderate_transform(config.image_size, config.image_mean, config.image_std, config.resize_interpolation)
    heavy_transform = build_heavy_transform(config.image_size, config.image_mean, config.image_std, config.resize_interpolation)
    val_transform = build_val_transform(config.image_size, config.image_mean, config.image_std, config.resize_interpolation)

    train_dataset = TrainSegmentationDataset(
        samples=train_samples,
        image_size=config.image_size,
        geom_transform=geom_transform,
        moderate_transform=moderate_transform,
        heavy_transform=heavy_transform,
        aug_mode=config.aug,
        copy_paste_p=config.copy_paste_p,
        copy_paste_two_objects_p=config.copy_paste_two_objects_p,
        copy_paste_scale_range=config.copy_paste_scale_range,
        boundary_kernel_size=config.boundary_kernel_size,
    )
    val_dataset = ValSegmentationDataset(
        samples=val_samples,
        transform=val_transform,
        boundary_kernel_size=config.boundary_kernel_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.physical_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
        drop_last=True,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.physical_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
        drop_last=False,
        persistent_workers=False,
    )
    return train_dataset, train_loader, val_loader


def train(config: TrainConfig) -> int:
    device = get_device()
    config.pin_memory = device.type == "cuda"
    if device.type == "mps":
        config.num_workers = 0

    set_seed(config.seed)
    cudnn.benchmark = device.type == "cuda"

    print(f"[main] project_root: {PROJECT_ROOT}")
    print(f"[main] device: {device}")
    print(f"[main] run_dir: {config.run_dir}")
    if config.fold == 0:
        print("[main] note: fold_0 is the large single-camera validation split and behaves like a domain-shift stress test.")
    if config.aug == "heavy" and config.epochs != 80:
        print(f"[main] note: heavy schedule was designed for 80 epochs, but got epochs={config.epochs}.")

    images_dir = LAB3_DATASET_ROOT / "train" / "images"
    masks_dir = LAB3_DATASET_ROOT / "train" / "masks"
    samples = collect_labeled_pairs(images_dir, masks_dir)
    val_sample_ids = load_val_sample_ids(PROJECT_ROOT, config.fold)
    train_samples, val_samples = split_samples_by_fold(samples, val_sample_ids)

    split_summary = {
        "fold": config.fold,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_cameras": sorted({infer_camera_group(image_path) for image_path, _ in train_samples}),
        "val_cameras": sorted({infer_camera_group(image_path) for image_path, _ in val_samples}),
    }
    print(
        f"[main] fold {config.fold} | train={split_summary['train_samples']} | val={split_summary['val_samples']} | "
        f"train_cameras={len(split_summary['train_cameras'])} | val_cameras={len(split_summary['val_cameras'])}"
    )

    config.run_dir.mkdir(parents=True, exist_ok=True)
    save_json(config.run_dir / "config.json", asdict(config))
    save_json(config.run_dir / "split_summary.json", split_summary)

    train_dataset, train_loader, val_loader = build_dataloaders(train_samples, val_samples, config, device)

    model = SegFormerV4(config.backbone).to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)

    if config.finetune_from:
        ft_path = Path(config.finetune_from)
        if not ft_path.exists():
            raise FileNotFoundError(f"finetune checkpoint not found: {ft_path}")
        print(f"[finetune] loading weights from {ft_path}")
        ft_ckpt = torch.load(ft_path, map_location=device, weights_only=False)
        model.load_state_dict(ft_ckpt["ema_model_state_dict"], strict=True)
        ema_model.load_state_dict(ft_ckpt["ema_model_state_dict"], strict=True)
        print(f"[finetune] loaded EMA weights, applying lr_scale={config.finetune_lr_scale}")
        config.base_lr = config.base_lr * config.finetune_lr_scale
        save_json(config.run_dir / "config.json", asdict(config))

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, total_epochs=config.epochs, warmup_epochs=config.warmup_epochs, min_lr=config.min_lr)
    scaler = get_grad_scaler(device)

    history: list[dict] = []
    best_score = -1.0
    start_epoch = 0

    if config.resume:
        resume_path = config.run_dir / "latest_model.pth"
        if resume_path.exists():
            print(f"[resume] loading checkpoint from {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_score = checkpoint.get("best_score", -1.0)
            history_path = config.run_dir / "history.json"
            if history_path.exists():
                history = json.loads(history_path.read_text(encoding="utf-8"))
            print(f"[resume] resuming from epoch {start_epoch + 1}, best_score={best_score:.4f}")
        else:
            print(f"[resume] no checkpoint found at {resume_path}, starting from scratch")

    print(
        f"[main] aug={config.aug} | image_size={config.image_size} | epochs={config.epochs} | "
        f"physical_bs={config.physical_batch_size} | effective_bs={config.effective_batch_size} | "
        f"accumulation={config.accumulation_steps} | mask_loss={config.mask_loss} | "
        f"resize_interp={config.resize_interpolation} | postprocess={'on' if config.postprocess_enabled else 'off'} | "
        f"tta_ops={config.tta_ops} | tta_scales={config.tta_scales}"
    )

    for epoch_idx in range(start_epoch, config.epochs):
        phase = get_epoch_phase(config, epoch_idx)
        train_dataset.set_phase(phase)

        t0 = time.time()
        train_metrics = train_one_epoch(
            model=model,
            ema_model=ema_model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            config=config,
            epoch_idx=epoch_idx,
        )
        current_lr = max(group["lr"] for group in optimizer.param_groups)
        scheduler.step()

        should_eval = (
            (epoch_idx + 1) % config.eval_every_n_epochs == 0
            or epoch_idx == 0
            or epoch_idx == config.epochs - 1
        )
        if should_eval:
            val_metrics = evaluate(ema_model, val_loader, device, config, use_tta=False)
        else:
            val_metrics = {"mIoU": 0.0, "dice": 0.0, "dice_tuned": 0.0, "best_threshold": 0.0, "pixel_acc": 0.0, "boundary_f1": 0.0, "loss": 0.0, "tta": False}

        elapsed = time.time() - t0

        row = {
            "epoch": epoch_idx,
            "phase": phase,
            "evaluated": should_eval,
            **train_metrics,
            **val_metrics,
            "learning_rate": current_lr,
            "time_sec": round(elapsed, 1),
        }
        history.append(row)

        is_best = False
        if should_eval:
            score = checkpoint_selection_score(row, config.selection_metric)
            is_best = score > best_score
            if is_best:
                best_score = score

        save_checkpoint(config, model, ema_model, optimizer, scheduler, epoch_idx, row, is_best=is_best, best_score=best_score)
        save_json(config.run_dir / "history.json", history)

        if should_eval:
            print(
                f"[epoch {epoch_idx + 1:03d}/{config.epochs}] phase={phase} "
                f"train_loss={row['train_loss']:.4f} "
                f"val_loss={row['loss']:.4f} "
                f"val_mIoU={row['mIoU']:.4f} "
                f"val_dice={row['dice']:.4f} "
                f"val_dice_tuned={row['dice_tuned']:.4f} "
                f"thr={row['best_threshold']:.2f} "
                f"boundary_f1={row['boundary_f1']:.4f} "
                f"lr={current_lr:.6e} "
                f"time={elapsed:.1f}s"
                + (" [best]" if is_best else "")
            )
        else:
            print(
                f"[epoch {epoch_idx + 1:03d}/{config.epochs}] phase={phase} "
                f"train_loss={row['train_loss']:.4f} "
                f"lr={current_lr:.6e} "
                f"time={elapsed:.1f}s [skip eval]"
            )

        empty_device_cache(device)

    if config.final_tta:
        best_path = config.run_dir / "best_model.pth"
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        ema_model.load_state_dict(checkpoint["ema_model_state_dict"], strict=True)
        final_tta_metrics = evaluate(ema_model, val_loader, device, config, use_tta=True)
        save_json(config.run_dir / "final_tta_metrics.json", final_tta_metrics)
        print(
            f"[final_tta] mIoU={final_tta_metrics['mIoU']:.4f} "
            f"dice={final_tta_metrics['dice']:.4f} "
            f"dice_tuned={final_tta_metrics['dice_tuned']:.4f} "
            f"thr={final_tta_metrics['best_threshold']:.2f}"
        )
        empty_device_cache(device)

    print(f"[main] training complete. Artifacts -> {config.run_dir}")
    return 0


def main() -> int:
    config = parse_args()
    return train(config)


if __name__ == "__main__":
    raise SystemExit(main())
