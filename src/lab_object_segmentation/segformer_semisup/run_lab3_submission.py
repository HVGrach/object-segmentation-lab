from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm.auto import tqdm
from transformers import SegformerModel

from lab_object_segmentation.common.paths import LAB3_DATASET_ROOT, PROJECT_ROOT, RUNS_ROOT

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CONFIG = {
    "gdrive_save_dir": "artifacts/runs/segformer_boundary_semisup_macos",
    "image_size": 224,
    "min_image_size": 64,
    "encoder": "nvidia/mit-b2",
    "num_classes": 2,
    "boundary_channels": 64,
    "decoder_hidden_size": 256,
    "tta_n_augments": 8,
    "seed": 42,
}


ARTIFACT_ROOT = RUNS_ROOT / "segformer_boundary_semisup_macos"
TEST_IMAGES_DIR = LAB3_DATASET_ROOT / "test_images"


def get_device() -> torch.device:
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_image_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_mask_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def collect_image_paths(input_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


def serialize_mask(mask2d: np.ndarray) -> str:
    return json.dumps(mask2d.astype(np.uint8).tolist(), separators=(",", ":"))


def _resize_if_too_small_image(image: np.ndarray, **kwargs) -> np.ndarray:
    min_side = min(image.shape[:2])
    if min_side >= DEFAULT_CONFIG["min_image_size"]:
        return image
    scale = DEFAULT_CONFIG["min_image_size"] / max(1, min_side)
    new_h = max(1, int(round(image.shape[0] * scale)))
    new_w = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def build_val_transform() -> A.Compose:
    return A.Compose(
        [
            A.Lambda(image=_resize_if_too_small_image),
            A.LongestMaxSize(
                max_size=DEFAULT_CONFIG["image_size"],
                interpolation=cv2.INTER_LINEAR,
            ),
            A.PadIfNeeded(
                min_height=DEFAULT_CONFIG["image_size"],
                min_width=DEFAULT_CONFIG["image_size"],
                border_mode=cv2.BORDER_REFLECT,
                fill=0,
                fill_mask=0,
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


class ConvMLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.proj(x)


class SegFormerAllMLPDecoder(nn.Module):
    def __init__(self, hidden_sizes: list[int], decoder_hidden_size: int, num_classes: int):
        super().__init__()
        self.projections = nn.ModuleList(
            [ConvMLP(hidden_size, decoder_hidden_size) for hidden_size in hidden_sizes]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_hidden_size * len(hidden_sizes), decoder_hidden_size, kernel_size=1),
            nn.BatchNorm2d(decoder_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.classifier = nn.Conv2d(decoder_hidden_size, num_classes, kernel_size=1)

    def forward(self, hidden_states):
        projected = []
        target_size = hidden_states[0].shape[-2:]
        for feature_map, projection in zip(hidden_states, self.projections):
            feat = projection(feature_map)
            feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            projected.append(feat)
        fused = self.fuse(torch.cat(projected, dim=1))
        return self.classifier(fused)


class BoundaryHead(nn.Module):
    def __init__(self, in_channels: int, boundary_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, boundary_channels, kernel_size=1),
            nn.BatchNorm2d(boundary_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(boundary_channels, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.layers(x)


class SegFormerWithBoundary(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.encoder = SegformerModel.from_pretrained(config["encoder"])
        hidden_sizes = list(self.encoder.config.hidden_sizes)
        self.decode_head = SegFormerAllMLPDecoder(
            hidden_sizes=hidden_sizes,
            decoder_hidden_size=config["decoder_hidden_size"],
            num_classes=config["num_classes"],
        )
        self.boundary_head = BoundaryHead(
            in_channels=hidden_sizes[0],
            boundary_channels=config["boundary_channels"],
        )

    def forward(self, pixel_values, return_boundary: bool = True):
        outputs = self.encoder(pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        seg_logits = self.decode_head(hidden_states)
        seg_logits = F.interpolate(
            seg_logits,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if return_boundary:
            boundary_logits = self.boundary_head(hidden_states[0])
            boundary_logits = F.interpolate(
                boundary_logits,
                size=pixel_values.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            return seg_logits, boundary_logits
        return seg_logits


def tensor_hflip(x):
    return torch.flip(x, dims=[-1])


def tensor_vflip(x):
    return torch.flip(x, dims=[-2])


def tensor_rot90(x):
    return torch.rot90(x, k=1, dims=[-2, -1])


def tensor_rot270(x):
    return torch.rot90(x, k=3, dims=[-2, -1])


TTA_OPS = [
    ("identity", lambda x: x, lambda y: y),
    ("hflip", tensor_hflip, tensor_hflip),
    ("vflip", tensor_vflip, tensor_vflip),
    ("hvflip", lambda x: tensor_hflip(tensor_vflip(x)), lambda y: tensor_vflip(tensor_hflip(y))),
    ("rot90", tensor_rot90, tensor_rot270),
    ("rot270", tensor_rot270, tensor_rot90),
    ("hflip_rot90", lambda x: tensor_hflip(tensor_rot90(x)), lambda y: tensor_rot270(tensor_hflip(y))),
    ("vflip_rot90", lambda x: tensor_vflip(tensor_rot90(x)), lambda y: tensor_rot270(tensor_vflip(y))),
]


def pick_best_checkpoint(metrics_root: Path, artifact_root: Path) -> tuple[Path, dict]:
    phase1_history_path = metrics_root / "phase1_history.json"
    iteration_summaries_path = metrics_root / "iteration_summaries.json"

    best_path = artifact_root / "phase1" / "best.pt"
    best_miou = -1.0
    best_meta = {"source": "phase1"}

    if phase1_history_path.exists():
        phase1_history = json.loads(phase1_history_path.read_text(encoding="utf-8"))
        phase1_best_row = max(
            phase1_history,
            key=lambda row: float(row.get("val_dice_tuned", row.get("val_dice", row.get("val_mIoU", -1.0))) or -1.0),
            default=None,
        )
        if phase1_best_row is not None:
            best_miou = float(
                phase1_best_row.get("val_dice_tuned", phase1_best_row.get("val_dice", phase1_best_row.get("val_mIoU", -1.0)))
            )
            best_meta = {
                "source": "phase1",
                "val_dice_tuned": phase1_best_row.get("val_dice_tuned"),
                "val_dice": phase1_best_row.get("val_dice"),
                "val_mIoU": phase1_best_row.get("val_mIoU"),
                "best_threshold": phase1_best_row.get("val_best_threshold"),
            }

    if iteration_summaries_path.exists():
        iteration_summaries = json.loads(iteration_summaries_path.read_text(encoding="utf-8"))
        for row in iteration_summaries:
            miou = float(row.get("phase3_best_dice", row.get("phase3_best_mIoU", -1.0)))
            iteration = int(row["iteration"])
            if miou > best_miou:
                best_miou = miou
                best_path = artifact_root / "phase3" / f"iteration_{iteration}" / "best.pt"
                best_meta = {
                    "source": f"phase3_iteration_{iteration}",
                    "iteration": iteration,
                    "score": miou,
                }

    return best_path, best_meta


def load_checkpoint_config(checkpoint_path: Path) -> tuple[dict, str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = dict(DEFAULT_CONFIG)
    config.update(checkpoint.get("config", {}))
    state_key = "ema_state_dict" if checkpoint.get("ema_state_dict") is not None else "model_state_dict"
    return config, state_key


def resolve_checkpoint_threshold(checkpoint_path: Path) -> float | None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metrics = checkpoint.get("metrics") or {}
    threshold = metrics.get("best_threshold")
    if threshold is None:
        threshold = checkpoint.get("config", {}).get("best_threshold")
    return None if threshold is None else float(threshold)


def load_model(checkpoint_path: Path) -> tuple[nn.Module, dict, str]:
    config, state_key = load_checkpoint_config(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = SegFormerWithBoundary(config)
    model.load_state_dict(checkpoint[state_key], strict=True)
    model.to(DEVICE)
    model.eval()
    return model, config, state_key


def resize_map_to_image(map_array: np.ndarray, image_shape_hw: tuple[int, int], interpolation: int) -> np.ndarray:
    target_h, target_w = image_shape_hw
    if map_array.shape[:2] == (target_h, target_w):
        return map_array
    return cv2.resize(map_array, (target_w, target_h), interpolation=interpolation)


@torch.no_grad()
def predict_mask_with_tta(
    model: nn.Module,
    image_rgb: np.ndarray,
    transform: A.Compose,
    tta_n: int,
    mask_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    base_tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(DEVICE)
    probs = []

    for _, forward_fn, inverse_fn in TTA_OPS[:tta_n]:
        augmented = forward_fn(base_tensor)
        logits = model(augmented, return_boundary=False)
        pred = F.softmax(logits, dim=1)
        pred = inverse_fn(pred)
        probs.append(pred.squeeze(0).cpu())

    stacked = torch.stack(probs, dim=0)
    mean_pred = stacked.mean(dim=0)
    positive_prob = mean_pred[1]
    if mask_threshold is None:
        confidence, pseudo_mask = mean_pred.max(dim=0)
    else:
        pseudo_mask = (positive_prob >= float(mask_threshold)).to(torch.uint8)
        confidence = torch.maximum(positive_prob, 1.0 - positive_prob)

    pseudo_mask_np = pseudo_mask.numpy().astype(np.uint8)
    confidence_np = confidence.numpy().astype(np.float32)

    orig_h, orig_w = image_rgb.shape[:2]
    pseudo_mask_np = resize_map_to_image(
        pseudo_mask_np,
        image_shape_hw=(orig_h, orig_w),
        interpolation=cv2.INTER_NEAREST,
    )
    confidence_np = resize_map_to_image(
        confidence_np,
        image_shape_hw=(orig_h, orig_w),
        interpolation=cv2.INTER_LINEAR,
    )
    return pseudo_mask_np, confidence_np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SegFormer semi-supervised checkpoint on lab3 test images and build a Kaggle submission."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=TEST_IMAGES_DIR,
        help="Directory with test images.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=ARTIFACT_ROOT,
        help="Directory with SegFormer training artifacts.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path. If omitted, the best checkpoint by validation mIoU is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACT_ROOT / "submission_outputs" / "test_images_lab3",
        help="Where to save predicted mask PNGs.",
    )
    parser.add_argument(
        "--submission-path",
        type=Path,
        default=ARTIFACT_ROOT / "submission_lab3_test_images_segformer.csv",
        help="Where to save the submission CSV.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=ARTIFACT_ROOT / "submission_lab3_test_images_segformer_summary.json",
        help="Where to save run metadata.",
    )
    parser.add_argument(
        "--tta-n",
        type=int,
        default=None,
        help="Override the number of TTA transforms.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional binary threshold for the positive-class probability. Defaults to the checkpoint threshold if present.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N images for a smoke test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG["seed"],
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    image_paths = collect_image_paths(args.input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {args.input_dir}")
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    checkpoint_meta = None
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path, checkpoint_meta = pick_best_checkpoint(
            metrics_root=args.artifact_root / "metrics",
            artifact_root=args.artifact_root,
        )
    if checkpoint_meta is None:
        checkpoint_meta = {"source": "manual", "path": str(checkpoint_path)}
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    model, model_config, state_key = load_model(checkpoint_path)
    transform = build_val_transform()
    tta_n = int(args.tta_n if args.tta_n is not None else model_config.get("tta_n_augments", 8))
    threshold = args.threshold if args.threshold is not None else resolve_checkpoint_threshold(checkpoint_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.submission_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device          : {DEVICE}")
    print(f"Checkpoint      : {checkpoint_path}")
    print(f"Checkpoint info : {checkpoint_meta}")
    print(f"State key       : {state_key}")
    print(f"Images          : {len(image_paths)}")
    print(f"TTA transforms  : {tta_n}")
    print(f"Mask threshold  : {threshold}")
    print(f"Output dir      : {args.output_dir}")
    print(f"Submission path : {args.submission_path}")

    stats_rows: list[dict] = []
    with args.submission_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["ImageId", "mask"])

        for index, image_path in enumerate(tqdm(image_paths, desc="SegFormer inference"), 1):
            image_rgb = read_image_rgb(image_path)
            mask, confidence = predict_mask_with_tta(
                model=model,
                image_rgb=image_rgb,
                transform=transform,
                tta_n=tta_n,
                mask_threshold=threshold,
            )

            mask_path = (args.output_dir / image_path.name).with_suffix(".png")
            save_mask_png(mask_path, mask)
            writer.writerow([image_path.name, serialize_mask(mask)])

            stats_rows.append(
                {
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "area_ratio": float(mask.mean()),
                    "mean_confidence": float(confidence.mean()),
                }
            )

            if index % 50 == 0 or index == len(image_paths):
                print(f"Processed {index}/{len(image_paths)}")

    save_json(
        args.summary_path,
        {
            "input_dir": str(args.input_dir),
            "artifact_root": str(args.artifact_root),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_meta": checkpoint_meta,
            "state_key": state_key,
            "output_dir": str(args.output_dir),
            "submission_path": str(args.submission_path),
            "num_images": len(stats_rows),
            "tta_n": tta_n,
            "threshold": threshold,
            "device": str(DEVICE),
            "stats_preview": stats_rows[:5],
        },
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
