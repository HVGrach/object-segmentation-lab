#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from lab_object_segmentation.common.paths import APPS_ROOT, DELIVERABLES_ROOT, LAB3_DATASET_ROOT, PROJECT_ROOT, RUNS_ROOT
from lab_object_segmentation.segformer_semisup.run_lab3_submission import (
    DEVICE,
    TTA_OPS,
    build_val_transform,
    load_model,
    read_image_rgb,
    resize_map_to_image,
    resolve_checkpoint_threshold,
)
from lab_object_segmentation.segformer_semisup.utils import (
    binary_dice_score,
    build_pseudo_stats,
    split_labeled_samples_grouped,
)


DEFAULT_PSEUDO_RUN = RUNS_ROOT / "segformer_boundary_semisup_macos_2026_03_31_cachefix"
APP_SOURCE_DIR = APPS_ROOT / "pseudo_label_tinder"
DEFAULT_OUTPUT_DIR = DELIVERABLES_ROOT / "pseudo_label_review_bundle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a review-ready pseudo-label dataset bundle with a Tinder-like web app."
    )
    parser.add_argument(
        "--pseudo-run-dir",
        type=Path,
        default=DEFAULT_PSEUDO_RUN,
        help="SegFormer pseudo-label run directory with metrics and pseudo masks.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional checkpoint path for Dice calibration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to assemble the bundle.",
    )
    parser.add_argument(
        "--final-archive",
        type=Path,
        default=DELIVERABLES_ROOT / "pseudo_label_review_bundle.7z",
        help="Path to the final compressed bundle.",
    )
    parser.add_argument(
        "--calibration-limit",
        type=int,
        default=240,
        help="Max labeled validation samples for fitting the expected Dice estimator.",
    )
    parser.add_argument(
        "--calibration-tta-n",
        type=int,
        default=4,
        help="TTA count used only for the Dice calibration pass.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "unknown"


def safe_text(value: str, limit: int = 48) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")
    return cleaned[:limit] or "item"


def stable_item_id(image_path: Path, source: str) -> str:
    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:10]
    return f"{slugify(source)}__{digest}__{safe_text(image_path.stem, limit=36)}"


def collect_labeled_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    image_map = {path.stem: path for path in images_dir.rglob("*") if path.is_file()}
    samples: list[tuple[Path, Path]] = []
    for mask_path in sorted(masks_dir.rglob("*")):
        if not mask_path.is_file():
            continue
        image_path = image_map.get(mask_path.stem)
        if image_path is not None:
            samples.append((image_path, mask_path))
    if not samples:
        raise FileNotFoundError(f"No labeled image/mask pairs found in {images_dir} and {masks_dir}")
    return samples


def read_binary_mask(path: Path) -> np.ndarray:
    mask = np.array(Image.open(path).convert("L"))
    return (mask > 127).astype(np.uint8)


@torch.no_grad()
def predict_mask_with_stats(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    *,
    transform,
    tta_n: int,
    mask_threshold: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    std_pred = stacked.std(dim=0)
    uncertainty = std_pred.max(dim=0).values
    positive_prob = mean_pred[1]

    if mask_threshold is None:
        confidence, pseudo_mask = mean_pred.max(dim=0)
    else:
        pseudo_mask = (positive_prob >= float(mask_threshold)).to(torch.uint8)
        confidence = torch.maximum(positive_prob, 1.0 - positive_prob)

    pseudo_mask_np = pseudo_mask.numpy().astype(np.uint8)
    confidence_np = confidence.numpy().astype(np.float32)
    uncertainty_np = uncertainty.numpy().astype(np.float32)

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
    uncertainty_np = resize_map_to_image(
        uncertainty_np,
        image_shape_hw=(orig_h, orig_w),
        interpolation=cv2.INTER_LINEAR,
    )
    return pseudo_mask_np, confidence_np, uncertainty_np


def regression_features_from_metrics(metrics: dict) -> np.ndarray:
    selection_score = float(metrics["selection_score"])
    avg_confidence = float(metrics["avg_confidence"])
    reliable_ratio = float(metrics["reliable_ratio"])
    fg_reliable_ratio = float(metrics["fg_reliable_ratio"])
    bg_reliable_ratio = float(metrics["bg_reliable_ratio"])
    object_ratio = float(metrics["object_ratio"])
    return np.asarray(
        [
            1.0,
            selection_score,
            avg_confidence,
            reliable_ratio,
            fg_reliable_ratio,
            bg_reliable_ratio,
            object_ratio,
            object_ratio * object_ratio,
            avg_confidence * fg_reliable_ratio,
        ],
        dtype=np.float64,
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if y_true.size == 0:
        return {"count": 0, "mae": None, "rmse": None, "r2": None}

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean(np.square(y_true - y_pred))))
    denom = float(np.sum(np.square(y_true - y_true.mean())))
    if denom <= 1e-12:
        r2 = 0.0
    else:
        r2 = float(1.0 - np.sum(np.square(y_true - y_pred)) / denom)
    return {
        "count": int(y_true.size),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "y_true_mean": float(y_true.mean()),
        "y_pred_mean": float(y_pred.mean()),
    }


def build_expected_dice_calibrator(
    *,
    checkpoint_path: Path,
    calibration_limit: int,
    calibration_tta_n: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    model, model_config, _ = load_model(checkpoint_path)
    mask_threshold = resolve_checkpoint_threshold(checkpoint_path)
    confidence_threshold = float(model_config.get("pseudo_confidence_threshold", 0.85))
    uncertainty_threshold = float(model_config.get("pseudo_uncertainty_threshold", 0.15))
    val_split = float(model_config.get("val_split", 0.15))
    split_seed = int(model_config.get("seed", seed))

    labeled_images = LAB3_DATASET_ROOT / "train" / "images"
    labeled_masks = LAB3_DATASET_ROOT / "train" / "masks"
    labeled_pairs = collect_labeled_pairs(labeled_images, labeled_masks)
    _, val_pairs = split_labeled_samples_grouped(labeled_pairs, val_split=val_split, seed=split_seed)

    rng = random.Random(seed)
    rng.shuffle(val_pairs)
    if calibration_limit > 0:
        val_pairs = val_pairs[: min(calibration_limit, len(val_pairs))]

    if len(val_pairs) < 16:
        raise RuntimeError("Not enough validation samples for Dice calibration.")

    transform = build_val_transform()
    rows: list[dict] = []
    for image_path, mask_path in tqdm(val_pairs, desc="Calibrating expected Dice"):
        image_rgb = read_image_rgb(image_path)
        pred_mask, confidence, uncertainty = predict_mask_with_stats(
            model=model,
            image_rgb=image_rgb,
            transform=transform,
            tta_n=calibration_tta_n,
            mask_threshold=mask_threshold,
        )
        target_mask = read_binary_mask(mask_path)
        stats = build_pseudo_stats(
            pseudo_mask=pred_mask,
            confidence=confidence,
            uncertainty=uncertainty,
            confidence_threshold=confidence_threshold,
            uncertainty_threshold=uncertainty_threshold,
        )
        row = {
            "image_name": image_path.name,
            "dice": binary_dice_score(pred_mask, target_mask),
            "selection_score": float(stats["selection_score"]),
            "avg_confidence": float(stats["avg_confidence"]),
            "reliable_ratio": float(stats["reliable_ratio"]),
            "fg_reliable_ratio": float(stats["fg_reliable_ratio"]),
            "bg_reliable_ratio": float(stats["bg_reliable_ratio"]),
            "object_ratio": float(stats["object_ratio"]),
        }
        rows.append(row)

    x = np.stack([regression_features_from_metrics(row) for row in rows], axis=0)
    y = np.asarray([float(row["dice"]) for row in rows], dtype=np.float64)

    indices = np.arange(len(rows))
    rng.shuffle(indices)
    split_idx = max(8, int(round(len(indices) * 0.8)))
    split_idx = min(split_idx, len(indices) - 4)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    coefficients, _, _, _ = np.linalg.lstsq(x[train_idx], y[train_idx], rcond=None)
    train_pred = np.clip(x[train_idx] @ coefficients, 0.0, 1.0)
    test_pred = np.clip(x[test_idx] @ coefficients, 0.0, 1.0)

    preview_rows = []
    for idx in test_idx[:10]:
        preview_rows.append(
            {
                "image_name": rows[idx]["image_name"],
                "dice_true": float(rows[idx]["dice"]),
                "dice_pred": float(np.clip(x[idx] @ coefficients, 0.0, 1.0)),
                "selection_score": float(rows[idx]["selection_score"]),
            }
        )

    calibration_summary = {
        "status": "ok",
        "checkpoint_path": str(checkpoint_path),
        "device": str(DEVICE),
        "mask_threshold": None if mask_threshold is None else float(mask_threshold),
        "confidence_threshold": confidence_threshold,
        "uncertainty_threshold": uncertainty_threshold,
        "calibration_tta_n": int(calibration_tta_n),
        "num_samples": int(len(rows)),
        "train_metrics": regression_metrics(y[train_idx], train_pred),
        "holdout_metrics": regression_metrics(y[test_idx], test_pred),
        "coefficients": [float(value) for value in coefficients.tolist()],
        "preview": preview_rows,
    }
    return coefficients, calibration_summary


def fallback_expected_dice(selection_score: float, avg_confidence: float) -> float:
    return float(np.clip(0.55 * selection_score + 0.35 * avg_confidence + 0.10, 0.0, 1.0))


def predict_expected_dice(metrics: dict, coefficients: np.ndarray | None) -> float:
    if coefficients is None:
        return fallback_expected_dice(
            selection_score=float(metrics["selection_score"]),
            avg_confidence=float(metrics["avg_confidence"]),
        )
    return float(np.clip(regression_features_from_metrics(metrics) @ coefficients, 0.0, 1.0))


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_overlay(image_path: Path, mask_path: Path, boundary_mask_path: Path | None) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    mask = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.float32)
    boundary = None
    if boundary_mask_path is not None and boundary_mask_path.exists():
        boundary = (np.array(Image.open(boundary_mask_path).convert("L")) > 127).astype(np.float32)

    image_arr = np.array(image).astype(np.float32)
    red = np.asarray([255.0, 79.0, 87.0], dtype=np.float32)
    cyan = np.asarray([29.0, 204.0, 255.0], dtype=np.float32)

    alpha = 0.42 * mask[..., None]
    overlaid = image_arr * (1.0 - alpha) + red * alpha

    if boundary is not None:
        boundary_alpha = 0.75 * boundary[..., None]
        overlaid = overlaid * (1.0 - boundary_alpha) + cyan * boundary_alpha

    return Image.fromarray(np.clip(overlaid, 0, 255).astype(np.uint8))


def render_demo_card(entry: dict, output_path: Path) -> None:
    background = Image.new("RGB", (1440, 1024), color=(245, 239, 230))
    draw = ImageDraw.Draw(background)
    title_font = load_font(46, bold=True)
    stat_font = load_font(28, bold=True)
    body_font = load_font(24)
    small_font = load_font(20)

    overlay = build_overlay(
        image_path=output_path.parent.parent / entry["image_relpath"],
        mask_path=output_path.parent.parent / entry["mask_relpath"],
        boundary_mask_path=(output_path.parent.parent / entry["boundary_mask_relpath"]),
    )

    overlay.thumbnail((780, 780), Image.Resampling.LANCZOS)
    card_w, card_h = 880, 920
    card_x = 100
    card_y = 52
    draw.rounded_rectangle(
        (card_x, card_y, card_x + card_w, card_y + card_h),
        radius=36,
        fill=(255, 252, 248),
        outline=(219, 207, 189),
        width=3,
    )

    badge_text = f"Expected Dice {entry['predicted_dice']:.3f}"
    draw.rounded_rectangle((150, 95, 530, 160), radius=24, fill=(18, 34, 41))
    draw.text((180, 110), badge_text, font=stat_font, fill=(245, 239, 230))

    image_x = card_x + (card_w - overlay.width) // 2
    image_y = 190
    background.paste(overlay, (image_x, image_y))

    button_y = 860
    draw.rounded_rectangle((170, button_y, 390, button_y + 82), radius=28, fill=(255, 99, 87))
    draw.rounded_rectangle((580, button_y, 800, button_y + 82), radius=28, fill=(38, 166, 120))
    draw.text((228, button_y + 20), "Reject", font=stat_font, fill=(255, 252, 248))
    draw.text((645, button_y + 20), "Accept", font=stat_font, fill=(255, 252, 248))

    side_x = 1040
    draw.text((side_x, 92), "Pseudo-label review", font=title_font, fill=(32, 42, 54))
    draw.text((side_x, 190), f"ID: {entry['item_id']}", font=body_font, fill=(62, 73, 85))
    draw.text((side_x, 240), f"Source: {entry['source']}", font=body_font, fill=(62, 73, 85))
    draw.text((side_x, 290), f"Original: {entry['original_name']}", font=small_font, fill=(62, 73, 85))
    draw.text((side_x, 370), f"Selection score: {entry['selection_score']:.3f}", font=body_font, fill=(32, 42, 54))
    draw.text((side_x, 420), f"Mean confidence: {entry['avg_confidence']:.3f}", font=body_font, fill=(32, 42, 54))
    draw.text((side_x, 470), f"Reliable ratio: {entry['reliable_ratio']:.3f}", font=body_font, fill=(32, 42, 54))
    draw.text((side_x, 520), f"FG reliable ratio: {entry['fg_reliable_ratio']:.3f}", font=body_font, fill=(32, 42, 54))
    draw.text((side_x, 570), f"Object ratio: {entry['object_ratio']:.3f}", font=body_font, fill=(32, 42, 54))
    draw.text((side_x, 690), "Keyboard:", font=stat_font, fill=(32, 42, 54))
    draw.text((side_x, 740), "Left: reject", font=body_font, fill=(62, 73, 85))
    draw.text((side_x, 785), "Right: accept", font=body_font, fill=(62, 73, 85))
    draw.text((side_x, 830), "Backspace: clear", font=body_font, fill=(62, 73, 85))
    draw.text((side_x, 875), "E: export CSV/JSON", font=body_font, fill=(62, 73, 85))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    background.save(output_path)


def create_zip_from_directory(source_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.relative_to(source_dir.parent))


def write_dataset_csv(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "item_id",
        "source",
        "original_name",
        "predicted_dice",
        "selection_score",
        "avg_confidence",
        "reliable_ratio",
        "fg_reliable_ratio",
        "bg_reliable_ratio",
        "object_ratio",
        "image_relpath",
        "mask_relpath",
        "boundary_mask_relpath",
        "original_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow({field: entry.get(field) for field in fieldnames})


def write_bundle_readme(path: Path, summary: dict) -> None:
    text = f"""# Pseudo-label Review Bundle

This bundle contains:

- `pseudo_label_dataset/` - extracted pseudo-label dataset used by the review app.
- `pseudo_labels_dataset.zip` - shareable ZIP archive with the same dataset for a classmate.
- `review_app/` - static Tinder-like web app for manual filtering.
- `smoke_tests/` - local validation outputs.
- `demo/` - rendered previews based on real images and masks.

Dataset summary:

- pseudo run: `{summary['pseudo_run_dir']}`
- checkpoint: `{summary['checkpoint_path']}`
- accepted pseudo-labels: `{summary['num_items']}`
- sources: `{json.dumps(summary['source_breakdown'], ensure_ascii=False)}`
- expected Dice estimator status: `{summary['dice_calibration']['status']}`

How to run:

1. Extract the outer archive.
2. From the extracted bundle root run:
   `python review_app/server.py --root .`
3. Open `http://127.0.0.1:8765/review_app/index.html`

Keyboard:

- Left arrow: reject
- Right arrow: accept
- Backspace: clear decision
- E: export decisions
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.pseudo_run_dir.exists():
        raise FileNotFoundError(f"Pseudo run directory does not exist: {args.pseudo_run_dir}")
    if not APP_SOURCE_DIR.exists():
        raise FileNotFoundError(f"Review app source directory does not exist: {APP_SOURCE_DIR}")

    metrics_path = args.pseudo_run_dir / "metrics" / "pseudo_iteration_0.json"
    pseudo_mask_dir = args.pseudo_run_dir / "pseudo_masks" / "iteration_0"
    boundary_mask_dir = args.pseudo_run_dir / "boundary_pseudo_masks" / "iteration_0"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    if not pseudo_mask_dir.exists():
        raise FileNotFoundError(f"Pseudo mask directory not found: {pseudo_mask_dir}")

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    accepted_rows = [row for row in payload["rows"] if row.get("accepted")]
    if not accepted_rows:
        raise RuntimeError("No accepted pseudo-label rows found.")

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint_path or (args.pseudo_run_dir / "phase3" / "iteration_0" / "best.pt")
    coefficients = None
    calibration_summary: dict
    try:
        coefficients, calibration_summary = build_expected_dice_calibrator(
            checkpoint_path=checkpoint_path,
            calibration_limit=args.calibration_limit,
            calibration_tta_n=args.calibration_tta_n,
            seed=args.seed,
        )
    except Exception as exc:
        calibration_summary = {
            "status": "fallback",
            "checkpoint_path": str(checkpoint_path),
            "reason": str(exc),
            "fallback_formula": "0.55 * selection_score + 0.35 * avg_confidence + 0.10",
        }

    dataset_root = args.output_dir / "pseudo_label_dataset"
    image_root = dataset_root / "images"
    mask_root = dataset_root / "masks"
    boundary_root = dataset_root / "boundary_masks"

    entries: list[dict] = []
    source_counter: Counter[str] = Counter()
    predicted_dice_values: list[float] = []

    for row in tqdm(accepted_rows, desc="Staging pseudo-label dataset"):
        image_path = Path(row["path"])
        source = str(row["source"])
        source_slug = slugify(source)
        item_id = stable_item_id(image_path, source)

        image_dst = image_root / source_slug / f"{item_id}{image_path.suffix.lower()}"
        mask_dst = mask_root / source_slug / f"{item_id}.png"
        boundary_dst = boundary_root / source_slug / f"{item_id}.png"

        pseudo_mask_src = pseudo_mask_dir / f"{image_path.stem}.png"
        boundary_mask_src = boundary_mask_dir / f"{image_path.stem}.png"
        if not pseudo_mask_src.exists():
            raise FileNotFoundError(f"Missing pseudo mask for accepted sample: {pseudo_mask_src}")

        image_dst.parent.mkdir(parents=True, exist_ok=True)
        mask_dst.parent.mkdir(parents=True, exist_ok=True)
        boundary_dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(image_path, image_dst)
        shutil.copy2(pseudo_mask_src, mask_dst)
        if boundary_mask_src.exists():
            shutil.copy2(boundary_mask_src, boundary_dst)

        predicted_dice = predict_expected_dice(row, coefficients)
        predicted_dice_values.append(predicted_dice)
        source_counter[source] += 1

        entry = {
            "item_id": item_id,
            "source": source,
            "source_slug": source_slug,
            "original_name": image_path.name,
            "original_path": str(image_path),
            "predicted_dice": predicted_dice,
            "dice_confidence": predicted_dice,
            "selection_score": float(row["selection_score"]),
            "avg_confidence": float(row["avg_confidence"]),
            "reliable_ratio": float(row["reliable_ratio"]),
            "fg_reliable_ratio": float(row["fg_reliable_ratio"]),
            "bg_reliable_ratio": float(row["bg_reliable_ratio"]),
            "object_ratio": float(row["object_ratio"]),
            "image_relpath": str(image_dst.relative_to(args.output_dir)),
            "mask_relpath": str(mask_dst.relative_to(args.output_dir)),
            "boundary_mask_relpath": str(boundary_dst.relative_to(args.output_dir)),
            "review_image_href": "../" + str(image_dst.relative_to(args.output_dir)),
            "review_mask_href": "../" + str(mask_dst.relative_to(args.output_dir)),
            "review_boundary_mask_href": "../" + str(boundary_dst.relative_to(args.output_dir)),
        }
        entries.append(entry)

    entries.sort(key=lambda row: (-row["predicted_dice"], row["source"], row["original_name"]))

    dataset_manifest = {
        "bundle_name": "pseudo_label_review_bundle",
        "pseudo_run_dir": str(args.pseudo_run_dir),
        "checkpoint_path": str(checkpoint_path),
        "num_items": len(entries),
        "source_breakdown": dict(sorted(source_counter.items())),
        "predicted_dice_summary": {
            "min": float(min(predicted_dice_values)),
            "mean": float(np.mean(predicted_dice_values)),
            "median": float(np.median(predicted_dice_values)),
            "max": float(max(predicted_dice_values)),
        },
        "dice_calibration": calibration_summary,
        "entries": entries,
    }
    save_json(dataset_root / "manifest.json", dataset_manifest)
    write_dataset_csv(dataset_root / "manifest.csv", entries)
    save_json(dataset_root / "dice_calibration.json", calibration_summary)

    review_app_dir = args.output_dir / "review_app"
    shutil.copytree(
        APP_SOURCE_DIR,
        review_app_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    save_json(
        review_app_dir / "data" / "manifest.json",
        {
            "dataset_name": "pseudo_label_dataset",
            "num_items": len(entries),
            "source_breakdown": dict(sorted(source_counter.items())),
            "predicted_dice_summary": dataset_manifest["predicted_dice_summary"],
            "dice_calibration": calibration_summary,
            "entries": entries,
        },
    )

    demo_dir = args.output_dir / "demo"
    if len(entries) >= 3:
        sample_indices = [0, len(entries) // 2, len(entries) - 1]
    else:
        sample_indices = list(range(len(entries)))
    for demo_idx, entry_idx in enumerate(sample_indices, start=1):
        render_demo_card(entries[entry_idx], demo_dir / f"demo_card_{demo_idx}.png")

    summary = {
        "pseudo_run_dir": str(args.pseudo_run_dir),
        "checkpoint_path": str(checkpoint_path),
        "num_items": len(entries),
        "source_breakdown": dict(sorted(source_counter.items())),
        "dice_calibration": calibration_summary,
    }
    save_json(args.output_dir / "bundle_manifest.json", summary)
    write_bundle_readme(args.output_dir / "README.md", summary)

    dataset_zip_path = args.output_dir / "pseudo_labels_dataset.zip"
    create_zip_from_directory(dataset_root, dataset_zip_path)

    args.final_archive.parent.mkdir(parents=True, exist_ok=True)
    if args.final_archive.exists():
        args.final_archive.unlink()
    subprocess.run(
        ["7z", "a", str(args.final_archive), str(args.output_dir)],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    print(json.dumps(
        {
            "bundle_dir": str(args.output_dir),
            "final_archive": str(args.final_archive),
            "dataset_zip": str(dataset_zip_path),
            "num_items": len(entries),
            "source_breakdown": dict(sorted(source_counter.items())),
            "dice_calibration_status": calibration_summary["status"],
        },
        indent=2,
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
