#!/usr/bin/env python3
"""Out-of-fold ensemble evaluation.

For each val sample, only models that did NOT train on it contribute predictions.
This gives an honest estimate of ensemble quality on all 2000 train images.

Usage:
    python scripts/eval_oof_ensemble.py
    python scripts/eval_oof_ensemble.py --presets lovasz3 wide6 --thresholds 0.40 0.45 0.50 0.55
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from lab_object_segmentation.common.paths import LAB3_DATASET_ROOT, RUNS_ROOT
from lab_object_segmentation.supervised_v4.train import (
    TrainConfig,
    SegFormerV4,
    build_val_transform,
    collect_labeled_pairs,
    empty_device_cache,
    get_device,
    infer_camera_group,
    load_val_sample_ids,
    postprocess_binary_mask,
    predict_with_tta,
    split_samples_by_fold,
    upsample_logits,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RUN_ROOT = RUNS_ROOT / "supervised_v4"

# Which fold each run was trained on (val fold = samples NOT seen during training)
RUN_FOLD_MAP = {
    "segformer_b2_fold0_320_moderate_ls003_ema": 0,
    "segformer_b2_fold0_320_moderate_lovasz_ls003_ema": 0,
    "segformer_b2_fold0_384_finetune": 0,
    "segformer_b2_fold0_384_finetune_lovasz": 0,
    "segformer_b2_fold1_320_moderate_ls003_ema": 1,
    "segformer_b2_fold1_320_moderate_lovasz_ls003_ema": 1,
    "segformer_b2_fold1_320_heavy80_sched_ls005_ema": 1,
    "segformer_b2_fold1_384_finetune_from_moderate": 1,
    "segformer_b2_fold1_384_finetune_lovasz": 1,
    "segformer_b2_fold2_320_moderate_ls003_ema": 2,
    "segformer_b2_fold2_320_moderate_lovasz_ls003_ema": 2,
    "segformer_b2_fold2_384_finetune": 2,
    "segformer_b2_fold2_384_finetune_lovasz": 2,
}

PRESET_MEMBERS = {
    "conservative3": [
        ("segformer_b2_fold0_320_moderate_lovasz_ls003_ema", 1.0 / 3.0),
        ("segformer_b2_fold1_384_finetune_from_moderate", 1.0 / 3.0),
        ("segformer_b2_fold2_320_moderate_lovasz_ls003_ema", 1.0 / 3.0),
    ],
    "blend4": [
        ("segformer_b2_fold0_320_moderate_lovasz_ls003_ema", 1.0 / 3.0),
        ("segformer_b2_fold1_384_finetune_from_moderate", 1.0 / 6.0),
        ("segformer_b2_fold1_320_moderate_lovasz_ls003_ema", 1.0 / 6.0),
        ("segformer_b2_fold2_320_moderate_lovasz_ls003_ema", 1.0 / 3.0),
    ],
    "lovasz3": [
        ("segformer_b2_fold0_320_moderate_lovasz_ls003_ema", 1.0 / 3.0),
        ("segformer_b2_fold1_320_moderate_lovasz_ls003_ema", 1.0 / 3.0),
        ("segformer_b2_fold2_320_moderate_lovasz_ls003_ema", 1.0 / 3.0),
    ],
    "wide6": [
        ("segformer_b2_fold0_320_moderate_lovasz_ls003_ema", 1.0 / 6.0),
        ("segformer_b2_fold0_320_moderate_ls003_ema", 1.0 / 6.0),
        ("segformer_b2_fold1_320_moderate_lovasz_ls003_ema", 1.0 / 6.0),
        ("segformer_b2_fold1_384_finetune_from_moderate", 1.0 / 6.0),
        ("segformer_b2_fold2_320_moderate_lovasz_ls003_ema", 1.0 / 6.0),
        ("segformer_b2_fold2_320_moderate_ls003_ema", 1.0 / 6.0),
    ],
}


@dataclass
class LoadedModel:
    run_name: str
    fold: int
    weight: float
    model: SegFormerV4
    config: TrainConfig
    transform: object


def load_model(run_name: str, weight: float, device: torch.device) -> LoadedModel:
    run_dir = RUN_ROOT / run_name
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "best_model.pth"

    config = TrainConfig(**json.loads(config_path.read_text(encoding="utf-8")))
    model = SegFormerV4(config.backbone).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("ema_model_state_dict") or checkpoint["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    fold = RUN_FOLD_MAP.get(run_name)
    if fold is None:
        fold = config.fold
    transform = build_val_transform(config.image_size, config.image_mean, config.image_std)
    return LoadedModel(run_name=run_name, fold=fold, weight=weight, model=model, config=config, transform=transform)


@torch.no_grad()
def predict_probability(member: LoadedModel, image_rgb: np.ndarray, device: torch.device, use_tta: bool) -> np.ndarray:
    transformed = member.transform(image=image_rgb)
    image_tensor = transformed["image"].unsqueeze(0).to(device)

    if use_tta:
        probs = predict_with_tta(member.model, image_tensor, image_tensor.shape[-2:], member.config)
    else:
        seg_logits_low, _ = member.model(image_tensor)
        seg_logits = upsample_logits(seg_logits_low, image_tensor.shape[-2:])
        probs = torch.sigmoid(seg_logits)

    prob_map = probs.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    orig_h, orig_w = image_rgb.shape[:2]
    if prob_map.shape != (orig_h, orig_w):
        prob_map = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return prob_map


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_flat = pred.ravel().astype(np.float64)
    gt_flat = gt.ravel().astype(np.float64)
    intersection = (pred_flat * gt_flat).sum()
    total = pred_flat.sum() + gt_flat.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_flat = pred.ravel().astype(bool)
    gt_flat = gt.ravel().astype(bool)
    intersection = (pred_flat & gt_flat).sum()
    union = (pred_flat | gt_flat).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OOF ensemble evaluation on val with ground truth.")
    parser.add_argument("--presets", nargs="+", default=list(PRESET_MEMBERS.keys()), help="Presets to evaluate")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.40, 0.45, 0.50, 0.55, 0.60], help="Thresholds to try")
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA")
    parser.add_argument("--disable-postprocess", action="store_true", help="Disable post-processing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    # Load all samples and build fold->val_ids mapping
    images_dir = LAB3_DATASET_ROOT / "train" / "images"
    masks_dir = LAB3_DATASET_ROOT / "train" / "masks"
    all_samples = collect_labeled_pairs(images_dir, masks_dir)
    print(f"Total labeled samples: {len(all_samples)}")

    fold_val_ids: dict[int, set[str]] = {}
    for fold in [0, 1, 2]:
        fold_val_ids[fold] = load_val_sample_ids(RUNS_ROOT, fold)
    print(f"Fold val sizes: {{{', '.join(f'{f}: {len(ids)}' for f, ids in fold_val_ids.items())}}}")

    # Map image_name -> (image_path, mask_path) for quick lookup
    sample_map: dict[str, tuple[Path, Path]] = {img.name: (img, msk) for img, msk in all_samples}

    # Collect all unique models needed across all presets
    all_run_names: set[str] = set()
    for preset_name in args.presets:
        if preset_name not in PRESET_MEMBERS:
            print(f"[WARN] Unknown preset: {preset_name}, skipping")
            continue
        for run_name, _ in PRESET_MEMBERS[preset_name]:
            all_run_names.add(run_name)

    print(f"\nLoading {len(all_run_names)} unique models...")
    models: dict[str, LoadedModel] = {}
    for run_name in sorted(all_run_names):
        print(f"  Loading {run_name}...")
        models[run_name] = load_model(run_name, 1.0, device)
    print("All models loaded.\n")

    # Pre-compute OOF probabilities: for each sample, predict with models that didn't see it
    # Key: image_name -> {run_name: prob_map}
    print("Computing OOF probability maps...")
    oof_probs: dict[str, dict[str, np.ndarray]] = {}
    oof_gt: dict[str, np.ndarray] = {}

    for idx, (image_path, mask_path) in enumerate(tqdm(all_samples, desc="OOF inference")):
        image_name = image_path.name
        image_rgb = np.array(Image.open(image_path).convert("RGB"))
        gt_mask = np.array(Image.open(mask_path).convert("L"))
        gt_mask = (gt_mask > 127).astype(np.uint8)
        oof_gt[image_name] = gt_mask

        # Which fold is this sample in the val set of?
        sample_val_fold = None
        for fold, val_ids in fold_val_ids.items():
            if image_name in val_ids:
                sample_val_fold = fold
                break

        # Predict with all loaded models that did NOT train on this fold
        # (i.e., models whose val_fold == sample_val_fold, meaning this sample was in their val set)
        # Wait — actually: a model trained on fold X has fold X as its val.
        # So sample is in val of fold X. The model trained on fold X did NOT see this sample.
        # Models trained on OTHER folds DID see this sample in training.
        # For OOF: we want predictions from models that did NOT see this sample in training.
        # A model trained on fold X excludes fold X val from training.
        # So if sample is in fold X val, the model for fold X did NOT see it -> use it.

        sample_probs: dict[str, np.ndarray] = {}
        for run_name, member in models.items():
            # Only predict if this model's fold matches (sample was in its val, not train)
            if member.fold == sample_val_fold:
                prob_map = predict_probability(member, image_rgb, device, use_tta=not args.no_tta)
                sample_probs[run_name] = prob_map

        oof_probs[image_name] = sample_probs

        if (idx + 1) % 100 == 0:
            empty_device_cache(device)

    empty_device_cache(device)
    print(f"OOF inference done: {len(oof_probs)} samples\n")

    # Evaluate each preset × threshold
    print("=" * 80)
    print(f"{'Preset':<20} {'Threshold':>10} {'Dice':>8} {'Dice_tuned':>12} {'IoU':>8} {'Samples':>8}")
    print("=" * 80)

    results: list[dict] = []

    for preset_name in args.presets:
        if preset_name not in PRESET_MEMBERS:
            continue
        members = PRESET_MEMBERS[preset_name]
        member_run_names = {run_name for run_name, _ in members}
        weight_map = {run_name: w for run_name, w in members}

        best_dice_tuned = -1.0
        best_threshold = -1.0

        for threshold in args.thresholds:
            dices = []
            ious = []

            for image_name, gt_mask in oof_gt.items():
                sample_probs = oof_probs[image_name]

                # Filter to only members in this preset that have predictions for this sample
                eligible = {rn: prob for rn, prob in sample_probs.items() if rn in member_run_names}
                if not eligible:
                    continue

                # Weighted average of eligible members (re-normalize weights)
                total_w = sum(weight_map[rn] for rn in eligible)
                if total_w <= 0:
                    continue

                ensemble_prob = np.zeros_like(gt_mask, dtype=np.float32)
                for rn, prob in eligible.items():
                    ensemble_prob += (weight_map[rn] / total_w) * prob

                binary_mask = (ensemble_prob >= threshold).astype(np.uint8)
                if not args.disable_postprocess:
                    pp_config = models[next(iter(eligible))].config
                    binary_mask = postprocess_binary_mask(binary_mask, pp_config)

                dices.append(compute_dice(binary_mask, gt_mask))
                ious.append(compute_iou(binary_mask, gt_mask))

            if not dices:
                continue

            mean_dice = float(np.mean(dices))
            mean_iou = float(np.mean(ious))

            if mean_dice > best_dice_tuned:
                best_dice_tuned = mean_dice
                best_threshold = threshold

            results.append({
                "preset": preset_name,
                "threshold": threshold,
                "dice": mean_dice,
                "iou": mean_iou,
                "n_samples": len(dices),
            })

            marker = " <-- best" if mean_dice >= best_dice_tuned else ""
            print(f"{preset_name:<20} {threshold:>10.2f} {mean_dice:>8.4f} {'':>12} {mean_iou:>8.4f} {len(dices):>8}{marker}")

        if best_dice_tuned > 0:
            print(f"{'':>20} {'BEST':>10} {'':>8} {best_dice_tuned:>12.4f} {'':>8} thr={best_threshold:.2f}")
        print("-" * 80)

    # Save results
    output_path = RUN_ROOT / "oof_ensemble_eval.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to: {output_path}")

    # Final leaderboard
    print("\n" + "=" * 60)
    print("LEADERBOARD (best threshold per preset)")
    print("=" * 60)
    best_per_preset: dict[str, dict] = {}
    for r in results:
        key = r["preset"]
        if key not in best_per_preset or r["dice"] > best_per_preset[key]["dice"]:
            best_per_preset[key] = r

    for rank, (preset, r) in enumerate(sorted(best_per_preset.items(), key=lambda x: -x[1]["dice"]), 1):
        print(f"  #{rank} {preset:<20} dice={r['dice']:.4f}  iou={r['iou']:.4f}  thr={r['threshold']:.2f}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
