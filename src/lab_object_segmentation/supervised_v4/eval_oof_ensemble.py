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
from lab_object_segmentation.supervised_v4.ensemble import (
    AGGREGATION_CHOICES,
    PRESET_MEMBERS,
    aggregate_probability_maps,
    resolve_member_specs,
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
    transform = build_val_transform(config.image_size, config.image_mean, config.image_std, config.resize_interpolation)
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
    parser.add_argument("--run-names", nargs="+", default=None, help="Optional explicit run names instead of presets.")
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="Optional weights for --run-names.")
    parser.add_argument("--aggregations", nargs="+", choices=AGGREGATION_CHOICES, default=["weighted_mean"], help="Aggregation rules to evaluate.")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.40, 0.45, 0.50, 0.55, 0.60], help="Thresholds to try")
    parser.add_argument("--member-threshold", type=float, default=0.5, help="Member threshold used by hard_vote aggregation.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of labeled samples for quick smoke screening.")
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA")
    parser.add_argument("--tta-scales", nargs="+", type=float, default=None, help="Optional override for member TTA scales.")
    parser.add_argument("--tta-ops", choices=("flips", "d4"), default=None, help="Optional override for member TTA transforms.")
    parser.add_argument("--disable-postprocess", action="store_true", help="Disable post-processing")
    parser.add_argument("--postprocess-min-component-area", type=int, default=128)
    parser.add_argument("--disable-postprocess-fill-holes", action="store_true")
    parser.add_argument("--n-splits", type=int, default=3, help="Number of grouped camera folds used by the evaluated runs.")
    parser.add_argument("--output-path", type=Path, default=RUN_ROOT / "oof_ensemble_search.json")
    return parser.parse_args()


def resolve_named_member_sets(args: argparse.Namespace) -> dict[str, list[tuple[str, float]]]:
    if args.run_names is not None:
        return {"custom": resolve_member_specs(run_names=args.run_names, weights=args.weights)}

    named_sets: dict[str, list[tuple[str, float]]] = {}
    for preset_name in args.presets:
        named_sets[preset_name] = resolve_member_specs(preset=preset_name)
    return named_sets


def apply_model_inference_overrides(model: LoadedModel, args: argparse.Namespace) -> LoadedModel:
    if args.tta_scales is not None:
        model.config.tta_scales = [float(scale) for scale in args.tta_scales]
    if args.tta_ops is not None:
        model.config.tta_ops = str(args.tta_ops)
    model.config.postprocess_enabled = not args.disable_postprocess
    model.config.postprocess_min_component_area = int(args.postprocess_min_component_area)
    model.config.postprocess_fill_holes = not args.disable_postprocess_fill_holes
    return model


def main() -> int:
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")
    named_member_sets = resolve_named_member_sets(args)

    # Load all samples and build fold->val_ids mapping
    images_dir = LAB3_DATASET_ROOT / "train" / "images"
    masks_dir = LAB3_DATASET_ROOT / "train" / "masks"
    all_samples = collect_labeled_pairs(images_dir, masks_dir, source_name="lab3_train", sample_weight=1.0)
    if args.limit is not None:
        all_samples = all_samples[: args.limit]
    print(f"Total labeled samples: {len(all_samples)}")

    fold_val_ids: dict[int, set[str]] = {}
    for fold in range(args.n_splits):
        fold_val_ids[fold] = load_val_sample_ids(RUNS_ROOT, fold, n_splits=args.n_splits)
    print(f"Fold val sizes: {{{', '.join(f'{f}: {len(ids)}' for f, ids in fold_val_ids.items())}}}")

    # Collect all unique models needed across all presets
    all_run_names: set[str] = set()
    for member_specs in named_member_sets.values():
        for run_name, _ in member_specs:
            all_run_names.add(run_name)

    print(f"\nLoading {len(all_run_names)} unique models...")
    models: dict[str, LoadedModel] = {}
    for run_name in sorted(all_run_names):
        print(f"  Loading {run_name}...")
        models[run_name] = apply_model_inference_overrides(load_model(run_name, 1.0, device), args)
    print("All models loaded.\n")

    # Pre-compute OOF probabilities: for each sample, predict with models that didn't see it
    # Key: image_name -> {run_name: prob_map}
    print("Computing OOF probability maps...")
    oof_probs: dict[str, dict[str, np.ndarray]] = {}
    oof_gt: dict[str, np.ndarray] = {}

    for idx, sample in enumerate(tqdm(all_samples, desc="OOF inference")):
        image_path = sample.image_path
        mask_path = sample.mask_path
        image_name = image_path.name
        image_rgb = np.array(Image.open(image_path).convert("RGB"))
        gt_mask = np.array(Image.open(mask_path).convert("L"))
        gt_mask = (gt_mask > 127).astype(np.uint8)
        oof_gt[image_name] = gt_mask

        sample_val_fold = None
        for fold, val_ids in fold_val_ids.items():
            if image_name in val_ids:
                sample_val_fold = fold
                break

        sample_probs: dict[str, np.ndarray] = {}
        for run_name, member in models.items():
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
    print(f"{'Ensemble':<20} {'Agg':<14} {'Threshold':>10} {'Dice':>8} {'IoU':>8} {'Samples':>8}")
    print("=" * 80)

    results: list[dict] = []

    for member_set_name, members in named_member_sets.items():
        member_run_names = {run_name for run_name, _ in members}
        weight_map = {run_name: w for run_name, w in members}
        representative_config = models[members[0][0]].config

        for aggregation in args.aggregations:
            best_dice = -1.0
            best_threshold = -1.0

            for threshold in args.thresholds:
                dices = []
                ious = []

                for image_name, gt_mask in oof_gt.items():
                    sample_probs = oof_probs[image_name]

                    eligible_items = [(rn, prob) for rn, prob in sample_probs.items() if rn in member_run_names]
                    if not eligible_items:
                        continue

                    ensemble_prob = aggregate_probability_maps(
                        probability_maps=[prob for _, prob in eligible_items],
                        weights=[weight_map[rn] for rn, _ in eligible_items],
                        aggregation=aggregation,
                        member_threshold=args.member_threshold,
                    )

                    binary_mask = (ensemble_prob >= threshold).astype(np.uint8)
                    if not args.disable_postprocess:
                        pp_config = models[eligible_items[0][0]].config
                        binary_mask = postprocess_binary_mask(binary_mask, pp_config)

                    dices.append(compute_dice(binary_mask, gt_mask))
                    ious.append(compute_iou(binary_mask, gt_mask))

                if not dices:
                    continue

                mean_dice = float(np.mean(dices))
                mean_iou = float(np.mean(ious))

                if mean_dice > best_dice:
                    best_dice = mean_dice
                    best_threshold = threshold

                results.append({
                    "ensemble": member_set_name,
                    "aggregation": aggregation,
                    "threshold": float(threshold),
                    "dice": mean_dice,
                    "iou": mean_iou,
                    "n_samples": len(dices),
                    "tta_enabled": not args.no_tta,
                    "tta_ops": None if args.no_tta else representative_config.tta_ops,
                    "tta_scales": None if args.no_tta else representative_config.tta_scales,
                    "member_threshold": float(args.member_threshold),
                    "postprocess_enabled": not args.disable_postprocess,
                    "postprocess_min_component_area": int(args.postprocess_min_component_area),
                    "postprocess_fill_holes": not args.disable_postprocess_fill_holes,
                })

                marker = " <-- best" if mean_dice >= best_dice else ""
                print(f"{member_set_name:<20} {aggregation:<14} {threshold:>10.2f} {mean_dice:>8.4f} {mean_iou:>8.4f} {len(dices):>8}{marker}")

            if best_dice > 0:
                print(f"{member_set_name:<20} {aggregation:<14} {'BEST':>10} {best_dice:>8.4f} {'':>8} thr={best_threshold:.2f}")
            print("-" * 80)

    # Save results
    best_per_combo: dict[tuple[str, str], dict] = {}
    for row in results:
        key = (row["ensemble"], row["aggregation"])
        if key not in best_per_combo or row["dice"] > best_per_combo[key]["dice"]:
            best_per_combo[key] = row

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(
            {
                "settings": {
                    "presets": args.presets,
                    "run_names": args.run_names,
                    "weights": args.weights,
                    "aggregations": args.aggregations,
                    "thresholds": args.thresholds,
                    "member_threshold": float(args.member_threshold),
                    "limit": args.limit,
                    "tta_enabled": not args.no_tta,
                    "tta_scales": args.tta_scales,
                    "tta_ops": args.tta_ops,
                    "postprocess_enabled": not args.disable_postprocess,
                    "postprocess_min_component_area": int(args.postprocess_min_component_area),
                    "postprocess_fill_holes": not args.disable_postprocess_fill_holes,
                },
                "results": results,
                "best_per_combo": list(best_per_combo.values()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nResults saved to: {args.output_path}")

    # Final leaderboard
    print("\n" + "=" * 60)
    print("LEADERBOARD (best threshold per ensemble x aggregation)")
    print("=" * 60)
    for rank, ((ensemble_name, aggregation), row) in enumerate(sorted(best_per_combo.items(), key=lambda item: -item[1]["dice"]), 1):
        print(
            f"  #{rank} {ensemble_name:<20} {aggregation:<14} "
            f"dice={row['dice']:.4f}  iou={row['iou']:.4f}  thr={row['threshold']:.2f}"
        )

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
