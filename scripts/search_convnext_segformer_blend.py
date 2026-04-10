#!/usr/bin/env python3
"""
Search ConvNeXt + SegFormer blend weights on a labeled validation split.

This is intended for practical blend tuning before full test inference.
By default it uses the fold-1 grouped validation split from lab3 train and the
strong single SegFormer mixed-data run as the anchor model.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

# Bootstrap
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from lab_object_segmentation.common.paths import LAB3_DATASET_ROOT, RUNS_ROOT
from lab_object_segmentation.convnext_inference import (
    ConvNeXtInferenceDataset,
    NOTEBOOK_IMAGE_SIZE,
    TTA_MODE_CHOICES,
    convnext_collate_fn,
    get_device,
    load_convnext_model,
    predict_with_tta as predict_convnext_with_tta,
)
from lab_object_segmentation.supervised_v4.ensemble import aggregate_probability_maps
from lab_object_segmentation.supervised_v4.predict_ensemble import load_member as load_segformer_member
from lab_object_segmentation.supervised_v4.train import (
    TrainConfig,
    collect_labeled_pairs,
    load_val_sample_ids,
    postprocess_binary_mask,
    predict_with_tta as predict_segformer_with_tta,
    upsample_logits,
)


CNXT_FOLD0 = "/Users/fgrach/Downloads/fold0_phase1_best_cnxt.pt"
CNXT_FOLD1 = "/Users/fgrach/Downloads/fold1_phase1_best_cnxt.pt"
DEPTH_ROOT = "/Users/fgrach/Downloads/content/depth_cache"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "artifacts/runs/convnext_ensemble/blend_weight_search.json"

SEGFORMER_RECIPE_SUMMARIES = {
    "top3_manual_teacher075": PROJECT_ROOT
    / "artifacts/runs/supervised_v4/submission_supervised_v4_top3_f014_manual_teacher075_weighted_mean_thr40_no_tta_summary.json",
    "single_manual_teacher075": PROJECT_ROOT
    / "artifacts/runs/supervised_v4/submission_supervised_v4_manual_teacher075_single_thr45_summary.json",
    "wide6": PROJECT_ROOT / "seg_runs/supervised_v4/submission_supervised_v4_wide6_thr50_summary.json",
}
SEGFORMER_RECIPE_CHOICES = tuple(SEGFORMER_RECIPE_SUMMARIES.keys())
SEGFORMER_TTA_CHOICES = ("auto", "on", "off")


@dataclass(frozen=True)
class ValSample:
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class SegFormerBlendRecipe:
    name: str
    summary_path: Path
    run_root: Path
    member_specs: list[tuple[str, float]]
    aggregation: str
    member_threshold: float
    tta_enabled: bool
    postprocess_enabled: bool
    postprocess_min_component_area: int
    postprocess_fill_holes: bool
    reference_threshold: float


class SegFormerInferenceDataset(Dataset):
    def __init__(self, samples: list[ValSample], transform: object):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_rgb = np.array(Image.open(sample.image_path).convert("RGB"))
        orig_h, orig_w = image_rgb.shape[:2]
        transformed = self.transform(image=image_rgb)
        return transformed["image"], sample.image_path.name, (orig_h, orig_w)


def segformer_collate_fn(batch):
    tensors, filenames, orig_hws = zip(*batch)
    return torch.stack(list(tensors)), list(filenames), list(orig_hws)


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


def _resolve_segformer_run_root(members_payload: list[dict]) -> Path:
    run_roots = set()
    for member in members_payload:
        run_dir = member.get("run_dir")
        if not run_dir:
            continue
        run_roots.add(Path(run_dir).expanduser().resolve().parent)
    if len(run_roots) == 1:
        return run_roots.pop()
    return PROJECT_ROOT / "artifacts/runs/supervised_v4"


def load_segformer_blend_recipe(recipe_name: str) -> SegFormerBlendRecipe:
    try:
        summary_path = SEGFORMER_RECIPE_SUMMARIES[recipe_name]
    except KeyError as exc:
        raise ValueError(f"Unknown SegFormer recipe: {recipe_name}") from exc
    if not summary_path.exists():
        raise FileNotFoundError(f"SegFormer recipe summary not found: {summary_path}")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    members_payload = payload.get("members", [])
    if not members_payload:
        raise RuntimeError(f"SegFormer recipe has no members: {summary_path}")

    member_specs = [(str(member["run_name"]), float(member["weight"])) for member in members_payload]
    return SegFormerBlendRecipe(
        name=recipe_name,
        summary_path=summary_path,
        run_root=_resolve_segformer_run_root(members_payload),
        member_specs=member_specs,
        aggregation=str(payload.get("aggregation", "weighted_mean")),
        member_threshold=float(payload.get("member_threshold", 0.5)),
        tta_enabled=bool(payload.get("tta_enabled", True)),
        postprocess_enabled=bool(payload.get("postprocess_enabled", True)),
        postprocess_min_component_area=int(payload.get("postprocess_min_component_area", 128)),
        postprocess_fill_holes=bool(payload.get("postprocess_fill_holes", True)),
        reference_threshold=float(payload.get("threshold", 0.5)),
    )


def resolve_segformer_tta(recipe: SegFormerBlendRecipe, tta_mode: str) -> bool:
    if tta_mode == "auto":
        return recipe.tta_enabled
    if tta_mode == "on":
        return True
    if tta_mode == "off":
        return False
    raise ValueError(f"Unsupported SegFormer TTA mode: {tta_mode}")


def build_weight_grid(step: float) -> list[float]:
    weights = []
    current = 0.0
    while current < 1.0 + step * 0.5:
        weights.append(round(min(1.0, current), 6))
        current += step
    if weights[-1] != 1.0:
        weights.append(1.0)
    return sorted(set(weights))


def collect_val_samples(
    fold: int,
    n_splits: int,
    limit: int | None,
    sample_seed: int,
) -> list[ValSample]:
    labeled = collect_labeled_pairs(
        LAB3_DATASET_ROOT / "train" / "images",
        LAB3_DATASET_ROOT / "train" / "masks",
        source_name="lab3_train",
        sample_weight=1.0,
    )
    val_ids = load_val_sample_ids(RUNS_ROOT, fold, n_splits=n_splits)
    samples = [ValSample(sample.image_path, sample.mask_path) for sample in labeled if sample.image_path.name in val_ids]
    if limit is not None and limit < len(samples):
        rng = random.Random(sample_seed)
        indices = list(range(len(samples)))
        rng.shuffle(indices)
        selected = sorted(indices[:limit])
        samples = [samples[idx] for idx in selected]
    return samples


def load_ground_truth_masks(samples: list[ValSample]) -> dict[str, np.ndarray]:
    gt_masks = {}
    for sample in samples:
        gt_mask = np.array(Image.open(sample.mask_path).convert("L"))
        gt_masks[sample.image_path.name] = (gt_mask > 127).astype(np.uint8)
    return gt_masks


def predict_convnext_probabilities(
    samples: list[ValSample],
    checkpoint_paths: list[str],
    depth_root: Path,
    image_size: int,
    tta_mode: str,
    batch_size: int,
    use_amp: bool,
    device: torch.device,
) -> dict[str, np.ndarray]:
    dataset = ConvNeXtInferenceDataset([sample.image_path for sample in samples], depth_root=depth_root, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=convnext_collate_fn,
    )
    models = [load_convnext_model(path, device) for path in checkpoint_paths]
    probabilities: dict[str, np.ndarray] = {}

    for rgbd_batch, filenames, orig_hws in tqdm(loader, desc=f"ConvNeXt val ({tta_mode})"):
        rgbd_batch = rgbd_batch.to(device)
        logit_accum = None
        for model in models:
            batch_logits = predict_convnext_with_tta(model, rgbd_batch, tta_mode, device, use_amp=use_amp)
            batch_logits = batch_logits.detach().cpu()
            logit_accum = batch_logits if logit_accum is None else logit_accum + batch_logits

        probs = torch.sigmoid(logit_accum / len(models))
        for idx, image_name in enumerate(filenames):
            prob = probs[idx, 0]
            orig_h, orig_w = orig_hws[idx]
            if tuple(prob.shape) != (orig_h, orig_w):
                prob = torch.nn.functional.interpolate(
                    prob.unsqueeze(0).unsqueeze(0),
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
            probabilities[image_name] = prob.numpy().astype(np.float16)
    return probabilities


def predict_segformer_member_probabilities(
    samples: list[ValSample],
    recipe: SegFormerBlendRecipe,
    segformer_use_tta: bool,
    batch_size: int,
    device: torch.device,
) -> dict[str, list[np.ndarray]]:
    sample_prob_lists: dict[str, list[np.ndarray]] = {sample.image_path.name: [] for sample in samples}

    for run_name, weight in recipe.member_specs:
        member = load_segformer_member(recipe.run_root, run_name, weight, device)
        dataset = SegFormerInferenceDataset(samples, member.transform)
        loader = DataLoader(
            dataset,
            batch_size=max(1, batch_size),
            shuffle=False,
            num_workers=0,
            collate_fn=segformer_collate_fn,
        )

        for image_batch, filenames, orig_hws in tqdm(loader, desc=f"SegFormer val ({run_name})", leave=False):
            image_batch = image_batch.to(device)
            if segformer_use_tta:
                probs = predict_segformer_with_tta(member.model, image_batch, image_batch.shape[-2:], member.config)
            else:
                seg_logits_low, _ = member.model(image_batch)
                seg_logits = upsample_logits(seg_logits_low, image_batch.shape[-2:])
                probs = torch.sigmoid(seg_logits)
            probs = probs.detach().cpu().numpy().astype(np.float32)

            for idx, image_name in enumerate(filenames):
                orig_h, orig_w = orig_hws[idx]
                prob_map = probs[idx, 0]
                if prob_map.shape != (orig_h, orig_w):
                    prob_map = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                sample_prob_lists[image_name].append(prob_map.astype(np.float16))
    return sample_prob_lists


def aggregate_segformer_probabilities(
    sample_prob_lists: dict[str, list[np.ndarray]],
    recipe: SegFormerBlendRecipe,
) -> dict[str, np.ndarray]:
    weights = [weight for _, weight in recipe.member_specs]
    aggregated = {}
    for image_name, prob_list in sample_prob_lists.items():
        aggregated[image_name] = aggregate_probability_maps(
            probability_maps=prob_list,
            weights=weights,
            aggregation=recipe.aggregation,
            member_threshold=recipe.member_threshold,
        ).astype(np.float16)
    return aggregated


def search_blend_grid(
    sample_names: list[str],
    gt_masks: dict[str, np.ndarray],
    convnext_probs: dict[str, np.ndarray],
    segformer_probs: dict[str, np.ndarray],
    convnext_weights: list[float],
    thresholds: list[float],
    postprocess_config: TrainConfig | None,
) -> list[dict]:
    results = []
    for convnext_weight in convnext_weights:
        segformer_weight = 1.0 - convnext_weight
        for threshold in thresholds:
            dices = []
            ious = []
            for image_name in sample_names:
                blended = (
                    convnext_weight * convnext_probs[image_name].astype(np.float32)
                    + segformer_weight * segformer_probs[image_name].astype(np.float32)
                )
                pred = (blended >= threshold).astype(np.uint8)
                if postprocess_config is not None and postprocess_config.postprocess_enabled:
                    pred = postprocess_binary_mask(pred, postprocess_config)
                gt = gt_masks[image_name]
                dices.append(compute_dice(pred, gt))
                ious.append(compute_iou(pred, gt))
            results.append(
                {
                    "convnext_weight": float(convnext_weight),
                    "segformer_weight": float(segformer_weight),
                    "threshold": float(threshold),
                    "dice": float(np.mean(dices)) if dices else 0.0,
                    "iou": float(np.mean(ious)) if ious else 0.0,
                    "n_samples": len(dices),
                }
            )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search ConvNeXt + SegFormer blend weights on labeled validation data.")
    parser.add_argument("--segformer-recipe", choices=SEGFORMER_RECIPE_CHOICES, default="single_manual_teacher075")
    parser.add_argument("--segformer-tta", choices=SEGFORMER_TTA_CHOICES, default="auto")
    parser.add_argument("--cnxt-tta-mode", choices=TTA_MODE_CHOICES, default="full14")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.35, 0.40, 0.45, 0.50, 0.55])
    parser.add_argument("--convnext-weights", nargs="+", type=float, default=None)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--val-fold", type=int, default=1)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="Optional number of val samples for smoke/proxy search.")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=NOTEBOOK_IMAGE_SIZE)
    parser.add_argument("--cnxt-batch-size", type=int, default=8)
    parser.add_argument("--segformer-batch-size", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--disable-postprocess", action="store_true")
    parser.add_argument("--postprocess-min-component-area", type=int, default=None)
    parser.add_argument("--disable-postprocess-fill-holes", action="store_true")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided")
    if args.cnxt_batch_size <= 0 or args.segformer_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if not 0.0 < args.weight_step <= 1.0:
        raise ValueError("--weight-step must be in (0, 1]")
    if any(not 0.0 <= thr <= 1.0 for thr in args.thresholds):
        raise ValueError("All thresholds must be in [0, 1]")

    device = get_device()
    recipe = load_segformer_blend_recipe(args.segformer_recipe)
    segformer_use_tta = resolve_segformer_tta(recipe, args.segformer_tta)

    samples = collect_val_samples(
        fold=args.val_fold,
        n_splits=args.n_splits,
        limit=args.limit,
        sample_seed=args.sample_seed,
    )
    if not samples:
        raise RuntimeError("No validation samples were selected")
    gt_masks = load_ground_truth_masks(samples)
    sample_names = [sample.image_path.name for sample in samples]

    print(f"Device             : {device}")
    print(f"Val fold           : {args.val_fold}/{args.n_splits - 1}")
    print(f"Samples            : {len(samples)}")
    print(f"SegFormer recipe   : {recipe.name}")
    print(f"SegFormer TTA      : {segformer_use_tta}")
    print(f"ConvNeXt TTA       : {args.cnxt_tta_mode}")
    print(f"Thresholds         : {args.thresholds}")

    convnext_weights = (
        sorted(set(float(weight) for weight in args.convnext_weights))
        if args.convnext_weights is not None
        else build_weight_grid(args.weight_step)
    )
    print(f"ConvNeXt weights   : {convnext_weights}")

    convnext_probs = predict_convnext_probabilities(
        samples=samples,
        checkpoint_paths=[CNXT_FOLD0, CNXT_FOLD1],
        depth_root=Path(DEPTH_ROOT),
        image_size=args.image_size,
        tta_mode=args.cnxt_tta_mode,
        batch_size=args.cnxt_batch_size,
        use_amp=not args.no_amp,
        device=device,
    )

    segformer_member_probs = predict_segformer_member_probabilities(
        samples=samples,
        recipe=recipe,
        segformer_use_tta=segformer_use_tta,
        batch_size=args.segformer_batch_size,
        device=device,
    )
    segformer_probs = aggregate_segformer_probabilities(segformer_member_probs, recipe)

    postprocess_config = None
    if recipe.postprocess_enabled and not args.disable_postprocess:
        first_member = load_segformer_member(recipe.run_root, recipe.member_specs[0][0], recipe.member_specs[0][1], device)
        postprocess_config = deepcopy(first_member.config)
        postprocess_config.postprocess_enabled = True
        if args.postprocess_min_component_area is not None:
            postprocess_config.postprocess_min_component_area = int(args.postprocess_min_component_area)
        else:
            postprocess_config.postprocess_min_component_area = recipe.postprocess_min_component_area
        postprocess_config.postprocess_fill_holes = not args.disable_postprocess_fill_holes and recipe.postprocess_fill_holes

    results = search_blend_grid(
        sample_names=sample_names,
        gt_masks=gt_masks,
        convnext_probs=convnext_probs,
        segformer_probs=segformer_probs,
        convnext_weights=convnext_weights,
        thresholds=[float(thr) for thr in args.thresholds],
        postprocess_config=postprocess_config,
    )
    ranked = sorted(results, key=lambda row: (row["dice"], row["iou"]), reverse=True)
    best = ranked[0]

    output_payload = {
        "status": "proxy_search" if args.limit is not None else "full_holdout_search",
        "warning": (
            "ConvNeXt blend weights are tuned on the selected validation split. "
            "This is a practical proxy unless the ConvNeXt checkpoints are confirmed to be fold-compatible with that split."
        ),
        "device": str(device),
        "val_fold": args.val_fold,
        "n_splits": args.n_splits,
        "n_samples": len(samples),
        "sample_seed": args.sample_seed,
        "segformer": {
            "recipe": recipe.name,
            "recipe_summary": str(recipe.summary_path),
            "tta_enabled": segformer_use_tta,
            "reference_threshold": recipe.reference_threshold,
            "aggregation": recipe.aggregation,
            "member_threshold": recipe.member_threshold,
            "members": [{"run_name": run_name, "weight": weight} for run_name, weight in recipe.member_specs],
        },
        "convnext": {
            "checkpoints": [CNXT_FOLD0, CNXT_FOLD1],
            "tta_mode": args.cnxt_tta_mode,
            "image_size": args.image_size,
            "use_amp": not args.no_amp,
        },
        "search_grid": {
            "thresholds": [float(thr) for thr in args.thresholds],
            "convnext_weights": convnext_weights,
        },
        "postprocess": (
            {
                "enabled": True,
                "min_component_area": postprocess_config.postprocess_min_component_area,
                "fill_holes": postprocess_config.postprocess_fill_holes,
            }
            if postprocess_config is not None
            else {"enabled": False}
        ),
        "best_result": best,
        "top10": ranked[:10],
        "all_results": ranked,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    print("\nBest blend:")
    print(
        f"  convnext_weight={best['convnext_weight']:.2f} "
        f"segformer_weight={best['segformer_weight']:.2f} "
        f"threshold={best['threshold']:.2f} "
        f"dice={best['dice']:.4f} iou={best['iou']:.4f}"
    )
    print(f"Saved search summary: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
