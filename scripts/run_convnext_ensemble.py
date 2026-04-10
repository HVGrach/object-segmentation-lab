#!/usr/bin/env python3
"""
ConvNeXt V2 inference with comparison TTA presets and a trusted SegFormer blend.

Usage:
  python scripts/run_convnext_ensemble.py --mode geometric
  python scripts/run_convnext_ensemble.py --mode scale_color
  python scripts/run_convnext_ensemble.py --mode full14
  python scripts/run_convnext_ensemble.py --mode both
  python scripts/run_convnext_ensemble.py --mode blend_segformer
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

# Bootstrap
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lab_object_segmentation.convnext_inference import (
    InferenceConfig,
    NOTEBOOK_IMAGE_SIZE,
    NOTEBOOK_THRESHOLD,
    TTA_MODE_CHOICES,
    get_device,
    load_convnext_model,
    load_rgbd,
    predict_with_tta,
    run_convnext_inference,
    serialize_mask,
    submission_stem,
    write_submission_csv,
)
from lab_object_segmentation.supervised_v4.ensemble import aggregate_probability_maps
from lab_object_segmentation.supervised_v4.predict_ensemble import (
    load_member as load_segformer_member,
    predict_member_probability,
    read_image_rgb,
)
from lab_object_segmentation.supervised_v4.train import postprocess_binary_mask

# ── Paths ──
CNXT_FOLD0 = "/Users/fgrach/Downloads/fold0_phase1_best_cnxt.pt"
CNXT_FOLD1 = "/Users/fgrach/Downloads/fold1_phase1_best_cnxt.pt"
TEST_DIR = str(PROJECT_ROOT / "data/raw/dl-lab-3-product-segmentation/test_images")
DEPTH_ROOT = "/Users/fgrach/Downloads/content/depth_cache"
OUTPUT_DIR = str(PROJECT_ROOT / "artifacts/runs/convnext_ensemble")
SEGFORMER_RUN_ROOT = PROJECT_ROOT / "artifacts/runs/supervised_v4"

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


def _resolve_segformer_run_root(members_payload: list[dict]) -> Path:
    run_roots = set()
    for member in members_payload:
        run_dir = member.get("run_dir")
        if not run_dir:
            continue
        run_roots.add(Path(run_dir).expanduser().resolve().parent)
    if len(run_roots) == 1:
        return run_roots.pop()
    return SEGFORMER_RUN_ROOT


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

    member_specs = [
        (str(member["run_name"]), float(member["weight"]))
        for member in members_payload
    ]
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


def blend_with_segformer(
    cnxt_tta_mode: str,
    threshold: float,
    segformer_weight: float,
    cnxt_weight: float,
    image_size: int,
    min_component_area: int | None,
    fill_holes: bool,
    use_amp: bool,
    limit: int,
    segformer_recipe_name: str,
    segformer_tta_mode: str,
) -> Path:
    """
    Blend ConvNeXt probabilities with a confirmed supervised_v4 SegFormer recipe.

    SegFormer members are loaded through the same helpers that generate the known
    supervised_v4 submissions, avoiding local reimplementation drift.
    """
    device = get_device()
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    recipe = load_segformer_blend_recipe(segformer_recipe_name)
    segformer_use_tta = resolve_segformer_tta(recipe, segformer_tta_mode)

    cnxt_models = [
        load_convnext_model(CNXT_FOLD0, device),
        load_convnext_model(CNXT_FOLD1, device),
    ]
    segformer_members = [
        load_segformer_member(recipe.run_root, run_name, weight, device)
        for run_name, weight in recipe.member_specs
    ]

    effective_min_component_area = (
        recipe.postprocess_min_component_area if min_component_area is None else int(min_component_area)
    )
    postprocess_config = deepcopy(segformer_members[0].config)
    postprocess_config.postprocess_enabled = recipe.postprocess_enabled
    postprocess_config.postprocess_min_component_area = effective_min_component_area
    postprocess_config.postprocess_fill_holes = fill_holes

    test_dir = Path(TEST_DIR)
    depth_root = Path(DEPTH_ROOT)
    image_paths = sorted(test_dir.glob("*.jpg"))
    if limit > 0:
        image_paths = image_paths[:limit]

    w_total = segformer_weight + cnxt_weight
    w_seg = segformer_weight / w_total
    w_cnxt = cnxt_weight / w_total

    print(f"\n[Blend] {len(image_paths)} images")
    print(f"  ConvNeXt: {len(cnxt_models)} models, TTA={cnxt_tta_mode}, weight={w_cnxt:.2f}")
    print(
        f"  SegFormer: {len(segformer_members)} members, recipe={recipe.name}, "
        f"agg={recipe.aggregation}, tta={segformer_use_tta}, weight={w_seg:.2f}"
    )
    print(
        f"  threshold={threshold:.2f}, postprocess={postprocess_config.postprocess_enabled}, "
        f"min_component_area={postprocess_config.postprocess_min_component_area}, "
        f"fill_holes={postprocess_config.postprocess_fill_holes}"
    )

    rows = []
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Blend inference"):
            image_rgb = read_image_rgb(img_path)
            orig_h, orig_w = image_rgb.shape[:2]

            rgbd = load_rgbd(img_path, depth_root, image_size).unsqueeze(0).to(device)
            cnxt_logit_accum = None
            for model in cnxt_models:
                logits = predict_with_tta(model, rgbd, cnxt_tta_mode, device, use_amp=use_amp)
                logits = logits.detach().cpu()
                cnxt_logit_accum = logits if cnxt_logit_accum is None else cnxt_logit_accum + logits
            cnxt_prob = torch.sigmoid(cnxt_logit_accum / len(cnxt_models))
            cnxt_prob = F.interpolate(
                cnxt_prob,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy().astype(np.float32)

            segformer_probability_maps = [
                predict_member_probability(member, image_rgb, device, use_tta=segformer_use_tta)
                for member in segformer_members
            ]
            segformer_prob = aggregate_probability_maps(
                probability_maps=segformer_probability_maps,
                weights=[member.weight for member in segformer_members],
                aggregation=recipe.aggregation,
                member_threshold=recipe.member_threshold,
            )

            blended_prob = (w_cnxt * cnxt_prob + w_seg * segformer_prob).astype(np.float32)
            mask = (blended_prob >= float(threshold)).astype(np.uint8)
            if postprocess_config.postprocess_enabled:
                mask = postprocess_binary_mask(mask, postprocess_config)

            rows.append({"ImageId": img_path.name, "mask": serialize_mask(mask)})

    recipe_tag = recipe.name.replace("manual_teacher075", "mt075")
    base_name = (
        f"submission_blend_cnxt{int(round(w_cnxt * 100)):02d}_"
        f"seg{int(round(w_seg * 100)):02d}_{recipe_tag}_{cnxt_tta_mode}"
    )
    if segformer_use_tta != recipe.tta_enabled:
        base_name += "_segtta" if segformer_use_tta else "_segno_tta"
    if image_size != NOTEBOOK_IMAGE_SIZE:
        base_name += f"_cnxtsz{image_size}"
    base_name += f"_thr{int(round(threshold * 100)):02d}"
    sub_name = submission_stem(base_name, limit=limit)
    sub_path = out_dir / f"{sub_name}.csv"
    write_submission_csv(sub_path, rows)

    cfg_path = out_dir / f"{sub_name}_config.json"
    cfg_payload = {
        "mode": "blend_segformer",
        "n_images": len(rows),
        "limit": limit,
        "threshold": threshold,
        "weights": {"convnext": w_cnxt, "segformer": w_seg},
        "convnext": {
            "checkpoints": [CNXT_FOLD0, CNXT_FOLD1],
            "tta_mode": cnxt_tta_mode,
            "image_size": image_size,
            "use_amp": use_amp,
        },
        "segformer": {
            "recipe": recipe.name,
            "recipe_summary": str(recipe.summary_path),
            "run_root": str(recipe.run_root),
            "members": [
                {"run_name": member.run_name, "weight": member.weight}
                for member in segformer_members
            ],
            "aggregation": recipe.aggregation,
            "member_threshold": recipe.member_threshold,
            "tta_enabled": segformer_use_tta,
            "reference_threshold": recipe.reference_threshold,
        },
        "postprocess": {
            "enabled": postprocess_config.postprocess_enabled,
            "min_component_area": postprocess_config.postprocess_min_component_area,
            "fill_holes": postprocess_config.postprocess_fill_holes,
        },
    }
    cfg_path.write_text(json.dumps(cfg_payload, indent=2), encoding="utf-8")

    print(f"\n[Blend] Saved: {sub_path}")
    return sub_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["geometric", "scale_color", "full14", "both", "blend_segformer"],
        default="both",
    )
    parser.add_argument("--threshold", type=float, default=NOTEBOOK_THRESHOLD)
    parser.add_argument("--image-size", type=int, default=NOTEBOOK_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 = all images")
    parser.add_argument(
        "--cnxt-tta-mode",
        choices=TTA_MODE_CHOICES,
        default="full14",
        help="ConvNeXt TTA preset used inside blend_segformer",
    )
    parser.add_argument(
        "--segformer-recipe",
        choices=SEGFORMER_RECIPE_CHOICES,
        default="top3_manual_teacher075",
        help="Confirmed supervised_v4 recipe used inside blend_segformer",
    )
    parser.add_argument(
        "--segformer-tta",
        choices=SEGFORMER_TTA_CHOICES,
        default="auto",
        help="Use recipe TTA, force it on, or force it off for SegFormer members",
    )
    parser.add_argument(
        "--segformer-weight",
        type=float,
        default=0.65,
        help="SegFormer weight in blend (ConvNeXt gets 1 - this)",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=None,
        help="Override min component area. Omit to keep recipe defaults for blend and 0 for standalone ConvNeXt.",
    )
    parser.add_argument("--no-fill-holes", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(f"--threshold must be in [0, 1], got {args.threshold}")
    if not 0.0 <= args.segformer_weight <= 1.0:
        raise ValueError(f"--segformer-weight must be in [0, 1], got {args.segformer_weight}")
    if args.image_size <= 0:
        raise ValueError(f"--image-size must be positive, got {args.image_size}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if args.limit < 0:
        raise ValueError(f"--limit must be >= 0, got {args.limit}")
    if args.min_component_area is not None and args.min_component_area < 0:
        raise ValueError(f"--min-component-area must be >= 0 when provided, got {args.min_component_area}")

    if args.image_size != NOTEBOOK_IMAGE_SIZE:
        print(
            f"[ConvNeXt] WARNING: notebook-faithful image_size is {NOTEBOOK_IMAGE_SIZE}, "
            f"but running with {args.image_size}."
        )
    if args.threshold != NOTEBOOK_THRESHOLD:
        print(
            f"[ConvNeXt] WARNING: notebook-faithful threshold is {NOTEBOOK_THRESHOLD:.2f}, "
            f"but running with {args.threshold:.2f}."
        )

    convnext_min_component_area = 0 if args.min_component_area is None else args.min_component_area

    if args.mode == "blend_segformer":
        cnxt_weight = 1.0 - args.segformer_weight
        sub = blend_with_segformer(
            cnxt_tta_mode=args.cnxt_tta_mode,
            threshold=args.threshold,
            segformer_weight=args.segformer_weight,
            cnxt_weight=cnxt_weight,
            image_size=args.image_size,
            min_component_area=args.min_component_area,
            fill_holes=not args.no_fill_holes,
            use_amp=not args.no_amp,
            limit=args.limit,
            segformer_recipe_name=args.segformer_recipe,
            segformer_tta_mode=args.segformer_tta,
        )
        print(f"\nBlend submission: {sub}")
        return

    cfgs = []
    if args.mode in ("geometric", "both"):
        cfgs.append(
            InferenceConfig(
                checkpoint_paths=[CNXT_FOLD0, CNXT_FOLD1],
                test_dir=TEST_DIR,
                depth_root=DEPTH_ROOT,
                output_dir=OUTPUT_DIR,
                image_size=args.image_size,
                threshold=args.threshold,
                tta_mode="geometric",
                min_component_area=convnext_min_component_area,
                fill_holes=not args.no_fill_holes,
                batch_size=args.batch_size,
                use_amp=not args.no_amp,
                limit=args.limit,
            )
        )
    if args.mode in ("scale_color", "both"):
        cfgs.append(
            InferenceConfig(
                checkpoint_paths=[CNXT_FOLD0, CNXT_FOLD1],
                test_dir=TEST_DIR,
                depth_root=DEPTH_ROOT,
                output_dir=OUTPUT_DIR,
                image_size=args.image_size,
                threshold=args.threshold,
                tta_mode="scale_color",
                min_component_area=convnext_min_component_area,
                fill_holes=not args.no_fill_holes,
                batch_size=args.batch_size,
                use_amp=not args.no_amp,
                limit=args.limit,
            )
        )
    if args.mode == "full14":
        cfgs.append(
            InferenceConfig(
                checkpoint_paths=[CNXT_FOLD0, CNXT_FOLD1],
                test_dir=TEST_DIR,
                depth_root=DEPTH_ROOT,
                output_dir=OUTPUT_DIR,
                image_size=args.image_size,
                threshold=args.threshold,
                tta_mode="full14",
                min_component_area=convnext_min_component_area,
                fill_holes=not args.no_fill_holes,
                batch_size=args.batch_size,
                use_amp=not args.no_amp,
                limit=args.limit,
            )
        )

    for cfg in cfgs:
        sub_path = run_convnext_inference(cfg)
        print(f"\nSubmission: {sub_path}")


if __name__ == "__main__":
    main()
