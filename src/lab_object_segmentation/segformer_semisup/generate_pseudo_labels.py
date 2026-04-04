#!/usr/bin/env python3
"""Generate pseudo labels with the existing SegFormer teacher checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from lab_object_segmentation.common.paths import LAB1_DATASET_ROOT, LAB3_DATASET_ROOT, RUNS_ROOT
from lab_object_segmentation.segformer_semisup.run_lab3_submission import (
    ARTIFACT_ROOT,
    DEFAULT_CONFIG,
    build_val_transform,
    collect_image_paths,
    load_model,
    pick_best_checkpoint,
    predict_mask_with_tta,
    read_image_rgb,
    save_json,
    save_mask_png,
)


def default_input_dir(dataset_name: str) -> Path:
    if dataset_name == "unlabeled":
        return LAB3_DATASET_ROOT / "unlabeled" / "images"
    if dataset_name == "lab1_train":
        return LAB1_DATASET_ROOT / "train" / "train"
    if dataset_name == "lab1_test":
        return LAB1_DATASET_ROOT / "test_images" / "test_images"
    raise ValueError(f"Unsupported dataset name: {dataset_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pseudo labels with the trained SegFormer teacher.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="unlabeled",
        choices=["unlabeled", "lab1_train", "lab1_test", "custom"],
        help="Named dataset shortcut. Use 'custom' with --input-dir for arbitrary folders.",
    )
    parser.add_argument("--input-dir", type=Path, default=None, help="Optional custom image directory.")
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT, help="SegFormer artifact directory.")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Explicit teacher checkpoint.")
    parser.add_argument("--output-root", type=Path, default=None, help="Directory to store masks, reliability, summary.")
    parser.add_argument("--tta-n", type=int, default=None, help="Override number of TTA transforms.")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N images.")
    parser.add_argument("--confidence-threshold", type=float, default=0.85, help="Pixel reliability threshold.")
    parser.add_argument("--min-reliable-ratio", type=float, default=0.5, help="Min fraction of reliable pixels.")
    parser.add_argument("--min-area-ratio", type=float, default=0.01, help="Min foreground area ratio.")
    parser.add_argument("--max-area-ratio", type=float, default=0.95, help="Max foreground area ratio.")
    parser.add_argument("--save-all", action="store_true", help="Save masks even for rejected samples.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.dataset_name == "custom":
        if args.input_dir is None:
            raise ValueError("--input-dir is required when --dataset-name custom")
        input_dir = args.input_dir
        dataset_name = args.input_dir.name
    else:
        input_dir = args.input_dir or default_input_dir(args.dataset_name)
        dataset_name = args.dataset_name

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_root = args.output_root or (
        RUNS_ROOT / "dinov2_research" / "pseudo_labels" / dataset_name
    )
    mask_dir = output_root / "masks"
    reliability_dir = output_root / "reliability"
    output_root.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    reliability_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint_path
    checkpoint_meta = None
    if checkpoint_path is None:
        checkpoint_path, checkpoint_meta = pick_best_checkpoint(
            metrics_root=args.artifact_root / "metrics",
            artifact_root=args.artifact_root,
        )
    if checkpoint_meta is None:
        checkpoint_meta = {"source": "manual", "path": str(checkpoint_path)}
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    model, model_config, _ = load_model(checkpoint_path)
    transform = build_val_transform()
    tta_n = int(args.tta_n if args.tta_n is not None else model_config.get("tta_n_augments", DEFAULT_CONFIG["tta_n_augments"]))

    image_paths = collect_image_paths(input_dir)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    rows: list[dict] = []
    accepted = 0

    print(f"[pseudo] checkpoint={checkpoint_path}")
    print(f"[pseudo] input_dir={input_dir}")
    print(f"[pseudo] output_root={output_root}")
    print(f"[pseudo] images={len(image_paths)} tta_n={tta_n}")

    for image_path in tqdm(image_paths, desc=f"Pseudo labels: {dataset_name}"):
        image_rgb = read_image_rgb(image_path)
        mask, confidence = predict_mask_with_tta(
            model=model,
            image_rgb=image_rgb,
            transform=transform,
            tta_n=tta_n,
        )

        reliability = (confidence >= args.confidence_threshold).astype(np.uint8)
        area_ratio = float(mask.mean())
        reliable_ratio = float(reliability.mean())
        mean_confidence = float(confidence.mean())
        is_accepted = (
            reliable_ratio >= args.min_reliable_ratio
            and area_ratio >= args.min_area_ratio
            and area_ratio <= args.max_area_ratio
        )

        rows.append(
            {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "accepted": bool(is_accepted),
                "area_ratio": area_ratio,
                "reliable_ratio": reliable_ratio,
                "mean_confidence": mean_confidence,
                "mask_path": str(mask_dir / f"{image_path.stem}.png"),
                "reliability_path": str(reliability_dir / f"{image_path.stem}.png"),
            }
        )

        if is_accepted:
            accepted += 1
        if is_accepted or args.save_all:
            save_mask_png(mask_dir / f"{image_path.stem}.png", mask.astype(np.uint8))
            save_mask_png(reliability_dir / f"{image_path.stem}.png", reliability.astype(np.uint8))

    accepted_rows = [row for row in rows if row["accepted"]]
    summary = {
        "dataset_name": dataset_name,
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_meta": checkpoint_meta,
        "tta_n": tta_n,
        "num_total": len(rows),
        "num_accepted": accepted,
        "accepted_ratio": accepted / max(1, len(rows)),
        "mean_confidence_accepted": (
            float(np.mean([row["mean_confidence"] for row in accepted_rows])) if accepted_rows else 0.0
        ),
        "mean_area_ratio_accepted": (
            float(np.mean([row["area_ratio"] for row in accepted_rows])) if accepted_rows else 0.0
        ),
        "confidence_threshold": args.confidence_threshold,
        "min_reliable_ratio": args.min_reliable_ratio,
        "min_area_ratio": args.min_area_ratio,
        "max_area_ratio": args.max_area_ratio,
    }

    with open(output_root / "stats.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    save_json(output_root / "summary.json", summary)
    save_json(output_root / "summary_preview.json", {"summary": summary, "rows": rows[:20]})

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
