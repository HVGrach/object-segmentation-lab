#!/usr/bin/env python3
"""Warm the base sample cache for the SegFormer semi-supervised pipeline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from lab_object_segmentation.common.paths import LAB1_DATASET_ROOT, LAB3_DATASET_ROOT, RUNS_ROOT
from lab_object_segmentation.segformer_semisup.sample_cache import warm_segformer_cache


def collect_image_paths(input_dir: Path) -> list[Path]:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in image_exts
    ]


def collect_labeled_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    image_map = {path.stem: path for path in collect_image_paths(images_dir)}
    pairs: list[tuple[Path, Path]] = []
    for mask_path in sorted(masks_dir.rglob("*")):
        if not mask_path.is_file():
            continue
        image_path = image_map.get(mask_path.stem)
        if image_path is not None:
            pairs.append((image_path, mask_path))
    return pairs


def limit_deterministically(items: list, limit: int | None, seed: int) -> list:
    if limit is None or len(items) <= limit:
        return items
    rng = random.Random(seed)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    selected = sorted(indices[:limit])
    return [items[idx] for idx in selected]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm disk cache for SegFormer base samples.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--min-image-size", type=int, default=64)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUNS_ROOT / "segformer_boundary_semisup_macos" / "sample_cache" / "size_224",
    )
    parser.add_argument("--include-external-unlabeled", action="store_true")
    parser.add_argument("--limit-labeled", type=int, default=None)
    parser.add_argument("--limit-unlabeled", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    labeled_samples = collect_labeled_pairs(
        LAB3_DATASET_ROOT / "train" / "images",
        LAB3_DATASET_ROOT / "train" / "masks",
    )
    unlabeled_paths = collect_image_paths(LAB3_DATASET_ROOT / "unlabeled" / "images")
    if args.include_external_unlabeled:
        unlabeled_paths.extend(collect_image_paths(LAB1_DATASET_ROOT / "train" / "train"))
        unlabeled_paths.extend(
            collect_image_paths(LAB1_DATASET_ROOT / "test_images" / "test_images")
        )

    labeled_samples = limit_deterministically(labeled_samples, args.limit_labeled, args.seed)
    unlabeled_paths = limit_deterministically(unlabeled_paths, args.limit_unlabeled, args.seed)

    print(f"[cache] output_dir={args.output_dir}")
    print(f"[cache] labeled={len(labeled_samples)} unlabeled={len(unlabeled_paths)}")

    manifest = warm_segformer_cache(
        cache_root=args.output_dir,
        labeled_samples=labeled_samples,
        unlabeled_paths=unlabeled_paths,
        image_size=args.image_size,
        min_image_size=args.min_image_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
