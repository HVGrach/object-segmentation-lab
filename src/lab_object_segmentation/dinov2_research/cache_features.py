#!/usr/bin/env python3
"""Precompute frozen DINOv2 patch tokens for fast ablations on Metal/CUDA/CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm

from lab_object_segmentation.common.paths import LAB3_DATASET_ROOT, RUNS_ROOT, guess_project_root
from lab_object_segmentation.dinov2_research.backbone import DINOv2Backbone

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
def get_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_image_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def build_transform(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def collect_paths(input_dir: Path, stems_file: Path | None, limit: int | None) -> list[Path]:
    if stems_file is None:
        paths = [p for p in sorted(input_dir.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        return paths[:limit] if limit else paths

    stems = [line.strip() for line in stems_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    paths: list[Path] = []
    for stem in stems:
        found = None
        for suffix in IMAGE_EXTS:
            candidate = input_dir / f"{stem}{suffix}"
            if candidate.exists():
                found = candidate
                break
        if found is None:
            raise FileNotFoundError(f"Could not resolve stem '{stem}' inside {input_dir}")
        paths.append(found)
    return paths[:limit] if limit else paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = guess_project_root()
    default_input = LAB3_DATASET_ROOT / "train" / "images"

    parser = argparse.ArgumentParser(description="Cache frozen DINOv2 features to disk.")
    parser.add_argument("--input-dir", type=Path, default=default_input, help="Directory with source images.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to save .pt feature files.")
    parser.add_argument("--stems-file", type=Path, default=None, help="Optional text file with stem names to cache.")
    parser.add_argument("--backbone-size", type=str, default="s", choices=["s", "b", "l"])
    parser.add_argument("--img-size", type=int, default=322, help="Resize before DINOv2 forward; must be /14.")
    parser.add_argument("--limit", type=int, default=None, help="Cache only first N images.")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--overwrite", action="store_true", help="Recompute features even if .pt already exists.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = guess_project_root()
    output_dir = args.output_dir or (
        RUNS_ROOT / "dinov2_research" / "feature_cache" / f"dinov2_{args.backbone_size}_{args.img_size}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.img_size % 14 != 0:
        raise ValueError(f"--img-size must be divisible by 14, got {args.img_size}")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    image_paths = collect_paths(args.input_dir, args.stems_file, args.limit)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    transform = build_transform(args.img_size)
    device = get_device()
    backbone = DINOv2Backbone(size=args.backbone_size, device=device)
    save_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    print(f"[cache] device={device}")
    print(f"[cache] input_dir={args.input_dir}")
    print(f"[cache] output_dir={output_dir}")
    print(f"[cache] images={len(image_paths)} backbone=DINOv2-{args.backbone_size.upper()} img_size={args.img_size}")

    saved = 0
    skipped = 0
    failures: list[dict[str, str]] = []

    for image_path in tqdm(image_paths, desc="Caching DINOv2 features"):
        out_path = output_dir / f"{image_path.stem}.pt"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            image_rgb = read_image_rgb(image_path)
            tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                tokens = backbone.extract_patch_tokens(tensor)[-1].squeeze(0)
            tokens = tokens.to(dtype=save_dtype).cpu().contiguous()
            torch.save(tokens, out_path)
            saved += 1
            if device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.synchronize()
        except Exception as exc:
            failures.append({"image": str(image_path), "error": str(exc)})

    manifest = {
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "backbone_size": args.backbone_size,
        "img_size": args.img_size,
        "dtype": args.dtype,
        "num_requested": len(image_paths),
        "num_saved": saved,
        "num_skipped": skipped,
        "num_failures": len(failures),
        "stems_file": None if args.stems_file is None else str(args.stems_file),
        "failures_preview": failures[:10],
    }
    manifest_path = output_dir / "cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[cache] saved={saved} skipped={skipped} failures={len(failures)}")
    print(f"[cache] manifest={manifest_path}")
    if failures:
        raise RuntimeError(f"Feature caching finished with {len(failures)} failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
