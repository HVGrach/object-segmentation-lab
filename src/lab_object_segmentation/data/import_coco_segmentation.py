#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from lab_object_segmentation.common.paths import PROJECT_ROOT


def slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    return slug.strip("._") or "imported_coco"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a COCO segmentation export into the project's images/masks format."
    )
    parser.add_argument(
        "--coco-root",
        required=True,
        help="Path to the COCO export root (or directly to a split folder with _annotations.coco.json).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split name inside --coco-root when the annotations live in a nested folder (default: train).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output dataset root. Defaults to data/derived/imported_coco/<source-name>.",
    )
    parser.add_argument(
        "--source-name",
        default="",
        help="Logical dataset name stored in manifest.json. Defaults to a slug based on the COCO folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output directory.",
    )
    return parser.parse_args()


def resolve_split_dir(coco_root: Path, split: str) -> Path:
    direct_annotations = coco_root / "_annotations.coco.json"
    if direct_annotations.exists():
        return coco_root

    split_dir = coco_root / split
    split_annotations = split_dir / "_annotations.coco.json"
    if split_annotations.exists():
        return split_dir

    raise FileNotFoundError(
        f"Could not find _annotations.coco.json in {coco_root} or {split_dir}."
    )


def resolve_output_image_name(image_payload: dict, used_names: set[str]) -> str:
    extra = image_payload.get("extra")
    preferred_name = (
        extra.get("name")
        if isinstance(extra, dict) and extra.get("name")
        else image_payload["file_name"]
    )
    candidate = preferred_name.replace("/", "_")
    path = Path(candidate)
    stem = path.stem
    suffix = path.suffix or ".jpg"
    final_name = f"{stem}{suffix}"
    if final_name in used_names:
        final_name = f"{stem}__id{image_payload['id']}{suffix}"
    used_names.add(final_name)
    return final_name


def decode_uncompressed_rle(segmentation: dict) -> np.ndarray:
    size = segmentation.get("size")
    counts = segmentation.get("counts")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError(f"Unsupported RLE size payload: {size}")
    if isinstance(counts, str):
        counts = decode_compressed_rle_string(counts)
    if not isinstance(counts, list):
        raise ValueError(f"Unsupported COCO RLE counts payload: {type(counts)!r}")

    total = int(size[0]) * int(size[1])
    flat = np.zeros(total, dtype=np.uint8)
    cursor = 0
    value = 0
    for run_length in counts:
        run_length = int(run_length)
        if run_length < 0:
            raise ValueError(f"Negative RLE run-length encountered: {run_length}")
        if value == 1 and run_length > 0:
            flat[cursor : cursor + run_length] = 1
        cursor += run_length
        value = 1 - value

    if cursor != total:
        raise ValueError(f"RLE decode length mismatch: expected {total}, got {cursor}")

    return flat.reshape((int(size[0]), int(size[1])), order="F")


def decode_compressed_rle_string(encoded: str) -> list[int]:
    counts: list[int] = []
    cursor = 0
    length = len(encoded)
    while cursor < length:
        value = 0
        shift = 0
        more = True
        while more:
            char_code = ord(encoded[cursor]) - 48
            cursor += 1
            value |= (char_code & 0x1F) << (5 * shift)
            more = bool(char_code & 0x20)
            shift += 1
            if not more and (char_code & 0x10):
                value |= -1 << (5 * shift)
        if len(counts) > 2:
            value += counts[-2]
        counts.append(int(value))
    return counts


def rasterize_segmentation(mask: np.ndarray, segmentation) -> None:
    if not segmentation:
        return

    if isinstance(segmentation, list):
        polygons = segmentation
        for polygon in polygons:
            if not polygon or len(polygon) < 6:
                continue
            pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [pts], color=1)
        return

    if isinstance(segmentation, dict):
        decoded = decode_uncompressed_rle(segmentation)
        mask[decoded > 0] = 1
        return

    raise ValueError(f"Unsupported segmentation payload type: {type(segmentation)!r}")


def import_dataset(
    *,
    coco_root: Path,
    split: str,
    output_dir: Path,
    source_name: str,
    overwrite: bool,
) -> dict:
    split_dir = resolve_split_dir(coco_root, split)
    annotations_path = split_dir / "_annotations.coco.json"
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    images_out_dir = output_dir / "images"
    masks_out_dir = output_dir / "masks"
    images_out_dir.mkdir(parents=True, exist_ok=True)
    masks_out_dir.mkdir(parents=True, exist_ok=True)

    annotations_by_image_id: dict[int, list[dict]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        annotations_by_image_id[int(annotation["image_id"])].append(annotation)

    used_output_names: set[str] = set()
    category_counter: Counter[int] = Counter()
    images_without_annotations: list[str] = []
    images_with_multiple_annotations = 0
    rendered_annotations = 0

    for image_payload in payload.get("images", []):
        image_id = int(image_payload["id"])
        source_image_path = split_dir / image_payload["file_name"]
        if not source_image_path.exists():
            raise FileNotFoundError(f"Source image not found: {source_image_path}")

        output_image_name = resolve_output_image_name(image_payload, used_output_names)
        output_image_path = images_out_dir / output_image_name
        shutil.copy2(source_image_path, output_image_path)

        height = int(image_payload["height"])
        width = int(image_payload["width"])
        mask = np.zeros((height, width), dtype=np.uint8)

        image_annotations = annotations_by_image_id.get(image_id, [])
        if not image_annotations:
            images_without_annotations.append(output_image_name)
        if len(image_annotations) > 1:
            images_with_multiple_annotations += 1

        for annotation in image_annotations:
            category_id = int(annotation.get("category_id", 0))
            category_counter[category_id] += 1
            if category_id <= 0:
                continue
            rasterize_segmentation(mask, annotation.get("segmentation"))
            rendered_annotations += 1

        mask_output_path = masks_out_dir / f"{Path(output_image_name).stem}.png"
        cv2.imwrite(str(mask_output_path), mask * 255)

    manifest = {
        "source_name": source_name,
        "default_sample_weight": 1.0,
        "input": {
            "coco_root": str(coco_root),
            "split": split,
            "split_dir": str(split_dir),
            "annotations_path": str(annotations_path),
        },
        "output": {
            "dataset_root": str(output_dir),
            "images_dir": str(images_out_dir),
            "masks_dir": str(masks_out_dir),
        },
        "summary": {
            "images": len(payload.get("images", [])),
            "annotations": len(payload.get("annotations", [])),
            "rendered_positive_annotations": rendered_annotations,
            "images_without_annotations": len(images_without_annotations),
            "images_with_multiple_annotations": images_with_multiple_annotations,
            "category_counts": dict(sorted(category_counter.items())),
        },
        "notes": {
            "images_without_annotations_preview": images_without_annotations[:10],
            "binary_mask_rule": "All annotations with category_id > 0 are merged into a single foreground mask.",
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return manifest


def main() -> int:
    args = parse_args()
    coco_root = Path(args.coco_root).expanduser().resolve()
    source_name = args.source_name.strip() or slugify(coco_root.name)
    default_output_dir = PROJECT_ROOT / "data" / "derived" / "imported_coco" / slugify(source_name)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir

    manifest = import_dataset(
        coco_root=coco_root,
        split=args.split,
        output_dir=output_dir,
        source_name=source_name,
        overwrite=args.overwrite,
    )

    summary = manifest["summary"]
    print(f"[import_coco] source_name={manifest['source_name']}")
    print(f"[import_coco] dataset_root={manifest['output']['dataset_root']}")
    print(
        "[import_coco] "
        f"images={summary['images']} "
        f"annotations={summary['annotations']} "
        f"rendered_positive_annotations={summary['rendered_positive_annotations']} "
        f"images_without_annotations={summary['images_without_annotations']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
