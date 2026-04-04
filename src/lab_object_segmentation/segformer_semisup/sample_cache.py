from __future__ import annotations

import hashlib
import json
from functools import partial
from pathlib import Path

import albumentations as A
import cv2
import numpy as np


def read_image_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_binary_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return (mask > 127).astype(np.uint8)


def _resize_if_too_small_image(image: np.ndarray, *, min_image_size: int, **kwargs) -> np.ndarray:
    h, w = image.shape[:2]
    min_side = min(h, w)
    if min_side >= min_image_size:
        return image
    scale = min_image_size / max(1, min_side)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _resize_if_too_small_mask(mask: np.ndarray, *, min_image_size: int, **kwargs) -> np.ndarray:
    h, w = mask.shape[:2]
    min_side = min(h, w)
    if min_side >= min_image_size:
        return mask
    scale = min_image_size / max(1, min_side)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def build_base_transform(
    image_size: int,
    min_image_size: int,
    additional_targets: dict | None = None,
) -> A.Compose:
    return A.Compose(
        [
            A.Lambda(
                image=partial(_resize_if_too_small_image, min_image_size=min_image_size),
                mask=partial(_resize_if_too_small_mask, min_image_size=min_image_size),
            ),
            A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_REFLECT,
                fill=0,
                fill_mask=0,
            ),
        ],
        additional_targets=additional_targets or {},
    )


def _cache_key(namespace: str, image_path: Path, related_paths: tuple[Path, ...]) -> str:
    payload = "\n".join([namespace, str(image_path.resolve()), *[str(path.resolve()) for path in related_paths]])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _cache_path(cache_root: Path, namespace: str, image_path: Path, related_paths: tuple[Path, ...]) -> Path:
    return cache_root / namespace / f"{_cache_key(namespace, image_path, related_paths)}.npz"


def prepare_cached_sample(
    *,
    cache_root: Path,
    namespace: str,
    image_path: Path,
    image_size: int,
    min_image_size: int,
    mask_path: Path | None = None,
    reliability_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    related_paths = tuple(path for path in [mask_path, reliability_path] if path is not None)
    cache_path = _cache_path(cache_root, namespace, image_path, related_paths)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not overwrite:
        with np.load(cache_path, allow_pickle=False) as payload:
            return {key: payload[key] for key in payload.files}

    image = read_image_rgb(image_path)
    original_hw = np.array(image.shape[:2], dtype=np.int32)

    additional_targets: dict[str, str] = {}
    transform_kwargs: dict[str, np.ndarray] = {"image": image}
    if mask_path is not None:
        additional_targets["mask"] = "mask"
        transform_kwargs["mask"] = read_binary_mask(mask_path)
    if reliability_path is not None:
        additional_targets["reliability"] = "mask"
        transform_kwargs["reliability"] = read_binary_mask(reliability_path)

    transform = build_base_transform(
        image_size=image_size,
        min_image_size=min_image_size,
        additional_targets=additional_targets,
    )
    transformed = transform(**transform_kwargs)

    payload: dict[str, np.ndarray] = {
        "image": transformed["image"].astype(np.uint8),
        "original_hw": original_hw,
    }
    if "mask" in transformed:
        payload["mask"] = transformed["mask"].astype(np.uint8)
    if "reliability" in transformed:
        payload["reliability"] = transformed["reliability"].astype(np.uint8)

    np.savez_compressed(cache_path, **payload)
    return payload


def warm_segformer_cache(
    *,
    cache_root: Path,
    labeled_samples: list[tuple[Path, Path]],
    unlabeled_paths: list[Path],
    image_size: int,
    min_image_size: int,
    overwrite: bool = False,
) -> dict:
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cache_root": str(cache_root),
        "image_size": image_size,
        "min_image_size": min_image_size,
        "num_labeled": len(labeled_samples),
        "num_unlabeled": len(unlabeled_paths),
        "overwrite": bool(overwrite),
    }

    for image_path, mask_path in labeled_samples:
        prepare_cached_sample(
            cache_root=cache_root,
            namespace="labeled",
            image_path=image_path,
            mask_path=mask_path,
            image_size=image_size,
            min_image_size=min_image_size,
            overwrite=overwrite,
        )

    for image_path in unlabeled_paths:
        prepare_cached_sample(
            cache_root=cache_root,
            namespace="unlabeled",
            image_path=image_path,
            image_size=image_size,
            min_image_size=min_image_size,
            overwrite=overwrite,
        )

    manifest_path = cache_root / "cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
