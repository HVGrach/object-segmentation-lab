from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from lab_object_segmentation.common.paths import NOTEBOOKS_ROOT

def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip("\n") + "\n",
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip("\n") + "\n",
    }


cells = [
    md(
        """
        # SegFormer-B2 Semantic Segmentation with Pseudo-Labeling for macOS (Metal / MPS)

        Ноутбук адаптирован под локальный layout этой лабораторной работы и запуск на Apple Silicon.
        """
    ),
    md(
        """
        ## 0. Установка зависимостей и импорты
        """
    ),
    code(
        """
        import os
        import subprocess
        import sys

        if os.getenv("SEGFORMER_SKIP_PIP", "0").strip().lower() in {"1", "true", "yes", "y", "on"}:
            print("Skipping pip bootstrap because SEGFORMER_SKIP_PIP=1")
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "pip"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "torch", "torchvision"])
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "transformers>=4.40.0",
                    "albumentations>=1.4.0",
                    "segmentation-models-pytorch>=0.3.3",
                    "gdown",
                    "timm",
                    "opencv-python-headless",
                    "scikit-learn",
                    "pandas",
                ]
            )
        """
    ),
    code(
        """
        from __future__ import annotations

        import copy
        import json
        import math
        import os
        import random
        import shutil
        import sys
        import time
        import warnings
        from contextlib import nullcontext
        from pathlib import Path

        import albumentations as A
        import cv2
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        import torch.backends.cudnn as cudnn
        import torch.nn as nn
        import torch.nn.functional as F
        import torchvision
        from albumentations.pytorch import ToTensorV2
        from PIL import Image
        from sklearn.metrics import confusion_matrix, f1_score
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm
        from transformers import SegformerModel

        try:
            sys.path.insert(0, str(Path.cwd().resolve() / "src"))

            from lab_object_segmentation.segformer_semisup.sample_cache import (
                prepare_cached_sample,
                warm_segformer_cache,
            )
            from lab_object_segmentation.segformer_semisup.utils import (
                binary_dice_score,
                boundary_loss_with_reliability,
                build_pseudo_stats,
                infer_camera_group,
                should_accept_pseudo_sample,
                split_labeled_samples_grouped,
                tune_binary_threshold,
            )
        warnings.filterwarnings("ignore")
        plt.style.use("seaborn-v0_8-whitegrid")
        """
    ),
    md(
        """
        ## 1. Конфигурация (CONFIG dict — все параметры в одном месте)
        """
    ),
    code(
        """
        CONFIG = {
            # --- Данные ---
            "gdrive_zip_url": "PASTE_GDOWN_URL_HERE",
            "labeled_dir": "data/raw/dl-lab-3-product-segmentation/train/images",
            "unlabeled_dir": "data/raw/dl-lab-3-product-segmentation/unlabeled/images",
            "masks_dir": "data/raw/dl-lab-3-product-segmentation/train/masks",
            "extra_unlabeled_dirs": [
                "data/raw/dl-lab-1-image-classification/train/train",
                "data/raw/dl-lab-1-image-classification/test_images/test_images",
            ],
            "use_external_unlabeled": True,
            "gdrive_save_dir": "artifacts/runs/segformer_boundary_semisup_macos",

            # --- Размер изображений ---
            "image_size": 224,
            "min_image_size": 64,

            # --- Модель ---
            "encoder": "nvidia/mit-b2",
            "num_classes": 2,
            "boundary_channels": 64,
            "decoder_hidden_size": 256,

            # --- Обучение ---
            "batch_size": 8,
            "unlabeled_batch_size": 8,
            "num_workers": 0,
            "pin_memory": True,
            "labeled_repeat": 6,
            "dataset_cache_enabled": True,
            "warm_dataset_cache": True,
            "dataset_cache_overwrite": False,
            "max_labeled_samples": None,
            "max_unlabeled_samples": None,

            # --- Оптимизатор ---
            "lr": 6e-5,
            "encoder_lr_scale": 0.1,
            "weight_decay": 0.01,
            "phase1_epochs": 50,
            "phase3_epochs": 80,
            "warmup_epochs": 5,
            "min_lr": 1e-6,
            "gradient_clip_norm": 1.0,

            # --- Лоссы ---
            "boundary_loss_weight": 0.1,
            "unsupervised_loss_weight": 0.5,
            "boundary_class_weights": None,
            "ignore_index": 255,

            # --- EMA ---
            "ema_decay": 0.999,
            "consistency_confidence_threshold": 0.90,

            # --- Псевдолейблинг ---
            "pseudo_confidence_threshold": 0.85,
            "pseudo_uncertainty_threshold": 0.15,
            "tta_n_augments": 8,
            "pseudo_min_reliable_ratio": 0.90,
            "pseudo_min_mean_confidence": 0.95,
            "pseudo_min_fg_reliable_ratio": 0.80,
            "pseudo_min_object_ratio": 0.01,
            "pseudo_max_object_ratio": 0.80,
            "pseudo_external_min_reliable_ratio": 0.94,
            "pseudo_external_min_mean_confidence": 0.98,
            "pseudo_external_min_fg_reliable_ratio": 0.88,
            "pseudo_external_max_object_ratio": 0.70,
            "pseudo_max_external_samples": 750,
            "pseudo_max_external_to_primary_ratio": 1.5,
            "pseudo_border_width": 3,
            "boundary_border_width": 3,

            # --- Итерации ---
            "n_pseudo_iterations": 1,

            # --- Качество внешних unlabeled ---
            "quality_filter_enabled": True,
            "keep_primary_unlabeled_always": True,
            "quality_min_score": 0.42,
            "quality_min_brightness": 0.10,
            "quality_max_brightness": 0.92,
            "quality_min_contrast": 0.06,
            "quality_preview_samples": 8,

            # --- Сохранение / валидация ---
            "save_every_n_epochs": 5,
            "eval_every_n_epochs": 2,
            "val_split": 0.15,
            "use_group_split": True,
            "selection_metric": "dice_tuned",
            "dice_eval_threshold": 0.50,
            "dice_threshold_grid": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
            "seed": 42,
        }

        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


        def _env_flag(name: str, default: bool | None = None) -> bool | None:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


        def apply_env_overrides(config: dict) -> None:
            smoke_mode = _env_flag("SEGFORMER_SMOKE_MODE", False)
            if smoke_mode:
                config.update(
                    {
                        "gdrive_save_dir": "artifacts/runs/segformer_boundary_semisup_smoke",
                        "phase1_epochs": 1,
                        "phase3_epochs": 1,
                        "n_pseudo_iterations": 1,
                        "tta_n_augments": 2,
                        "use_external_unlabeled": False,
                        "warm_dataset_cache": True,
                        "max_labeled_samples": 24,
                        "max_unlabeled_samples": 4,
                    }
                )

            env_overrides = {
                "SEGFORMER_GDRIVE_SAVE_DIR": ("gdrive_save_dir", str),
                "SEGFORMER_IMAGE_SIZE": ("image_size", int),
                "SEGFORMER_BATCH_SIZE": ("batch_size", int),
                "SEGFORMER_UNLABELED_BATCH_SIZE": ("unlabeled_batch_size", int),
                "SEGFORMER_PHASE1_EPOCHS": ("phase1_epochs", int),
                "SEGFORMER_PHASE3_EPOCHS": ("phase3_epochs", int),
                "SEGFORMER_TTA_N": ("tta_n_augments", int),
                "SEGFORMER_N_PSEUDO_ITERATIONS": ("n_pseudo_iterations", int),
                "SEGFORMER_DATASET_CACHE_ENABLED": ("dataset_cache_enabled", lambda raw: raw.strip().lower() in {"1", "true", "yes", "y", "on"}),
                "SEGFORMER_WARM_CACHE": ("warm_dataset_cache", lambda raw: raw.strip().lower() in {"1", "true", "yes", "y", "on"}),
                "SEGFORMER_DATASET_CACHE_OVERWRITE": ("dataset_cache_overwrite", lambda raw: raw.strip().lower() in {"1", "true", "yes", "y", "on"}),
                "SEGFORMER_MAX_LABELED_SAMPLES": ("max_labeled_samples", int),
                "SEGFORMER_MAX_UNLABELED_SAMPLES": ("max_unlabeled_samples", int),
                "SEGFORMER_USE_EXTERNAL_UNLABELED": ("use_external_unlabeled", lambda raw: raw.strip().lower() in {"1", "true", "yes", "y", "on"}),
            }
            for env_name, (config_key, caster) in env_overrides.items():
                raw_value = os.getenv(env_name)
                if raw_value is None or raw_value == "":
                    continue
                config[config_key] = caster(raw_value)


        apply_env_overrides(CONFIG)
        """
    ),
    md(
        """
        ## 2. Подготовка окружения, путей и quality-aware unlabeled pool
        """
    ),
    code(
        """
        def guess_project_root() -> Path:
            cwd = Path.cwd().resolve()
            for candidate in [cwd, *cwd.parents]:
                if (candidate / "README.md").exists() and (candidate / "src").exists() and (candidate / "configs").exists():
                    return candidate
            return cwd


        PROJECT_ROOT = guess_project_root()
        ARTIFACT_ROOT = PROJECT_ROOT / CONFIG["gdrive_save_dir"]
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        DATASET_CACHE_ROOT = ARTIFACT_ROOT / "sample_cache" / f"size_{CONFIG['image_size']}"

        LABELED_DIR = PROJECT_ROOT / CONFIG["labeled_dir"]
        MASKS_DIR = PROJECT_ROOT / CONFIG["masks_dir"]
        PRIMARY_UNLABELED_DIR = PROJECT_ROOT / CONFIG["unlabeled_dir"]
        EXTRA_UNLABELED_DIRS = [PROJECT_ROOT / p for p in CONFIG["extra_unlabeled_dirs"]]


        def set_seed(seed: int) -> None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)


        def get_device() -> torch.device:
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")


        DEVICE = get_device()
        set_seed(CONFIG["seed"])
        cudnn.benchmark = DEVICE.type == "cuda"


        def maybe_download_from_gdrive(config: dict, project_root: Path) -> None:
            url = config["gdrive_zip_url"]
            if not url or "PASTE_GDOWN_URL_HERE" in url:
                print("gdown step skipped: local dataset layout is used.")
                return

            import gdown

            download_dir = project_root / "downloads"
            download_dir.mkdir(parents=True, exist_ok=True)
            zip_path = download_dir / "dataset.zip"

            if not zip_path.exists():
                print(f"Downloading dataset archive from Google Drive -> {zip_path}")
                gdown.download(url=url, output=str(zip_path), quiet=False, fuzzy=True)

            extract_dir = project_root / "downloaded_data"
            if not extract_dir.exists():
                print(f"Extracting archive -> {extract_dir}")
                shutil.unpack_archive(str(zip_path), str(extract_dir))
            else:
                print(f"Archive already extracted: {extract_dir}")


        def collect_image_paths(input_dir: Path) -> list[Path]:
            if not input_dir.exists():
                return []
            return sorted(
                [
                    p
                    for p in input_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                ]
            )


        def collect_labeled_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
            image_map = {
                p.stem: p
                for p in collect_image_paths(images_dir)
            }
            samples = []
            missing_images = []
            for mask_path in sorted(masks_dir.rglob("*")):
                if not mask_path.is_file() or mask_path.suffix.lower() not in MASK_EXTS:
                    continue
                image_path = image_map.get(mask_path.stem)
                if image_path is None:
                    missing_images.append(mask_path.name)
                    continue
                samples.append((image_path, mask_path))
            if missing_images:
                print(f"[WARN] masks without matching images: {len(missing_images)}")
            if not samples:
                raise RuntimeError("No labeled image/mask pairs were found.")
            return samples


        def limit_samples_deterministically(items: list, max_items: int | None, seed: int) -> list:
            if max_items is None or len(items) <= max_items:
                return items
            rng = random.Random(seed)
            indices = list(range(len(items)))
            rng.shuffle(indices)
            selected = sorted(indices[:max_items])
            return [items[idx] for idx in selected]


        def estimate_image_quality(image_path: Path) -> dict:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                return {
                    "path": str(image_path),
                    "source": "unknown",
                    "height": 0,
                    "width": 0,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "sharpness": 0.0,
                    "quality_score": 0.0,
                    "keep": False,
                }

            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            h, w = gray.shape

            brightness = float(rgb.mean() / 255.0)
            contrast = float(gray.std() / 255.0)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())

            sharpness_score = min(1.0, math.log1p(sharpness) / 6.0)
            contrast_score = min(1.0, contrast / 0.18)
            brightness_score = 1.0 - min(1.0, abs(brightness - 0.5) / 0.5)
            size_score = min(1.0, min(h, w) / max(1, CONFIG["min_image_size"]))

            quality_score = (
                0.40 * sharpness_score
                + 0.20 * contrast_score
                + 0.20 * brightness_score
                + 0.20 * size_score
            )
            keep = (
                quality_score >= CONFIG["quality_min_score"]
                and CONFIG["quality_min_brightness"] <= brightness <= CONFIG["quality_max_brightness"]
                and contrast >= CONFIG["quality_min_contrast"]
            )

            return {
                "path": str(image_path),
                "source": "unknown",
                "height": int(h),
                "width": int(w),
                "brightness": brightness,
                "contrast": contrast,
                "sharpness": sharpness,
                "quality_score": quality_score,
                "keep": bool(keep),
            }


        def build_unlabeled_pool(config: dict) -> tuple[list[Path], pd.DataFrame]:
            rows = []
            primary_paths = collect_image_paths(PRIMARY_UNLABELED_DIR)
            for path in primary_paths:
                payload = estimate_image_quality(path)
                payload["source"] = "lab_unlabeled"
                if config["keep_primary_unlabeled_always"]:
                    payload["keep"] = True
                rows.append(payload)

            if config["use_external_unlabeled"]:
                for extra_dir in EXTRA_UNLABELED_DIRS:
                    if len(extra_dir.parts) >= 3:
                        source_name = f"{extra_dir.parts[-3]}::{extra_dir.parts[-2]}"
                    else:
                        source_name = extra_dir.name
                    for path in collect_image_paths(extra_dir):
                        payload = estimate_image_quality(path)
                        payload["source"] = source_name
                        if not config["quality_filter_enabled"]:
                            payload["keep"] = True
                        rows.append(payload)

            quality_df = pd.DataFrame(rows)
            if quality_df.empty:
                raise RuntimeError("No unlabeled images were found.")

            keep_paths = [Path(p) for p in quality_df.loc[quality_df["keep"], "path"].tolist()]
            return keep_paths, quality_df


        maybe_download_from_gdrive(CONFIG, PROJECT_ROOT)

        LABELED_SAMPLES = collect_labeled_pairs(LABELED_DIR, MASKS_DIR)
        LABELED_SAMPLES = limit_samples_deterministically(
            LABELED_SAMPLES,
            CONFIG["max_labeled_samples"],
            CONFIG["seed"],
        )
        UNLABELED_PATHS, UNLABELED_QUALITY_DF = build_unlabeled_pool(CONFIG)
        UNLABELED_PATHS = limit_samples_deterministically(
            UNLABELED_PATHS,
            CONFIG["max_unlabeled_samples"],
            CONFIG["seed"],
        )
        UNLABELED_PATH_SET = {str(path) for path in UNLABELED_PATHS}
        UNLABELED_QUALITY_DF = UNLABELED_QUALITY_DF[UNLABELED_QUALITY_DF["path"].isin(UNLABELED_PATH_SET)].copy()
        UNLABELED_SOURCE_BY_PATH = {
            str(Path(row["path"])): row["source"]
            for _, row in UNLABELED_QUALITY_DF.iterrows()
        }

        print(f"Project root: {PROJECT_ROOT}")
        print(f"Device: {DEVICE}")
        print(f"Labeled pairs: {len(LABELED_SAMPLES)}")
        print(f"Primary unlabeled dir: {PRIMARY_UNLABELED_DIR}")
        print(f"Selected unlabeled images after quality filter: {len(UNLABELED_PATHS)}")
        print()
        print(
            UNLABELED_QUALITY_DF.groupby("source")[["quality_score", "brightness", "contrast", "keep"]]
            .agg({"quality_score": ["count", "mean"], "brightness": "mean", "contrast": "mean", "keep": "mean"})
            .round(3)
        )
        """
    ),
    code(
        """
        preview_df = (
            UNLABELED_QUALITY_DF.sort_values("quality_score", ascending=True)
            .groupby("source", group_keys=False)
            .head(CONFIG["quality_preview_samples"])
        )
        display(preview_df[["source", "path", "quality_score", "brightness", "contrast", "sharpness", "keep"]].head(20))
        """
    ),
    md(
        """
        ## 3. Аугментации
        """
    ),
    code(
        """
        def _resize_if_too_small_image(image: np.ndarray, **kwargs) -> np.ndarray:
            h, w = image.shape[:2]
            min_side = min(h, w)
            if min_side >= CONFIG["min_image_size"]:
                return image
            scale = CONFIG["min_image_size"] / max(1, min_side)
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


        def _resize_if_too_small_mask(mask: np.ndarray, **kwargs) -> np.ndarray:
            h, w = mask.shape[:2]
            min_side = min(h, w)
            if min_side >= CONFIG["min_image_size"]:
                return mask
            scale = CONFIG["min_image_size"] / max(1, min_side)
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            return cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


        def build_base_resize_ops() -> list:
            return [
                A.Lambda(image=_resize_if_too_small_image, mask=_resize_if_too_small_mask),
                A.LongestMaxSize(max_size=CONFIG["image_size"], interpolation=cv2.INTER_LINEAR),
                A.PadIfNeeded(
                    min_height=CONFIG["image_size"],
                    min_width=CONFIG["image_size"],
                    border_mode=cv2.BORDER_REFLECT,
                    fill=0,
                    fill_mask=0,
                ),
            ]


        def build_weak_transform(additional_targets: dict | None = None, include_base_resize: bool = True):
            return A.Compose(
                (build_base_resize_ops() if include_base_resize else [])
                + [
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
                additional_targets=additional_targets or {},
            )


        def build_val_transform(additional_targets: dict | None = None, include_base_resize: bool = True):
            return A.Compose(
                (build_base_resize_ops() if include_base_resize else [])
                + [
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
                additional_targets=additional_targets or {},
            )


        def build_strong_transform(additional_targets: dict | None = None, include_base_resize: bool = True):
            return A.Compose(
                (build_base_resize_ops() if include_base_resize else [])
                + [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.1,
                        p=0.8,
                    ),
                    A.GaussNoise(std_range=(0.04, 0.12), p=0.3),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
                    A.CoarseDropout(
                        num_holes_range=(1, 8),
                        hole_height_range=(8, 16),
                        hole_width_range=(8, 16),
                        fill=0,
                        fill_mask=255,
                        p=0.3,
                    ),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
                additional_targets=additional_targets or {},
            )


        def build_consistency_strong_transform(additional_targets: dict | None = None, include_base_resize: bool = True):
            return A.Compose(
                (build_base_resize_ops() if include_base_resize else [])
                + [
                    A.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.1,
                        p=0.8,
                    ),
                    A.GaussNoise(std_range=(0.04, 0.12), p=0.3),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                    A.CoarseDropout(
                        num_holes_range=(1, 8),
                        hole_height_range=(8, 16),
                        hole_width_range=(8, 16),
                        fill=0,
                        fill_mask=255,
                        p=0.3,
                    ),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
                additional_targets=additional_targets or {},
            )


        def denormalize_image_tensor(image_tensor: torch.Tensor) -> np.ndarray:
            mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(3, 1, 1)
            image = image_tensor * std + mean
            image = image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            return image


        def cache_enabled() -> bool:
            return bool(CONFIG["dataset_cache_enabled"])


        def get_cached_labeled_sample(image_path: Path, mask_path: Path) -> dict[str, np.ndarray]:
            return prepare_cached_sample(
                cache_root=DATASET_CACHE_ROOT,
                namespace="labeled",
                image_path=image_path,
                mask_path=mask_path,
                image_size=CONFIG["image_size"],
                min_image_size=CONFIG["min_image_size"],
                overwrite=CONFIG["dataset_cache_overwrite"],
            )


        def get_cached_unlabeled_image(image_path: Path) -> dict[str, np.ndarray]:
            return prepare_cached_sample(
                cache_root=DATASET_CACHE_ROOT,
                namespace="unlabeled",
                image_path=image_path,
                image_size=CONFIG["image_size"],
                min_image_size=CONFIG["min_image_size"],
                overwrite=CONFIG["dataset_cache_overwrite"],
            )


        def get_cached_pseudo_sample(image_path: Path, pseudo_mask_path: Path, reliability_path: Path) -> dict[str, np.ndarray]:
            return prepare_cached_sample(
                cache_root=DATASET_CACHE_ROOT,
                namespace="pseudo",
                image_path=image_path,
                mask_path=pseudo_mask_path,
                reliability_path=reliability_path,
                image_size=CONFIG["image_size"],
                min_image_size=CONFIG["min_image_size"],
                overwrite=CONFIG["dataset_cache_overwrite"],
            )


        def maybe_warm_dataset_cache() -> None:
            if not cache_enabled() or not CONFIG["warm_dataset_cache"]:
                return
            manifest = warm_segformer_cache(
                cache_root=DATASET_CACHE_ROOT,
                labeled_samples=LABELED_SAMPLES,
                unlabeled_paths=UNLABELED_PATHS,
                image_size=CONFIG["image_size"],
                min_image_size=CONFIG["min_image_size"],
                overwrite=CONFIG["dataset_cache_overwrite"],
            )
            print("Dataset cache ready:")
            print(json.dumps(manifest, indent=2, ensure_ascii=False))
        """
    ),
    md(
        """
        ## 4. Датасеты и DataLoader'ы
        """
    ),
    code(
        """
        def split_labeled_samples(samples: list[tuple[Path, Path]], val_split: float, seed: int):
            if CONFIG["use_group_split"]:
                return split_labeled_samples_grouped(samples, val_split=val_split, seed=seed)

            indices = list(range(len(samples)))
            rng = random.Random(seed)
            rng.shuffle(indices)
            val_size = max(1, int(round(len(samples) * val_split)))
            val_indices = set(indices[:val_size])
            train_samples = [samples[i] for i in range(len(samples)) if i not in val_indices]
            val_samples = [samples[i] for i in range(len(samples)) if i in val_indices]
            return train_samples, val_samples


        def generate_boundary_mask(binary_mask: np.ndarray, border_width: int = 3) -> np.ndarray:
            binary_mask = (binary_mask > 0).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_width, border_width))
            gradient = cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, kernel)
            return (gradient > 0).astype(np.uint8)


        class LabeledDataset(Dataset):
            def __init__(self, image_paths, mask_paths, transform=None, boundary_masks_dir: Path | None = None):
                self.image_paths = [Path(p) for p in image_paths]
                self.mask_paths = [Path(p) for p in mask_paths]
                self.transform = transform
                self.boundary_masks_dir = Path(boundary_masks_dir) if boundary_masks_dir is not None else None

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                image_path = self.image_paths[idx]
                mask_path = self.mask_paths[idx]

                if cache_enabled():
                    cached = get_cached_labeled_sample(image_path, mask_path)
                    image = cached["image"].copy()
                    mask = cached["mask"].copy()
                else:
                    image = np.array(Image.open(image_path).convert("RGB"))
                    mask = np.array(Image.open(mask_path).convert("L"))
                    mask = (mask > 127).astype(np.uint8)

                loaded_boundary = None
                if self.boundary_masks_dir is not None:
                    boundary_path = self.boundary_masks_dir / f"{mask_path.stem}.png"
                    if boundary_path.exists():
                        loaded_boundary = (np.array(Image.open(boundary_path).convert("L")) > 127).astype(np.uint8)

                if self.transform is not None:
                    transformed = self.transform(image=image, mask=mask)
                    image = transformed["image"]
                    mask = transformed["mask"]
                    if torch.is_tensor(mask):
                        mask = mask.long()
                    else:
                        mask = torch.from_numpy(mask).long()
                else:
                    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                    mask = torch.from_numpy(mask).long()

                mask_np = mask.cpu().numpy().astype(np.uint8)
                binary_mask_np = (mask_np == 1).astype(np.uint8)
                if loaded_boundary is not None and mask_np.max() <= 1:
                    boundary_np = loaded_boundary.astype(np.uint8)
                    if boundary_np.shape != binary_mask_np.shape:
                        boundary_np = generate_boundary_mask(binary_mask_np, CONFIG["boundary_border_width"])
                else:
                    boundary_np = generate_boundary_mask(binary_mask_np, CONFIG["boundary_border_width"])
                boundary_np[mask_np == CONFIG["ignore_index"]] = 0
                boundary = torch.from_numpy(boundary_np).float().unsqueeze(0)

                return {
                    "image": image.float(),
                    "mask": mask.long(),
                    "boundary": boundary.float(),
                    "path": str(image_path),
                }


        class UnlabeledDataset(Dataset):
            def __init__(self, image_paths, weak_transform, strong_transform):
                self.image_paths = [Path(p) for p in image_paths]
                self.weak_transform = weak_transform
                self.strong_transform = strong_transform

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                image_path = self.image_paths[idx]
                if cache_enabled():
                    cached = get_cached_unlabeled_image(image_path)
                    image = cached["image"].copy()
                else:
                    image = np.array(Image.open(image_path).convert("RGB"))
                weak = self.weak_transform(image=image)["image"]
                strong = self.strong_transform(image=image)["image"]
                return {
                    "weak": weak.float(),
                    "strong": strong.float(),
                    "path": str(image_path),
                }


        class PseudoLabeledDataset(Dataset):
            def __init__(self, image_paths, pseudo_mask_paths, reliability_masks_paths, transform=None):
                self.image_paths = [Path(p) for p in image_paths]
                self.pseudo_mask_paths = [Path(p) for p in pseudo_mask_paths]
                self.reliability_masks_paths = [Path(p) for p in reliability_masks_paths]
                self.transform = transform

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                image_path = self.image_paths[idx]
                pseudo_mask_path = self.pseudo_mask_paths[idx]
                reliability_path = self.reliability_masks_paths[idx]

                if cache_enabled():
                    cached = get_cached_pseudo_sample(image_path, pseudo_mask_path, reliability_path)
                    image = cached["image"].copy()
                    mask = cached["mask"].copy()
                    reliability = cached["reliability"].copy()
                else:
                    image = np.array(Image.open(image_path).convert("RGB"))
                    mask = (np.array(Image.open(pseudo_mask_path).convert("L")) > 127).astype(np.uint8)
                    reliability = (np.array(Image.open(reliability_path).convert("L")) > 127).astype(np.uint8)

                    image_h, image_w = image.shape[:2]
                    if mask.shape[:2] != (image_h, image_w):
                        mask = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
                    if reliability.shape[:2] != (image_h, image_w):
                        reliability = cv2.resize(reliability, (image_w, image_h), interpolation=cv2.INTER_NEAREST)

                if self.transform is not None:
                    transformed = self.transform(image=image, mask=mask, reliability=reliability)
                    image = transformed["image"]
                    mask = transformed["mask"]
                    reliability = transformed["reliability"]
                    if not torch.is_tensor(mask):
                        mask = torch.from_numpy(mask)
                    if not torch.is_tensor(reliability):
                        reliability = torch.from_numpy(reliability)
                    mask = mask.long()
                    reliability = reliability.long()
                else:
                    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                    mask = torch.from_numpy(mask).long()
                    reliability = torch.from_numpy(reliability).long()

                reliability = torch.where(reliability == CONFIG["ignore_index"], 0, reliability)
                reliability = (reliability > 0).long()

                mask_np = mask.cpu().numpy().astype(np.uint8)
                binary_mask_np = (mask_np == 1).astype(np.uint8)
                boundary_np = generate_boundary_mask(binary_mask_np, CONFIG["boundary_border_width"])
                boundary_np[mask_np == CONFIG["ignore_index"]] = 0
                boundary = torch.from_numpy(boundary_np).float().unsqueeze(0)

                return {
                    "image": image.float(),
                    "mask": mask.long(),
                    "boundary": boundary.float(),
                    "reliability": reliability.float(),
                    "path": str(image_path),
                }


        def resolve_num_workers(requested_num_workers: int) -> int:
            if sys.platform == "darwin" and ("ipykernel" in sys.modules or DEVICE.type == "mps"):
                if requested_num_workers != 0:
                    print(
                        f"[INFO] Overriding num_workers={requested_num_workers} -> 0 for macOS/Jupyter compatibility."
                    )
                return 0
            return max(0, int(requested_num_workers))


        def make_loader(dataset, batch_size, shuffle, config):
            pin_memory = bool(config["pin_memory"] and DEVICE.type == "cuda")
            effective_num_workers = resolve_num_workers(config["num_workers"])
            drop_last = bool(shuffle and len(dataset) >= batch_size)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=effective_num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                persistent_workers=effective_num_workers > 0,
            )


        TRAIN_SAMPLES, VAL_SAMPLES = split_labeled_samples(LABELED_SAMPLES, CONFIG["val_split"], CONFIG["seed"])
        print(f"Train labeled samples: {len(TRAIN_SAMPLES)}")
        print(f"Val labeled samples: {len(VAL_SAMPLES)}")
        print(f"Train groups: {len({infer_camera_group(path) for path, _ in TRAIN_SAMPLES})}")
        print(f"Val groups: {len({infer_camera_group(path) for path, _ in VAL_SAMPLES})}")
        """
    ),
    md(
        """
        ## 5. Генерация граничных масок (из бинарных)
        """
    ),
    code(
        """
        def generate_all_boundaries(mask_paths: list[Path], boundary_output_dir: Path, border_width: int):
            boundary_output_dir.mkdir(parents=True, exist_ok=True)
            unique_mask_paths = sorted({Path(mask_path) for mask_path in mask_paths})
            for mask_path in tqdm(unique_mask_paths, desc="Generating labeled boundaries"):
                mask = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)
                boundary = generate_boundary_mask(mask, border_width=border_width)
                out_path = boundary_output_dir / f"{mask_path.stem}.png"
                Image.fromarray((boundary * 255).astype(np.uint8)).save(out_path)


        LABELED_BOUNDARY_DIR = ARTIFACT_ROOT / "derived" / "labeled_boundaries"
        generate_all_boundaries(
            [mask_path for _, mask_path in LABELED_SAMPLES],
            LABELED_BOUNDARY_DIR,
            CONFIG["boundary_border_width"],
        )


        def compute_boundary_pos_weight(boundary_dir: Path) -> float:
            pos_pixels = 0
            total_pixels = 0
            for boundary_path in sorted(boundary_dir.glob("*.png")):
                boundary = (np.array(Image.open(boundary_path).convert("L")) > 127).astype(np.uint8)
                pos_pixels += int(boundary.sum())
                total_pixels += int(boundary.size)
            neg_pixels = max(1, total_pixels - pos_pixels)
            pos_pixels = max(1, pos_pixels)
            return float(neg_pixels / pos_pixels)


        CONFIG["boundary_class_weights"] = compute_boundary_pos_weight(LABELED_BOUNDARY_DIR)
        print(f"Boundary pos_weight: {CONFIG['boundary_class_weights']:.4f}")
        """
    ),
    md(
        """
        ## 6. Архитектура модели (SegFormer-B2 + boundary auxiliary head)
        """
    ),
    code(
        """
        class ConvMLP(nn.Module):
            def __init__(self, in_channels: int, out_channels: int):
                super().__init__()
                self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

            def forward(self, x):
                return self.proj(x)


        class SegFormerAllMLPDecoder(nn.Module):
            def __init__(self, hidden_sizes, decoder_hidden_size: int, num_classes: int):
                super().__init__()
                self.projections = nn.ModuleList(
                    [ConvMLP(hidden_size, decoder_hidden_size) for hidden_size in hidden_sizes]
                )
                self.fuse = nn.Sequential(
                    nn.Conv2d(decoder_hidden_size * len(hidden_sizes), decoder_hidden_size, kernel_size=1),
                    nn.BatchNorm2d(decoder_hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.1),
                )
                self.classifier = nn.Conv2d(decoder_hidden_size, num_classes, kernel_size=1)

            def forward(self, hidden_states):
                projected = []
                target_size = hidden_states[0].shape[-2:]
                for feature_map, projection in zip(hidden_states, self.projections):
                    feat = projection(feature_map)
                    feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
                    projected.append(feat)
                fused = self.fuse(torch.cat(projected, dim=1))
                return self.classifier(fused)


        class BoundaryHead(nn.Module):
            def __init__(self, in_channels: int, boundary_channels: int):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Conv2d(in_channels, boundary_channels, kernel_size=1),
                    nn.BatchNorm2d(boundary_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(boundary_channels, 1, kernel_size=1),
                )

            def forward(self, x):
                return self.layers(x)


        class SegFormerWithBoundary(nn.Module):
            def __init__(self, config: dict):
                super().__init__()
                self.config = config
                self.encoder = SegformerModel.from_pretrained(config["encoder"])
                hidden_sizes = list(self.encoder.config.hidden_sizes)
                self.decode_head = SegFormerAllMLPDecoder(
                    hidden_sizes=hidden_sizes,
                    decoder_hidden_size=config["decoder_hidden_size"],
                    num_classes=config["num_classes"],
                )
                self.boundary_head = BoundaryHead(
                    in_channels=hidden_sizes[0],
                    boundary_channels=config["boundary_channels"],
                )

            def forward(self, pixel_values, return_boundary: bool = True):
                outputs = self.encoder(pixel_values, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                seg_logits = self.decode_head(hidden_states)
                seg_logits = F.interpolate(
                    seg_logits,
                    size=pixel_values.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                if return_boundary:
                    boundary_logits = self.boundary_head(hidden_states[0])
                    boundary_logits = F.interpolate(
                        boundary_logits,
                        size=pixel_values.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    return seg_logits, boundary_logits
                return seg_logits


        def freeze_encoder_stages(model: SegFormerWithBoundary, n_stages: int = 2, freeze: bool = True):
            freeze_tokens = []
            for stage_idx in range(n_stages):
                freeze_tokens.extend(
                    [
                        f"encoder.patch_embeddings.{stage_idx}",
                        f"encoder.block.{stage_idx}",
                        f"encoder.layer_norm.{stage_idx}",
                    ]
                )
            for name, param in model.encoder.named_parameters():
                if any(token in name for token in freeze_tokens):
                    param.requires_grad = not freeze


        def count_parameters(model: nn.Module) -> tuple[int, int]:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total, trainable


        def build_optimizer(model: SegFormerWithBoundary, config: dict):
            encoder_params = []
            decoder_params = list(model.decode_head.parameters())
            boundary_params = list(model.boundary_head.parameters())
            decoder_param_ids = {id(p) for p in decoder_params}
            boundary_param_ids = {id(p) for p in boundary_params}

            for p in model.encoder.parameters():
                if id(p) not in decoder_param_ids and id(p) not in boundary_param_ids and p.requires_grad:
                    encoder_params.append(p)

            param_groups = [
                {"params": encoder_params, "lr": config["lr"] * config["encoder_lr_scale"]},
                {"params": [p for p in decoder_params if p.requires_grad], "lr": config["lr"]},
                {"params": [p for p in boundary_params if p.requires_grad], "lr": config["lr"]},
            ]
            return AdamW(param_groups, weight_decay=config["weight_decay"])
        """
    ),
    md(
        """
        ## 7. Вспомогательные функции (EMA, метрики, лосс-функции)
        """
    ),
    code(
        """
        def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0, ignore_index: int = 255):
            num_classes = pred.shape[1]
            probs = F.softmax(pred, dim=1)
            valid_mask = (target != ignore_index).unsqueeze(1)
            target_clamped = target.clone()
            target_clamped[target_clamped == ignore_index] = 0
            target_one_hot = F.one_hot(target_clamped, num_classes=num_classes).permute(0, 3, 1, 2).float()
            probs = probs * valid_mask
            target_one_hot = target_one_hot * valid_mask
            intersection = (probs * target_one_hot).sum(dim=(2, 3))
            cardinality = probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
            dice = (2.0 * intersection + smooth) / (cardinality + smooth)
            return 1.0 - dice.mean()


        def seg_loss(pred_logits: torch.Tensor, target_mask: torch.Tensor, ignore_index: int = 255):
            ce = F.cross_entropy(pred_logits, target_mask, ignore_index=ignore_index)
            dsc = dice_loss(pred_logits, target_mask, ignore_index=ignore_index)
            return 0.5 * ce + 0.5 * dsc


        def boundary_loss(pred_boundary: torch.Tensor, target_boundary: torch.Tensor):
            target_boundary = target_boundary.float()
            pos_pixels = target_boundary.sum()
            neg_pixels = target_boundary.numel() - pos_pixels
            pos_weight = neg_pixels / (pos_pixels + 1e-6)
            if CONFIG["boundary_class_weights"] is not None:
                pos_weight = torch.tensor(CONFIG["boundary_class_weights"], device=pred_boundary.device)
            else:
                pos_weight = torch.tensor(float(pos_weight), device=pred_boundary.device)
            return F.binary_cross_entropy_with_logits(
                pred_boundary,
                target_boundary,
                pos_weight=pos_weight,
            )


        def pseudo_boundary_loss(pred_boundary: torch.Tensor, target_boundary: torch.Tensor, reliability_mask: torch.Tensor):
            pos_weight = CONFIG["boundary_class_weights"]
            if pos_weight is None:
                target_boundary_float = target_boundary.float()
                pos_pixels = target_boundary_float.sum()
                neg_pixels = target_boundary_float.numel() - pos_pixels
                pos_weight = float(neg_pixels / (pos_pixels + 1e-6))
            return boundary_loss_with_reliability(
                pred_boundary=pred_boundary,
                target_boundary=target_boundary,
                reliability_mask=reliability_mask,
                pos_weight=pos_weight,
            )


        def total_supervised_loss(seg_logits, boundary_pred, seg_target, boundary_target):
            return seg_loss(seg_logits, seg_target, ignore_index=CONFIG["ignore_index"]) + (
                CONFIG["boundary_loss_weight"] * boundary_loss(boundary_pred, boundary_target)
            )


        def seg_loss_with_reliability(pred_logits, target_mask, reliability_mask, ignore_index: int = 255):
            valid = (target_mask != ignore_index).float()
            reliability = reliability_mask.float() * valid

            ce_map = F.cross_entropy(pred_logits, target_mask.clamp_max(1), reduction="none")
            ce = (ce_map * reliability).sum() / reliability.sum().clamp_min(1.0)

            probs = F.softmax(pred_logits, dim=1)
            target_clamped = target_mask.clone().clamp_max(1)
            target_one_hot = F.one_hot(target_clamped, num_classes=pred_logits.shape[1]).permute(0, 3, 1, 2).float()
            reliability_expanded = reliability.unsqueeze(1)
            probs = probs * reliability_expanded
            target_one_hot = target_one_hot * reliability_expanded
            intersection = (probs * target_one_hot).sum(dim=(2, 3))
            cardinality = probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
            dice = 1.0 - ((2.0 * intersection + 1.0) / (cardinality + 1.0)).mean()

            return 0.5 * ce + 0.5 * dice


        def pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
            valid = target != CONFIG["ignore_index"]
            if valid.sum() == 0:
                return 0.0
            return float((pred[valid] == target[valid]).mean())


        def mean_iou(pred: np.ndarray, target: np.ndarray, num_classes: int = 2) -> float:
            valid = target != CONFIG["ignore_index"]
            pred = pred[valid]
            target = target[valid]
            if pred.size == 0:
                return 0.0
            ious = []
            for cls_idx in range(num_classes):
                pred_mask = pred == cls_idx
                target_mask = target == cls_idx
                intersection = np.logical_and(pred_mask, target_mask).sum()
                union = np.logical_or(pred_mask, target_mask).sum()
                if union == 0:
                    ious.append(1.0)
                else:
                    ious.append(intersection / union)
            return float(np.mean(ious))


        def boundary_f1(pred_boundary: torch.Tensor, target_boundary: torch.Tensor, threshold: float = 0.5) -> float:
            pred_binary = (torch.sigmoid(pred_boundary).detach().cpu().numpy() > threshold).astype(np.uint8)
            target_binary = (target_boundary.detach().cpu().numpy() > 0.5).astype(np.uint8)
            return float(f1_score(target_binary.reshape(-1), pred_binary.reshape(-1), zero_division=0))


        def checkpoint_selection_score(metrics: dict) -> float:
            metric_name = str(CONFIG.get("selection_metric", "dice_tuned"))
            if metric_name == "dice_tuned":
                return float(metrics.get("dice_tuned", metrics.get("dice", metrics.get("mIoU", -1.0))))
            if metric_name == "dice":
                return float(metrics.get("dice", metrics.get("mIoU", -1.0)))
            return float(metrics.get("mIoU", -1.0))


        def update_ema(teacher_model, student_model, decay: float = 0.999):
            for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
                t_param.data = decay * t_param.data + (1.0 - decay) * s_param.data
            for t_buffer, s_buffer in zip(teacher_model.buffers(), student_model.buffers()):
                t_buffer.data.copy_(s_buffer.data)


        def get_autocast_context(device: torch.device):
            if device.type == "cuda":
                return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            return nullcontext()


        def get_grad_scaler(device: torch.device):
            return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")


        def build_scheduler(optimizer, total_epochs: int, warmup_epochs: int, min_lr: float):
            if total_epochs <= 1:
                return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

            if warmup_epochs > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lambda epoch: (
                        0.01 + 0.99 * ((epoch + 1) / max(1, warmup_epochs))
                        if epoch < warmup_epochs
                        else 1.0
                    ),
                )
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, total_epochs - warmup_epochs),
                    eta_min=min_lr,
                )
                return torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs],
                )

            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_epochs),
                eta_min=min_lr,
            )


        def save_json(path: Path, payload: dict | list):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)


        def save_checkpoint(model, optimizer, scheduler, epoch, phase_name, metrics, is_best, save_dir, extra: dict | None = None):
            save_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "epoch": epoch,
                "phase": phase_name,
                "metrics": metrics,
                "config": CONFIG,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            }
            if extra:
                payload.update(extra)

            latest_path = save_dir / "latest.pt"
            torch.save(payload, latest_path)
            if is_best:
                torch.save(payload, save_dir / "best.pt")
            if (epoch + 1) % CONFIG["save_every_n_epochs"] == 0:
                torch.save(payload, save_dir / f"epoch_{epoch + 1:03d}.pt")


        def load_checkpoint_to_model(checkpoint_path: Path, model: nn.Module, map_location: str | torch.device = "cpu", state_key: str = "model_state_dict"):
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            model.load_state_dict(checkpoint[state_key], strict=True)
            return checkpoint


        @torch.no_grad()
        def evaluate(model, dataloader, device):
            model.eval()
            losses = []
            pixel_accs = []
            mious = []
            dices = []
            boundary_scores = []
            positive_probabilities = []
            targets = []

            for batch in tqdm(dataloader, desc="Validation", leave=False):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                boundaries = batch["boundary"].to(device)

                seg_logits, boundary_pred = model(images)
                loss = total_supervised_loss(seg_logits, boundary_pred, masks, boundaries)

                probs_pos = F.softmax(seg_logits, dim=1)[:, 1].detach().cpu().numpy()
                pred_mask = seg_logits.argmax(dim=1).detach().cpu().numpy()
                target_mask = masks.detach().cpu().numpy()

                losses.append(float(loss.item()))
                pixel_accs.append(pixel_accuracy(pred_mask, target_mask))
                mious.append(mean_iou(pred_mask, target_mask, num_classes=CONFIG["num_classes"]))
                for sample_prob, sample_target in zip(probs_pos, target_mask):
                    positive_probabilities.append(sample_prob.astype(np.float32))
                    targets.append(sample_target.astype(np.uint8))
                    dices.append(
                        binary_dice_score(
                            pred_mask=(sample_prob >= float(CONFIG["dice_eval_threshold"])).astype(np.uint8),
                            target_mask=sample_target,
                            ignore_index=CONFIG["ignore_index"],
                        )
                    )
                boundary_scores.append(boundary_f1(boundary_pred, boundaries))

            threshold_metrics = tune_binary_threshold(
                probabilities=positive_probabilities,
                targets=targets,
                threshold_grid=[float(x) for x in CONFIG["dice_threshold_grid"]],
                ignore_index=CONFIG["ignore_index"],
            )
            return {
                "mIoU": float(np.mean(mious)) if mious else 0.0,
                "dice": float(np.mean(dices)) if dices else 0.0,
                "dice_eval_threshold": float(CONFIG["dice_eval_threshold"]),
                "dice_tuned": float(threshold_metrics["best_dice"]),
                "best_threshold": float(threshold_metrics["best_threshold"]),
                "pixel_acc": float(np.mean(pixel_accs)) if pixel_accs else 0.0,
                "boundary_f1": float(np.mean(boundary_scores)) if boundary_scores else 0.0,
                "loss": float(np.mean(losses)) if losses else 0.0,
            }


        def infinite_loader(loader):
            while True:
                for batch in loader:
                    yield batch


        def format_minutes(minutes: float) -> str:
            hours = int(minutes // 60)
            mins = int(minutes % 60)
            if hours > 0:
                return f"{hours}h {mins}m"
            return f"{mins}m"


        def estimate_phase_minutes(num_samples: int, batch_size: int, epochs: int, device_type: str, complexity_factor: float = 1.0) -> float:
            steps = math.ceil(num_samples / max(1, batch_size)) * max(1, epochs)
            sec_per_step = {"cuda": 0.30, "mps": 0.55, "cpu": 1.80}.get(device_type, 1.80)
            return steps * sec_per_step * complexity_factor / 60.0


        def phase_banner(phase_name: str, dataset_sizes: dict, model: nn.Module, batch_size: int, epochs: int, complexity_factor: float = 1.0):
            total_params, trainable_params = count_parameters(model)
            est_minutes = estimate_phase_minutes(
                num_samples=max(dataset_sizes.values()) if dataset_sizes else 0,
                batch_size=batch_size,
                epochs=epochs,
                device_type=DEVICE.type,
                complexity_factor=complexity_factor,
            )
            print("=" * 80)
            print(f"{phase_name}")
            print(f"Device: {DEVICE}")
            print(f"Dataset sizes: {dataset_sizes}")
            print(f"Parameters: total={total_params:,} | trainable={trainable_params:,}")
            print(f"Estimated time on current device: {format_minutes(est_minutes)}")
            print("=" * 80)
        """
    ),
    md(
        """
        ## 8. Фаза 1 — Supervised training
        """
    ),
    code(
        """
        maybe_warm_dataset_cache()

        train_transform_labeled = build_strong_transform(include_base_resize=not cache_enabled())
        val_transform_labeled = build_val_transform(include_base_resize=not cache_enabled())

        train_dataset = LabeledDataset(
            image_paths=[x[0] for x in TRAIN_SAMPLES],
            mask_paths=[x[1] for x in TRAIN_SAMPLES],
            transform=train_transform_labeled,
            boundary_masks_dir=LABELED_BOUNDARY_DIR,
        )
        val_dataset = LabeledDataset(
            image_paths=[x[0] for x in VAL_SAMPLES],
            mask_paths=[x[1] for x in VAL_SAMPLES],
            transform=val_transform_labeled,
            boundary_masks_dir=LABELED_BOUNDARY_DIR,
        )

        train_loader = make_loader(train_dataset, CONFIG["batch_size"], shuffle=True, config=CONFIG)
        val_loader = make_loader(val_dataset, CONFIG["batch_size"], shuffle=False, config=CONFIG)


        def train_one_epoch_supervised(model, dataloader, optimizer, scaler, device):
            model.train()
            running_loss = 0.0

            for batch in tqdm(dataloader, desc="Phase 1 Train", leave=False):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                boundaries = batch["boundary"].to(device)

                optimizer.zero_grad(set_to_none=True)

                with get_autocast_context(device):
                    seg_logits, boundary_pred = model(images)
                    loss = total_supervised_loss(seg_logits, boundary_pred, masks, boundaries)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip_norm"])
                scaler.step(optimizer)
                scaler.update()

                running_loss += float(loss.item())

            return running_loss / max(1, len(dataloader))


        phase1_model = SegFormerWithBoundary(CONFIG).to(DEVICE)
        freeze_encoder_stages(phase1_model, n_stages=2, freeze=True)
        phase1_optimizer = build_optimizer(phase1_model, CONFIG)
        phase1_scheduler = build_scheduler(
            phase1_optimizer,
            total_epochs=CONFIG["phase1_epochs"],
            warmup_epochs=CONFIG["warmup_epochs"],
            min_lr=CONFIG["min_lr"],
        )
        phase1_scaler = get_grad_scaler(DEVICE)

        phase_banner(
            "Phase 1 - Supervised training",
            dataset_sizes={"train_labeled": len(train_dataset), "val": len(val_dataset)},
            model=phase1_model,
            batch_size=CONFIG["batch_size"],
            epochs=CONFIG["phase1_epochs"],
            complexity_factor=1.0,
        )

        phase1_dir = ARTIFACT_ROOT / "phase1"
        phase1_history = []
        all_metrics = []
        best_phase1_score = -1.0

        for epoch in range(CONFIG["phase1_epochs"]):
            train_loss = train_one_epoch_supervised(
                phase1_model,
                train_loader,
                phase1_optimizer,
                phase1_scaler,
                DEVICE,
            )
            phase1_scheduler.step()

            metrics = {"loss": None, "mIoU": None, "dice": None, "dice_tuned": None, "best_threshold": None, "pixel_acc": None, "boundary_f1": None}
            if (epoch + 1) % CONFIG["eval_every_n_epochs"] == 0 or epoch == 0 or epoch == CONFIG["phase1_epochs"] - 1:
                metrics = evaluate(phase1_model, val_loader, DEVICE)
                current_score = checkpoint_selection_score(metrics)
                is_best = current_score > best_phase1_score
                if is_best:
                    best_phase1_score = current_score
                save_checkpoint(
                    model=phase1_model,
                    optimizer=phase1_optimizer,
                    scheduler=phase1_scheduler,
                    epoch=epoch,
                    phase_name="phase1",
                    metrics=metrics,
                    is_best=is_best,
                    save_dir=phase1_dir,
                )
            else:
                is_best = False

            current_lr = max(group["lr"] for group in phase1_optimizer.param_groups)
            row = {
                "phase": "phase1",
                "iteration": 0,
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": None if metrics["loss"] is None else float(metrics["loss"]),
                "val_mIoU": None if metrics["mIoU"] is None else float(metrics["mIoU"]),
                "val_dice": None if metrics["dice"] is None else float(metrics["dice"]),
                "val_dice_tuned": None if metrics["dice_tuned"] is None else float(metrics["dice_tuned"]),
                "val_best_threshold": None if metrics["best_threshold"] is None else float(metrics["best_threshold"]),
                "val_pixel_acc": None if metrics["pixel_acc"] is None else float(metrics["pixel_acc"]),
                "val_boundary_f1": None if metrics["boundary_f1"] is None else float(metrics["boundary_f1"]),
                "learning_rate": float(current_lr),
            }
            phase1_history.append(row)
            if metrics["dice_tuned"] is not None:
                all_metrics.append(row)
            print(
                f"[Phase 1][Epoch {epoch + 1:03d}] "
                f"train_loss={train_loss:.4f} "
                f"val_dice_tuned={0.0 if metrics['dice_tuned'] is None else metrics['dice_tuned']:.4f} "
                f"val_dice={0.0 if metrics['dice'] is None else metrics['dice']:.4f} "
                f"val_mIoU={0.0 if metrics['mIoU'] is None else metrics['mIoU']:.4f} "
                f"val_pixel_acc={0.0 if metrics['pixel_acc'] is None else metrics['pixel_acc']:.4f} "
                f"val_boundary_f1={0.0 if metrics['boundary_f1'] is None else metrics['boundary_f1']:.4f} "
                f"thr={0.0 if metrics['best_threshold'] is None else metrics['best_threshold']:.2f} "
                f"lr={current_lr:.6e}"
            )

        save_json(ARTIFACT_ROOT / "metrics" / "phase1_history.json", phase1_history)
        PHASE1_BEST_CKPT = phase1_dir / "best.pt"
        print(f"Best Phase 1 checkpoint: {PHASE1_BEST_CKPT}")
        """
    ),
    code(
        """
        phase1_df = pd.DataFrame(phase1_history)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(phase1_df["epoch"], phase1_df["train_loss"], label="train_loss")
        axes[0].plot(phase1_df["epoch"], phase1_df["val_loss"], label="val_loss")
        axes[0].set_title("Phase 1 Loss Curve")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(phase1_df["epoch"], phase1_df["val_dice_tuned"], label="val_dice_tuned")
        axes[1].plot(phase1_df["epoch"], phase1_df["val_dice"], label="val_dice")
        axes[1].plot(phase1_df["epoch"], phase1_df["val_mIoU"], label="val_mIoU")
        axes[1].plot(phase1_df["epoch"], phase1_df["val_pixel_acc"], label="val_pixel_acc")
        axes[1].plot(phase1_df["epoch"], phase1_df["val_boundary_f1"], label="val_boundary_f1")
        axes[1].set_title("Phase 1 Validation Metrics")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 9. Фаза 2 — Генерация и фильтрация псевдолейблов
        """
    ),
    code(
        """
        inference_transform = build_val_transform(include_base_resize=not cache_enabled())


        def tensor_hflip(x): return torch.flip(x, dims=[-1])
        def tensor_vflip(x): return torch.flip(x, dims=[-2])
        def tensor_rot90(x): return torch.rot90(x, k=1, dims=[-2, -1])
        def tensor_rot270(x): return torch.rot90(x, k=3, dims=[-2, -1])


        TTA_OPS = [
            ("identity", lambda x: x, lambda y: y),
            ("hflip", tensor_hflip, tensor_hflip),
            ("vflip", tensor_vflip, tensor_vflip),
            ("hvflip", lambda x: tensor_hflip(tensor_vflip(x)), lambda y: tensor_vflip(tensor_hflip(y))),
            ("rot90", tensor_rot90, tensor_rot270),
            ("rot270", tensor_rot270, tensor_rot90),
            ("hflip_rot90", lambda x: tensor_hflip(tensor_rot90(x)), lambda y: tensor_rot270(tensor_hflip(y))),
            ("vflip_rot90", lambda x: tensor_vflip(tensor_rot90(x)), lambda y: tensor_rot270(tensor_vflip(y))),
        ]


        @torch.no_grad()
        def predict_with_tta(model, image_rgb: np.ndarray, device: torch.device, tta_n: int):
            base_tensor = inference_transform(image=image_rgb)["image"].unsqueeze(0).to(device)
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
            confidence, pseudo_mask = mean_pred.max(dim=0)
            return mean_pred, std_pred, uncertainty, confidence, pseudo_mask


        def resize_map_to_image(map_array: np.ndarray, image_shape_hw: tuple[int, int], interpolation: int) -> np.ndarray:
            target_h, target_w = image_shape_hw
            if map_array.shape[:2] == (target_h, target_w):
                return map_array
            return cv2.resize(map_array, (target_w, target_h), interpolation=interpolation)


        def save_mask(mask: np.ndarray, out_path: Path):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8)).save(out_path)


        def generate_pseudo_labels(model, unlabeled_paths, config, device, iteration=0):
            model.eval()
            pseudo_dir = ARTIFACT_ROOT / "pseudo_masks" / f"iteration_{iteration}"
            reliability_dir = ARTIFACT_ROOT / "reliability_masks" / f"iteration_{iteration}"
            boundary_dir = ARTIFACT_ROOT / "boundary_pseudo_masks" / f"iteration_{iteration}"
            stats_rows = []
            vis_examples = []
            for image_path in tqdm(unlabeled_paths, desc=f"Pseudo labeling iteration {iteration}"):
                if cache_enabled():
                    cached = get_cached_unlabeled_image(image_path)
                    image_rgb = cached["image"].copy()
                    orig_h, orig_w = [int(x) for x in cached["original_hw"]]
                else:
                    image_rgb = np.array(Image.open(image_path).convert("RGB"))
                    orig_h, orig_w = image_rgb.shape[:2]
                mean_pred, std_pred, uncertainty, confidence, pseudo_mask = predict_with_tta(
                    model=model,
                    image_rgb=image_rgb,
                    device=device,
                    tta_n=config["tta_n_augments"],
                )

                pseudo_mask_np = pseudo_mask.numpy().astype(np.uint8)
                confidence_np = confidence.numpy().astype(np.float32)
                uncertainty_np = uncertainty.numpy().astype(np.float32)

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

                source_name = UNLABELED_SOURCE_BY_PATH.get(str(image_path), "unknown")
                is_primary_source = source_name == "lab_unlabeled"
                stats = build_pseudo_stats(
                    pseudo_mask=pseudo_mask_np,
                    confidence=confidence_np,
                    uncertainty=uncertainty_np,
                    confidence_threshold=config["pseudo_confidence_threshold"],
                    uncertainty_threshold=config["pseudo_uncertainty_threshold"],
                )
                reliable = stats["reliable"]
                base_accept = should_accept_pseudo_sample(
                    stats,
                    min_reliable_ratio=(
                        config["pseudo_min_reliable_ratio"]
                        if is_primary_source
                        else max(config["pseudo_min_reliable_ratio"], config["pseudo_external_min_reliable_ratio"])
                    ),
                    min_mean_confidence=(
                        config["pseudo_min_mean_confidence"]
                        if is_primary_source
                        else max(config["pseudo_min_mean_confidence"], config["pseudo_external_min_mean_confidence"])
                    ),
                    min_fg_reliable_ratio=(
                        config["pseudo_min_fg_reliable_ratio"]
                        if is_primary_source
                        else max(config["pseudo_min_fg_reliable_ratio"], config["pseudo_external_min_fg_reliable_ratio"])
                    ),
                    min_object_ratio=config["pseudo_min_object_ratio"],
                    max_object_ratio=(
                        config["pseudo_max_object_ratio"]
                        if is_primary_source
                        else min(config["pseudo_max_object_ratio"], config["pseudo_external_max_object_ratio"])
                    ),
                )

                stats_rows.append(
                    {
                        "path": str(image_path),
                        "source": source_name,
                        "accepted": False,
                        "base_accept": bool(base_accept),
                        "reliable_ratio": stats["reliable_ratio"],
                        "object_ratio": stats["object_ratio"],
                        "avg_confidence": stats["avg_confidence"],
                        "fg_reliable_ratio": stats["fg_reliable_ratio"],
                        "bg_reliable_ratio": stats["bg_reliable_ratio"],
                        "selection_score": stats["selection_score"],
                    }
                )

                if not base_accept:
                    continue

                pseudo_mask_path = pseudo_dir / f"{image_path.stem}.png"
                reliability_path = reliability_dir / f"{image_path.stem}.png"
                boundary_path = boundary_dir / f"{image_path.stem}.png"

                boundary_np = generate_boundary_mask(pseudo_mask_np, border_width=config["pseudo_border_width"])
                save_mask(pseudo_mask_np * 255, pseudo_mask_path)
                save_mask(reliable * 255, reliability_path)
                save_mask(boundary_np * 255, boundary_path)

                if len(vis_examples) < 8:
                    vis_examples.append(
                        {
                            "image": image_rgb,
                            "pseudo_mask": pseudo_mask_np,
                            "confidence": confidence_np,
                            "reliability": reliable,
                            "path": str(image_path),
                        }
                    )

            stats_df = pd.DataFrame(stats_rows)
            candidate_df = stats_df[stats_df["base_accept"]].copy()
            if not candidate_df.empty:
                primary_mask = candidate_df["source"] == "lab_unlabeled"
                primary_df = candidate_df[primary_mask].copy()
                external_df = candidate_df[~primary_mask].copy()
                external_limit = config["pseudo_max_external_samples"]
                external_ratio = config.get("pseudo_max_external_to_primary_ratio")
                if external_ratio is not None and external_ratio >= 0:
                    ratio_limit = int(round(len(primary_df) * float(external_ratio)))
                    external_limit = ratio_limit if external_limit is None else min(int(external_limit), ratio_limit)
                if external_limit is not None and external_limit >= 0 and len(external_df) > external_limit:
                    external_df = external_df.sort_values(
                        by=["selection_score", "fg_reliable_ratio", "avg_confidence"],
                        ascending=False,
                    ).head(external_limit)
                accepted_df = pd.concat([primary_df, external_df], ignore_index=True)
            else:
                accepted_df = candidate_df

            accepted_paths = set(accepted_df["path"].tolist()) if not accepted_df.empty else set()
            stats_df["accepted"] = stats_df["path"].isin(accepted_paths)
            rejected_after_cap_df = stats_df[stats_df["base_accept"] & (~stats_df["accepted"])]
            for _, rejected_row in rejected_after_cap_df.iterrows():
                stem = Path(rejected_row["path"]).stem
                for file_path in [
                    pseudo_dir / f"{stem}.png",
                    reliability_dir / f"{stem}.png",
                    boundary_dir / f"{stem}.png",
                ]:
                    if file_path.exists():
                        file_path.unlink()

            summary = {
                "iteration": iteration,
                "num_total": int(len(stats_df)),
                "num_candidates_after_thresholds": int(candidate_df.shape[0]),
                "num_accepted": int(accepted_df.shape[0]),
                "accepted_percent": float(accepted_df.shape[0] / max(1, len(stats_df))),
                "avg_confidence_accepted": float(accepted_df["avg_confidence"].mean()) if not accepted_df.empty else 0.0,
                "avg_object_ratio_accepted": float(accepted_df["object_ratio"].mean()) if not accepted_df.empty else 0.0,
                "avg_fg_reliable_ratio_accepted": float(accepted_df["fg_reliable_ratio"].mean()) if not accepted_df.empty else 0.0,
                "accepted_by_source": accepted_df["source"].value_counts().to_dict() if not accepted_df.empty else {},
            }

            save_json(
                ARTIFACT_ROOT / "metrics" / f"pseudo_iteration_{iteration}.json",
                {"summary": summary, "rows": stats_df.to_dict(orient="records")},
            )
            return summary, stats_df, vis_examples


        phase1_inference_model = SegFormerWithBoundary(CONFIG).to(DEVICE)
        _ = load_checkpoint_to_model(PHASE1_BEST_CKPT, phase1_inference_model, map_location=DEVICE)

        phase2_iter0_summary, phase2_iter0_stats_df, phase2_iter0_examples = generate_pseudo_labels(
            model=phase1_inference_model,
            unlabeled_paths=UNLABELED_PATHS,
            config=CONFIG,
            device=DEVICE,
            iteration=0,
        )

        print(json.dumps(phase2_iter0_summary, indent=2))
        """
    ),
    code(
        """
        if not phase2_iter0_examples:
            print("No pseudo-label visualization examples passed the current filter.")
        else:
            fig, axes = plt.subplots(len(phase2_iter0_examples), 4, figsize=(14, 3 * max(1, len(phase2_iter0_examples))))
            if len(phase2_iter0_examples) == 1:
                axes = np.expand_dims(axes, axis=0)

            for row_idx, sample in enumerate(phase2_iter0_examples):
                axes[row_idx, 0].imshow(sample["image"])
                axes[row_idx, 0].set_title("Image")
                axes[row_idx, 1].imshow(sample["pseudo_mask"], cmap="gray")
                axes[row_idx, 1].set_title("Pseudo mask")
                axes[row_idx, 2].imshow(sample["confidence"], cmap="viridis")
                axes[row_idx, 2].set_title("Confidence")
                axes[row_idx, 3].imshow(sample["reliability"], cmap="gray")
                axes[row_idx, 3].set_title("Reliability")
                for col_idx in range(4):
                    axes[row_idx, col_idx].axis("off")

            plt.tight_layout()
            plt.show()
        """
    ),
    md(
        """
        ## 10. Фаза 3 — Semi-supervised training (EMA Teacher-Student)
        """
    ),
    code(
        """
        pseudo_transform = build_strong_transform(
            additional_targets={"reliability": "mask"},
            include_base_resize=not cache_enabled(),
        )
        unlabeled_weak_transform = build_weak_transform(include_base_resize=not cache_enabled())
        unlabeled_strong_transform = build_consistency_strong_transform(include_base_resize=not cache_enabled())


        def build_iteration_pseudo_triplets(iteration: int):
            pseudo_dir = ARTIFACT_ROOT / "pseudo_masks" / f"iteration_{iteration}"
            reliability_dir = ARTIFACT_ROOT / "reliability_masks" / f"iteration_{iteration}"
            pseudo_mask_paths = sorted(pseudo_dir.glob("*.png"))
            image_paths = []
            reliability_paths = []
            filtered_mask_paths = []

            all_unlabeled_map = {p.stem: p for p in UNLABELED_PATHS}
            for pseudo_mask_path in pseudo_mask_paths:
                image_path = all_unlabeled_map.get(pseudo_mask_path.stem)
                reliability_path = reliability_dir / pseudo_mask_path.name
                if image_path is None or not reliability_path.exists():
                    continue
                image_paths.append(image_path)
                filtered_mask_paths.append(pseudo_mask_path)
                reliability_paths.append(reliability_path)
            return image_paths, filtered_mask_paths, reliability_paths


        def train_one_epoch_semi(student, teacher, labeled_loader, pseudo_loader, unlabeled_loader, optimizer, scaler, device):
            student.train()
            teacher.eval()

            labeled_iter = infinite_loader(labeled_loader)
            pseudo_iter = infinite_loader(pseudo_loader) if pseudo_loader is not None else None
            unlabeled_iter = infinite_loader(unlabeled_loader)

            num_steps = max(len(labeled_loader), len(unlabeled_loader), len(pseudo_loader) if pseudo_loader is not None else 0)

            total_loss_meter = []
            labeled_loss_meter = []
            pseudo_loss_meter = []
            consistency_loss_meter = []
            consistency_acceptance_meter = []

            for _ in tqdm(range(num_steps), desc="Phase 3 Train", leave=False):
                batch_l = next(labeled_iter)
                batch_u = next(unlabeled_iter)
                batch_p = next(pseudo_iter) if pseudo_iter is not None else None

                optimizer.zero_grad(set_to_none=True)

                labeled_images = batch_l["image"].to(device)
                labeled_masks = batch_l["mask"].to(device)
                labeled_boundaries = batch_l["boundary"].to(device)

                weak_images = batch_u["weak"].to(device)
                strong_images = batch_u["strong"].to(device)

                with get_autocast_context(device):
                    seg_logits_l, boundary_pred_l = student(labeled_images)
                    l_labeled = total_supervised_loss(seg_logits_l, boundary_pred_l, labeled_masks, labeled_boundaries)

                    l_pseudo = torch.tensor(0.0, device=device)
                    if batch_p is not None:
                        pseudo_images = batch_p["image"].to(device)
                        pseudo_masks = batch_p["mask"].to(device)
                        pseudo_boundaries = batch_p["boundary"].to(device)
                        reliability_masks = batch_p["reliability"].to(device)

                        seg_logits_p, boundary_pred_p = student(pseudo_images)
                        l_pseudo = seg_loss_with_reliability(seg_logits_p, pseudo_masks, reliability_masks)
                        l_pseudo = l_pseudo + (
                            CONFIG["boundary_loss_weight"]
                            * pseudo_boundary_loss(boundary_pred_p, pseudo_boundaries, reliability_masks)
                        )

                    with torch.no_grad():
                        teacher_logits = teacher(weak_images, return_boundary=False)
                        teacher_probs = F.softmax(teacher_logits, dim=1)
                        pseudo_from_teacher = teacher_probs.argmax(dim=1)
                        teacher_confidence = teacher_probs.max(dim=1).values
                        consistency_mask = (teacher_confidence > CONFIG["consistency_confidence_threshold"]).float()

                    student_logits = student(strong_images, return_boundary=False)
                    ce_map = F.cross_entropy(student_logits, pseudo_from_teacher, reduction="none")
                    l_consistency = (ce_map * consistency_mask).sum() / consistency_mask.sum().clamp_min(1.0)

                    total_loss = (
                        l_labeled
                        + CONFIG["unsupervised_loss_weight"] * l_pseudo
                        + CONFIG["unsupervised_loss_weight"] * l_consistency
                    )

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), CONFIG["gradient_clip_norm"])
                scaler.step(optimizer)
                scaler.update()

                update_ema(teacher, student, decay=CONFIG["ema_decay"])

                total_loss_meter.append(float(total_loss.item()))
                labeled_loss_meter.append(float(l_labeled.item()))
                pseudo_loss_meter.append(float(l_pseudo.item()))
                consistency_loss_meter.append(float(l_consistency.item()))
                consistency_acceptance_meter.append(float(consistency_mask.mean().item()))

            return {
                "total_loss": float(np.mean(total_loss_meter)),
                "labeled_loss": float(np.mean(labeled_loss_meter)),
                "pseudo_loss": float(np.mean(pseudo_loss_meter)),
                "consistency_loss": float(np.mean(consistency_loss_meter)),
                "consistency_acceptance": float(np.mean(consistency_acceptance_meter)),
            }


        def run_phase3_iteration(iteration: int, init_checkpoint_path: Path):
            pseudo_image_paths, pseudo_mask_paths, reliability_paths = build_iteration_pseudo_triplets(iteration)

            repeated_train_images = [x[0] for x in TRAIN_SAMPLES] * CONFIG["labeled_repeat"]
            repeated_train_masks = [x[1] for x in TRAIN_SAMPLES] * CONFIG["labeled_repeat"]

            labeled_dataset_phase3 = LabeledDataset(
                image_paths=repeated_train_images,
                mask_paths=repeated_train_masks,
                transform=train_transform_labeled,
                boundary_masks_dir=LABELED_BOUNDARY_DIR,
            )
            pseudo_dataset_phase3 = (
                PseudoLabeledDataset(
                    image_paths=pseudo_image_paths,
                    pseudo_mask_paths=pseudo_mask_paths,
                    reliability_masks_paths=reliability_paths,
                    transform=pseudo_transform,
                )
                if len(pseudo_image_paths) > 0
                else None
            )
            unlabeled_dataset_phase3 = UnlabeledDataset(
                image_paths=UNLABELED_PATHS,
                weak_transform=unlabeled_weak_transform,
                strong_transform=unlabeled_strong_transform,
            )

            labeled_loader_phase3 = make_loader(labeled_dataset_phase3, CONFIG["batch_size"], shuffle=True, config=CONFIG)
            pseudo_loader_phase3 = (
                make_loader(pseudo_dataset_phase3, CONFIG["batch_size"], shuffle=True, config=CONFIG)
                if pseudo_dataset_phase3 is not None and len(pseudo_dataset_phase3) > 0
                else None
            )
            unlabeled_loader_phase3 = make_loader(
                unlabeled_dataset_phase3,
                CONFIG["unlabeled_batch_size"],
                shuffle=True,
                config=CONFIG,
            )

            student = SegFormerWithBoundary(CONFIG).to(DEVICE)
            init_ckpt = torch.load(init_checkpoint_path, map_location=DEVICE)
            init_state_key = "ema_state_dict" if "ema_state_dict" in init_ckpt else "model_state_dict"
            student.load_state_dict(init_ckpt[init_state_key], strict=True)
            freeze_encoder_stages(student, n_stages=2, freeze=False)

            teacher = copy.deepcopy(student).to(DEVICE)
            for param in teacher.parameters():
                param.requires_grad = False

            optimizer = build_optimizer(student, CONFIG)
            scheduler = build_scheduler(
                optimizer,
                total_epochs=CONFIG["phase3_epochs"],
                warmup_epochs=CONFIG["warmup_epochs"],
                min_lr=CONFIG["min_lr"],
            )
            scaler = get_grad_scaler(DEVICE)

            phase_banner(
                f"Phase 3 - Semi-supervised training (iteration {iteration})",
                dataset_sizes={
                    "labeled_repeated": len(labeled_dataset_phase3),
                    "pseudo_labeled": 0 if pseudo_dataset_phase3 is None else len(pseudo_dataset_phase3),
                    "unlabeled": len(unlabeled_dataset_phase3),
                    "val": len(val_dataset),
                },
                model=student,
                batch_size=CONFIG["batch_size"],
                epochs=CONFIG["phase3_epochs"],
                complexity_factor=1.5,
            )

            save_dir = ARTIFACT_ROOT / "phase3" / f"iteration_{iteration}"
            history = []
            best_selection_score = -1.0
            low_acceptance_streak = 0

            for epoch in range(CONFIG["phase3_epochs"]):
                train_stats = train_one_epoch_semi(
                    student=student,
                    teacher=teacher,
                    labeled_loader=labeled_loader_phase3,
                    pseudo_loader=pseudo_loader_phase3,
                    unlabeled_loader=unlabeled_loader_phase3,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=DEVICE,
                )
                scheduler.step()

                if train_stats["consistency_acceptance"] < 0.30:
                    low_acceptance_streak += 1
                else:
                    low_acceptance_streak = 0
                if low_acceptance_streak >= 5:
                    print("Low pseudo-label acceptance rate, consider lowering threshold")

                metrics = {"loss": None, "mIoU": None, "dice": None, "dice_tuned": None, "best_threshold": None, "pixel_acc": None, "boundary_f1": None}
                if (epoch + 1) % CONFIG["eval_every_n_epochs"] == 0 or epoch == 0 or epoch == CONFIG["phase3_epochs"] - 1:
                    metrics = evaluate(teacher, val_loader, DEVICE)
                    current_score = checkpoint_selection_score(metrics)
                    is_best = current_score > best_selection_score
                    if is_best:
                        best_selection_score = current_score
                    save_checkpoint(
                        model=student,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        phase_name=f"phase3_iteration_{iteration}",
                        metrics=metrics,
                        is_best=is_best,
                        save_dir=save_dir,
                        extra={"ema_state_dict": teacher.state_dict()},
                    )
                else:
                    is_best = False

                row = {
                    "phase": "phase3",
                    "iteration": iteration,
                    "epoch": epoch + 1,
                    "train_total_loss": train_stats["total_loss"],
                    "train_labeled_loss": train_stats["labeled_loss"],
                    "train_pseudo_loss": train_stats["pseudo_loss"],
                    "train_consistency_loss": train_stats["consistency_loss"],
                    "consistency_acceptance": train_stats["consistency_acceptance"],
                    "val_loss": metrics["loss"],
                    "val_mIoU": metrics["mIoU"],
                    "val_dice": metrics["dice"],
                    "val_dice_tuned": metrics["dice_tuned"],
                    "val_best_threshold": metrics["best_threshold"],
                    "val_pixel_acc": metrics["pixel_acc"],
                    "val_boundary_f1": metrics["boundary_f1"],
                    "learning_rate": max(group["lr"] for group in optimizer.param_groups),
                }
                history.append(row)
                if metrics["dice_tuned"] is not None:
                    all_metrics.append(row)

                print(
                    f"[Phase 3][Iter {iteration}][Epoch {epoch + 1:03d}] "
                    f"total={train_stats['total_loss']:.4f} "
                    f"L_sup={train_stats['labeled_loss']:.4f} "
                    f"L_pseudo={train_stats['pseudo_loss']:.4f} "
                    f"L_cons={train_stats['consistency_loss']:.4f} "
                    f"consistency_accept={train_stats['consistency_acceptance']:.3f} "
                    f"val_dice_tuned={0.0 if metrics['dice_tuned'] is None else metrics['dice_tuned']:.4f} "
                    f"val_dice={0.0 if metrics['dice'] is None else metrics['dice']:.4f} "
                    f"val_mIoU={0.0 if metrics['mIoU'] is None else metrics['mIoU']:.4f} "
                    f"thr={0.0 if metrics['best_threshold'] is None else metrics['best_threshold']:.2f}"
                )

            save_json(ARTIFACT_ROOT / "metrics" / f"phase3_iteration_{iteration}.json", history)
            return save_dir / "best.pt", history


        PHASE3_ITER0_BEST_CKPT, phase3_iter0_history = run_phase3_iteration(
            iteration=0,
            init_checkpoint_path=PHASE1_BEST_CKPT,
        )
        """
    ),
    md(
        """
        ## 11. Фаза 4 — Итеративный псевдолейблинг (повтор фаз 2-3)
        """
    ),
    code(
        """
        iteration_summaries = []
        previous_best_dice = max([row["val_dice_tuned"] for row in phase1_history if row["val_dice_tuned"] is not None], default=0.0)
        current_checkpoint = PHASE3_ITER0_BEST_CKPT

        phase3_iter0_best = max([row["val_dice_tuned"] for row in phase3_iter0_history if row["val_dice_tuned"] is not None], default=0.0)
        iteration_summaries.append(
            {
                "iteration": 0,
                "phase2_summary": phase2_iter0_summary,
                "phase3_best_dice": phase3_iter0_best,
                "delta_dice": phase3_iter0_best - previous_best_dice,
            }
        )
        previous_best_dice = phase3_iter0_best

        for iteration in range(1, CONFIG["n_pseudo_iterations"]):
            iteration_model = SegFormerWithBoundary(CONFIG).to(DEVICE)
            iteration_ckpt = torch.load(current_checkpoint, map_location=DEVICE)
            state_key = "ema_state_dict" if "ema_state_dict" in iteration_ckpt else "model_state_dict"
            iteration_model.load_state_dict(iteration_ckpt[state_key], strict=True)

            phase2_summary, _, _ = generate_pseudo_labels(
                model=iteration_model,
                unlabeled_paths=UNLABELED_PATHS,
                config=CONFIG,
                device=DEVICE,
                iteration=iteration,
            )
            best_ckpt, history = run_phase3_iteration(iteration=iteration, init_checkpoint_path=current_checkpoint)
            current_best = max([row["val_dice_tuned"] for row in history if row["val_dice_tuned"] is not None], default=0.0)
            delta_dice = current_best - previous_best_dice
            iteration_summaries.append(
                {
                    "iteration": iteration,
                    "phase2_summary": phase2_summary,
                    "phase3_best_dice": current_best,
                    "delta_dice": delta_dice,
                }
            )
            current_checkpoint = best_ckpt
            previous_best_dice = current_best

            print(f"[Iter {iteration}] delta_dice={delta_dice:.4f}")
            if delta_dice < 0.002:
                print("Converged, stopping iterations")
                break

        save_json(ARTIFACT_ROOT / "metrics" / "iteration_summaries.json", iteration_summaries)
        FINAL_BEST_CKPT = current_checkpoint
        print(f"Final checkpoint for evaluation: {FINAL_BEST_CKPT}")
        """
    ),
    md(
        """
        ## 12. Финальная оценка и визуализация предсказаний
        """
    ),
    code(
        """
        final_model = SegFormerWithBoundary(CONFIG).to(DEVICE)
        final_ckpt = torch.load(FINAL_BEST_CKPT, map_location=DEVICE)
        final_state_key = "ema_state_dict" if "ema_state_dict" in final_ckpt else "model_state_dict"
        final_model.load_state_dict(final_ckpt[final_state_key], strict=True)

        final_metrics = evaluate(final_model, val_loader, DEVICE)
        final_threshold = float(final_metrics["best_threshold"])
        print("Final validation metrics:")
        print(json.dumps(final_metrics, indent=2))


        @torch.no_grad()
        def collect_predictions_and_targets(model, dataloader, device, max_examples: int = 16):
            model.eval()
            all_preds = []
            all_targets = []
            examples = []

            for batch in tqdm(dataloader, desc="Collect final predictions", leave=False):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(images, return_boundary=False)
                pos_probs = F.softmax(logits, dim=1)[:, 1]
                preds = (pos_probs >= final_threshold).long()

                all_preds.append(preds.cpu().numpy().reshape(-1))
                all_targets.append(masks.cpu().numpy().reshape(-1))

                if len(examples) < max_examples:
                    for i in range(images.shape[0]):
                        if len(examples) >= max_examples:
                            break
                        examples.append(
                            {
                                "image": denormalize_image_tensor(images[i].cpu()),
                                "gt": masks[i].cpu().numpy(),
                                "pred": preds[i].cpu().numpy(),
                                "path": batch["path"][i],
                            }
                        )

            pred_flat = np.concatenate(all_preds)
            target_flat = np.concatenate(all_targets)
            valid = target_flat != CONFIG["ignore_index"]
            return pred_flat[valid], target_flat[valid], examples


        pred_flat, target_flat, final_examples = collect_predictions_and_targets(final_model, val_loader, DEVICE, max_examples=16)
        cm = confusion_matrix(target_flat, pred_flat, labels=[0, 1])
        print("Confusion matrix:")
        print(cm)

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title("Validation Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()


        def build_diff_overlay(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
            gt = gt.astype(bool)
            pred = pred.astype(bool)
            tp = gt & pred
            fp = (~gt) & pred
            fn = gt & (~pred)
            overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
            overlay[tp] = np.array([0, 255, 0], dtype=np.uint8)
            overlay[fp] = np.array([255, 0, 0], dtype=np.uint8)
            overlay[fn] = np.array([0, 0, 255], dtype=np.uint8)
            return overlay


        n_examples = len(final_examples)
        if n_examples == 0:
            print("No final validation examples collected for visualization.")
        else:
            fig, axes = plt.subplots(n_examples, 4, figsize=(14, 3 * max(1, n_examples)))
            if n_examples == 1:
                axes = np.expand_dims(axes, axis=0)
            for row_idx, sample in enumerate(final_examples):
                axes[row_idx, 0].imshow(sample["image"])
                axes[row_idx, 0].set_title("Image")
                axes[row_idx, 1].imshow(sample["gt"], cmap="gray")
                axes[row_idx, 1].set_title("GT")
                axes[row_idx, 2].imshow(sample["pred"], cmap="gray")
                axes[row_idx, 2].set_title("Prediction")
                axes[row_idx, 3].imshow(build_diff_overlay(sample["gt"], sample["pred"]))
                axes[row_idx, 3].set_title("Diff: FP red / FN blue / TP green")
                for col_idx in range(4):
                    axes[row_idx, col_idx].axis("off")
            plt.tight_layout()
            plt.show()


        metrics_table = pd.DataFrame(all_metrics)
        display(metrics_table.tail(20))
        save_json(ARTIFACT_ROOT / "metrics" / "all_metrics.json", all_metrics)
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


output_path = Path(__file__).resolve().parents[1] / "scripts" / "segformer_boundary_semisup_macos.ipynb"
output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Notebook generated at: {output_path}")
