"""Datasets for one-shot segmentation with camera-based episode sampling."""

from __future__ import annotations

import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class _BaseOneShotDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        img_size: int,
        cached_features_dir: str | Path | None,
        train_mode: bool,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.cache_dir = Path(cached_features_dir) if cached_features_dir else None
        self.use_cache = self.cache_dir is not None and self.cache_dir.exists()

        aug = []
        if train_mode:
            aug.extend(
                [
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
                ]
            )
        aug.extend(
            [
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
        self.transform = A.Compose(aug)

    def _read_rgb(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _read_mask(self, mask_path: Path) -> np.ndarray:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        return (mask > 127).astype(np.float32)

    def _load_image_mask(self, stem: str) -> tuple[torch.Tensor, torch.Tensor]:
        image = self._read_rgb(self.image_dir / f"{stem}.jpg")
        mask = self._read_mask(self.mask_dir / f"{stem}.png")
        transformed = self.transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"].unsqueeze(0).float()

    def _load_cached(self, stem: str) -> torch.Tensor:
        path = self.cache_dir / f"{stem}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing cached feature: {path}")
        return torch.load(path, weights_only=True, map_location="cpu")


# ======================================================================
# Training dataset — random episodes (support != query camera)
# ======================================================================
class OneShotTrainDataset(_BaseOneShotDataset):
    """Each item is one episode: support/query drawn from different cameras."""

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        camera_ips: list[str],
        img_size: int = 322,
        cached_features_dir: str | Path | None = None,
        samples_per_epoch: int = 1000,
        debug_limit: int | None = None,
    ):
        super().__init__(
            image_dir=image_dir,
            mask_dir=mask_dir,
            img_size=img_size,
            cached_features_dir=cached_features_dir,
            train_mode=True,
        )
        self.samples_per_epoch = samples_per_epoch

        self.camera_groups: dict[str, list[str]] = {}
        for img_path in sorted(self.image_dir.glob("*.jpg")):
            camera = img_path.stem.split("_")[0]
            if camera in camera_ips:
                self.camera_groups.setdefault(camera, []).append(img_path.stem)

        if debug_limit:
            per_camera = max(2, debug_limit // max(1, len(self.camera_groups)))
            for camera in list(self.camera_groups):
                self.camera_groups[camera] = self.camera_groups[camera][:per_camera]

        self.camera_list = [camera for camera, stems in sorted(self.camera_groups.items()) if stems]
        if len(self.camera_list) < 2:
            raise ValueError(f"Need at least 2 camera groups, got {len(self.camera_list)}")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        support_camera = random.choice(self.camera_list)
        support_stem = random.choice(self.camera_groups[support_camera])

        other_cameras = [camera for camera in self.camera_list if camera != support_camera]
        query_camera = random.choice(other_cameras)
        query_stem = random.choice(self.camera_groups[query_camera])

        support_img, support_mask = self._load_image_mask(support_stem)
        query_img, query_mask = self._load_image_mask(query_stem)

        batch = {
            "support_mask": support_mask,
            "query_mask": query_mask,
            "support_id": support_stem,
            "query_id": query_stem,
            "support_camera": support_camera,
            "query_camera": query_camera,
        }
        if self.use_cache:
            batch["support_feat"] = self._load_cached(support_stem)
            batch["query_feat"] = self._load_cached(query_stem)
            return batch

        batch["support_img"] = support_img
        batch["query_img"] = query_img
        return batch


# ======================================================================
# Validation dataset — deterministic fixed support
# ======================================================================
class OneShotValDataset(_BaseOneShotDataset):
    """Fixed support image; iterates over validation-camera images as queries."""

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        val_camera_ips: list[str],
        train_camera_ips: list[str],
        img_size: int = 322,
        cached_features_dir: str | Path | None = None,
        debug_limit: int | None = None,
    ):
        super().__init__(
            image_dir=image_dir,
            mask_dir=mask_dir,
            img_size=img_size,
            cached_features_dir=cached_features_dir,
            train_mode=False,
        )

        val_stems: list[str] = []
        train_stems: list[str] = []
        for image_path in sorted(self.image_dir.glob("*.jpg")):
            camera = image_path.stem.split("_")[0]
            if camera in val_camera_ips:
                val_stems.append(image_path.stem)
            elif camera in train_camera_ips:
                train_stems.append(image_path.stem)

        if debug_limit:
            val_stems = val_stems[:debug_limit]
            train_stems = train_stems[: max(1, debug_limit)]

        if not val_stems:
            raise ValueError("Validation set is empty for the selected fold.")

        self.val_stems = val_stems
        self.support_stem = train_stems[0] if train_stems else val_stems[0]

    def __len__(self):
        return len(self.val_stems)

    def __getitem__(self, idx):
        query_stem = self.val_stems[idx]
        support_img, support_mask = self._load_image_mask(self.support_stem)
        query_img, query_mask = self._load_image_mask(query_stem)

        batch = {
            "support_mask": support_mask,
            "query_mask": query_mask,
            "support_id": self.support_stem,
            "query_id": query_stem,
        }
        if self.use_cache:
            batch["support_feat"] = self._load_cached(self.support_stem)
            batch["query_feat"] = self._load_cached(query_stem)
            return batch

        batch["support_img"] = support_img
        batch["query_img"] = query_img
        return batch
