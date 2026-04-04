#!/usr/bin/env python3
"""Main entry point for DINOv2 one-shot segmentation ablation.

Usage:
    python train_dinov2.py --config configs/dinov2/ablation_film.yaml
    python train_dinov2.py --config configs/dinov2/smoke_test.yaml
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from lab_object_segmentation.common.paths import guess_project_root
from lab_object_segmentation.dinov2_research.config import Config
from lab_object_segmentation.dinov2_research.dataset import OneShotTrainDataset, OneShotValDataset
from lab_object_segmentation.dinov2_research.model import build_model
from lab_object_segmentation.dinov2_research.trainer import Trainer


def get_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="DINOv2 One-Shot Segmentation Trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args(argv)

    # Load config
    cfg = Config.from_yaml(args.config)
    project_root = guess_project_root()
    cfg.set_project_root(project_root)

    # Device & seed
    device = get_device()
    seed_everything(cfg.seed)
    print(f"[main] project_root: {project_root}")
    print(f"[main] device: {device}")

    # Camera split
    train_cams, val_cams = cfg.load_camera_split()
    print(f"[main] fold {cfg.val_fold}: {len(train_cams)} train cameras, {len(val_cams)} val cameras")

    # Datasets
    debug_limit = cfg.debug_samples if cfg.debug else None
    cache_dir = cfg.cache_dir if cfg.use_cached_features else None

    train_ds = OneShotTrainDataset(
        image_dir=cfg.train_images_dir,
        mask_dir=cfg.train_masks_dir,
        camera_ips=train_cams,
        img_size=cfg.img_size,
        cached_features_dir=cache_dir,
        samples_per_epoch=cfg.samples_per_epoch,
        debug_limit=debug_limit,
    )
    val_ds = OneShotValDataset(
        image_dir=cfg.train_images_dir,
        mask_dir=cfg.train_masks_dir,
        val_camera_ips=val_cams,
        train_camera_ips=train_cams,
        img_size=cfg.img_size,
        cached_features_dir=cache_dir,
        debug_limit=debug_limit,
    )

    print(f"[main] train: {len(train_ds)} episodes/epoch, val: {len(val_ds)} images")
    print(f"[main] using cached features: {train_ds.use_cache}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )

    # Build model
    model = build_model(
        backbone_size=cfg.backbone_size,
        fusion_type=cfg.fusion_type,
        decoder_channels=cfg.decoder_channels,
        num_classes=cfg.num_classes,
        img_size=cfg.img_size,
        device=device,
        use_cached=cfg.use_cached_features and train_ds.use_cache,
    )

    # Trainer
    trainer = Trainer(model=model, cfg=cfg, device=device)
    if args.resume:
        trainer.resume_from(args.resume)

    trainer.train(train_loader, val_loader)

    return 0


if __name__ == "__main__":
    sys.exit(main())
