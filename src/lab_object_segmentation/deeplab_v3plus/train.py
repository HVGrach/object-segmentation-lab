#!/usr/bin/env python3
"""Train a DeepLabV3+ candidate on one grouped camera fold.

This trainer intentionally reuses the project's proven conventions:
- grouped-by-camera fold split from supervised_v4;
- image_size=384 and resnet34 encoder from advanced_baseline;
- EMA, threshold tuning, TTA, and post-processing for ensemble friendliness;
- moderate augmentations and lovasz_focal support from supervised_v4.

Example:

python scripts/train_deeplab_v3plus.py \
    --run-name deeplabv3p_r34_fold1_384_lovasz \
    --fold 1
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim import AdamW
import segmentation_models_pytorch as smp

from lab_object_segmentation.common.paths import LAB3_DATASET_ROOT, PROJECT_ROOT, RUNS_ROOT
from lab_object_segmentation.segformer_semisup.utils import infer_camera_group
from lab_object_segmentation.supervised_v4.train import (
    build_dataloaders,
    build_scheduler as build_cosine_scheduler,
    checkpoint_selection_score,
    collect_dataset_root_pairs,
    collect_labeled_pairs,
    empty_device_cache,
    evaluate,
    get_device,
    get_epoch_phase,
    load_val_sample_ids,
    resolve_extra_holdout_overrides,
    resolve_extra_weight_overrides,
    save_checkpoint,
    save_json,
    set_seed,
    split_samples_by_fold,
    split_samples_random,
    summarize_depth_coverage,
    train_one_epoch,
)


@dataclass
class TrainConfig:
    run_name: str
    fold: int
    n_splits: int
    image_size: int
    epochs: int
    aug: str
    label_smoothing: float
    physical_batch_size: int

    effective_batch_size: int = 16
    encoder_name: str = "resnet34"
    encoder_weights: str = "imagenet"
    base_lr: float = 3e-4
    encoder_lr_scale: float = 0.5
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    min_lr: float = 1e-6
    grad_clip_norm: float = 1.0
    boundary_loss_weight: float = 0.10
    focal_gamma: float = 2.0
    focal_alpha_pos: float = 0.75
    mask_loss: str = "lovasz_focal"
    resize_interpolation: str = "linear"
    ema_decay: float = 0.999
    save_every_n_epochs: int = 5
    eval_every_n_epochs: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    seed: int = 42
    selection_metric: str = "dice_tuned"
    threshold_grid: list[float] = field(
        default_factory=lambda: [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    )
    image_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    image_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    copy_paste_p: float = 0.30
    copy_paste_two_objects_p: float = 0.30
    copy_paste_scale_range: tuple[float, float] = (0.6, 1.4)
    boundary_kernel_size: int = 3
    tta_scales: list[float] = field(default_factory=lambda: [0.75, 1.0, 1.25])
    tta_ops: str = "flips"
    postprocess_enabled: bool = True
    postprocess_min_component_area: int = 128
    postprocess_fill_holes: bool = True
    resume: bool = False
    finetune_from: str = ""
    finetune_lr_scale: float = 0.30
    final_tta: bool = True
    extra_labeled_roots: list[str] = field(default_factory=list)
    extra_labeled_weights: list[float] = field(default_factory=list)
    extra_labeled_holdout_ratios: list[float] = field(default_factory=list)
    extra_holdout_ratio: float = 0.0
    extra_holdout_seed: int = 42
    depth_root: str = ""
    depth_mean: float = 0.5
    depth_std: float = 0.25
    depth_fill_value: float = 0.5
    decoder_channels: int = 256
    decoder_atrous_rates: tuple[int, int, int] = (12, 24, 36)

    @property
    def accumulation_steps(self) -> int:
        return max(1, self.effective_batch_size // max(1, self.physical_batch_size))

    @property
    def run_dir(self) -> Path:
        return RUNS_ROOT / "deeplab_v3plus" / self.run_name

    @property
    def depth_enabled(self) -> bool:
        return bool(self.depth_root)

    @property
    def input_channels(self) -> int:
        return 4 if self.depth_enabled else 3


class BoundaryHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DeepLabV3PlusBoundary(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights,
            in_channels=config.input_channels,
            classes=1,
            activation=None,
            decoder_channels=config.decoder_channels,
            decoder_atrous_rates=config.decoder_atrous_rates,
        )
        low_level_channels = int(self.model.encoder.out_channels[1])
        self.boundary_head = BoundaryHead(low_level_channels)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model.encoder(pixel_values)
        decoder_output = self.model.decoder(features)
        seg_logits = self.model.segmentation_head(decoder_output)
        boundary_logits = self.boundary_head(features[1])
        return seg_logits, boundary_logits


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a DeepLabV3+ fold candidate for later ensemble use.")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--n-splits", type=int, default=3, help="Number of grouped camera folds used for train/val split.")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--aug", choices=["geom", "moderate", "heavy"], default="moderate")
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--mask-loss", choices=["dice_focal", "lovasz_focal"], default="lovasz_focal")
    parser.add_argument("--resize-interpolation", choices=["nearest", "linear", "cubic", "area", "lanczos4"], default="linear")
    parser.add_argument("--physical-batch-size", type=int, default=8)
    parser.add_argument("--effective-batch-size", type=int, default=16)
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every-n-epochs", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Resume from latest_model.pth in run_dir")
    parser.add_argument("--finetune-from", type=str, default="", help="Path to checkpoint .pth to load model+ema weights from.")
    parser.add_argument("--finetune-lr-scale", type=float, default=0.3, help="Scale base_lr by this factor for fine-tuning.")
    parser.add_argument("--tta-scales", type=float, nargs="+", default=[0.75, 1.0, 1.25], help="Multi-scale TTA scales for final evaluation.")
    parser.add_argument("--tta-ops", choices=("flips", "d4"), default="flips", help="Test-time augmentation transform family.")
    parser.add_argument("--disable-postprocess", action="store_true", help="Disable mask post-processing during evaluation.")
    parser.add_argument("--postprocess-min-component-area", type=int, default=128, help="Remove connected components smaller than this area.")
    parser.add_argument("--disable-postprocess-fill-holes", action="store_true", help="Disable filling holes inside predicted masks.")
    parser.add_argument(
        "--extra-labeled-root",
        action="append",
        default=[],
        help="Optional extra labeled dataset root with images/ and masks/ folders. Can be passed multiple times.",
    )
    parser.add_argument(
        "--extra-labeled-weight",
        action="append",
        type=float,
        default=[],
        help="Optional sample-weight override(s) for --extra-labeled-root. One value broadcasts to all roots.",
    )
    parser.add_argument(
        "--extra-holdout-ratio",
        type=float,
        default=0.0,
        help="Optional random holdout ratio reserved from each extra labeled root for validation.",
    )
    parser.add_argument(
        "--extra-labeled-holdout-ratio",
        action="append",
        type=float,
        default=[],
        help="Optional per-root holdout ratio(s) for --extra-labeled-root. Overrides --extra-holdout-ratio when provided.",
    )
    parser.add_argument("--extra-holdout-seed", type=int, default=42, help="Seed for extra labeled holdout splitting.")
    parser.add_argument("--depth-root", type=str, default="", help="Optional root with depth .npz files keyed by image stem.")
    parser.add_argument("--depth-mean", type=float, default=0.5, help="Depth normalization mean applied after spatial transforms.")
    parser.add_argument("--depth-std", type=float, default=0.25, help="Depth normalization std applied after spatial transforms.")
    parser.add_argument("--depth-fill-value", type=float, default=0.5, help="Raw depth value used when a sample has no matching depth file.")
    parser.add_argument("--decoder-channels", type=int, default=256)
    args = parser.parse_args()

    if args.n_splits < 2:
        raise ValueError("n-splits must be >= 2.")
    if args.fold < 0 or args.fold >= args.n_splits:
        raise ValueError(f"Unsupported fold index {args.fold} for n-splits={args.n_splits}.")
    if args.physical_batch_size <= 0:
        raise ValueError("physical-batch-size must be positive.")
    if args.effective_batch_size < args.physical_batch_size:
        raise ValueError("effective-batch-size must be >= physical-batch-size.")
    if args.effective_batch_size % args.physical_batch_size != 0:
        raise ValueError("effective-batch-size must be divisible by physical-batch-size.")
    if args.encoder_lr_scale <= 0:
        raise ValueError("encoder-lr-scale must be positive.")
    if not 0.0 <= args.extra_holdout_ratio < 1.0:
        raise ValueError("extra-holdout-ratio must be in [0, 1).")
    if any(weight <= 0 for weight in args.extra_labeled_weight):
        raise ValueError("extra-labeled-weight values must be positive.")
    if any(not 0.0 <= ratio < 1.0 for ratio in args.extra_labeled_holdout_ratio):
        raise ValueError("extra-labeled-holdout-ratio values must be in [0, 1).")
    if args.depth_std <= 0:
        raise ValueError("depth-std must be positive.")

    return TrainConfig(
        run_name=args.run_name,
        fold=args.fold,
        n_splits=args.n_splits,
        image_size=args.image_size,
        epochs=args.epochs,
        aug=args.aug,
        label_smoothing=args.label_smoothing,
        physical_batch_size=args.physical_batch_size,
        effective_batch_size=args.effective_batch_size,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        base_lr=args.base_lr,
        encoder_lr_scale=args.encoder_lr_scale,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        mask_loss=args.mask_loss,
        resize_interpolation=args.resize_interpolation,
        seed=args.seed,
        eval_every_n_epochs=args.eval_every_n_epochs,
        tta_scales=[float(scale) for scale in args.tta_scales],
        tta_ops=args.tta_ops,
        postprocess_enabled=not args.disable_postprocess,
        postprocess_min_component_area=int(args.postprocess_min_component_area),
        postprocess_fill_holes=not args.disable_postprocess_fill_holes,
        resume=args.resume,
        finetune_from=args.finetune_from,
        finetune_lr_scale=args.finetune_lr_scale,
        extra_labeled_roots=[str(Path(root).expanduser().resolve()) for root in args.extra_labeled_root],
        extra_labeled_weights=[float(weight) for weight in args.extra_labeled_weight],
        extra_labeled_holdout_ratios=[float(ratio) for ratio in args.extra_labeled_holdout_ratio],
        extra_holdout_ratio=float(args.extra_holdout_ratio),
        extra_holdout_seed=int(args.extra_holdout_seed),
        depth_root=str(Path(args.depth_root).expanduser().resolve()) if args.depth_root else "",
        depth_mean=float(args.depth_mean),
        depth_std=float(args.depth_std),
        depth_fill_value=float(args.depth_fill_value),
        decoder_channels=int(args.decoder_channels),
    )


def build_optimizer(model: DeepLabV3PlusBoundary, config: TrainConfig) -> AdamW:
    groups: dict[tuple[str, str], list[nn.Parameter]] = {
        ("encoder", "decay"): [],
        ("encoder", "no_decay"): [],
        ("decoder", "decay"): [],
        ("decoder", "no_decay"): [],
    }

    def is_no_decay(name: str, param: nn.Parameter) -> bool:
        name_l = name.lower()
        return param.ndim <= 1 or "bias" in name_l or "bn" in name_l or "norm" in name_l

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        branch = "encoder" if name.startswith("model.encoder.") else "decoder"
        bucket = "no_decay" if is_no_decay(name, param) else "decay"
        groups[(branch, bucket)].append(param)

    param_groups = []
    for (branch, bucket), params in groups.items():
        if not params:
            continue
        lr_scale = config.encoder_lr_scale if branch == "encoder" else 1.0
        param_groups.append(
            {
                "params": params,
                "lr": config.base_lr * lr_scale,
                "weight_decay": config.weight_decay if bucket == "decay" else 0.0,
                "group_name": f"{branch}_{bucket}",
            }
        )
    return AdamW(param_groups)


def train(config: TrainConfig) -> int:
    device = get_device()
    config.pin_memory = device.type == "cuda"
    if device.type == "mps":
        config.num_workers = 0

    set_seed(config.seed)
    cudnn.benchmark = device.type == "cuda"

    print(f"[main] project_root: {PROJECT_ROOT}")
    print(f"[main] device: {device}")
    print(f"[main] run_dir: {config.run_dir}")
    if config.n_splits == 3 and config.fold == 0:
        print("[main] note: fold_0 is the large single-camera validation split and behaves like a domain-shift stress test.")

    images_dir = LAB3_DATASET_ROOT / "train" / "images"
    masks_dir = LAB3_DATASET_ROOT / "train" / "masks"
    samples = collect_labeled_pairs(images_dir, masks_dir, source_name="lab3_train", sample_weight=1.0)
    val_sample_ids = load_val_sample_ids(PROJECT_ROOT, config.fold, n_splits=config.n_splits)
    train_samples, val_samples = split_samples_by_fold(samples, val_sample_ids)
    base_train_count = len(train_samples)
    base_val_count = len(val_samples)

    extra_holdout_samples = []
    extra_sources_summary: list[dict] = []
    extra_weight_overrides = resolve_extra_weight_overrides(config.extra_labeled_roots, config.extra_labeled_weights)
    extra_holdout_overrides = resolve_extra_holdout_overrides(
        config.extra_labeled_roots,
        config.extra_labeled_holdout_ratios,
        config.extra_holdout_ratio,
    )
    for source_idx, (root_str, weight, holdout_ratio) in enumerate(
        zip(config.extra_labeled_roots, extra_weight_overrides, extra_holdout_overrides),
        start=1,
    ):
        dataset_root = Path(root_str)
        extra_samples = collect_dataset_root_pairs(dataset_root, sample_weight=weight)
        extra_train_samples, extra_val_samples = split_samples_random(
            extra_samples,
            holdout_ratio,
            seed=config.extra_holdout_seed + source_idx - 1,
        )
        train_samples.extend(extra_train_samples)
        extra_holdout_samples.extend(extra_val_samples)
        extra_sources_summary.append(
            {
                "dataset_root": str(dataset_root),
                "source_name": extra_samples[0].source_name if extra_samples else dataset_root.name,
                "sample_weight": float(weight),
                "holdout_ratio": float(holdout_ratio),
                "train_samples": len(extra_train_samples),
                "holdout_samples": len(extra_val_samples),
                "total_samples": len(extra_samples),
            }
        )

    split_summary = {
        "n_splits": config.n_splits,
        "fold": config.fold,
        "base_train_samples": base_train_count,
        "base_val_samples": base_val_count,
        "extra_train_samples": len(train_samples) - base_train_count,
        "extra_holdout_samples": len(extra_holdout_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_cameras": sorted({infer_camera_group(sample.image_path) for sample in train_samples}),
        "val_cameras": sorted({infer_camera_group(sample.image_path) for sample in val_samples}),
        "extra_sources": extra_sources_summary,
    }
    print(
        f"[main] fold {config.fold}/{config.n_splits - 1} | train={split_summary['train_samples']} | "
        f"val={split_summary['val_samples']} | train_cameras={len(split_summary['train_cameras'])} | "
        f"val_cameras={len(split_summary['val_cameras'])}"
    )

    if config.depth_enabled:
        depth_root = Path(config.depth_root)
        if not depth_root.exists():
            raise FileNotFoundError(f"depth root not found: {depth_root}")
        train_depth_summary = summarize_depth_coverage(train_samples, depth_root)
        val_depth_summary = summarize_depth_coverage(val_samples, depth_root)
        extra_holdout_depth_summary = summarize_depth_coverage(extra_holdout_samples, depth_root) if extra_holdout_samples else None
        split_summary["depth"] = {
            "depth_root": str(depth_root),
            "depth_mean": float(config.depth_mean),
            "depth_std": float(config.depth_std),
            "depth_fill_value": float(config.depth_fill_value),
            "train": train_depth_summary,
            "val": val_depth_summary,
            "extra_holdout": extra_holdout_depth_summary,
        }
        print(
            "[main] depth=on | "
            f"train direct={train_depth_summary['direct']} heuristic={train_depth_summary['heuristic']} missing={train_depth_summary['missing']} | "
            f"val direct={val_depth_summary['direct']} heuristic={val_depth_summary['heuristic']} missing={val_depth_summary['missing']}"
        )

    config.run_dir.mkdir(parents=True, exist_ok=True)
    save_json(config.run_dir / "config.json", asdict(config))
    save_json(config.run_dir / "split_summary.json", split_summary)

    train_dataset, train_loader, val_loader, extra_holdout_loader = build_dataloaders(
        train_samples,
        val_samples,
        extra_holdout_samples,
        config,
        device,
    )

    model = DeepLabV3PlusBoundary(config).to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)

    if config.finetune_from:
        ft_path = Path(config.finetune_from)
        if not ft_path.exists():
            raise FileNotFoundError(f"finetune checkpoint not found: {ft_path}")
        print(f"[finetune] loading weights from {ft_path}")
        ft_ckpt = torch.load(ft_path, map_location=device, weights_only=False)
        model.load_state_dict(ft_ckpt["ema_model_state_dict"], strict=True)
        ema_model.load_state_dict(ft_ckpt["ema_model_state_dict"], strict=True)
        config.base_lr = config.base_lr * config.finetune_lr_scale
        save_json(config.run_dir / "config.json", asdict(config))
        print(f"[finetune] loaded EMA weights, applying lr_scale={config.finetune_lr_scale}")

    optimizer = build_optimizer(model, config)
    scheduler = build_cosine_scheduler(
        optimizer,
        total_epochs=config.epochs,
        warmup_epochs=config.warmup_epochs,
        min_lr=config.min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history: list[dict] = []
    best_score = -1.0
    start_epoch = 0

    if config.resume:
        resume_path = config.run_dir / "latest_model.pth"
        if resume_path.exists():
            print(f"[resume] loading checkpoint from {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_score = float(checkpoint.get("best_score", -1.0))
            history_path = config.run_dir / "history.json"
            if history_path.exists():
                history = json.loads(history_path.read_text(encoding="utf-8"))
            print(f"[resume] resuming from epoch {start_epoch + 1}, best_score={best_score:.4f}")
        else:
            print(f"[resume] no checkpoint found at {resume_path}, starting from scratch")

    print(
        f"[main] encoder={config.encoder_name} | image_size={config.image_size} | epochs={config.epochs} | "
        f"aug={config.aug} | mask_loss={config.mask_loss} | physical_bs={config.physical_batch_size} | "
        f"effective_bs={config.effective_batch_size} | accumulation={config.accumulation_steps} | "
        f"base_lr={config.base_lr:.2e} | encoder_lr_scale={config.encoder_lr_scale:.2f} | "
        f"postprocess={'on' if config.postprocess_enabled else 'off'} | "
        f"tta_ops={config.tta_ops} | tta_scales={config.tta_scales}"
    )

    for epoch_idx in range(start_epoch, config.epochs):
        phase = get_epoch_phase(config, epoch_idx)
        train_dataset.set_phase(phase)

        t0 = time.time()
        train_metrics = train_one_epoch(
            model=model,
            ema_model=ema_model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            config=config,
            epoch_idx=epoch_idx,
        )
        current_lr = max(group["lr"] for group in optimizer.param_groups)
        scheduler.step()

        should_eval = (
            (epoch_idx + 1) % config.eval_every_n_epochs == 0
            or epoch_idx == 0
            or epoch_idx == config.epochs - 1
        )
        if should_eval:
            val_metrics = evaluate(ema_model, val_loader, device, config, use_tta=False)
            extra_holdout_metrics = (
                evaluate(ema_model, extra_holdout_loader, device, config, use_tta=False)
                if extra_holdout_loader is not None
                else None
            )
        else:
            val_metrics = {
                "mIoU": 0.0,
                "dice": 0.0,
                "dice_tuned": 0.0,
                "best_threshold": 0.0,
                "pixel_acc": 0.0,
                "boundary_f1": 0.0,
                "loss": 0.0,
                "tta": False,
            }
            extra_holdout_metrics = None

        elapsed = time.time() - t0
        row = {
            "epoch": epoch_idx,
            "phase": phase,
            "evaluated": should_eval,
            **train_metrics,
            **val_metrics,
            "learning_rate": current_lr,
            "time_sec": round(elapsed, 1),
        }
        if extra_holdout_metrics is not None:
            row["extra_holdout"] = extra_holdout_metrics
            row["extra_holdout_dice_tuned"] = extra_holdout_metrics["dice_tuned"]
            row["extra_holdout_mIoU"] = extra_holdout_metrics["mIoU"]
        history.append(row)

        is_best = False
        if should_eval:
            score = checkpoint_selection_score(row, config.selection_metric)
            is_best = score > best_score
            if is_best:
                best_score = score

        save_checkpoint(
            config,
            model,
            ema_model,
            optimizer,
            scheduler,
            epoch_idx,
            row,
            is_best=is_best,
            best_score=best_score,
        )
        save_json(config.run_dir / "history.json", history)

        if should_eval:
            extra_holdout_suffix = ""
            if extra_holdout_metrics is not None:
                extra_holdout_suffix = (
                    f" extra_holdout_dice_tuned={extra_holdout_metrics['dice_tuned']:.4f}"
                    f" extra_holdout_mIoU={extra_holdout_metrics['mIoU']:.4f}"
                )
            print(
                f"[epoch {epoch_idx + 1:03d}/{config.epochs}] phase={phase} "
                f"train_loss={row['train_loss']:.4f} "
                f"val_loss={row['loss']:.4f} "
                f"val_mIoU={row['mIoU']:.4f} "
                f"val_dice={row['dice']:.4f} "
                f"val_dice_tuned={row['dice_tuned']:.4f} "
                f"thr={row['best_threshold']:.2f} "
                f"boundary_f1={row['boundary_f1']:.4f} "
                f"lr={current_lr:.6e} "
                f"time={elapsed:.1f}s"
                f"{extra_holdout_suffix}"
                + (" [best]" if is_best else "")
            )
        else:
            print(
                f"[epoch {epoch_idx + 1:03d}/{config.epochs}] phase={phase} "
                f"train_loss={row['train_loss']:.4f} "
                f"lr={current_lr:.6e} "
                f"time={elapsed:.1f}s [skip eval]"
            )

        empty_device_cache(device)

    if config.final_tta:
        best_path = config.run_dir / "best_model.pth"
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        ema_model.load_state_dict(checkpoint["ema_model_state_dict"], strict=True)
        final_tta_metrics = evaluate(ema_model, val_loader, device, config, use_tta=True)
        save_json(config.run_dir / "final_tta_metrics.json", final_tta_metrics)
        print(
            f"[final_tta] mIoU={final_tta_metrics['mIoU']:.4f} "
            f"dice={final_tta_metrics['dice']:.4f} "
            f"dice_tuned={final_tta_metrics['dice_tuned']:.4f} "
            f"thr={final_tta_metrics['best_threshold']:.2f}"
        )
        if extra_holdout_loader is not None:
            extra_holdout_final_tta_metrics = evaluate(ema_model, extra_holdout_loader, device, config, use_tta=True)
            save_json(config.run_dir / "extra_holdout_final_tta_metrics.json", extra_holdout_final_tta_metrics)
            print(
                f"[final_tta extra_holdout] mIoU={extra_holdout_final_tta_metrics['mIoU']:.4f} "
                f"dice={extra_holdout_final_tta_metrics['dice']:.4f} "
                f"dice_tuned={extra_holdout_final_tta_metrics['dice_tuned']:.4f} "
                f"thr={extra_holdout_final_tta_metrics['best_threshold']:.2f}"
            )
        empty_device_cache(device)

    print(f"[main] training complete. Artifacts -> {config.run_dir}")
    return 0


def main() -> int:
    config = parse_args()
    return train(config)


if __name__ == "__main__":
    raise SystemExit(main())
