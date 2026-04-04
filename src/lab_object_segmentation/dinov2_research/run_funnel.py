#!/usr/bin/env python3
"""Automate the proxy ablation funnel: DINOv2-S -> B -> L."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from lab_object_segmentation.common.paths import LAB3_DATASET_ROOT, PROJECT_ROOT, RUNS_ROOT
from lab_object_segmentation.dinov2_research.cache_features import main as cache_features_main
from lab_object_segmentation.dinov2_research.train import main as train_main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DINOv2 proxy ablation funnel.")
    parser.add_argument("--img-size", type=int, default=322)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--samples-per-epoch", type=int, default=256)
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--cache-missing", action="store_true", help="Auto-cache full train set when cache is missing.")
    parser.add_argument("--skip-stage3", action="store_true", help="Stop after S/B ranking validation.")
    return parser.parse_args(argv)


def write_yaml(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def config_payload(backbone_size: str, fusion_type: str, img_size: int, epochs: int, samples_per_epoch: int, val_fold: int) -> dict:
    batch_sizes = {"s": 8, "b": 6, "l": 4}
    return {
        "experiment_name": f"funnel_{backbone_size}_{fusion_type}",
        "seed": 42,
        "backbone_size": backbone_size,
        "fusion_type": fusion_type,
        "decoder_channels": 256,
        "num_classes": 1,
        "img_size": img_size,
        "batch_size": batch_sizes[backbone_size],
        "num_workers": 0,
        "samples_per_epoch": samples_per_epoch,
        "epochs": epochs,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "scheduler": "cosine",
        "run_preflight": True,
        "bce_weight": 0.5,
        "dice_weight": 0.5,
        "checkpoint_every": 5,
        "use_cached_features": True,
        "cache_dtype": "float16",
        "val_fold": val_fold,
        "debug": False,
        "debug_samples": 8,
    }


def cache_dir(project_root: Path, backbone_size: str, img_size: int) -> Path:
    return RUNS_ROOT / "dinov2_research" / "feature_cache" / f"dinov2_{backbone_size}_{img_size}"


def ensure_cache(project_root: Path, backbone_size: str, img_size: int, enabled: bool):
    cdir = cache_dir(project_root, backbone_size, img_size)
    num_cached = len(list(cdir.glob("*.pt"))) if cdir.exists() else 0
    num_train = len(list((LAB3_DATASET_ROOT / "train" / "images").glob("*.jpg")))
    if num_cached >= num_train:
        print(f"[funnel] cache ready for DINOv2-{backbone_size.upper()} ({num_cached} files)")
        return
    if not enabled:
        raise RuntimeError(
            f"Missing full cache for DINOv2-{backbone_size.upper()} at {cdir}. "
            "Run python scripts/cache_dinov2_features.py first or use --cache-missing."
        )
    cache_features_main(
        [
            "--input-dir",
            str(LAB3_DATASET_ROOT / "train" / "images"),
            "--backbone-size",
            backbone_size,
            "--img-size",
            str(img_size),
        ]
    )


def best_val_iou(project_root: Path, experiment_name: str) -> float:
    history_path = RUNS_ROOT / "dinov2_research" / "experiments" / experiment_name / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    return float(max(row["val_iou"] for row in history))


def stage_run(project_root: Path, config_path: Path) -> float:
    train_main(["--config", str(config_path)])
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return best_val_iou(project_root, payload["experiment_name"])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = PROJECT_ROOT
    generated_dir = RUNS_ROOT / "dinov2_research" / "generated_configs"
    funnel_dir = RUNS_ROOT / "dinov2_research" / "funnel_runs"
    funnel_dir.mkdir(parents=True, exist_ok=True)

    fusion_types = ["prototype_cosine", "concat", "film", "correlation_4d"]
    summary: dict[str, dict[str, float | list[str] | str]] = {"stage1_s": {}, "stage2_b": {}, "stage3_l": {}}

    ensure_cache(project_root, "s", args.img_size, args.cache_missing)
    for fusion in fusion_types:
        config_path = generated_dir / f"stage1_s_{fusion}.yaml"
        write_yaml(
            config_path,
            config_payload(
                backbone_size="s",
                fusion_type=fusion,
                img_size=args.img_size,
                epochs=args.epochs,
                samples_per_epoch=args.samples_per_epoch,
                val_fold=args.val_fold,
            ),
        )
        summary["stage1_s"][fusion] = stage_run(project_root, config_path)

    ranking = sorted(summary["stage1_s"], key=lambda key: float(summary["stage1_s"][key]), reverse=True)
    top2 = ranking[:2]
    worst = ranking[-1]
    summary["stage1_ranking"] = {"ordered": ranking, "top2": top2, "worst": worst}

    ensure_cache(project_root, "b", args.img_size, args.cache_missing)
    for fusion in [*top2, worst]:
        config_path = generated_dir / f"stage2_b_{fusion}.yaml"
        write_yaml(
            config_path,
            config_payload(
                backbone_size="b",
                fusion_type=fusion,
                img_size=args.img_size,
                epochs=args.epochs,
                samples_per_epoch=args.samples_per_epoch,
                val_fold=args.val_fold,
            ),
        )
        summary["stage2_b"][fusion] = stage_run(project_root, config_path)

    ranking_b = sorted(summary["stage2_b"], key=lambda key: float(summary["stage2_b"][key]), reverse=True)
    summary["stage2_ranking"] = {"ordered": ranking_b}

    if not args.skip_stage3:
        winner = ranking_b[0]
        finalists = [winner]
        if winner != "prototype_cosine":
            finalists.append("prototype_cosine")

        ensure_cache(project_root, "l", args.img_size, args.cache_missing)
        for fusion in finalists:
            config_path = generated_dir / f"stage3_l_{fusion}.yaml"
            write_yaml(
                config_path,
                config_payload(
                    backbone_size="l",
                    fusion_type=fusion,
                    img_size=args.img_size,
                    epochs=args.epochs,
                    samples_per_epoch=args.samples_per_epoch,
                    val_fold=args.val_fold,
                ),
            )
            summary["stage3_l"][fusion] = stage_run(project_root, config_path)

        summary["stage3_ranking"] = {
            "ordered": sorted(summary["stage3_l"], key=lambda key: float(summary["stage3_l"][key]), reverse=True)
        }

    summary_path = funnel_dir / "latest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[funnel] summary saved to {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
