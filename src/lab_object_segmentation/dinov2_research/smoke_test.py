#!/usr/bin/env python3
"""End-to-end smoke test for the DINOv2 research pipeline on Metal/CUDA/CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lab_object_segmentation.common.paths import CONFIGS_ROOT, PROJECT_ROOT, RUNS_ROOT
from lab_object_segmentation.dinov2_research.cache_features import main as cache_features_main
from lab_object_segmentation.dinov2_research.config import Config
from lab_object_segmentation.dinov2_research.train import main as train_main
from lab_object_segmentation.segformer_semisup.generate_pseudo_labels import main as generate_pseudo_labels_main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a full smoke test for the DINOv2 Metal pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_ROOT / "dinov2" / "smoke_test.yaml",
        help="Training config used for the smoke run.",
    )
    parser.add_argument("--pseudo-limit", type=int, default=2, help="How many unlabeled images to pseudo-label.")
    parser.add_argument("--overwrite-cache", action="store_true", help="Force recaching of smoke features.")
    return parser.parse_args(argv)


def build_smoke_stems(cfg: Config) -> list[str]:
    train_cameras, val_cameras = cfg.load_camera_split()
    image_paths = sorted(cfg.train_images_dir.glob("*.jpg"))

    train_groups: dict[str, list[str]] = {}
    all_train_stems: list[str] = []
    all_val_stems: list[str] = []
    for path in image_paths:
        camera = path.stem.split("_")[0]
        if camera in train_cameras:
            train_groups.setdefault(camera, []).append(path.stem)
            all_train_stems.append(path.stem)
        elif camera in val_cameras:
            all_val_stems.append(path.stem)

    per_camera = max(2, cfg.debug_samples // max(1, len(train_groups)))
    stems = set()
    for camera in sorted(train_groups):
        stems.update(train_groups[camera][:per_camera])

    stems.update(all_train_stems[: max(1, cfg.debug_samples)])
    stems.update(all_val_stems[: cfg.debug_samples])
    return sorted(stems)


def verify_outputs(cfg: Config, pseudo_output_root: Path, expected_cache_stems: list[str]):
    cache_manifest = cfg.cache_dir / "cache_manifest.json"
    best_ckpt = cfg.save_dir / "checkpoints" / "best.pt"
    latest_ckpt = cfg.save_dir / "checkpoints" / "latest.pt"
    history = cfg.save_dir / "history.json"
    preflight = cfg.preflight_summary_path
    results_table = cfg.results_table_path
    pseudo_summary = pseudo_output_root / "summary.json"

    required_files = [cache_manifest, best_ckpt, latest_ckpt, history, preflight, results_table, pseudo_summary]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Smoke test missing expected outputs:\n" + "\n".join(missing))

    cached_paths = [cfg.cache_dir / f"{stem}.pt" for stem in expected_cache_stems]
    missing_cached = [str(path) for path in cached_paths if not path.exists()]
    if missing_cached:
        raise FileNotFoundError("Smoke test missing cached feature files:\n" + "\n".join(missing_cached[:20]))

    summary = {
        "config": str(cfg.save_dir / "config.yaml"),
        "experiment_dir": str(cfg.save_dir),
        "cache_dir": str(cfg.cache_dir),
        "num_cached_expected": len(expected_cache_stems),
        "pseudo_output_root": str(pseudo_output_root),
        "results_table": str(results_table),
    }
    summary_path = cfg.save_dir / "smoke_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[smoke] validation summary saved to {summary_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = PROJECT_ROOT
    cfg = Config.from_yaml(args.config)
    cfg.set_project_root(project_root)

    smoke_root = RUNS_ROOT / "dinov2_research" / "smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    stems_file = smoke_root / "smoke_cache_stems.txt"
    stems = build_smoke_stems(cfg)
    stems_file.write_text("\n".join(stems) + "\n", encoding="utf-8")

    cache_args = [
        "--input-dir",
        str(cfg.train_images_dir),
        "--stems-file",
        str(stems_file),
        "--backbone-size",
        cfg.backbone_size,
        "--img-size",
        str(cfg.img_size),
        "--dtype",
        cfg.cache_dtype,
    ]
    if args.overwrite_cache:
        cache_args.append("--overwrite")
    cache_features_main(cache_args)

    pseudo_output_root = smoke_root / "pseudo_unlabeled"
    generate_pseudo_labels_main(
        [
            "--dataset-name",
            "unlabeled",
            "--output-root",
            str(pseudo_output_root),
            "--limit",
            str(args.pseudo_limit),
            "--tta-n",
            "2",
            "--save-all",
        ]
    )

    train_main(["--config", str(args.config)])

    verify_outputs(cfg, pseudo_output_root, stems)
    print("[smoke] pipeline smoke test completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
