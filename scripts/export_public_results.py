#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "artifacts" / "runs"
DELIVERABLES_ROOT = PROJECT_ROOT / "artifacts" / "deliverables"
PUBLIC_ROOT = PROJECT_ROOT / "artifacts" / "public"
SUPERVISED_V4_HYPOTHESIS_PUBLIC_PATH = PUBLIC_ROOT / "supervised_v4_hypothesis_suite_latest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export publish-safe experiment metadata from local artifacts.")
    parser.add_argument(
        "--allow-missing-sources",
        action="store_true",
        help="Reuse the previously exported public metadata when a local source directory is missing.",
    )
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sanitize_path_text(text: str) -> str:
    normalized = text.replace("\\", "/")
    project_root = PROJECT_ROOT.resolve().as_posix()
    if normalized.startswith(project_root):
        normalized = normalized[len(project_root) :].lstrip("/")

    if normalized.startswith("seg_runs/"):
        normalized = normalized.replace("seg_runs/", "artifacts/runs/", 1)
    if normalized.startswith("deliverables/"):
        normalized = normalized.replace("deliverables/", "artifacts/deliverables/", 1)

    if "dl-lab-3-product-segmentation" in normalized:
        return f"dataset::lab3/{Path(normalized).name}"
    if "dl-lab-1-image-classification" in normalized:
        return f"dataset::lab1/{Path(normalized).name}"

    if normalized.startswith("artifacts/") or normalized.startswith("notebooks/") or normalized.startswith("scripts/"):
        return normalized

    if normalized.startswith("/"):
        return Path(normalized).name

    return normalized


def sanitize_value(value):
    if isinstance(value, dict):
        return {key: sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, str) and ("/" in value or "\\" in value):
        return sanitize_path_text(value)
    return value


def to_float(value, default: float = 0.0) -> float:
    if value in (None, "", "null"):
        return default
    return float(value)


def infer_mask_loss(config: dict, run_name: str) -> str:
    explicit = config.get("mask_loss")
    if explicit:
        return explicit
    if "lovasz" in run_name:
        return "lovasz_focal"
    return "dice_focal"


def pick_best(rows: list[dict], key: str) -> dict:
    return max(rows, key=lambda item: to_float(item.get(key)))


def relative_artifact(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def load_existing_registry() -> dict:
    registry_path = PUBLIC_ROOT / "experiment_registry.json"
    if not registry_path.exists():
        return {}
    return read_json(registry_path)


def load_existing_supervised_v4_hypothesis_summary() -> dict | None:
    if not SUPERVISED_V4_HYPOTHESIS_PUBLIC_PATH.exists():
        return None
    return read_json(SUPERVISED_V4_HYPOTHESIS_PUBLIC_PATH)


def load_existing_tracks(existing_registry: dict) -> dict[str, dict]:
    if not existing_registry:
        return {}
    registry = existing_registry
    return {track["id"]: track for track in registry.get("tracks", [])}


def with_fallback(track_id: str, exporter, existing_tracks: dict[str, dict], allow_missing_sources: bool) -> dict:
    try:
        return exporter()
    except FileNotFoundError:
        if allow_missing_sources and track_id in existing_tracks:
            return deepcopy(existing_tracks[track_id])
        raise


def export_advanced_baseline() -> dict:
    run_root = RUNS_ROOT / "advanced_baseline"
    ensemble_summary = read_json(run_root / "ensemble_summary.json")
    model_rows = read_csv_rows(run_root / "model_summaries.csv")
    submission_summary = sanitize_value(read_json(run_root / "submission_lab3_test_images_mps_summary.json"))
    best_single = pick_best(model_rows, "oof_best_dice")

    return {
        "id": "advanced_baseline",
        "title": "Advanced Baseline",
        "status": "working",
        "category": "supervised",
        "headline": "Safest reproducible pipeline with grouped-by-camera CV, threshold tuning, TTA, and a lightweight ensemble.",
        "key_results": {
            "best_single_model": {
                "name": best_single["model_name"],
                "oof_dice": to_float(best_single["oof_best_dice"]),
                "oof_iou": to_float(best_single["oof_best_iou"]),
                "best_threshold": to_float(best_single["best_threshold"]),
            },
            "best_ensemble": {
                "candidate_name": ensemble_summary["candidate_name"],
                "oof_dice": ensemble_summary["oof_best_dice"],
                "oof_iou": ensemble_summary["oof_best_iou"],
                "best_threshold": ensemble_summary["best_threshold"],
            },
            "submission": {
                "num_images": submission_summary["num_images"],
                "device": submission_summary["device"],
                "tta_enabled": submission_summary["tta_enabled"],
            },
        },
        "artifacts": [
            relative_artifact(run_root / "ensemble_summary.json"),
            relative_artifact(run_root / "model_summaries.csv"),
            relative_artifact(run_root / "folds_summary.json"),
            relative_artifact(run_root / "submission_lab3_test_images_mps_summary.json"),
        ],
        "commands": {
            "smoke": "python scripts/run_lab3_ensemble_submission.py --limit 5",
            "full_inference": "python scripts/run_lab3_ensemble_submission.py",
        },
        "details": {
            "selected_models": ensemble_summary["selected_models"],
            "weights_by_model": ensemble_summary["weights_by_model"],
            "candidate_rankings": sanitize_value(ensemble_summary["candidate_rankings"]),
        },
    }


def export_supervised_v4() -> dict:
    run_root = RUNS_ROOT / "supervised_v4"
    run_rows = []
    for config_path in sorted(run_root.glob("*/config.json")):
        run_dir = config_path.parent
        metrics_path = run_dir / "final_tta_metrics.json"
        if not metrics_path.exists():
            continue
        config = read_json(config_path)
        metrics = read_json(metrics_path)
        run_rows.append(
            {
                "run_name": config.get("run_name", run_dir.name),
                "fold": config.get("fold"),
                "image_size": config.get("image_size"),
                "aug": config.get("aug", "unknown"),
                "mask_loss": infer_mask_loss(config, config.get("run_name", run_dir.name)),
                "dice_tuned": metrics["dice_tuned"],
                "dice": metrics["dice"],
                "mIoU": metrics["mIoU"],
                "best_threshold": metrics["best_threshold"],
                "tta": metrics["tta"],
            }
        )
    if not run_rows:
        raise FileNotFoundError("No final_tta_metrics.json files found under artifacts/runs/supervised_v4")

    best_run = max(run_rows, key=lambda item: (item["dice_tuned"], item["mIoU"]))
    oof_rows = read_json(run_root / "oof_ensemble_eval.json")
    best_oof = max(oof_rows, key=lambda item: (item["dice"], item["iou"]))
    soup_eval = read_json(run_root / "checkpoint_soup_fold1_eval.json")
    best_soup_name, best_soup_metrics = max(
        soup_eval["results"].items(),
        key=lambda item: (item[1]["dice_tuned"], item[1]["mIoU"]),
    )
    submission_summary = sanitize_value(read_json(run_root / "submission_supervised_v4_wide6_thr50_summary.json"))
    latest_hypothesis_suite = export_supervised_v4_hypothesis_suite()

    artifacts = [
        relative_artifact(run_root / "oof_ensemble_eval.json"),
        relative_artifact(run_root / "checkpoint_soup_fold1_eval.json"),
        relative_artifact(run_root / "submission_supervised_v4_wide6_thr50_summary.json"),
    ]
    key_results = {
        "best_run": best_run,
        "best_oof_ensemble": best_oof,
        "best_checkpoint_soup": {
            "name": best_soup_name,
            **best_soup_metrics,
        },
        "submission": {
            "preset": "wide6",
            "num_images": submission_summary["num_images"],
            "device": submission_summary["device"],
            "tta_enabled": submission_summary["tta_enabled"],
            "threshold": submission_summary["threshold"],
        },
    }

    if latest_hypothesis_suite is not None:
        write_json(SUPERVISED_V4_HYPOTHESIS_PUBLIC_PATH, sanitize_value(latest_hypothesis_suite))
        artifacts.extend(latest_hypothesis_suite["published_artifacts"])
        key_results["latest_hypothesis_suite"] = {
            "suite_tag": latest_hypothesis_suite["suite_tag"],
            "takeaway": latest_hypothesis_suite["takeaway"],
            "best_oof_overall": latest_hypothesis_suite["best_oof_overall"],
            "best_oof_with_tta": latest_hypothesis_suite["best_oof_with_tta"],
            "best_training_screen": latest_hypothesis_suite["best_training_screen"],
        }

    return {
        "id": "supervised_v4",
        "title": "Supervised V4",
        "status": "working",
        "category": "supervised",
        "headline": "Strongest supervised-only track built around SegFormer-B2 and a custom V4 decoder.",
        "key_results": key_results,
        "artifacts": artifacts,
        "commands": {
            "smoke": "python scripts/predict_supervised_v4_ensemble.py --preset wide6 --limit 5",
            "train_help": "python scripts/train_supervised_v4.py --help",
        },
        "details": {
            "top_runs_by_dice_tuned": sorted(run_rows, key=lambda item: (item["dice_tuned"], item["mIoU"]), reverse=True)[:6],
            "oof_grid": oof_rows,
            "latest_hypothesis_suite_path": (
                relative_artifact(SUPERVISED_V4_HYPOTHESIS_PUBLIC_PATH) if latest_hypothesis_suite is not None else None
            ),
        },
    }


def _best_metric_row(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    return max(rows, key=lambda row: (to_float(row.get("dice")), to_float(row.get("iou"))))


def _latest_hypothesis_suite_dir() -> Path | None:
    suite_root = RUNS_ROOT / "supervised_v4" / "hypothesis_suite"
    if not suite_root.exists():
        return None
    suite_dirs = [path for path in suite_root.iterdir() if path.is_dir()]
    if not suite_dirs:
        return None
    return sorted(suite_dirs, key=lambda path: (path.name, path.stat().st_mtime))[-1]


def _summarize_oof_search_file(path: Path) -> dict:
    payload = read_json(path)
    settings = payload.get("settings", {})
    results = payload.get("results", [])
    best_per_combo = payload.get("best_per_combo") or []
    best_result = _best_metric_row(results)
    if best_result is not None:
        best_result = {
            **best_result,
            "source_file": relative_artifact(path),
        }

    return {
        "source_file": relative_artifact(path),
        "settings": settings,
        "best_result": best_result,
        "best_per_combo": sorted(
            best_per_combo,
            key=lambda row: (to_float(row.get("dice")), to_float(row.get("iou"))),
            reverse=True,
        )[:6],
    }


def _summarize_training_screen_run(run_dir: Path) -> dict:
    config = read_json(run_dir / "config.json")
    metrics = read_json(run_dir / "final_tta_metrics.json")
    return {
        "run_name": config.get("run_name", run_dir.name),
        "fold": config.get("fold"),
        "image_size": config.get("image_size"),
        "aug": config.get("aug", "unknown"),
        "mask_loss": infer_mask_loss(config, config.get("run_name", run_dir.name)),
        "resize_interpolation": config.get("resize_interpolation", "linear"),
        "finetune_from_checkpoint": bool(config.get("finetune_from")),
        "dice_tuned": metrics["dice_tuned"],
        "dice": metrics["dice"],
        "mIoU": metrics["mIoU"],
        "best_threshold": metrics["best_threshold"],
        "tta": metrics["tta"],
        "source_metrics_file": relative_artifact(run_dir / "final_tta_metrics.json"),
    }


def export_supervised_v4_hypothesis_suite() -> dict | None:
    suite_dir = _latest_hypothesis_suite_dir()
    if suite_dir is None:
        return load_existing_supervised_v4_hypothesis_summary()

    oof_paths = sorted(suite_dir.glob("oof_search*.json"))
    if not oof_paths:
        return load_existing_supervised_v4_hypothesis_summary()

    oof_file_summaries = [_summarize_oof_search_file(path) for path in oof_paths]
    oof_best_rows = [summary["best_result"] for summary in oof_file_summaries if summary["best_result"] is not None]
    best_oof_overall = _best_metric_row(oof_best_rows)
    best_oof_with_tta = _best_metric_row(
        [
            summary["best_result"]
            for summary in oof_file_summaries
            if summary["best_result"] is not None and summary["settings"].get("tta_enabled")
        ]
    )

    training_screen_rows = []
    for metrics_path in sorted((RUNS_ROOT / "supervised_v4").glob(f"screen_{suite_dir.name}_*/final_tta_metrics.json")):
        training_screen_rows.append(_summarize_training_screen_run(metrics_path.parent))
    training_screen_rows.sort(key=lambda row: (to_float(row.get("dice_tuned")), to_float(row.get("mIoU"))), reverse=True)
    best_training_screen = training_screen_rows[0] if training_screen_rows else None

    takeaway_parts = []
    if best_oof_overall is not None:
        tta_label = "with TTA" if best_oof_overall.get("tta_enabled") else "without TTA"
        takeaway_parts.append(
            f"{best_oof_overall['ensemble']} {best_oof_overall['aggregation']} {tta_label} reached Dice {best_oof_overall['dice']:.4f}"
        )
    if best_training_screen is not None:
        takeaway_parts.append(
            f"the strongest short screen was {best_training_screen['run_name']} with dice_tuned {best_training_screen['dice_tuned']:.4f}"
        )

    published_artifacts = [relative_artifact(SUPERVISED_V4_HYPOTHESIS_PUBLIC_PATH), relative_artifact(suite_dir / "summary.txt")]
    if best_oof_overall is not None:
        published_artifacts.append(best_oof_overall["source_file"])
    if best_training_screen is not None:
        published_artifacts.append(best_training_screen["source_metrics_file"])

    return {
        "suite_tag": suite_dir.name,
        "suite_dir": relative_artifact(suite_dir),
        "summary_file": relative_artifact(suite_dir / "summary.txt"),
        "takeaway": "; ".join(takeaway_parts) if takeaway_parts else "Latest supervised_v4 hypothesis suite summary.",
        "best_oof_overall": best_oof_overall,
        "best_oof_with_tta": best_oof_with_tta,
        "best_training_screen": best_training_screen,
        "published_artifacts": published_artifacts,
        "oof_file_summaries": oof_file_summaries,
        "training_screen_leaderboard": training_screen_rows,
    }


def export_segformer_semisup() -> dict:
    run_root = RUNS_ROOT / "segformer_boundary_semisup_macos"
    phase1_history = read_json(run_root / "metrics" / "phase1_history.json")
    best_phase1 = max(
        (row for row in phase1_history if row.get("val_mIoU") is not None),
        key=lambda row: row["val_mIoU"],
    )
    iteration_summaries = read_json(run_root / "metrics" / "iteration_summaries.json")
    best_iteration = max(iteration_summaries, key=lambda row: row["phase3_best_mIoU"])
    stricter_pseudo = read_json(RUNS_ROOT / "segformer_boundary_semisup_macos_2026_03_31_cachefix" / "metrics" / "pseudo_iteration_0.json")
    bundle_manifest = sanitize_value(read_json(DELIVERABLES_ROOT / "pseudo_label_review_bundle" / "bundle_manifest.json"))
    submission_summary = sanitize_value(read_json(run_root / "submission_lab3_test_images_segformer_summary.json"))

    return {
        "id": "segformer_boundary_semisup_macos",
        "title": "SegFormer Boundary Semi-Supervised",
        "status": "research",
        "category": "semi-supervised",
        "headline": "Boundary-aware SegFormer-B2 teacher-student pipeline with pseudo-label filtering and EMA.",
        "key_results": {
            "best_supervised_phase": {
                "epoch": best_phase1["epoch"],
                "val_mIoU": best_phase1["val_mIoU"],
                "val_pixel_acc": best_phase1["val_pixel_acc"],
                "val_boundary_f1": best_phase1["val_boundary_f1"],
            },
            "best_semi_supervised_iteration": best_iteration,
            "stricter_pseudo_filtering": stricter_pseudo["summary"],
            "review_bundle": {
                "num_items": bundle_manifest["num_items"],
                "source_breakdown": bundle_manifest["source_breakdown"],
                "dice_holdout_r2": bundle_manifest["dice_calibration"]["holdout_metrics"]["r2"],
                "dice_holdout_rmse": bundle_manifest["dice_calibration"]["holdout_metrics"]["rmse"],
            },
            "submission": {
                "num_images": submission_summary["num_images"],
                "device": submission_summary["device"],
                "tta_n": submission_summary["tta_n"],
            },
        },
        "artifacts": [
            relative_artifact(run_root / "metrics" / "phase1_history.json"),
            relative_artifact(run_root / "metrics" / "iteration_summaries.json"),
            relative_artifact(run_root / "submission_lab3_test_images_segformer_summary.json"),
            relative_artifact(DELIVERABLES_ROOT / "pseudo_label_review_bundle" / "bundle_manifest.json"),
        ],
        "commands": {
            "smoke": "python scripts/run_segformer_semisup_smoke_test.py",
            "submission_smoke": "python scripts/run_lab3_segformer_submission.py --limit 5",
        },
        "details": {
            "iteration_summaries": iteration_summaries,
            "submission_preview": submission_summary["stats_preview"],
        },
    }


def export_dinov2_research() -> dict:
    results_path = RUNS_ROOT / "dinov2_research" / "results_table.csv"
    rows = read_csv_rows(results_path)
    deduped: dict[str, dict] = {}
    for row in rows:
        key = row["experiment"]
        if key not in deduped or to_float(row["best_val_iou"]) > to_float(deduped[key]["best_val_iou"]):
            deduped[key] = row
    best_row = max(deduped.values(), key=lambda row: to_float(row["best_val_iou"]))
    leaderboard = sorted(deduped.values(), key=lambda row: to_float(row["best_val_iou"]), reverse=True)

    return {
        "id": "dinov2_research",
        "title": "DINOv2 Research",
        "status": "research",
        "category": "proxy-research",
        "headline": "Frozen DINOv2 feature-cache track used for low-cost fusion ablations and hypothesis screening.",
        "key_results": {
            "best_ablation": {
                "experiment": best_row["experiment"],
                "fusion": best_row["fusion"],
                "backbone": best_row["backbone"],
                "best_val_iou": to_float(best_row["best_val_iou"]),
                "best_val_dice": to_float(best_row["best_val_dice"]),
                "best_epoch": int(float(best_row["best_epoch"])),
            }
        },
        "artifacts": [relative_artifact(results_path)],
        "commands": {
            "smoke": "python scripts/run_dinov2_smoke_test.py --overwrite-cache",
            "funnel": "python scripts/run_dinov2_funnel.py --cache-missing",
        },
        "details": {
            "leaderboard": [
                {
                    "experiment": row["experiment"],
                    "fusion": row["fusion"],
                    "backbone": row["backbone"],
                    "best_val_iou": to_float(row["best_val_iou"]),
                    "best_val_dice": to_float(row["best_val_dice"]),
                }
                for row in leaderboard
            ]
        },
    }


def build_benchmark_snapshot(tracks: list[dict]) -> dict:
    track_map = {track["id"]: track for track in tracks}
    advanced = track_map["advanced_baseline"]
    supervised = track_map["supervised_v4"]
    semisup = track_map["segformer_boundary_semisup_macos"]
    dinov2 = track_map["dinov2_research"]
    latest_hypothesis_suite = supervised["key_results"].get("latest_hypothesis_suite")

    highlights = [
        {
            "label": "Safest reproducible path",
            "track_id": advanced["id"],
            "track_title": advanced["title"],
            "metric": f"OOF Dice {advanced['key_results']['best_ensemble']['oof_dice']:.4f}",
            "note": "Grouped-by-camera 3-fold ensemble with EMA, threshold tuning, and TTA.",
        },
        {
            "label": "Strongest supervised signal",
            "track_id": supervised["id"],
            "track_title": supervised["title"],
            "metric": f"dice_tuned {supervised['key_results']['best_run']['dice_tuned']:.4f}",
            "note": supervised["key_results"]["best_run"]["run_name"],
        },
    ]

    if latest_hypothesis_suite is not None:
        best_oof = latest_hypothesis_suite.get("best_oof_overall") or {}
        best_screen = latest_hypothesis_suite.get("best_training_screen") or {}
        highlights.append(
            {
                "label": "Latest aug sweep",
                "track_id": supervised["id"],
                "track_title": supervised["title"],
                "metric": f"OOF Dice {to_float(best_oof.get('dice')):.4f}",
                "note": (
                    f"{best_oof.get('ensemble', 'custom')} {best_oof.get('aggregation', 'weighted_mean')} "
                    f"{'with TTA' if best_oof.get('tta_enabled') else 'without TTA'}; "
                    f"best short screen {best_screen.get('run_name', 'n/a')} "
                    f"dice_tuned {to_float(best_screen.get('dice_tuned')):.4f}."
                ),
            }
        )

    highlights.extend(
        [
            {
                "label": "Best semi-supervised local result",
                "track_id": semisup["id"],
                "track_title": semisup["title"],
                "metric": f"val_mIoU {semisup['key_results']['best_semi_supervised_iteration']['phase3_best_mIoU']:.4f}",
                "note": "Iteration 0 improved over the supervised phase, iteration 1 regressed.",
            },
            {
                "label": "Best proxy-research result",
                "track_id": dinov2["id"],
                "track_title": dinov2["title"],
                "metric": f"best_val_iou {dinov2['key_results']['best_ablation']['best_val_iou']:.4f}",
                "note": f"{dinov2['key_results']['best_ablation']['fusion']} fusion on {dinov2['key_results']['best_ablation']['backbone']}.",
            },
        ]
    )

    return {
        "highlights": highlights,
        "table": [
            {
                "track": advanced["title"],
                "status": advanced["status"],
                "headline_metric": f"OOF Dice {advanced['key_results']['best_ensemble']['oof_dice']:.4f}",
                "entrypoint": advanced["commands"]["smoke"],
            },
            {
                "track": supervised["title"],
                "status": supervised["status"],
                "headline_metric": f"dice_tuned {supervised['key_results']['best_run']['dice_tuned']:.4f}",
                "entrypoint": supervised["commands"]["smoke"],
            },
            {
                "track": semisup["title"],
                "status": semisup["status"],
                "headline_metric": f"val_mIoU {semisup['key_results']['best_semi_supervised_iteration']['phase3_best_mIoU']:.4f}",
                "entrypoint": semisup["commands"]["smoke"],
            },
            {
                "track": dinov2["title"],
                "status": dinov2["status"],
                "headline_metric": f"best_val_iou {dinov2['key_results']['best_ablation']['best_val_iou']:.4f}",
                "entrypoint": dinov2["commands"]["smoke"],
            },
        ],
    }


def copy_selected_previews(existing_registry: dict, allow_missing_sources: bool) -> list[dict]:
    preview_specs = [
        (
            RUNS_ROOT / "segformer_boundary_semisup_macos" / "preview" / "phase3_iter0_best4.png",
            PUBLIC_ROOT / "previews" / "segformer" / "phase3_iter0_best4.png",
        ),
        (
            RUNS_ROOT / "segformer_boundary_semisup_macos" / "preview" / "phase3_iter0_worst4.png",
            PUBLIC_ROOT / "previews" / "segformer" / "phase3_iter0_worst4.png",
        ),
        (
            DELIVERABLES_ROOT / "pseudo_label_review_bundle" / "demo" / "demo_card_1.png",
            PUBLIC_ROOT / "previews" / "review_bundle" / "demo_card_1.png",
        ),
        (
            DELIVERABLES_ROOT / "pseudo_label_review_bundle" / "demo" / "demo_card_2.png",
            PUBLIC_ROOT / "previews" / "review_bundle" / "demo_card_2.png",
        ),
        (
            DELIVERABLES_ROOT / "pseudo_label_review_bundle" / "demo" / "demo_card_3.png",
            PUBLIC_ROOT / "previews" / "review_bundle" / "demo_card_3.png",
        ),
    ]

    copied = []
    missing_sources = False
    for source, destination in preview_specs:
        if not source.exists():
            missing_sources = True
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(
            {
                "source": relative_artifact(source),
                "public_path": relative_artifact(destination),
                "size_bytes": destination.stat().st_size,
            }
        )
    if copied:
        return copied
    if allow_missing_sources and missing_sources:
        return deepcopy(existing_registry.get("copied_previews", []))
    return copied


def main() -> int:
    args = parse_args()
    existing_registry = load_existing_registry()
    existing_tracks = load_existing_tracks(existing_registry)

    tracks = [
        with_fallback("advanced_baseline", export_advanced_baseline, existing_tracks, args.allow_missing_sources),
        with_fallback("supervised_v4", export_supervised_v4, existing_tracks, args.allow_missing_sources),
        with_fallback(
            "segformer_boundary_semisup_macos",
            export_segformer_semisup,
            existing_tracks,
            args.allow_missing_sources,
        ),
        with_fallback("dinov2_research", export_dinov2_research, existing_tracks, args.allow_missing_sources),
    ]

    benchmark_snapshot = build_benchmark_snapshot(tracks)
    copied_previews = copy_selected_previews(existing_registry, args.allow_missing_sources)

    experiment_registry = {
        "schema_version": 1,
        "tracks": tracks,
        "copied_previews": copied_previews,
    }

    write_json(PUBLIC_ROOT / "experiment_registry.json", sanitize_value(experiment_registry))
    write_json(PUBLIC_ROOT / "benchmark_snapshot.json", sanitize_value(benchmark_snapshot))
    print(f"Exported public metadata to {PUBLIC_ROOT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
