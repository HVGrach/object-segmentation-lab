#!/usr/bin/env python3
"""Run a lightweight smoke test for the SegFormer semi-supervised notebook pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from lab_object_segmentation.common.paths import LAB3_DATASET_ROOT, NOTEBOOKS_ROOT, RUNS_ROOT
from lab_object_segmentation.segformer_semisup.run_notebook import main as run_notebook_main
from lab_object_segmentation.segformer_semisup.utils import (
    boundary_loss_with_reliability,
    build_pseudo_stats,
    should_accept_pseudo_sample,
    split_labeled_samples_grouped,
)


NOTEBOOK_PATH = NOTEBOOKS_ROOT / "research" / "segformer_boundary_semisup_macos.ipynb"
SMOKE_ARTIFACT_ROOT = RUNS_ROOT / "segformer_boundary_semisup_smoke"


def collect_labeled_pairs() -> list[tuple[Path, Path]]:
    images_dir = LAB3_DATASET_ROOT / "train" / "images"
    masks_dir = LAB3_DATASET_ROOT / "train" / "masks"
    image_map = {path.stem: path for path in sorted(images_dir.rglob("*")) if path.is_file()}
    pairs = []
    for mask_path in sorted(masks_dir.rglob("*")):
        if not mask_path.is_file():
            continue
        image_path = image_map.get(mask_path.stem)
        if image_path is not None:
            pairs.append((image_path, mask_path))
    if not pairs:
        raise RuntimeError("No labeled pairs found for smoke test.")
    return pairs


def run_synthetic_checks() -> None:
    pairs = collect_labeled_pairs()
    train_samples, val_samples = split_labeled_samples_grouped(pairs, val_split=0.15, seed=42)
    train_groups = {sample[0].name.split("_", 1)[0] for sample in train_samples}
    val_groups = {sample[0].name.split("_", 1)[0] for sample in val_samples}
    overlap = train_groups & val_groups
    if overlap:
        raise AssertionError(f"Grouped split leaked groups into validation: {sorted(overlap)[:5]}")

    pseudo_mask = torch.tensor([[0, 1], [1, 1]], dtype=torch.uint8).numpy()
    confidence = torch.tensor([[0.99, 0.97], [0.96, 0.95]], dtype=torch.float32).numpy()
    uncertainty = torch.tensor([[0.01, 0.02], [0.03, 0.04]], dtype=torch.float32).numpy()
    stats = build_pseudo_stats(
        pseudo_mask=pseudo_mask,
        confidence=confidence,
        uncertainty=uncertainty,
        confidence_threshold=0.85,
        uncertainty_threshold=0.15,
    )
    accepted = should_accept_pseudo_sample(
        stats,
        min_reliable_ratio=0.90,
        min_mean_confidence=0.95,
        min_fg_reliable_ratio=0.80,
        min_object_ratio=0.01,
        max_object_ratio=0.80,
    )
    if not accepted:
        raise AssertionError("Synthetic high-confidence pseudo sample should have been accepted.")

    pred_boundary = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    target_boundary = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    zero_reliability = torch.zeros((1, 2, 2), dtype=torch.float32)
    masked_loss = boundary_loss_with_reliability(
        pred_boundary=pred_boundary,
        target_boundary=target_boundary,
        reliability_mask=zero_reliability,
        pos_weight=1.0,
    )
    if float(masked_loss.item()) != 0.0:
        raise AssertionError("Boundary loss should be zero when reliability mask is empty.")

    print("[smoke] synthetic checks passed")


def verify_outputs(executed_notebook: Path) -> Path:
    required_files = [
        NOTEBOOK_PATH,
        executed_notebook,
        SMOKE_ARTIFACT_ROOT / "metrics" / "phase1_history.json",
        SMOKE_ARTIFACT_ROOT / "metrics" / "pseudo_iteration_0.json",
        SMOKE_ARTIFACT_ROOT / "metrics" / "phase3_iteration_0.json",
        SMOKE_ARTIFACT_ROOT / "phase1" / "best.pt",
        SMOKE_ARTIFACT_ROOT / "phase3" / "iteration_0" / "best.pt",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Smoke test missing expected outputs:\n" + "\n".join(missing))

    summary_path = SMOKE_ARTIFACT_ROOT / "metrics" / "pseudo_iteration_0.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))["summary"]
    print("[smoke] pseudo iteration summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary_path


def main() -> int:
    run_synthetic_checks()

    if SMOKE_ARTIFACT_ROOT.exists():
        shutil.rmtree(SMOKE_ARTIFACT_ROOT)

    executed_notebook = SMOKE_ARTIFACT_ROOT / "segformer_boundary_semisup_smoke.executed.ipynb"
    SMOKE_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    run_notebook_main(
        [
            "--smoke",
            "--skip-pip",
            "--output-notebook",
            str(executed_notebook),
        ]
    )

    summary_path = verify_outputs(executed_notebook)
    print(f"[smoke] validation summary path: {summary_path}")
    print("[smoke] SegFormer semi-supervised smoke test completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
