#!/usr/bin/env python3
"""Build and execute the SegFormer semi-supervised notebook headlessly."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from lab_object_segmentation.common.paths import NOTEBOOKS_ROOT, PROJECT_ROOT, RUNS_ROOT
from lab_object_segmentation.segformer_semisup.build_notebook import main as build_notebook_main

NOTEBOOK_PATH = NOTEBOOKS_ROOT / "research" / "segformer_boundary_semisup_macos.ipynb"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the SegFormer semi-supervised notebook headlessly.")
    parser.add_argument("--smoke", action="store_true", help="Run with SEGFORMER_SMOKE_MODE=1.")
    parser.add_argument(
        "--output-notebook",
        type=Path,
        default=None,
        help="Path to save the executed notebook. Defaults under the active artifact root.",
    )
    parser.add_argument(
        "--artifact-subdir",
        type=str,
        default=None,
        help="Optional override for SEGFORMER_GDRIVE_SAVE_DIR.",
    )
    parser.add_argument("--skip-pip", action="store_true", help="Set SEGFORMER_SKIP_PIP=1 during notebook execution.")
    return parser.parse_args(argv)


def build_notebook() -> None:
    build_notebook_main()


def active_artifact_root(smoke: bool, artifact_subdir: str | None) -> Path:
    if artifact_subdir:
        return PROJECT_ROOT / artifact_subdir
    if smoke:
        return RUNS_ROOT / "segformer_boundary_semisup_smoke"
    return RUNS_ROOT / "segformer_boundary_semisup_macos"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_notebook()

    if args.smoke:
        os.environ["SEGFORMER_SMOKE_MODE"] = "1"
    if args.artifact_subdir:
        os.environ["SEGFORMER_GDRIVE_SAVE_DIR"] = args.artifact_subdir
    if args.skip_pip:
        os.environ["SEGFORMER_SKIP_PIP"] = "1"
    os.environ["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    output_path = args.output_notebook
    if output_path is None:
        output_path = active_artifact_root(args.smoke, args.artifact_subdir) / (
            "segformer_boundary_semisup_smoke.executed.ipynb" if args.smoke else "segformer_boundary_semisup.executed.ipynb"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with NOTEBOOK_PATH.open("r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()

    with output_path.open("w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    print(f"Executed notebook saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
