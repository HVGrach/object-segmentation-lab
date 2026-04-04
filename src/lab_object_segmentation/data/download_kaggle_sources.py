from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import kagglehub

from lab_object_segmentation.common.paths import LAB1_DATASET_ROOT, LAB3_DATASET_ROOT

LAB3_TARGET = LAB3_DATASET_ROOT
LAB1_TARGET = LAB1_DATASET_ROOT

LAB3_HANDLE = "dl-lab-3-product-segmentation"
LAB1_NOTEBOOK_HANDLE = "packagemanager/pm-113281741-at-03-23-2026-18-04-38"


def ensure_empty_or_forced(path: Path, force: bool) -> None:
    if not path.exists():
        return
    if not any(path.iterdir()):
        return
    if not force:
        raise FileExistsError(
            f"{path} already contains files. Re-run with --force to replace it."
        )
    shutil.rmtree(path)


def copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def materialize_download(source: Path, target: Path) -> None:
    if source.is_dir():
        children = [p for p in source.iterdir()]
        if len(children) == 1 and children[0].is_dir() and children[0].name == target.name:
            copy_tree_contents(children[0], target)
            return

        if source.name == target.name:
            copy_tree_contents(source, target)
            return

        copy_tree_contents(source, target)
        return

    if source.suffix.lower() == ".zip":
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with ZipFile(source) as archive:
                archive.extractall(tmp_path)
            materialize_download(tmp_path, target)
        return

    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target / source.name)


def download_lab3(force: bool) -> None:
    ensure_empty_or_forced(LAB3_TARGET, force)
    print(f"[lab3] downloading competition source: {LAB3_HANDLE}")
    source_path = Path(kagglehub.competition_download(LAB3_HANDLE, force_download=force))
    print(f"[lab3] cached source: {source_path}")
    materialize_download(source_path, LAB3_TARGET)
    print(f"[lab3] ready at: {LAB3_TARGET}")


def download_lab1(force: bool) -> None:
    ensure_empty_or_forced(LAB1_TARGET, force)
    print(f"[lab1] downloading notebook output: {LAB1_NOTEBOOK_HANDLE}")
    source_path = Path(
        kagglehub.notebook_output_download(LAB1_NOTEBOOK_HANDLE, force_download=force)
    )
    print(f"[lab1] cached source: {source_path}")
    materialize_download(source_path, LAB1_TARGET)
    print(f"[lab1] ready at: {LAB1_TARGET}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Kaggle sources that are intentionally excluded from the Docker image."
    )
    parser.add_argument(
        "--only",
        choices=["all", "lab1", "lab3"],
        default="all",
        help="Choose which source to fetch.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing downloaded data inside the target directories.",
    )
    args = parser.parse_args()

    try:
        if args.only in {"all", "lab3"}:
            download_lab3(force=args.force)
        if args.only in {"all", "lab1"}:
            download_lab1(force=args.force)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
