from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a clean folder with only accepted teacher predictions from a completed run."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Completed run directory containing teacher_cache/summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder. Defaults to <run-dir>/accepted_teacher_predictions.",
    )
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy files instead of creating symlinks.",
    )
    return parser.parse_args()


def ensure_link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        import shutil

        shutil.copy2(src, dst)
        return
    os.symlink(src.resolve(), dst)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "teacher_cache" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing teacher cache summary: {summary_path}")

    output_dir = args.output_dir.resolve() if args.output_dir is not None else run_dir / "accepted_teacher_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    confidence_dir = output_dir / "confidence"

    teacher_cache = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = teacher_cache["rows"]
    accepted_rows = [row for row in rows if bool(row["accepted"])]
    if not accepted_rows:
        raise RuntimeError("No accepted teacher predictions found in the provided run.")

    manifest_rows: list[dict] = []
    for row in accepted_rows:
        image_path = Path(row["image_path"]).resolve()
        mask_path = Path(row["mask_path"]).resolve()
        confidence_path = Path(row["confidence_path"]).resolve()

        image_out = images_dir / image_path.name
        mask_out = masks_dir / mask_path.name
        confidence_out = confidence_dir / confidence_path.name

        ensure_link_or_copy(image_path, image_out, args.copy_files)
        ensure_link_or_copy(mask_path, mask_out, args.copy_files)
        ensure_link_or_copy(confidence_path, confidence_out, args.copy_files)

        manifest_rows.append(
            {
                "image_name": image_path.name,
                "image_path": str(image_out),
                "mask_path": str(mask_out),
                "confidence_path": str(confidence_out),
                "reliable_ratio": float(row["reliable_ratio"]),
                "mean_confidence": float(row["mean_confidence"]),
                "fg_reliable_ratio": float(row["fg_reliable_ratio"]),
                "area_ratio": float(row["area_ratio"]),
                "selection_score": float(row["selection_score"]),
                "accept_reason": str(row["accept_reason"]),
            }
        )

    summary = {
        "run_dir": str(run_dir),
        "teacher_cache_summary": str(summary_path),
        "output_dir": str(output_dir),
        "num_accepted": len(manifest_rows),
        "mode": "copy" if args.copy_files else "symlink",
        "source_teacher_summary": teacher_cache.get("summary", {}),
    }

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    save_json(output_dir / "manifest.json", {"summary": summary, "rows": manifest_rows})
    save_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
