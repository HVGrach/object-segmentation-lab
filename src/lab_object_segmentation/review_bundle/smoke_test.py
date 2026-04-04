#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the pseudo-label review bundle.")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        required=True,
        help="Path to the extracted pseudo-label review bundle.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the temporary HTTP server.",
    )
    return parser.parse_args()


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=20) as response:
        return response.read().decode("utf-8")


def main() -> int:
    args = parse_args()
    bundle_dir = args.bundle_dir.resolve()
    review_dir = bundle_dir / "review_app"
    dataset_dir = bundle_dir / "pseudo_label_dataset"
    dataset_zip = bundle_dir / "pseudo_labels_dataset.zip"
    smoke_dir = bundle_dir / "smoke_tests"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    required_paths = [
        review_dir / "index.html",
        review_dir / "app.js",
        review_dir / "styles.css",
        review_dir / "server.py",
        review_dir / "data" / "manifest.json",
        dataset_dir / "manifest.json",
        dataset_dir / "manifest.csv",
        dataset_zip,
        bundle_dir / "README.md",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Bundle is missing required files: {missing}")

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    if not entries:
        raise RuntimeError("Dataset manifest contains no entries.")

    for entry in entries[:5]:
        for key in ["image_relpath", "mask_relpath", "boundary_mask_relpath"]:
            path = bundle_dir / entry[key]
            if not path.exists():
                raise FileNotFoundError(f"Missing asset referenced by manifest: {path}")

    with zipfile.ZipFile(dataset_zip) as archive:
        members = set(archive.namelist())
        expected_member = "pseudo_label_dataset/manifest.json"
        if expected_member not in members:
            raise RuntimeError(f"ZIP archive is missing {expected_member}")

    server_cmd = [
        sys.executable,
        str(review_dir / "server.py"),
        "--root",
        str(bundle_dir),
        "--port",
        str(args.port),
    ]
    server = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    base_url = f"http://127.0.0.1:{args.port}"
    try:
        time.sleep(2.0)
        index_html = fetch_text(f"{base_url}/review_app/index.html")
        app_manifest = json.loads(fetch_text(f"{base_url}/review_app/data/manifest.json"))
        if "Pseudo-label review" not in index_html:
            raise RuntimeError("Index page did not contain the expected title.")
        if int(app_manifest.get("num_items", 0)) != int(manifest["num_items"]):
            raise RuntimeError("App manifest item count does not match dataset manifest.")
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)

    summary = {
        "status": "ok",
        "bundle_dir": str(bundle_dir),
        "num_items": int(manifest["num_items"]),
        "checked_entries": 5,
        "zip_archive": str(dataset_zip),
        "server_url": f"{base_url}/review_app/index.html",
    }
    (smoke_dir / "smoke_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
