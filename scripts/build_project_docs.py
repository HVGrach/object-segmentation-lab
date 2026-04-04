#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = PROJECT_ROOT / "artifacts" / "public"
DOCS_GENERATED_ROOT = PROJECT_ROOT / "docs" / "generated"
README_PATH = PROJECT_ROOT / "README.md"
REGISTRY_MD_PATH = PROJECT_ROOT / "docs" / "experiment_registry.md"

README_MARKER_START = "<!-- AUTO-GENERATED:BENCHMARK_SNAPSHOT_START -->"
README_MARKER_END = "<!-- AUTO-GENERATED:BENCHMARK_SNAPSHOT_END -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build generated docs from publish-safe metadata.")
    parser.add_argument("--check", action="store_true", help="Fail instead of writing when generated files are stale.")
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def render_benchmark_markdown(snapshot: dict) -> str:
    lines = []
    for item in snapshot["highlights"]:
        lines.append(f"- **{item['label']}**: `{item['track_title']}` with {item['metric']}. {item['note']}")
    lines.append("")
    lines.append("| Track | Status | Headline metric | Quick entrypoint |")
    lines.append("| --- | --- | --- | --- |")
    for row in snapshot["table"]:
        lines.append(
            f"| {row['track']} | {row['status']} | {row['headline_metric']} | `{row['entrypoint']}` |"
        )
    return "\n".join(lines)


def render_registry_markdown(registry: dict) -> str:
    lines = [
        "# Experiment Registry",
        "",
        "This document is generated from the publish-safe metadata in `artifacts/public/`.",
        "",
    ]
    for track in registry["tracks"]:
        lines.append(f"## {track['title']}")
        lines.append("")
        lines.append(f"- Status: `{track['status']}`")
        lines.append(f"- Category: `{track['category']}`")
        lines.append(f"- Summary: {track['headline']}")
        lines.append("")
        lines.append("### Key Results")
        lines.append("")
        key_results = json.dumps(track["key_results"], indent=2, ensure_ascii=False)
        lines.append("```json")
        lines.append(key_results)
        lines.append("```")
        lines.append("")
        lines.append("### Commands")
        lines.append("")
        for label, command in track["commands"].items():
            lines.append(f"- `{label}`: `{command}`")
        lines.append("")
        lines.append("### Source Artifacts")
        lines.append("")
        for artifact in track["artifacts"]:
            lines.append(f"- `{artifact}`")
        warnings = track.get("warnings", [])
        if warnings:
            lines.append("")
            lines.append("### Warnings")
            lines.append("")
            for warning in warnings:
                lines.append(f"- {warning}")
        lines.append("")
    if registry.get("copied_previews"):
        lines.append("## Selected Preview Assets")
        lines.append("")
        for item in registry["copied_previews"]:
            lines.append(f"- `{item['public_path']}` (from `{item['source']}`)")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def replace_marker_block(text: str, start_marker: str, end_marker: str, replacement: str) -> str:
    if start_marker not in text or end_marker not in text:
        raise ValueError(f"Could not locate markers {start_marker} / {end_marker}")
    prefix, rest = text.split(start_marker, 1)
    _, suffix = rest.split(end_marker, 1)
    return f"{prefix}{start_marker}\n{replacement}\n{end_marker}{suffix}"


def write_if_changed(path: Path, content: str, check: bool) -> bool:
    existing = path.read_text(encoding="utf-8") if path.exists() else None
    if existing == content:
        return False
    if check:
        raise SystemExit(f"Generated file is stale: {path.relative_to(PROJECT_ROOT)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def main() -> int:
    registry = read_json(PUBLIC_ROOT / "experiment_registry.json")
    snapshot = read_json(PUBLIC_ROOT / "benchmark_snapshot.json")

    registry_json = json.dumps(registry, indent=2, ensure_ascii=False) + "\n"
    snapshot_json = json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n"
    registry_md = render_registry_markdown(registry)

    write_if_changed(DOCS_GENERATED_ROOT / "experiment_registry.json", registry_json, check=args.check)
    write_if_changed(DOCS_GENERATED_ROOT / "benchmark_snapshot.json", snapshot_json, check=args.check)
    write_if_changed(REGISTRY_MD_PATH, registry_md, check=args.check)

    readme = README_PATH.read_text(encoding="utf-8")
    benchmark_block = render_benchmark_markdown(snapshot)
    updated_readme = replace_marker_block(readme, README_MARKER_START, README_MARKER_END, benchmark_block)
    write_if_changed(README_PATH, updated_readme, check=args.check)

    print("Generated docs are up to date.")
    return 0


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main())
