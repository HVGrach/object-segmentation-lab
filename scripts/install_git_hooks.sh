#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

git config core.hooksPath .githooks
chmod +x .githooks/pre-push

echo "Configured Git hooks for $(basename "$ROOT_DIR")."
echo "Pre-push will refresh public metadata and generated docs before every push."
