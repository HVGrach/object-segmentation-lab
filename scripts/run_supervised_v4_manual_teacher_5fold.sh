#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p artifacts/runs/supervised_v4/logs

PYTHON_BIN="${PYTHON_BIN:-python}"
N_SPLITS="${N_SPLITS:-5}"
FOLDS="${FOLDS:-0 1 2 3 4}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MANUAL_ROOT="${MANUAL_ROOT:-data/derived/imported_coco/manual_products_segmentation_v1}"
TEACHER_ROOT="${TEACHER_ROOT:-artifacts/runs/unimatch_v2_wide6/night_20260405_unimatchv2_wide6_tta_sahi/accepted_teacher_predictions}"
IMAGE_SIZE="${IMAGE_SIZE:-320}"
EPOCHS="${EPOCHS:-35}"
PHYSICAL_BATCH_SIZE="${PHYSICAL_BATCH_SIZE:-8}"

if [[ ! -d "$MANUAL_ROOT" ]]; then
    echo "Manual dataset root not found: $MANUAL_ROOT" >&2
    exit 1
fi

if [[ ! -d "$TEACHER_ROOT" ]]; then
    echo "Teacher-approved pseudo root not found: $TEACHER_ROOT" >&2
    exit 1
fi

STARTED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
echo " SUPERVISED_V4 5-FOLD MANUAL+TEACHER STARTED: $STARTED_AT"
echo "=============================================="
echo "n_splits=$N_SPLITS"
echo "folds=$FOLDS"
echo "manual_root=$MANUAL_ROOT"
echo "teacher_root=$TEACHER_ROOT"
echo ""

run_fold() {
    local fold="$1"
    local run_name="segformer_b2_${N_SPLITS}fold_fold${fold}_320_manual_v1_plus_teacher075"
    local run_dir="artifacts/runs/supervised_v4/${run_name}"
    local final_metrics="${run_dir}/final_tta_metrics.json"

    if [[ "$SKIP_EXISTING" == "1" && -f "$final_metrics" ]]; then
        echo "[skip] fold=$fold run_name=$run_name already has final_tta_metrics.json"
        return
    fi

    echo ""
    echo "====== FOLD $fold / $((N_SPLITS - 1)) ======"
    echo "  started: $(date '+%H:%M:%S')"
    echo "  run_name: $run_name"
    echo ""

    "$PYTHON_BIN" -u scripts/train_supervised_v4.py \
        --run-name "$run_name" \
        --n-splits "$N_SPLITS" \
        --fold "$fold" \
        --image-size "$IMAGE_SIZE" \
        --epochs "$EPOCHS" \
        --aug moderate \
        --label-smoothing 0.03 \
        --mask-loss lovasz_focal \
        --physical-batch-size "$PHYSICAL_BATCH_SIZE" \
        --extra-labeled-root "$MANUAL_ROOT" \
        --extra-labeled-root "$TEACHER_ROOT" \
        --extra-labeled-weight 1.0 \
        --extra-labeled-weight 0.75 \
        --extra-labeled-holdout-ratio 0.15 \
        --extra-labeled-holdout-ratio 0.0

    echo ""
    echo "  finished: $(date '+%H:%M:%S')"
    if [[ -f "$final_metrics" ]]; then
        "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
p = Path("$final_metrics")
metrics = json.loads(p.read_text())
print(f"  final_tta: dice_tuned={metrics['dice_tuned']:.4f} mIoU={metrics['mIoU']:.4f} thr={metrics['best_threshold']:.2f}")
PY
    fi
    echo "====== FOLD $fold DONE ======"
}

for fold in $FOLDS; do
    run_fold "$fold"
done

echo ""
echo "=============================================="
echo " SUPERVISED_V4 5-FOLD MANUAL+TEACHER COMPLETE"
echo " Started:  $STARTED_AT"
echo " Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
