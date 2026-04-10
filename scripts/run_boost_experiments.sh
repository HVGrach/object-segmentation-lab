#!/usr/bin/env bash
# Two parallel experiments to push past 0.914 on public LB.
# Run on MacBook Pro M2 Max 32GB (MPS backend).
#
# Experiment A: MIT-B3 backbone (stronger features, same resolution)
# Experiment B: MIT-B2 at 384px (better boundary precision)
#
# Usage:
#   # Run both sequentially (safest for 32GB RAM):
#   bash scripts/run_boost_experiments.sh
#
#   # Run only experiment A:
#   bash scripts/run_boost_experiments.sh a
#
#   # Run only experiment B:
#   bash scripts/run_boost_experiments.sh b

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

MANUAL_ROOT="data/derived/imported_coco/manual_products_segmentation_v1"
TEACHER_ROOT="artifacts/runs/unimatch_v2_wide6/night_20260405_unimatchv2_wide6_tta_sahi/accepted_teacher_predictions"

if [[ ! -d "$MANUAL_ROOT" ]]; then
    echo "ERROR: Manual dataset not found: $MANUAL_ROOT" >&2
    exit 1
fi
if [[ ! -d "$TEACHER_ROOT" ]]; then
    echo "ERROR: Teacher pseudo root not found: $TEACHER_ROOT" >&2
    exit 1
fi

MODE="${1:-all}"

# ─── Experiment A: MIT-B3 backbone, 320px ───────────────────────────
run_experiment_a() {
    local run_name="segformer_b3_5fold_fold1_320_manual_v1_plus_teacher075"
    echo ""
    echo "=============================================="
    echo " EXPERIMENT A: MIT-B3 @ 320px (fold1)"
    echo " run_name: $run_name"
    echo " started:  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="

    "$PYTHON_BIN" -u scripts/train_supervised_v4.py \
        --run-name "$run_name" \
        --backbone "nvidia/mit-b3" \
        --n-splits 5 \
        --fold 1 \
        --image-size 320 \
        --epochs 35 \
        --aug moderate \
        --label-smoothing 0.03 \
        --mask-loss lovasz_focal \
        --physical-batch-size 6 \
        --effective-batch-size 18 \
        --extra-labeled-root "$MANUAL_ROOT" \
        --extra-labeled-root "$TEACHER_ROOT" \
        --extra-labeled-weight 1.0 \
        --extra-labeled-weight 0.75 \
        --extra-labeled-holdout-ratio 0.15 \
        --extra-labeled-holdout-ratio 0.0

    echo ""
    echo " EXPERIMENT A FINISHED: $(date '+%Y-%m-%d %H:%M:%S')"
    local metrics="artifacts/runs/supervised_v4/${run_name}/final_tta_metrics.json"
    if [[ -f "$metrics" ]]; then
        "$PYTHON_BIN" -c "
import json
m = json.loads(open('$metrics').read())
print(f'  >> dice_tuned={m[\"dice_tuned\"]:.4f}  mIoU={m[\"mIoU\"]:.4f}  thr={m[\"best_threshold\"]:.2f}')
"
    fi
    echo "=============================================="
}

# ─── Experiment B: MIT-B2 backbone, 384px ───────────────────────────
run_experiment_b() {
    local run_name="segformer_b2_5fold_fold1_384_manual_v1_plus_teacher075"
    echo ""
    echo "=============================================="
    echo " EXPERIMENT B: MIT-B2 @ 384px (fold1)"
    echo " run_name: $run_name"
    echo " started:  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="

    "$PYTHON_BIN" -u scripts/train_supervised_v4.py \
        --run-name "$run_name" \
        --backbone "nvidia/mit-b2" \
        --n-splits 5 \
        --fold 1 \
        --image-size 384 \
        --epochs 35 \
        --aug moderate \
        --label-smoothing 0.03 \
        --mask-loss lovasz_focal \
        --physical-batch-size 6 \
        --effective-batch-size 18 \
        --extra-labeled-root "$MANUAL_ROOT" \
        --extra-labeled-root "$TEACHER_ROOT" \
        --extra-labeled-weight 1.0 \
        --extra-labeled-weight 0.75 \
        --extra-labeled-holdout-ratio 0.15 \
        --extra-labeled-holdout-ratio 0.0

    echo ""
    echo " EXPERIMENT B FINISHED: $(date '+%Y-%m-%d %H:%M:%S')"
    local metrics="artifacts/runs/supervised_v4/${run_name}/final_tta_metrics.json"
    if [[ -f "$metrics" ]]; then
        "$PYTHON_BIN" -c "
import json
m = json.loads(open('$metrics').read())
print(f'  >> dice_tuned={m[\"dice_tuned\"]:.4f}  mIoU={m[\"mIoU\"]:.4f}  thr={m[\"best_threshold\"]:.2f}')
"
    fi
    echo "=============================================="
}

case "$MODE" in
    a|A)
        run_experiment_a
        ;;
    b|B)
        run_experiment_b
        ;;
    all|"")
        run_experiment_a
        run_experiment_b
        echo ""
        echo "====== BOTH EXPERIMENTS COMPLETE ======"
        echo "Compare results:"
        echo "  B3@320: artifacts/runs/supervised_v4/segformer_b3_5fold_fold1_320_manual_v1_plus_teacher075/final_tta_metrics.json"
        echo "  B2@384: artifacts/runs/supervised_v4/segformer_b2_5fold_fold1_384_manual_v1_plus_teacher075/final_tta_metrics.json"
        echo ""
        echo "Next: submit the winner, then run 5-fold if it beats 0.9127 (fold1 baseline)."
        ;;
    *)
        echo "Usage: $0 [a|b|all]"
        exit 1
        ;;
esac
