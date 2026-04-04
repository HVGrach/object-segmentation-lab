#!/usr/bin/env bash
# Overnight training script — ~7-8 hours on M2 Max 32GB
# Trains best recipes on all 3 folds with both loss variants
#
# Usage:
#   caffeinate -dimsu bash scripts/run_overnight.sh 2>&1 | tee artifacts/runs/supervised_v4/logs/overnight.log

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p artifacts/runs/supervised_v4/logs

SCRIPT="python -u scripts/train_supervised_v4.py"
STARTED_AT=$(date '+%Y-%m-%d %H:%M:%S')
echo "=============================================="
echo " OVERNIGHT TRAINING STARTED: $STARTED_AT"
echo "=============================================="
echo ""

run_step() {
    local step_num="$1"
    local desc="$2"
    shift 2
    echo ""
    echo "====== STEP $step_num: $desc ======"
    echo "  started: $(date '+%H:%M:%S')"
    echo "  command: $SCRIPT $*"
    echo ""
    $SCRIPT "$@"
    echo ""
    echo "  finished: $(date '+%H:%M:%S')"
    echo "====== STEP $step_num DONE ======"
    echo ""
}

# ============================================================
# PHASE 1: Finish Lovász experiment on fold 1
# ============================================================

echo ">>> PHASE 1: Complete Lovász fold 1"

# Step 1: Resume Lovász fold_1 320 training (from epoch 6 → 35)
run_step 1 "Lovász fold_1 320 (resume)" \
    --run-name segformer_b2_fold1_320_moderate_lovasz_ls003_ema \
    --fold 1 --image-size 320 --epochs 35 --aug moderate \
    --label-smoothing 0.03 --mask-loss lovasz_focal \
    --physical-batch-size 8 --resume

# Step 2: Finetune Lovász fold_1 → 384
run_step 2 "Lovász fold_1 384 finetune" \
    --run-name segformer_b2_fold1_384_finetune_lovasz \
    --fold 1 --image-size 384 --epochs 8 --aug moderate \
    --label-smoothing 0.03 --mask-loss lovasz_focal \
    --physical-batch-size 4 --eval-every-n-epochs 1 \
    --finetune-from artifacts/runs/supervised_v4/segformer_b2_fold1_320_moderate_lovasz_ls003_ema/best_model.pth

# ============================================================
# PHASE 2: dice_focal on folds 0 and 2 (proven recipe)
# ============================================================

echo ">>> PHASE 2: dice_focal on folds 0, 2"

# Step 3: dice_focal fold_0 320
run_step 3 "dice_focal fold_0 320" \
    --run-name segformer_b2_fold0_320_moderate_ls003_ema \
    --fold 0 --image-size 320 --epochs 35 --aug moderate \
    --label-smoothing 0.03 --mask-loss dice_focal \
    --physical-batch-size 8

# Step 4: Finetune fold_0 → 384
run_step 4 "dice_focal fold_0 384 finetune" \
    --run-name segformer_b2_fold0_384_finetune \
    --fold 0 --image-size 384 --epochs 8 --aug moderate \
    --label-smoothing 0.03 --mask-loss dice_focal \
    --physical-batch-size 4 --eval-every-n-epochs 1 \
    --finetune-from artifacts/runs/supervised_v4/segformer_b2_fold0_320_moderate_ls003_ema/best_model.pth

# Step 5: dice_focal fold_2 320
run_step 5 "dice_focal fold_2 320" \
    --run-name segformer_b2_fold2_320_moderate_ls003_ema \
    --fold 2 --image-size 320 --epochs 35 --aug moderate \
    --label-smoothing 0.03 --mask-loss dice_focal \
    --physical-batch-size 8

# Step 6: Finetune fold_2 → 384
run_step 6 "dice_focal fold_2 384 finetune" \
    --run-name segformer_b2_fold2_384_finetune \
    --fold 2 --image-size 384 --epochs 8 --aug moderate \
    --label-smoothing 0.03 --mask-loss dice_focal \
    --physical-batch-size 4 --eval-every-n-epochs 1 \
    --finetune-from artifacts/runs/supervised_v4/segformer_b2_fold2_320_moderate_ls003_ema/best_model.pth

# ============================================================
# PHASE 3: Lovász on folds 0 and 2 (for ensemble diversity)
# ============================================================

echo ">>> PHASE 3: Lovász on folds 0, 2"

# Step 7: Lovász fold_0 320
run_step 7 "Lovász fold_0 320" \
    --run-name segformer_b2_fold0_320_moderate_lovasz_ls003_ema \
    --fold 0 --image-size 320 --epochs 35 --aug moderate \
    --label-smoothing 0.03 --mask-loss lovasz_focal \
    --physical-batch-size 8

# Step 8: Finetune Lovász fold_0 → 384
run_step 8 "Lovász fold_0 384 finetune" \
    --run-name segformer_b2_fold0_384_finetune_lovasz \
    --fold 0 --image-size 384 --epochs 8 --aug moderate \
    --label-smoothing 0.03 --mask-loss lovasz_focal \
    --physical-batch-size 4 --eval-every-n-epochs 1 \
    --finetune-from artifacts/runs/supervised_v4/segformer_b2_fold0_320_moderate_lovasz_ls003_ema/best_model.pth

# Step 9: Lovász fold_2 320
run_step 9 "Lovász fold_2 320" \
    --run-name segformer_b2_fold2_320_moderate_lovasz_ls003_ema \
    --fold 2 --image-size 320 --epochs 35 --aug moderate \
    --label-smoothing 0.03 --mask-loss lovasz_focal \
    --physical-batch-size 8

# Step 10: Finetune Lovász fold_2 → 384
run_step 10 "Lovász fold_2 384 finetune" \
    --run-name segformer_b2_fold2_384_finetune_lovasz \
    --fold 2 --image-size 384 --epochs 8 --aug moderate \
    --label-smoothing 0.03 --mask-loss lovasz_focal \
    --physical-batch-size 4 --eval-every-n-epochs 1 \
    --finetune-from artifacts/runs/supervised_v4/segformer_b2_fold2_320_moderate_lovasz_ls003_ema/best_model.pth

# ============================================================
# SUMMARY
# ============================================================

echo ""
echo "=============================================="
echo " OVERNIGHT TRAINING COMPLETE"
echo " Started:  $STARTED_AT"
echo " Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
echo ""
echo "Results summary (final TTA metrics):"
echo ""

for run_dir in artifacts/runs/supervised_v4/*/; do
    tta_file="$run_dir/final_tta_metrics.json"
    if [ -f "$tta_file" ]; then
        dice_tuned=$(python3 -c "import json; d=json.load(open('$tta_file')); print(f\"{d['dice_tuned']:.4f}\")" 2>/dev/null || echo "N/A")
        echo "  $(basename "$run_dir"): dice_tuned=$dice_tuned"
    fi
done

echo ""
echo "All artifacts in: artifacts/runs/supervised_v4/"
echo "Good morning!"
