#!/usr/bin/env bash
# Overnight hypothesis suite for supervised_v4.
#
# What it does:
# - runs OOF ensemble searches over aggregation / threshold / TTA variants;
# - retunes postprocess for wide6;
# - launches short fold-1 screening runs for geom / heavy / interpolation / finetune hypotheses;
# - writes everything into a timestamped suite directory under artifacts/runs/supervised_v4/hypothesis_suite/.
#
# Usage:
#   bash scripts/run_supervised_v4_hypothesis_suite.sh
#   SUITE_TAG=manual_rerun bash scripts/run_supervised_v4_hypothesis_suite.sh
#   SUITE_PROFILE=oof bash scripts/run_supervised_v4_hypothesis_suite.sh
#
# Profiles:
#   full  - OOF search + postprocess retune + short training screens (default)
#   oof   - only OOF search + postprocess retune
#   train - only short training screens

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "${LAB_SEG_CAFFEINATED:-0}" != "1" ]]; then
  export LAB_SEG_CAFFEINATED=1
  exec caffeinate -imsu bash "$0" "$@"
fi

PROFILE="${SUITE_PROFILE:-${1:-full}}"
case "$PROFILE" in
  full|oof|train) ;;
  *)
    echo "Unsupported SUITE_PROFILE: $PROFILE"
    echo "Expected one of: full, oof, train"
    exit 2
    ;;
esac

RUN_ROOT="artifacts/runs/supervised_v4"
SUITE_TAG="${SUITE_TAG:-$(date '+%Y%m%d_%H%M%S')}"
SUITE_DIR="$RUN_ROOT/hypothesis_suite/$SUITE_TAG"
LOG_DIR="$RUN_ROOT/logs"
LOG_PATH="$LOG_DIR/hypothesis_suite_${SUITE_TAG}.log"
mkdir -p "$SUITE_DIR" "$LOG_DIR"

exec > >(tee -a "$LOG_PATH") 2>&1

echo "======================================================"
echo " SUPERVISED_V4 HYPOTHESIS SUITE"
echo " profile    : $PROFILE"
echo " suite_tag  : $SUITE_TAG"
echo " suite_dir  : $SUITE_DIR"
echo " log_path   : $LOG_PATH"
echo " started_at : $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"
echo
echo "Important:"
echo "- caffeinate is active for this process."
echo "- On MacBook, lid-close sleep is usually NOT bypassed by caffeinate alone."
echo "- For closed-lid overnight runs, use external power and clamshell mode."
echo

BASE_CKPT="${BASE_CKPT:-$RUN_ROOT/segformer_b2_fold1_320_moderate_lovasz_ls003_ema/best_model.pth}"
if [[ ! -f "$BASE_CKPT" ]]; then
  echo "Missing base checkpoint: $BASE_CKPT"
  exit 3
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

COMMON_THRESHOLDS=(0.35 0.40 0.45 0.50 0.55 0.60 0.65)
COMMON_AGGREGATIONS=(weighted_mean mean hard_vote max_prob)
COMMON_PRESETS=(lovasz3 blend4 wide6)

declare -a TRAIN_RUN_NAMES=()

run_step() {
  local step_num="$1"
  local desc="$2"
  local sentinel="$3"
  shift 3

  echo
  echo "====== STEP $step_num: $desc ======"
  echo "started: $(date '+%H:%M:%S')"
  if [[ -n "$sentinel" ]]; then
    echo "sentinel: $sentinel"
  fi
  echo "command: $*"
  echo

  if [[ -n "$sentinel" && -f "$sentinel" ]]; then
    echo "[skip] sentinel already exists"
  else
    "$@"
  fi

  echo
  echo "finished: $(date '+%H:%M:%S')"
  echo "====== STEP $step_num DONE ======"
  echo
}

append_train_run() {
  TRAIN_RUN_NAMES+=("$1")
}

run_oof_suite() {
  local step=1

  run_step "$step" "OOF search | no TTA" \
    "$SUITE_DIR/oof_search_no_tta.json" \
    "$PYTHON_BIN" scripts/eval_oof_ensemble.py \
      --presets "${COMMON_PRESETS[@]}" \
      --aggregations "${COMMON_AGGREGATIONS[@]}" \
      --thresholds "${COMMON_THRESHOLDS[@]}" \
      --no-tta \
      --output-path "$SUITE_DIR/oof_search_no_tta.json"
  step=$((step + 1))

  run_step "$step" "OOF search | flips x1" \
    "$SUITE_DIR/oof_search_flips_x1.json" \
    "$PYTHON_BIN" scripts/eval_oof_ensemble.py \
      --presets "${COMMON_PRESETS[@]}" \
      --aggregations "${COMMON_AGGREGATIONS[@]}" \
      --thresholds "${COMMON_THRESHOLDS[@]}" \
      --tta-scales 1.0 \
      --tta-ops flips \
      --output-path "$SUITE_DIR/oof_search_flips_x1.json"
  step=$((step + 1))

  run_step "$step" "OOF search | d4 x1" \
    "$SUITE_DIR/oof_search_d4_x1.json" \
    "$PYTHON_BIN" scripts/eval_oof_ensemble.py \
      --presets "${COMMON_PRESETS[@]}" \
      --aggregations "${COMMON_AGGREGATIONS[@]}" \
      --thresholds "${COMMON_THRESHOLDS[@]}" \
      --tta-scales 1.0 \
      --tta-ops d4 \
      --output-path "$SUITE_DIR/oof_search_d4_x1.json"
  step=$((step + 1))

  run_step "$step" "OOF search | flips multiscale" \
    "$SUITE_DIR/oof_search_flips_multiscale.json" \
    "$PYTHON_BIN" scripts/eval_oof_ensemble.py \
      --presets "${COMMON_PRESETS[@]}" \
      --aggregations "${COMMON_AGGREGATIONS[@]}" \
      --thresholds "${COMMON_THRESHOLDS[@]}" \
      --tta-scales 0.75 1.0 1.25 \
      --tta-ops flips \
      --output-path "$SUITE_DIR/oof_search_flips_multiscale.json"
  step=$((step + 1))

  run_step "$step" "OOF search | d4 multiscale" \
    "$SUITE_DIR/oof_search_d4_multiscale.json" \
    "$PYTHON_BIN" scripts/eval_oof_ensemble.py \
      --presets "${COMMON_PRESETS[@]}" \
      --aggregations "${COMMON_AGGREGATIONS[@]}" \
      --thresholds "${COMMON_THRESHOLDS[@]}" \
      --tta-scales 0.75 1.0 1.25 \
      --tta-ops d4 \
      --output-path "$SUITE_DIR/oof_search_d4_multiscale.json"
  step=$((step + 1))

  run_step "$step" "wide6 postprocess retune | no postprocess" \
    "$SUITE_DIR/oof_search_wide6_no_post.json" \
    "$PYTHON_BIN" scripts/eval_oof_ensemble.py \
      --presets wide6 \
      --aggregations "${COMMON_AGGREGATIONS[@]}" \
      --thresholds "${COMMON_THRESHOLDS[@]}" \
      --tta-scales 0.75 1.0 1.25 \
      --tta-ops d4 \
      --disable-postprocess \
      --output-path "$SUITE_DIR/oof_search_wide6_no_post.json"
  step=$((step + 1))

  run_step "$step" "wide6 postprocess retune | min area 64" \
    "$SUITE_DIR/oof_search_wide6_post64.json" \
    "$PYTHON_BIN" scripts/eval_oof_ensemble.py \
      --presets wide6 \
      --aggregations "${COMMON_AGGREGATIONS[@]}" \
      --thresholds "${COMMON_THRESHOLDS[@]}" \
      --tta-scales 0.75 1.0 1.25 \
      --tta-ops d4 \
      --postprocess-min-component-area 64 \
      --output-path "$SUITE_DIR/oof_search_wide6_post64.json"
  step=$((step + 1))

  run_step "$step" "wide6 postprocess retune | min area 256" \
    "$SUITE_DIR/oof_search_wide6_post256.json" \
    "$PYTHON_BIN" scripts/eval_oof_ensemble.py \
      --presets wide6 \
      --aggregations "${COMMON_AGGREGATIONS[@]}" \
      --thresholds "${COMMON_THRESHOLDS[@]}" \
      --tta-scales 0.75 1.0 1.25 \
      --tta-ops d4 \
      --postprocess-min-component-area 256 \
      --output-path "$SUITE_DIR/oof_search_wide6_post256.json"
}

run_training_suite() {
  local train_tag="screen_${SUITE_TAG}"
  local step=101

  local run_name

  run_name="${train_tag}_fold1_320_geom_lovasz_ls003_e12"
  append_train_run "$run_name"
  run_step "$step" "train screen | geom 320" \
    "$RUN_ROOT/$run_name/final_tta_metrics.json" \
    "$PYTHON_BIN" scripts/train_supervised_v4.py \
      --run-name "$run_name" \
      --fold 1 --image-size 320 --epochs 12 --aug geom \
      --label-smoothing 0.03 --mask-loss lovasz_focal \
      --resize-interpolation linear \
      --physical-batch-size 8 --eval-every-n-epochs 1 \
      --tta-scales 1.0 --tta-ops flips
  step=$((step + 1))

  run_name="${train_tag}_fold1_320_heavy_lovasz_ls003_e20"
  append_train_run "$run_name"
  run_step "$step" "train screen | heavy 320" \
    "$RUN_ROOT/$run_name/final_tta_metrics.json" \
    "$PYTHON_BIN" scripts/train_supervised_v4.py \
      --run-name "$run_name" \
      --fold 1 --image-size 320 --epochs 20 --aug heavy \
      --label-smoothing 0.03 --mask-loss lovasz_focal \
      --resize-interpolation linear \
      --physical-batch-size 8 --eval-every-n-epochs 1 \
      --tta-scales 1.0 --tta-ops flips
  step=$((step + 1))

  run_name="${train_tag}_fold1_320_moderate_lovasz_ls003_cubic_e12"
  append_train_run "$run_name"
  run_step "$step" "train screen | moderate 320 cubic" \
    "$RUN_ROOT/$run_name/final_tta_metrics.json" \
    "$PYTHON_BIN" scripts/train_supervised_v4.py \
      --run-name "$run_name" \
      --fold 1 --image-size 320 --epochs 12 --aug moderate \
      --label-smoothing 0.03 --mask-loss lovasz_focal \
      --resize-interpolation cubic \
      --physical-batch-size 8 --eval-every-n-epochs 1 \
      --tta-scales 1.0 --tta-ops flips
  step=$((step + 1))

  run_name="${train_tag}_fold1_320_moderate_lovasz_ls003_area_e12"
  append_train_run "$run_name"
  run_step "$step" "train screen | moderate 320 area" \
    "$RUN_ROOT/$run_name/final_tta_metrics.json" \
    "$PYTHON_BIN" scripts/train_supervised_v4.py \
      --run-name "$run_name" \
      --fold 1 --image-size 320 --epochs 12 --aug moderate \
      --label-smoothing 0.03 --mask-loss lovasz_focal \
      --resize-interpolation area \
      --physical-batch-size 8 --eval-every-n-epochs 1 \
      --tta-scales 1.0 --tta-ops flips
  step=$((step + 1))

  run_name="${train_tag}_fold1_384_ft_moderate_lovasz_ls003_e8"
  append_train_run "$run_name"
  run_step "$step" "train screen | finetune 384" \
    "$RUN_ROOT/$run_name/final_tta_metrics.json" \
    "$PYTHON_BIN" scripts/train_supervised_v4.py \
      --run-name "$run_name" \
      --fold 1 --image-size 384 --epochs 8 --aug moderate \
      --label-smoothing 0.03 --mask-loss lovasz_focal \
      --resize-interpolation linear \
      --physical-batch-size 4 --eval-every-n-epochs 1 \
      --finetune-from "$BASE_CKPT" \
      --tta-scales 1.0 --tta-ops flips
  step=$((step + 1))

  run_name="${train_tag}_fold1_448_ft_moderate_lovasz_ls003_e6"
  append_train_run "$run_name"
  run_step "$step" "train screen | finetune 448" \
    "$RUN_ROOT/$run_name/final_tta_metrics.json" \
    "$PYTHON_BIN" scripts/train_supervised_v4.py \
      --run-name "$run_name" \
      --fold 1 --image-size 448 --epochs 6 --aug moderate \
      --label-smoothing 0.03 --mask-loss lovasz_focal \
      --resize-interpolation linear \
      --physical-batch-size 2 --eval-every-n-epochs 1 \
      --finetune-from "$BASE_CKPT" \
      --tta-scales 1.0 --tta-ops flips
}

build_summary() {
  local summary_path="$SUITE_DIR/summary.txt"
  "$PYTHON_BIN" - "$SUITE_DIR" "$RUN_ROOT" "${TRAIN_RUN_NAMES[@]}" > "$summary_path" <<'PY'
import json
import sys
from pathlib import Path

suite_dir = Path(sys.argv[1])
run_root = Path(sys.argv[2])
train_run_names = sys.argv[3:]

print(f"Suite dir: {suite_dir}")
print()
print("OOF best results:")
best_rows = []
for path in sorted(suite_dir.glob("oof_search*.json")):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    rows = payload.get("best_per_combo") or payload.get("results") or []
    for row in rows:
        row = dict(row)
        row["_source"] = path.name
        best_rows.append(row)

best_rows.sort(key=lambda row: row.get("dice", -1.0), reverse=True)
for row in best_rows[:10]:
    ensemble = row.get("ensemble", "?")
    aggregation = row.get("aggregation", "?")
    dice = row.get("dice", 0.0)
    iou = row.get("iou", 0.0)
    thr = row.get("threshold", 0.0)
    source = row.get("_source", "?")
    print(f"- {source}: {ensemble} | {aggregation} | dice={dice:.4f} | iou={iou:.4f} | thr={thr:.2f}")

print()
print("Training screen results:")
train_rows = []
for run_name in train_run_names:
    metrics_path = run_root / run_name / "final_tta_metrics.json"
    if not metrics_path.exists():
        continue
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    train_rows.append(
        {
            "run_name": run_name,
            "dice_tuned": payload.get("dice_tuned", -1.0),
            "dice": payload.get("dice", -1.0),
            "miou": payload.get("mIoU", -1.0),
            "best_threshold": payload.get("best_threshold", -1.0),
        }
    )

train_rows.sort(key=lambda row: row["dice_tuned"], reverse=True)
for row in train_rows:
    print(
        f"- {row['run_name']}: dice_tuned={row['dice_tuned']:.4f} | "
        f"dice={row['dice']:.4f} | miou={row['miou']:.4f} | thr={row['best_threshold']:.2f}"
    )
PY

  echo
  echo "Summary saved to: $summary_path"
  echo
  cat "$summary_path"
}

case "$PROFILE" in
  full)
    run_oof_suite
    run_training_suite
    ;;
  oof)
    run_oof_suite
    ;;
  train)
    run_training_suite
    ;;
esac

build_summary

echo
echo "======================================================"
echo " SUITE COMPLETE"
echo " finished_at : $(date '+%Y-%m-%d %H:%M:%S')"
echo " suite_dir   : $SUITE_DIR"
echo " log_path    : $LOG_PATH"
echo "======================================================"
