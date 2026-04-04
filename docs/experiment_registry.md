# Experiment Registry

This document is generated from the publish-safe metadata in `artifacts/public/`.

## Advanced Baseline

- Status: `working`
- Category: `supervised`
- Summary: Safest reproducible pipeline with grouped-by-camera CV, threshold tuning, TTA, and a lightweight ensemble.

### Key Results

```json
{
  "best_single_model": {
    "name": "fpn_resnet34",
    "oof_dice": 0.8380353893339634,
    "oof_iou": 0.7581554671525955,
    "best_threshold": 0.65
  },
  "best_ensemble": {
    "candidate_name": "regression_weights",
    "oof_dice": 0.8390599061250686,
    "oof_iou": 0.7599045597910881,
    "best_threshold": 0.4
  },
  "submission": {
    "num_images": 2000,
    "device": "mps",
    "tta_enabled": true
  }
}
```

### Commands

- `smoke`: `python scripts/run_lab3_ensemble_submission.py --limit 5`
- `full_inference`: `python scripts/run_lab3_ensemble_submission.py`

### Source Artifacts

- `artifacts/runs/advanced_baseline/ensemble_summary.json`
- `artifacts/runs/advanced_baseline/model_summaries.csv`
- `artifacts/runs/advanced_baseline/folds_summary.json`
- `artifacts/runs/advanced_baseline/submission_lab3_test_images_mps_summary.json`

## Supervised V4

- Status: `working`
- Category: `supervised`
- Summary: Strongest supervised-only track built around SegFormer-B2 and a custom V4 decoder.

### Key Results

```json
{
  "best_run": {
    "run_name": "segformer_b2_fold1_384_finetune_from_moderate",
    "fold": 1,
    "image_size": 384,
    "aug": "moderate",
    "mask_loss": "dice_focal",
    "dice_tuned": 0.8958750763299083,
    "dice": 0.8898094141333135,
    "mIoU": 0.9013437319030186,
    "best_threshold": 0.75,
    "tta": true
  },
  "best_oof_ensemble": {
    "preset": "wide6",
    "threshold": 0.5,
    "dice": 0.866759883242149,
    "iou": 0.7930754936197527,
    "n_samples": 2000
  },
  "best_checkpoint_soup": {
    "name": "moderate_0.75_heavy_0.25",
    "mIoU": 0.9008051427341905,
    "dice": 0.8887924016961049,
    "dice_tuned": 0.893569225238452,
    "best_threshold": 0.85,
    "pixel_acc": 0.9846971998632155,
    "boundary_f1": 0.07065178976483284,
    "loss": 0.14024657398462295,
    "tta": false
  },
  "submission": {
    "preset": "wide6",
    "num_images": 2000,
    "device": "mps",
    "tta_enabled": true,
    "threshold": 0.5
  }
}
```

### Commands

- `smoke`: `python scripts/predict_supervised_v4_ensemble.py --preset wide6 --limit 5`
- `train_help`: `python scripts/train_supervised_v4.py --help`

### Source Artifacts

- `artifacts/runs/supervised_v4/oof_ensemble_eval.json`
- `artifacts/runs/supervised_v4/checkpoint_soup_fold1_eval.json`
- `artifacts/runs/supervised_v4/submission_supervised_v4_wide6_thr50_summary.json`

## SegFormer Boundary Semi-Supervised

- Status: `research`
- Category: `semi-supervised`
- Summary: Boundary-aware SegFormer-B2 teacher-student pipeline with pseudo-label filtering and EMA.

### Key Results

```json
{
  "best_supervised_phase": {
    "epoch": 50,
    "val_mIoU": 0.9031246800649837,
    "val_pixel_acc": 0.9810050202235499,
    "val_boundary_f1": 0.07409165679631424
  },
  "best_semi_supervised_iteration": {
    "iteration": 0,
    "phase2_summary": {
      "iteration": 0,
      "num_total": 12742,
      "num_accepted": 12702,
      "accepted_percent": 0.9968607753884791,
      "avg_confidence_accepted": 0.9536282104772764,
      "avg_object_ratio_accepted": 0.32127732457015046
    },
    "phase3_best_mIoU": 0.9047450744081909,
    "delta_mIoU": 0.0016203943432071544
  },
  "stricter_pseudo_filtering": {
    "iteration": 0,
    "num_total": 12742,
    "num_candidates_after_thresholds": 4925,
    "num_accepted": 3215,
    "accepted_percent": 0.2523151781509967,
    "avg_confidence_accepted": 0.97608855729155,
    "avg_object_ratio_accepted": 0.26653892259684603,
    "avg_fg_reliable_ratio_accepted": 0.8915333129636245,
    "accepted_by_source": {
      "dl-lab-1-image-classification::train": 2379,
      "dl-lab-1-image-classification::test_images": 621,
      "lab_unlabeled": 215
    }
  },
  "review_bundle": {
    "num_items": 3215,
    "source_breakdown": {
      "dl-lab-1-image-classification::test_images": 621,
      "dl-lab-1-image-classification::train": 2379,
      "lab_unlabeled": 215
    },
    "dice_holdout_r2": 0.8733402064485057,
    "dice_holdout_rmse": 0.050272551879223144
  },
  "submission": {
    "num_images": 2000,
    "device": "mps",
    "tta_n": 8
  }
}
```

### Commands

- `smoke`: `python scripts/run_segformer_semisup_smoke_test.py`
- `submission_smoke`: `python scripts/run_lab3_segformer_submission.py --limit 5`

### Source Artifacts

- `artifacts/runs/segformer_boundary_semisup_macos/metrics/phase1_history.json`
- `artifacts/runs/segformer_boundary_semisup_macos/metrics/iteration_summaries.json`
- `artifacts/runs/segformer_boundary_semisup_macos/submission_lab3_test_images_segformer_summary.json`
- `artifacts/deliverables/pseudo_label_review_bundle/bundle_manifest.json`

## DINOv2 Research

- Status: `research`
- Category: `proxy-research`
- Summary: Frozen DINOv2 feature-cache track used for low-cost fusion ablations and hypothesis screening.

### Key Results

```json
{
  "best_ablation": {
    "experiment": "ablation_concat_s",
    "fusion": "concat",
    "backbone": "dinov2_s",
    "best_val_iou": 0.7224,
    "best_val_dice": 0.8375,
    "best_epoch": 9
  }
}
```

### Commands

- `smoke`: `python scripts/run_dinov2_smoke_test.py --overwrite-cache`
- `funnel`: `python scripts/run_dinov2_funnel.py --cache-missing`

### Source Artifacts

- `artifacts/runs/dinov2_research/results_table.csv`

## Selected Preview Assets

- `artifacts/public/previews/segformer/phase3_iter0_best4.png` (from `artifacts/runs/segformer_boundary_semisup_macos/preview/phase3_iter0_best4.png`)
- `artifacts/public/previews/segformer/phase3_iter0_worst4.png` (from `artifacts/runs/segformer_boundary_semisup_macos/preview/phase3_iter0_worst4.png`)
- `artifacts/public/previews/review_bundle/demo_card_1.png` (from `artifacts/deliverables/pseudo_label_review_bundle/demo/demo_card_1.png`)
- `artifacts/public/previews/review_bundle/demo_card_2.png` (from `artifacts/deliverables/pseudo_label_review_bundle/demo/demo_card_2.png`)
- `artifacts/public/previews/review_bundle/demo_card_3.png` (from `artifacts/deliverables/pseudo_label_review_bundle/demo/demo_card_3.png`)
