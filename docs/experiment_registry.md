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

## ConvNeXt + SegFormer Blend

- Status: `working`
- Category: `ensemble`
- Summary: Current best public blend: notebook-faithful ConvNeXt full14 mixed with a top-3 manual+teacher075 SegFormer recipe.

### Key Results

```json
{
  "best_public_submission": {
    "submission_name": "submission_blend_cnxt35_seg65_top3_mt075_full14_thr40.csv",
    "kaggle_public_lb": 0.91662,
    "kaggle_public_lb_date": "2026-04-10",
    "num_images": 2000,
    "threshold": 0.4,
    "convnext_weight": 0.35,
    "segformer_weight": 0.65,
    "convnext_tta_mode": "full14",
    "convnext_image_size": 420,
    "segformer_recipe": "top3_manual_teacher075",
    "segformer_tta_enabled": false,
    "postprocess_enabled": true,
    "postprocess_min_component_area": 128,
    "postprocess_fill_holes": true
  },
  "full_holdout_weight_search": {
    "warning": "ConvNeXt blend weights are tuned on the selected validation split. This is a practical proxy unless the ConvNeXt checkpoints are confirmed to be fold-compatible with that split.",
    "val_fold": 1,
    "n_splits": 3,
    "n_samples": 594,
    "best_result": {
      "convnext_weight": 0.35,
      "segformer_weight": 0.65,
      "threshold": 0.4,
      "dice": 0.9279756682745803,
      "iou": 0.8751655499496511,
      "n_samples": 594
    }
  },
  "segformer_anchor": {
    "recipe": "top3_manual_teacher075",
    "aggregation": "weighted_mean",
    "reference_threshold": 0.4,
    "tta_enabled": false,
    "members": [
      {
        "run_name": "segformer_b2_5fold_fold0_320_manual_v1_plus_teacher075",
        "weight": 0.3333333333333333
      },
      {
        "run_name": "segformer_b2_5fold_fold1_320_manual_v1_plus_teacher075",
        "weight": 0.3333333333333333
      },
      {
        "run_name": "segformer_b2_5fold_fold4_320_manual_v1_plus_teacher075",
        "weight": 0.3333333333333333
      }
    ]
  },
  "kaggle_submission_history": {
    "best_public_score": 0.91662,
    "top_completed_submissions": [
      {
        "submission_name": "submission_blend_cnxt35_seg65_top3_mt075_full14_thr40.csv",
        "status": "complete",
        "public_score": 0.91662,
        "uploaded_by": "Fedor Grach",
        "approx_date": "2026-04-10",
        "track": "convnext_ensemble",
        "note": "Current best public result."
      },
      {
        "submission_name": "submission_supervised_v4_top3_f014_manual_teacher075_weighted_mean_thr40_no_tta.csv",
        "status": "complete",
        "public_score": 0.91439,
        "uploaded_by": "Fedor Grach",
        "approx_date": "2026-04-09",
        "track": "supervised_v4",
        "note": "Best pure SegFormer ensemble among the tested public submissions."
      },
      {
        "submission_name": "submission_supervised_v4_5fold_manual_teacher075_weighted_mean_thr40_no_tta.csv",
        "status": "complete",
        "public_score": 0.91432,
        "uploaded_by": "Fedor Grach",
        "approx_date": "2026-04-09",
        "track": "supervised_v4",
        "occurrences": 2,
        "note": "Repeated upload of the same 5-fold equal-weight SegFormer ensemble."
      },
      {
        "submission_name": "submission_supervised_v4_manual_teacher075_single_thr45.csv",
        "status": "complete",
        "public_score": 0.91242,
        "uploaded_by": "Fedor Grach",
        "approx_date": "2026-04-08",
        "track": "supervised_v4",
        "note": "Strongest single-run SegFormer baseline."
      },
      {
        "submission_name": "submission_supervised_v4_wide6_thr50.csv",
        "status": "complete",
        "public_score": 0.9078,
        "uploaded_by": "Fedor Grach",
        "approx_date": "2026-04-04",
        "track": "supervised_v4",
        "note": "Earlier wide6 supervised_v4 ensemble."
      },
      {
        "submission_name": "submission_lab3_test_images_mps.csv",
        "status": "complete",
        "public_score": 0.89694,
        "uploaded_by": "Fedor Grach",
        "approx_date": "2026-03-24",
        "track": "advanced_baseline",
        "note": "Classic ensemble MPS inference with TTA."
      }
    ]
  }
}
```

### Commands

- `smoke`: `PYTHONPATH=src python scripts/run_convnext_ensemble.py --mode blend_segformer --cnxt-tta-mode full14 --segformer-recipe top3_manual_teacher075 --segformer-weight 0.65 --threshold 0.40 --limit 5`
- `full_inference`: `PYTHONPATH=src python scripts/run_convnext_ensemble.py --mode blend_segformer --cnxt-tta-mode full14 --segformer-recipe top3_manual_teacher075 --segformer-weight 0.65 --threshold 0.40`
- `weight_search_full`: `PYTHONPATH=src python scripts/search_convnext_segformer_blend.py --segformer-recipe single_manual_teacher075 --cnxt-tta-mode full14 --thresholds 0.35 0.40 0.45 0.50 0.55 --weight-step 0.05 --output-path artifacts/runs/convnext_ensemble/blend_weight_search_full.json`

### Source Artifacts

- `artifacts/runs/convnext_ensemble/submission_blend_cnxt35_seg65_top3_mt075_full14_thr40_config.json`
- `artifacts/runs/convnext_ensemble/blend_weight_search_full.json`
- `artifacts/public/convnext_ensemble_best_blend.json`
- `artifacts/public/convnext_ensemble_best_blend.md`
- `artifacts/public/kaggle_submission_history.json`
- `artifacts/public/external_artifact_links.json`

### Warnings

- ConvNeXt checkpoints are external notebook-trained artifacts; the strongest practical evidence is the public leaderboard score plus the local holdout search, not a clean in-repo CV benchmark.
- Do not compare the old standalone ConvNeXt uploads directly to the notebook-faithful blend path; the early exporter produced malformed or semantically broken submissions.

## Supervised V4

- Status: `working`
- Category: `supervised`
- Summary: Current strongest practical SegFormer-B2/V4 baseline with clean manual labels and lower-weight teacher-approved pseudo-labels.

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
  },
  "latest_hypothesis_suite": {
    "suite_tag": "night_20260404_223806",
    "takeaway": "wide6 weighted_mean without TTA reached Dice 0.8684; the strongest short screen was screen_night_20260404_223806_fold1_384_ft_moderate_lovasz_ls003_e8 with dice_tuned 0.8912",
    "best_oof_overall": {
      "ensemble": "wide6",
      "aggregation": "weighted_mean",
      "threshold": 0.55,
      "dice": 0.8684194043723789,
      "iou": 0.7945301031333779,
      "n_samples": 2000,
      "tta_enabled": false,
      "tta_ops": null,
      "tta_scales": null,
      "member_threshold": 0.5,
      "postprocess_enabled": true,
      "postprocess_min_component_area": 128,
      "postprocess_fill_holes": true,
      "source_file": "artifacts/runs/supervised_v4/hypothesis_suite/night_20260404_223806/oof_search_no_tta.json"
    },
    "best_oof_with_tta": {
      "ensemble": "wide6",
      "aggregation": "max_prob",
      "threshold": 0.55,
      "dice": 0.8668311325136824,
      "iou": 0.792659169291871,
      "n_samples": 2000,
      "tta_enabled": true,
      "tta_ops": "flips",
      "tta_scales": [
        0.75,
        1.0,
        1.25
      ],
      "member_threshold": 0.5,
      "postprocess_enabled": true,
      "postprocess_min_component_area": 128,
      "postprocess_fill_holes": true,
      "source_file": "artifacts/runs/supervised_v4/hypothesis_suite/night_20260404_223806/oof_search_flips_multiscale.json"
    },
    "best_training_screen": {
      "run_name": "screen_night_20260404_223806_fold1_384_ft_moderate_lovasz_ls003_e8",
      "fold": 1,
      "image_size": 384,
      "aug": "moderate",
      "mask_loss": "lovasz_focal",
      "resize_interpolation": "linear",
      "finetune_from_checkpoint": true,
      "dice_tuned": 0.8912327114442993,
      "dice": 0.8886892502282749,
      "mIoU": 0.900741398642644,
      "best_threshold": 0.6,
      "tta": true,
      "source_metrics_file": "artifacts/runs/supervised_v4/screen_night_20260404_223806_fold1_384_ft_moderate_lovasz_ls003_e8/final_tta_metrics.json"
    }
  },
  "current_kaggle_baseline": {
    "run_name": "segformer_b2_fold1_320_manual_v1_plus_teacher075",
    "summary": "segformer_b2_fold1_320_manual_v1_plus_teacher075 became the current Kaggle-backed baseline after mixing base lab3 train, imported manual labels, and teacher-approved pseudo-labels with weight 0.75.",
    "kaggle_public_lb": 0.91242,
    "threshold": 0.45,
    "aggregation": "weighted_mean",
    "tta_enabled": true,
    "fold1_dice_tuned": 0.9199229259300161,
    "fold1_mIoU": 0.9257426211571912,
    "manual_holdout_dice_tuned": 0.9758237623320052,
    "manual_holdout_mIoU": 0.9650304318958995
  }
}
```

### Commands

- `smoke`: `python scripts/predict_supervised_v4_ensemble.py --run-names segformer_b2_fold1_320_manual_v1_plus_teacher075 --weights 1.0 --aggregation weighted_mean --threshold 0.45 --limit 5`
- `train_help`: `python scripts/train_supervised_v4.py --help`
- `current_baseline_inference`: `python scripts/predict_supervised_v4_ensemble.py --run-names segformer_b2_fold1_320_manual_v1_plus_teacher075 --weights 1.0 --aggregation weighted_mean --threshold 0.45`
- `current_baseline_train`: `python scripts/train_supervised_v4.py --run-name segformer_b2_fold1_320_manual_v1_plus_teacher075 --fold 1 --image-size 320 --epochs 35 --aug moderate --label-smoothing 0.03 --mask-loss lovasz_focal --physical-batch-size 8 --extra-labeled-root data/derived/imported_coco/manual_products_segmentation_v1 --extra-labeled-root artifacts/runs/unimatch_v2_wide6/night_20260405_unimatchv2_wide6_tta_sahi/accepted_teacher_predictions --extra-labeled-weight 1.0 --extra-labeled-weight 0.75 --extra-labeled-holdout-ratio 0.15 --extra-labeled-holdout-ratio 0.0`

### Source Artifacts

- `artifacts/runs/supervised_v4/oof_ensemble_eval.json`
- `artifacts/runs/supervised_v4/checkpoint_soup_fold1_eval.json`
- `artifacts/runs/supervised_v4/submission_supervised_v4_wide6_thr50_summary.json`
- `artifacts/public/supervised_v4_hypothesis_suite_latest.json`
- `artifacts/runs/supervised_v4/hypothesis_suite/night_20260404_223806/summary.txt`
- `artifacts/runs/supervised_v4/hypothesis_suite/night_20260404_223806/oof_search_no_tta.json`
- `artifacts/runs/supervised_v4/screen_night_20260404_223806_fold1_384_ft_moderate_lovasz_ls003_e8/final_tta_metrics.json`
- `artifacts/public/supervised_v4_manual_teacher075_baseline.json`
- `artifacts/public/supervised_v4_manual_teacher075_baseline.md`

### Warnings

- Local fold1 validation is optimistic for clean CV comparison because the teacher-approved pseudo-labels were accepted by a wide6 teacher spanning other folds.
- The public leaderboard score is an external checkpoint, but it should still be cross-checked against future private-leaderboard-safe variants.

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
