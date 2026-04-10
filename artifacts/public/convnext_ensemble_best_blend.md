# ConvNeXt + SegFormer Current Best Blend

`submission_blend_cnxt35_seg65_top3_mt075_full14_thr40.csv` is the current best public Kaggle result in this repository.

## Recipe

- ConvNeXt branch: notebook-faithful `RGBD UPerNet` inference with `full14` TTA at `420`
- ConvNeXt checkpoints: `fold0_phase1_best_cnxt.pt`, `fold1_phase1_best_cnxt.pt`
- SegFormer branch: `top3_manual_teacher075`
- Blend weights: `ConvNeXt 0.35`, `SegFormer 0.65`
- Final threshold: `0.40`
- Postprocess: enabled, `min_component_area = 128`, hole filling on

## Public Result

- Public leaderboard: `0.91662`
- Submission date: `2026-04-10`
- Test coverage: `2000 / 2000` images

## Why This Version Matters

- It replaced the early reverse-engineered ConvNeXt exporter with a notebook-faithful inference path.
- The final weights came from a full labeled holdout search on `594` validation images.
- The resulting blend beat the strongest pure SegFormer candidates tested so far.

## Local Weight Search

- Best holdout point: `ConvNeXt 0.35`, `SegFormer 0.65`, `threshold 0.40`
- Holdout score: `dice 0.9280`, `iou 0.8752`
- Search scope: `fold 1`, `3` grouped splits, `594` validation images

## Comparisons

- `submission_supervised_v4_top3_f014_manual_teacher075_weighted_mean_thr40_no_tta.csv`: `0.91439`
- `submission_supervised_v4_5fold_manual_teacher075_weighted_mean_thr40_no_tta.csv`: `0.91432`
- `submission_supervised_v4_manual_teacher075_single_thr45.csv`: `0.91242`

## External Assets

The required Google Drive links for competition validation are listed in `artifacts/public/external_artifact_links.json`.

## Caution

The old standalone ConvNeXt uploads should not be treated as the quality of the current notebook-faithful branch. Earlier files went through a broken exporter path and were useful only for debugging the format and geometry issues.
