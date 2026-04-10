# Supervised V4 Current Baseline

`segformer_b2_fold1_320_manual_v1_plus_teacher075` is the current Kaggle-backed baseline for `supervised_v4`.

## Recipe

- Base source: original `lab3` training split with sample weight `1.0`
- Clean labels: imported `manual_products_segmentation_v1` with sample weight `1.0`
- Teacher-approved pseudo-labels: `accepted_teacher_predictions` with sample weight `0.75`
- Manual holdout: `16` images reserved from the imported clean set

## Local Metrics

- Fold 1 final TTA: `mIoU 0.9257`, `dice_tuned 0.9199`, threshold `0.45`
- Manual extra-holdout final TTA: `mIoU 0.9650`, `dice_tuned 0.9758`, threshold `0.30`

## Kaggle Submission

- Public leaderboard: `0.91242`
- Inference: single-run weighted-mean submission with TTA and threshold `0.45`
- Test coverage: `2000 / 2000` images

## Validation Notes

- Full training completed on `mps`
- Full test inference completed on `mps`
- Submission CSV row count was checked before packaging
- A compact ZIP artifact was created for upload

## Caution

The public score is a strong practical signal, but the local fold metric is not a clean CV benchmark: the teacher-approved pseudo-labels were accepted by a `wide6` teacher that spans other folds.
