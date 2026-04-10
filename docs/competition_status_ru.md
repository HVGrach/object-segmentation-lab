# Competition Status (RU)

Краткая сводка для отправки организатору по состоянию на `2026-04-10`.

## Текущий лучший submit

- Лучший подтвержденный public score: `0.91662`
- Файл: `submission_blend_cnxt35_seg65_top3_mt075_full14_thr40.csv`
- Рецепт: `ConvNeXt full14` + `SegFormer top3 manual_teacher075`
- Веса: `ConvNeXt 0.35`, `SegFormer 0.65`
- Threshold: `0.40`

Repro command:

```bash
PYTHONPATH=src python scripts/run_convnext_ensemble.py \
  --mode blend_segformer \
  --cnxt-tta-mode full14 \
  --segformer-recipe top3_manual_teacher075 \
  --segformer-weight 0.65 \
  --threshold 0.40
```

## Подтвержденные Kaggle submissions

| Rank | Submission | Public | Комментарий |
| --- | --- | --- | --- |
| 1 | `submission_blend_cnxt35_seg65_top3_mt075_full14_thr40.csv` | `0.91662` | Текущий лучший blend |
| 2 | `submission_supervised_v4_top3_f014_manual_teacher075_weighted_mean_thr40_no_tta.csv` | `0.91439` | Лучший чистый SegFormer ensemble |
| 3 | `submission_supervised_v4_5fold_manual_teacher075_weighted_mean_thr40_no_tta.csv` | `0.91432` | 5-fold equal-weight SegFormer ensemble |
| 4 | `submission_supervised_v4_manual_teacher075_single_thr45.csv` | `0.91242` | Лучший single-run SegFormer baseline |
| 5 | `submission_supervised_v4_wide6_thr50.csv` | `0.90780` | Ранний `wide6` supervised ensemble |
| 6 | `submission_lab3_test_images_mps.csv` | `0.89694` | Классический `advanced_baseline` |
| 7 | `submission_ans_16TTAfill.csv` | `0.89613` | Старый semi-supervised кандидат |
| 8 | `submission_resized.csv` | `0.88909` | Внешний Colab ensemble try: `ConvNextV2-Base + UPerNet, 420x420` + `SegFormerV4, 320x320` |
| 9 | `submission_lab3_test_images_best_dice_thr040.csv` | `0.88757` | Исторический intermediate submit |
| 10 | `submission_lab3_test_images_segformer.csv` | `0.87092` | Старый notebook-centered semi-sup submit |
| 11 | `submission_blend_cnxt60_seg40_thr45.csv` | `0.82689` | Legacy blend path до notebook-faithful cleanup |
| 12 | `submission_cnxt_geometric_thr45.csv` | `0.43449` | Legacy standalone ConvNeXt path, не считать честной оценкой текущего branch |

## Что сломалось по пути

- `submission_cnxt_geometric_thr45.csv` был несколько раз отклонен до исправления exporter path.
- Подтвержденные Kaggle ошибки:
  - `Submission contains null values`
  - `ID column ImageId not found in submission`
- Корень проблемы: ранний reverse-engineered ConvNeXt exporter писал некорректный CSV contract и ломал geometry/serialization масок.
- После переписывания inference/export path под notebook-faithful рецепт загрузки стали валидными, а blend вышел на `0.91662`.

## Главные выводы

- Лучший общий practical result сейчас дает не standalone ConvNeXt, а его blend с сильным SegFormer anchor.
- Лучший чистый supervised сигнал по Kaggle все еще идет из `supervised_v4`.
- Старые ConvNeXt standalone/public scores ниже `0.9` не отражают качество текущего notebook-faithful branch и нужны только как debug history.

## Resolved external notebook check

Последний внешний try уже подтвержден Kaggle и теперь больше не считается `pending`.

- Ноутбуки:
  - `/Users/fgrach/convnext_upernet.ipynb`
  - `/Users/fgrach/inference_ensemble.ipynb`
- Upload:
  - `submission_resized.csv`
- Description:
  - `ConvNextV2-Base + UPerNet, 420x420`
  - `SegFormerV4, 320x320`
  - `Ensemble`
- Public score:
  - `0.88909`

Вывод: это полезный подтвержденный внешний try, но он заметно ниже текущего лидера `0.91662`, поэтому канонический best submit не меняется.

## External assets для валидации соревнования

Эти ссылки обязательно нужны для зачета и должны быть приложены вместе с репозиторием:

- `weights_cnxt`: [Google Drive](https://drive.google.com/file/d/1IDmrtqNOVTTDqG5x6em-Ti5VmC6s4Lew/view?usp=sharing)
- `to_pseudolabel`: [Google Drive](https://drive.google.com/file/d/1ZEB4zUigsEeG_Qd0FWpchuzxx9n5EGq_/view?usp=drive_link)
- `depth_cache`: [Google Drive](https://drive.google.com/file/d/1SQZG7DqUDy-_Ri1MU8LZCtjl6XNfCAHU/view?usp=sharing)
- `segformer`: [Google Drive](https://drive.google.com/file/d/1xRw_PhkSCwMUmKrdXwn9uPGfOKz3YncM/view?usp=sharing)
- `dataset`: [Google Drive](https://drive.google.com/file/d/1WvM42wCxmNwVlxAvqVbBGtfp78PyI403/view?usp=sharing)

Машинно-читаемая фиксация этих ссылок лежит в `artifacts/public/external_artifact_links.json`.
