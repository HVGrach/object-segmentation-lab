# Подробный разбор проекта

Этот документ оставляет русскоязычный контекст проекта в нейтральной форме без локальных handoff-зависимостей.

## Что это за проект

Лабораторная работа посвящена бинарной сегментации товара на изображении:

- вход: RGB-изображение;
- выход: бинарная маска объекта;
- основной датасет: `dl-lab-3-product-segmentation`;
- внешний unlabeled-пул для semi-supervised экспериментов: `dl-lab-1-image-classification`.

## Канонические локальные пути

- код: `src/lab_object_segmentation/`
- CLI-скрипты: `scripts/`
- ноутбуки: `notebooks/`
- локальные полные артефакты запусков: `artifacts/runs/`
- локальные deliverables: `artifacts/deliverables/`
- publish-safe metadata для GitHub: `artifacts/public/`

## Треки проекта

### 1. `advanced_baseline`

Статус: `рабочее`

- supervised pipeline на `segmentation_models_pytorch`
- grouped-by-camera split
- 3-fold CV, EMA, threshold tuning, TTA и ансамбль
- safest reproducible path для повторения готового результата

Ключевой подтверждённый результат:

- лучший single model: `fpn_resnet34`, `OOF Dice = 0.8380`, `OOF IoU = 0.7582`
- лучший ансамбль: `regression_weights`, `OOF Dice = 0.8391`, `OOF IoU = 0.7599`

Типовой smoke-run:

```bash
python scripts/run_lab3_ensemble_submission.py --limit 5
```

### 2. `supervised_v4`

Статус: `рабочее`

- SegFormer-B2 + V4 decoder
- grouped-by-camera supervision
- реальные fold runs, OOF ensemble evaluation и checkpoint soup evaluation
- strongest current supervised signal в проекте

Ключевой подтверждённый результат:

- лучший fold run: `segformer_b2_fold1_384_finetune_from_moderate`, `dice_tuned = 0.8959`
- лучший OOF ensemble: preset `wide6`, `dice = 0.8668`, `IoU = 0.7931`

Типовой smoke-run:

```bash
python scripts/predict_supervised_v4_ensemble.py --preset wide6 --limit 5
```

### 3. `segformer_boundary_semisup_macos`

Статус: `R&D`

- SegFormer-B2 + boundary head
- teacher-student semi-supervised pipeline
- pseudo-label generation, reliability masks, EMA

Ключевые подтверждённые выводы:

- supervised phase: `val_mIoU = 0.9031`
- iteration 0: `val_mIoU = 0.9047`, небольшой прирост
- iteration 1: деградация до `val_mIoU = 0.8656`

Вывод:

- трек сильный исследовательски, но не production default

### 4. `dinov2_research`

Статус: `R&D`

- frozen DINOv2 backbone
- feature cache и ablation fusion-голов
- быстрый proxy-track для проверки гипотез

Ключевой подтверждённый результат:

- лучший fusion: `concat`, `best_val_iou = 0.7224`

Вывод:

- полезен как R&D-track, но не заменяет основной supervised pipeline

## Что публикуется в GitHub

В GitHub-репозиторий попадают:

- код;
- notebook history;
- публичная документация;
- sanitized summary JSON/CSV;
- generated experiment registry;
- selected preview assets.

Не публикуются:

- raw datasets;
- чекпоинты и веса;
- probability maps и caches;
- крупные архивы и docker exports;
- локальные planning/handoff документы.

## Как обновлять public metadata после нового эксперимента

1. Запустить локальный training или inference.
2. Убедиться, что подтверждённые артефакты появились в `artifacts/runs/` или `artifacts/deliverables/`.
3. Обновить publish-safe слой:

```bash
python scripts/export_public_results.py
python scripts/build_project_docs.py
```

4. Проверить изменения в `artifacts/public/`, `docs/generated/`, `docs/experiment_registry.md` и `README.md`.

## Что считать главным

- safest reproducible path: `advanced_baseline`
- strongest current supervised signal: `supervised_v4`
- strongest semi-supervised local result: `segformer_boundary_semisup_macos` iteration 0
- proxy-research track: `dinov2_research`
