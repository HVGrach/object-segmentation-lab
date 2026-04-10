# Experimental Entrypoints (RU)

Этот файл нужен как короткий индекс для новых R&D-скриптов и утилит, которые уже лежат в репозитории, но еще не должны автоматически считаться подтвержденными результатами проекта.

Использовать его для навигации по коду. Не использовать как замену `READ_FIRST_PROJECT_CONTEXT_RU.md`.

## Supervised V4 Utilities

### `scripts/run_supervised_v4_manual_teacher_5fold.sh`

- Назначение: пакетный запуск `5-fold` рецепта `manual + teacher075` для `supervised_v4`.
- Когда полезно: нужно повторить или досчитать все `5` fold-run’ов одной командой.
- Статус интерпретации: сам скрипт полезный и рабочий как launcher, но не является отдельным подтвержденным benchmark.

### `scripts/run_boost_experiments.sh`

- Назначение: запуск двух более агрессивных `supervised_v4`-экспериментов для попытки пробить public plateau.
- Что внутри:
  - `MIT-B3 @ 320`
  - `MIT-B2 @ 384`
- Статус интерпретации: `R&D / не запускалось` как канонический baseline; держать как экспериментальный launcher.

## UniMatch V2 Utilities

### `scripts/run_unimatch_v2_wide6.py`

- Назначение: CLI entrypoint для notebook-derived ветки `unimatch_v2_wide6`.
- Код: `src/lab_object_segmentation/unimatch_v2_wide6.py`
- Когда полезно: нужен повторяемый headless запуск `phase1 -> teacher cache -> phase2 -> phase3` без ноутбука.

### `scripts/export_accepted_teacher_predictions.py`

- Назначение: собрать чистую папку только с accepted teacher predictions из завершенного run’а.
- Код: `src/lab_object_segmentation/export_accepted_teacher_predictions.py`
- Когда полезно: нужно переиспользовать teacher-approved pseudo-labels как отдельный labeled source в `supervised_v4` или другом пайплайне.

## DeepLabV3+ Candidate

### `scripts/train_deeplab_v3plus.py`

- Назначение: отдельный `DeepLabV3+` кандидат, переиспользующий проверенные проектные конвенции.
- Код: `src/lab_object_segmentation/deeplab_v3plus/train.py`
- Что уже встроено:
  - grouped-by-camera split
  - EMA / threshold tuning / TTA-friendly evaluation
  - опциональный depth-root для `RGB + D`
- Статус интерпретации: `R&D / не запускалось` как подтвержденный leaderboard path.

## DINOv2 Research Maintenance

### `src/lab_object_segmentation/dinov2_research/*`

- Назначение: поддержка proxy-ablation funnel и более стабильных repeatable R&D запусков.
- Что лежит в текущем пакете:
  - обновления `config.py`
  - доработки dataset/fusion/model/trainer
  - улучшения `run_funnel.py`
- Статус интерпретации: это улучшение исследовательской инфраструктуры, а не новый подтвержденный результат.

## Ops Helper

### `scripts/keep_mac_awake.sh`

- Назначение: удерживать macOS от сна через `caffeinate` во время длинных MPS-run’ов.
- Когда полезно: долгие local train/inference ночью или без присмотра.
- Статус интерпретации: operational helper, не ML-эксперимент.

## Правило для агентов

- Если по этим entrypoint’ам еще нет подтвержденных артефактов уровня проекта, не поднимать их до канонического статуса в `READ_FIRST_PROJECT_CONTEXT_RU.md`.
- Для навигации и будущего доразвития ссылаться на этот файл и `AI_AGENT_RUNBOOK_RU.md`.
