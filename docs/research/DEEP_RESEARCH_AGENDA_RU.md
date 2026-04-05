# DEEP_RESEARCH_AGENDA_RU

Этот файл не является factual-status документом проекта.

Он нужен как операционный research agenda для внешнего deep research и для следующих AI-агентов: что именно исследовать, в каком порядке, по каким критериям фильтровать найденные методы и в каком формате возвращать результат. Канонический factual context по проекту остается в `READ_FIRST_PROJECT_CONTEXT_RU.md`.

Этот документ опирается на:

- `READ_FIRST_PROJECT_CONTEXT_RU.md`
- `artifacts/runs/segformer_boundary_semisup_macos/metrics/iteration_summaries.json`
- `artifacts/runs/segformer_boundary_semisup_macos_2026_03_31_cachefix/metrics/pseudo_iteration_0.json`
- `artifacts/deliverables/pseudo_label_review_bundle/bundle_manifest.json`
- `artifacts/runs/supervised_v4/oof_ensemble_eval.json`
- `docs/research/SOTA_IDEAS_RU.md`

## Summary

- Главный подтвержденный bottleneck проекта сейчас связан не с выбором "самого нового backbone", а с качеством и управлением pseudo-labels в semi-supervised цикле.
- В старой ветке `segformer_boundary_semisup_macos` acceptance был слишком широким: iteration 0 приняла `99.7%`, iteration 1 — `88.3%`, после чего iteration 1 деградировала до `mIoU = 0.8656`.
- Более строгий rerun уже сдвинул проект в правильную сторону: `segformer_boundary_semisup_macos_2026_03_31_cachefix` принял только `25.2%` кандидатов и подготовил quality-aware review bundle.
- `review_bundle` уже содержит working expected-Dice estimator, значит research по calibration и mask QA опирается не на голую идею, а на существующий рабочий задел.
- `supervised_v4` уже силен: OOF-ансамбль `wide6` дает `Dice = 0.8668`, `IoU = 0.7931`, поэтому generic search в духе "найти новый backbone" имеет более низкий ROI, чем исследование pseudo-label governance, OOD unlabeled usage и domain robustness.

Операционный вывод:

- research с наивысшим ROI должен улучшать pseudo-label filtering, mask quality estimation, OOD-safe unlabeled usage, boundary refinement и camera/domain robustness;
- research, который уводит проект в full one-shot / few-shot / counting-first paradigm, имеет более низкий приоритет и не должен становиться mainline без отдельного scoped validation plan.

## Priority Research Tracks

### 1. `Pseudo-label selection beyond fixed confidence thresholds`

- Deep-research question: какие методы 2024-2026 реально помогают бороться с overconfidence, sample-specific thresholding, сохранением полезного low-confidence контекста и накоплением ошибок между pseudo-итерациями?
- Почему это высокий приоритет: project artifacts уже показывают, что широкая приемка псевдомасок ломает следующий self-training cycle.
- Expected fit: `segformer_semisup`, reliability-mask logic, `review_bundle`.
- Что особенно искать: adaptive acceptance, rank-based filtering, pixel-wise reliability use, conformal or calibrated selection, curriculum-based pseudo-label governance.
- Seed sources:
  - [When Confidence Fails, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_When_Confidence_Fails_Revisiting_Pseudo-Label_Selection_in_Semi-supervised_Semantic_Segmentation_ICCV_2025_paper.html)
  - [Confidence-Weighted Boundary-Aware Learning, 2025](https://arxiv.org/abs/2502.15152)
  - [Pseudo-label survey, 2024](https://arxiv.org/abs/2403.01909)

### 2. `Mask quality estimation and calibration without ground truth`

- Deep-research question: какие современные методы умеют оценивать expected mask quality, доверительные интервалы или risk score для сегментационной маски без GT в inference-time?
- Почему это высокий приоритет: у проекта уже есть working expected-Dice estimator; next step — сделать его более risk-aware и полезным для автоматической фильтрации, а не только для review.
- Expected fit: `review_bundle`, pseudo-label ranking, строгая селекция перед semi-sup train.
- Что особенно искать: mask quality estimation, risk calibration, ranking under uncertainty, uncertainty decomposition, conformal scoring для segmentation outputs.
- Seed source:
  - [ConfIC-RCA, 2025/2026](https://arxiv.org/abs/2503.04522)

### 3. `SAM2 as teacher, refiner, or proposal generator`

- Deep-research question: как использовать promptable foundation segmentation не как "замену всему", а как узкий инструмент для boundary refinement, proposal masks, hard-case correction и ускорения human review?
- Почему это высокий приоритет: SAM2 в проекте еще не использовался, но он может быть полезен как second opinion для сложных масок без полной миграции пайплайна.
- Expected fit: pseudo-label refinement, review acceleration, boundary repair для hard cases.
- Что особенно искать: SAM2 as refiner, proposal generator, calibration on small labeled target domain, lightweight adapters, practical constraints on small data and offline preprocessing.
- Seed sources:
  - [SAM 2](https://arxiv.org/abs/2408.00714)
  - [CAT-SAM, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05662.pdf)
  - [SAM2-Adapter](https://arxiv.org/abs/2408.04579)

### 4. `Safe use of OOD unlabeled images`

- Deep-research question: как современные методы определяют, когда OOD unlabeled данные помогают, как их безопасно pseudo-labelить и какие признаки указывают, что источник лучше исключить заранее?
- Почему это высокий приоритет: в более строгом pseudo run accepted pool уже включает много элементов из `dl-lab-1-image-classification`, то есть вопрос source suitability у проекта реальный, а не теоретический.
- Expected fit: source filtering перед pseudo-label generation, DINOv2-based clustering/ranking, source-aware pseudo-label governance.
- Что особенно искать: ID/OOD screening, selective pseudo-labeling, safe unlabeled mixing, domain similarity estimation, lightweight feature-based source filtering.
- Seed source:
  - [SemiOVS, 2025](https://arxiv.org/abs/2507.03302)

### 5. `Camera/domain robustness and lightweight test-time adaptation`

- Deep-research question: какие методы domain generalization, source-free adaptation или lightweight test-time calibration совместимы с grouped-by-camera split и бинарной segmentation задачей?
- Почему это высокий приоритет: camera grouping уже является core assumption проекта и влияет как на воспроизводимость, так и на реальное качество generalized inference.
- Expected fit: `advanced_baseline`, `supervised_v4`, post-hoc calibration or lightweight TTA.
- Что особенно искать: grouped-domain robustness, source-free adaptation, calibration at test time, camera-aware generalization, safe low-compute TTA.
- Seed source:
  - [Domain Generalization for Semantic Segmentation survey, 2024](https://link.springer.com/article/10.1007/s10462-024-10817-z)

### 6. `Lower-priority track: DINOv2 modernization`

- Deep-research question: если нужен research branch, какие parameter-efficient методы dense adaptation для DINOv2 или DINO+SAM hybrids реально стоят внимания, и как не повторять frozen-feature ablation loop без practical payoff?
- Почему это lower priority: текущий proxy ceiling в `dinov2_research` далеко ниже `supervised_v4`, поэтому это не shortest path to project gain.
- Expected fit: только R&D/proxy branch, а не mainline replacement.
- Что особенно искать: dense adapters, LoRA-style adaptation, DINO as ranking/filtering backbone, DINO+SAM narrow integrations.

## Expected Outputs

Каждый отдельный deep-research run должен возвращать не длинный обзор, а короткий structured brief по одной идее или одному paper family.

Обязательная схема brief:

```text
problem ->
key idea ->
why it fits this repo ->
exact hook point (supervised_v4 / segformer_semisup / review_bundle / dinov2_research / advanced_baseline) ->
compute cost ->
implementation risk ->
expected upside ->
go / no-go verdict
```

Требования к таким brief:

- не ограничиваться paper summary;
- обязательно давать одну конкретную "translation to project" фразу;
- указывать точку встраивания в текущий репозиторий;
- разделять `интересно как идея` и `реально стоит пробовать сейчас`.

В этой фазе не требуется:

- писать код;
- менять публичные интерфейсы;
- запускать дорогие train/inference циклы;
- обновлять `READ_FIRST_PROJECT_CONTEXT_RU.md`.

## Evaluation Criteria

Оставлять в high-priority shortlist только методы, которые plausibly улучшают хотя бы один из реальных bottleneck проекта:

- pseudo-label filtering;
- mask quality estimation / QA;
- OOD-safe unlabeled usage;
- camera/domain robustness;
- boundary refinement;
- hard-case correction без полного переписывания mainline.

Нужно явно отбрасывать:

- методы, завязанные на большие multi-class benchmarks без чистого пути переноса в бинарную segmentation задачу;
- методы, предполагающие дорогую онлайн-аннотацию или heavy human-in-the-loop;
- методы, требующие большого retraining effort без понятного first-step prototype;
- generic "новый encoder/decoder/backbone" идеи, если они не адресуют подтвержденный bottleneck.

При прочих равных выше оценивать методы, которые:

- имеют open code или понятный implementation sketch;
- совместимы с small-data adaptation;
- допускают lightweight offline preprocessing;
- не конфликтуют с `mps`-ориентированным локальным workflow;
- можно проверить через ограниченный prototype, а не только через full retrain.

## Assumptions

- Это документ для research-planning, а не для фиксации выполненных экспериментов.
- Приоритет здесь grounded в project artifacts, а не в abstract-level SOTA claims.
- Следующий крупный прирост в проекте вероятнее придет от лучшего управления pseudo-labels и более безопасного использования unlabeled data, чем от прямой замены основного supervised backbone.
- `SOTA_IDEAS_RU.md` остается широким банком гипотез, а этот документ — более узким operational shortlist для deep research.

## Анти-ловушки для deep research

- Не трактовать любые обещания вида `+X mIoU` как переносимые на этот проект.
- Не путать one-shot / few-shot / counting papers с прямой дорожной картой для текущего supervised/semi-sup пайплайна.
- Не предлагать `SAM2`, `FSSAM`, `SANSA`, `FS-SAM2`, `FS-DINO` как mainline replacement без scoped validation plan.
- Если paper звучит сильно, но не имеет ясного hook point в `supervised_v4`, `segformer_semisup`, `review_bundle`, `advanced_baseline` или `dinov2_research`, он не должен попадать в ближайший shortlist.
