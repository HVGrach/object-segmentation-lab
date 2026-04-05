# SOTA_IDEAS_RU

Этот файл не является factual-status документом проекта.

Он хранит только курированные идеи из внешних research-черновиков и нужен как банк гипотез для следующего агента или исследователя. Любые утверждения здесь считаются рабочими гипотезами, пока они не подтверждены артефактами проекта в `artifacts/runs/*`, `scripts/*`, `deliverables/*` или не отражены в `READ_FIRST_PROJECT_CONTEXT_RU.md`.

Приоритет здесь выставлен не по paper leaderboard сам по себе, а по текущему состоянию проекта:

- `supervised_v4` уже является strongest current supervised signal.
- старый EMA teacher-student в `segformer_boundary_semisup_macos` показал слабый прирост на iteration 0 и сильную деградацию на iteration 1.
- более строгая селекция pseudo-labels и review bundle уже дали полезный сигнал, значит главный узкий момент сейчас не "любой новый backbone", а качество pseudo-label governance.
- `dinov2_research` пока полезен как R&D/proxy-track, но не как основной путь.

Внешние входные заметки для этой сводки:

- `Semi-Supervised Segmentation  SOTA методы и причины отказа Teacher-Student EMA.md`
- `SOTA One-Shot Semantic Segmentation 2025–2026.md`

## Контекст проекта для отбора идей

- Подтвержденный supervised ориентир сейчас идет из `supervised_v4`: OOF-ансамбль `wide6` дал `dice = 0.866760`, `IoU = 0.793075`.
- Подтвержденная semi-sup проблема видна в старой SegFormer-ветке: iteration 0 приняла `12702 / 12742 = 99.686%`, iteration 1 приняла `11249 / 12742 = 88.283%`, после чего `val_mIoU` упал до `0.865590`.
- Более строгий rerun `segformer_boundary_semisup_macos_2026_03_31_cachefix` показал, что узкое место действительно в качестве и отборе pseudo-labels: после порогов осталось `4925`, принято `3215`, то есть `25.232%`.
- Review bundle уже доказал практическую ценность quality estimation: holdout calibration для expected Dice дала `MAE = 0.034190`, `RMSE = 0.050273`, `R² = 0.873340`.
- `dinov2_research` пока не подтверждает замену основного пайплайна: лучший proxy-результат `concat = 0.7224 IoU`, что сильно ниже основного SegFormer-трека.

Главный вывод для отбора идей:

- высокий приоритет получают идеи, которые улучшают pseudo-label selection, calibration, unlabeled pool filtering и устойчивость semi-sup цикла;
- средний приоритет получают идеи, которые можно надстроить над текущей архитектурой без полного переписывания основного трека;
- низкий приоритет получают идеи, которые уводят проект в другой класс задач: one-shot / few-shot / counting-first / benchmark chasing.

## Идеи с высоким fit

### Карточка 1

- Идея: адаптивный отбор pseudo-labels и оценка качества маски вместо фиксированного `confidence >= const`.
- Откуда взято: semi-supervised note; идеи про calibration, conformal filtering, quality-aware pseudo-labeling; частично подтверждается уже существующим review bundle.
- Fit: высокий.
- Предварительная оценка: самый прямой и наименее рискованный путь улучшить текущую semi-sup ветку без смены главной архитектуры.
- Почему это может помочь именно нам: проект уже показал, что широкая приемка псевдомасок ломает iteration 1, а более строгая селекция и expected Dice estimator уже дали полезный сигнал. Значит здесь не теоретический fit, а прямое продолжение подтвержденной ветки.
- Куда встраивается в проект: `src/lab_object_segmentation/segformer_semisup/generate_pseudo_labels.py`, notebook logic в `src/lab_object_segmentation/segformer_semisup/build_notebook.py`, quality ranking в `src/lab_object_segmentation/review_bundle/prepare_bundle.py`.
- Главные риски / почему не факт: внешние заметки местами обещают слишком большой прирост; calibration может переобучиться на holdout и не дать выигрыша в новом pseudo-цикле.
- Стартовый способ проверки: оставить teacher и decoder как есть, но заменить fixed-threshold приемку на score-based ranking с калибровкой по labeled validation и сравнить acceptance / quality для iteration 0 без полного retrain.

### Карточка 2

- Идея: unreliable pixels не выбрасывать целиком, а использовать мягко: soft supervision, masking loss по надежным пикселям, отдельная обработка ненадежных областей.
- Откуда взято: semi-supervised note; линия про Unreliable Pseudo-Labels и отказ от наивной схемы "accepted / rejected целиком".
- Fit: высокий.
- Предварительная оценка: сильная идея для проекта, потому что текущая логика во многом работает на уровне sample-level acceptance, а не pixel-level usefulness.
- Почему это может помочь именно нам: у нас уже сохраняются reliability masks, но основная ценность может быть не только в отсеве картинок, а в использовании частично качественных масок. Это особенно важно для retail-изображений, где объект и фон могут быть неоднородно надежны.
- Куда встраивается в проект: reliability mask pipeline в `src/lab_object_segmentation/segformer_semisup/build_notebook.py`, pseudo-mask generation и последующая training loss логика в semi-sup ветке.
- Главные риски / почему не факт: это уже глубже меняет loss и sampling; можно случайно добавить сложность без реального выигрыша, если teacher по факту ошибается не локально, а структурно.
- Стартовый способ проверки: сделать дешёвую ablation на одной pseudo-итерации, где sample не отбрасывается полностью, а loss считается только по надежным пикселям и сравнивается с текущим hard-accept вариантом.

### Карточка 3

- Идея: curriculum self-training с ужесточением acceptance по итерациям, а не повторением одной и той же схемы pseudo-labeling.
- Откуда взято: semi-supervised note; идеи curriculum labeling и более осторожного multi-stage self-training.
- Fit: высокий.
- Предварительная оценка: высокий practical fit, потому что текущий проект уже увидел, что iteration 1 может деградировать; следующая версия цикла должна быть существенно более консервативной, чем iteration 0.
- Почему это может помочь именно нам: проектный факт уже есть: старая ветка деградировала на второй итерации, а cachefix-ветка пошла в сторону более жесткой фильтрации. Значит идея не новая "снаружи", а логичное продолжение собственной истории.
- Куда встраивается в проект: semi-sup orchestration вокруг `segformer_semisup` notebook/build pipeline и summary metrics в `artifacts/runs/segformer_boundary_semisup_macos*/metrics/*`.
- Главные риски / почему не факт: curriculum сам по себе не спасет, если teacher переобучен или poorly calibrated; можно потерять recall и получить слишком маленький pseudo-pool.
- Стартовый способ проверки: формализовать schedule acceptance для iteration 0/1, например более жесткий score cutoff на следующей итерации и обязательное сравнение quality distribution до train.

### Карточка 4

- Идея: фильтрация OOD / low-fit unlabeled данных до pseudo-labeling, а не только после него.
- Откуда взято: semi-supervised note; идеи selective pseudo-labeling и OOD-aware unlabeled usage.
- Fit: высокий.
- Предварительная оценка: это одна из самых практичных идей для проекта, потому что уже используется смесь `lab_unlabeled`, `dl-lab-1-image-classification::train` и `::test_images`, а не один чистый пул.
- Почему это может помочь именно нам: строгий run уже показал source breakdown accepted pseudo-labels, значит у нас есть реальная почва для source-aware отбора. Если часть источников системно хуже, их лучше фильтровать до expensive training.
- Куда встраивается в проект: pre-filtering перед `src/lab_object_segmentation/segformer_semisup/generate_pseudo_labels.py`, clustering / feature-based filtering можно завязать на `src/lab_object_segmentation/dinov2_research/cache_features.py` и `run_funnel.py`.
- Главные риски / почему не факт: можно случайно выкинуть полезные редкие кейсы и сузить domain coverage; внешний note не доказывает, что OOD-фильтр обязательно нужен именно на нашем датасете.
- Стартовый способ проверки: сравнить selection score и accepted-ratio по источникам, затем сделать простой DINOv2-based source screening без обучения и посмотреть, меняется ли качество top-ranked pseudo-label pool.

### Карточка 5

- Идея: заменить plain EMA teacher-student на UniMatch-style weak-to-strong pseudo-supervision без отдельного EMA teacher.
- Откуда взято: semi-supervised note; блок про UniMatch как альтернативу классическому Mean Teacher.
- Fit: высокий.
- Предварительная оценка: это самый содержательный архитектурный кандидат на замену текущей деградировавшей EMA-логики, но уже дороже предыдущих идей по внедрению.
- Почему это может помочь именно нам: старый проектный pain point уже связан именно с teacher-student coupling и overconfident pseudo-labeling. Если уходить от EMA, это должно быть не "новое всё подряд", а конкретная weak-to-strong схема, где coupling убирается по построению.
- Куда встраивается в проект: ближе всего к `src/lab_object_segmentation/supervised_v4/train.py` как к более чистому современному training entrypoint, но с semi-sup расширением; частично reuse аугментаций и TTA logic из текущего supervised and semisup code.
- Главные риски / почему не факт: paper-level SOTA не гарантирует выигрыша на бинарной retail segmentation; заметка слишком оптимистична по ожидаемому приросту; понадобится аккуратная реализация данных и аугментаций.
- Стартовый способ проверки: не переписывать весь semi-sup notebook, а собрать минимальный prototype на одном fold и одном unlabeled subset с weak-view pseudo-label + strong-view consistency.

### Карточка 6

- Идея: Dual Teacher как адресный фикс именно для coupling problem, если проект все же остается в teacher-student парадигме.
- Откуда взято: semi-supervised note; блок про Dual Teacher.
- Fit: высокий.
- Предварительная оценка: хороший backup-путь, если команда хочет сохранить текущую teacher-centered архитектуру, но убрать главный дефект plain EMA.
- Почему это может помочь именно нам: в текущем проекте уже есть прямой сигнал, что одна teacher-student линия нестабильна на повторной pseudo-итерации. Dual Teacher стоит рассматривать не как "новый default", а как targeted intervention в уже найденную проблему.
- Куда встраивается в проект: semi-sup ветка `segformer_semisup`, если сохранять teacher-generated pseudo-label cycle.
- Главные риски / почему не факт: это архитектурно сложнее, чем просто починить фильтрацию; diversity двух teachers легко оказаться формальной, а не реальной.
- Стартовый способ проверки: прежде чем внедрять полную схему, проверить, действительно ли можно стабильно разнести два teacher checkpoint по validation behavior и pseudo-label ranking на одном и том же unlabeled slice.

### Карточка 7

- Идея: CorrMatch-style propagation использовать как refinement слоя для pseudo-labels, а не как полную замену current pipeline.
- Откуда взято: semi-supervised note; блок про CorrMatch и correlation-based propagation.
- Fit: высокий.
- Предварительная оценка: идея выглядит полезной именно как refinement-stage, потому что проект уже работает с boundary / reliability / pseudo masks, а shape-aware propagation может закрыть слабые участки масок.
- Почему это может помочь именно нам: в top-down retail сценах много похожих по форме объектов, и pseudo-mask refinement по similarity/correlation может оказаться уместнее, чем еще одна грубая threshold heuristic.
- Куда встраивается в проект: post-processing/refinement между pseudo-label generation и final acceptance в `segformer_semisup`, а также как quality-upgrade step перед `review_bundle`.
- Главные риски / почему не факт: correlation-heavy методы могут оказаться слишком дорогими для реального цикла; есть риск повторить судьбу `correlation_4d` из `dinov2_research`, где сложный fusion не подтвердился.
- Стартовый способ проверки: не переносить full CorrMatch, а попробовать локальный refinement только для top-ranked pseudo-labels и измерить, меняется ли expected Dice / boundary quality.

## Идеи со средним fit

### Карточка 8

- Идея: SWSEG-style feature regularization как добавка к semi-sup или supervised training, а не как новый основной метод.
- Откуда взято: semi-supervised note; блок про Sliced-Wasserstein regularization.
- Fit: средний.
- Предварительная оценка: интересный add-on, но это уже second-order optimization после того, как будет стабилизирован pseudo-label pipeline.
- Почему это может помочь именно нам: у проекта есть однородные текстуры и возможный feature collapse на semi-sup данных; дополнительная регуляризация признаков теоретически может улучшить устойчивость.
- Куда встраивается в проект: скорее в новый training entrypoint поверх `supervised_v4` или UniMatch-like prototype, чем в старый notebook.
- Главные риски / почему не факт: внешний документ даёт слишком сильные обещания по приросту; может добавить complexity без явной пользы на бинарной задаче.
- Стартовый способ проверки: пробовать только после стабилизации базовой semi-sup схемы и только как isolated add-on ablation.

### Карточка 9

- Идея: AllSpark-style архитектурное взаимодействие labeled / unlabeled features, но только как research-ответвление.
- Откуда взято: semi-supervised note; блок про AllSpark.
- Fit: средний.
- Предварительная оценка: идея интересная, но уже лезет в decoder/feature interaction и потому заметно дороже, чем проблемы фильтрации и pseudo-governance.
- Почему это может помочь именно нам: если выяснится, что bottleneck не только в pseudo-label quality, но и в способе совместного обучения labeled/unlabeled потоков, такая идея может стать следующим уровнем улучшения.
- Куда встраивается в проект: скорее рядом с `supervised_v4` decoder path, чем в `advanced_baseline`; потребует отдельного экспериментального training entrypoint.
- Главные риски / почему не факт: высокий engineering overhead и слабая гарантия выигрыша именно на текущем бинарном домене.
- Стартовый способ проверки: сначала paper-to-code decomposition и микро-prototype на одном fold без полного migration основного пайплайна.

### Карточка 10

- Идея: DINOv2 adapters / DINO-backed SSL upgrades, но только как надстройка к текущему стеку, а не как новая главная ветка.
- Откуда взято: semi-supervised note и частично one-shot note; блоки про DINOv2 как сильный backbone и adapter-friendly foundation.
- Fit: средний.
- Предварительная оценка: полезное направление, но оно должно быть подчинено текущему факту, что `dinov2_research` пока намного слабее main SegFormer track.
- Почему это может помочь именно нам: DINOv2 уже присутствует в кодовой базе, есть feature cache и опыт быстрых ablation. Это делает DINO-backed эксперименты дешёвыми для proxy research.
- Куда встраивается в проект: `src/lab_object_segmentation/dinov2_research/*`, а также потенциально как feature-based filter для unlabeled pool.
- Главные риски / почему не факт: заметки про DINOv2 часто относятся к другим task families; перенос идеи "foundation backbone = сразу лучше" может не сработать на нашем датасете.
- Стартовый способ проверки: использовать DINOv2 сначала не для замены main model, а для ranking / clustering / OOD screening и только потом для новой segmentation head.

### Карточка 11

- Идея: Grounded SAM 2 / SAM 2 использовать как pseudo-label refiner, proposal generator или review accelerator, а не как немедленную mainline replacement.
- Откуда взято: обе внешние заметки; вторая заметка подробно обсуждает SAM2/FSS-пайплайны.
- Fit: средний.
- Предварительная оценка: идея потенциально сильная, но только если внедрять её узко и прагматично, а не переносить в проект весь one-shot/FSS стек.
- Почему это может помочь именно нам: SAM2 сейчас не использовался в проекте, но может дать полезный второй взгляд на сложные boundary cases, улучшить review bundle или дать proposal masks для hard examples.
- Куда встраивается в проект: как offline refinement stage рядом с `review_bundle`, либо как внешний pseudo-label proposal tool до semi-sup training.
- Главные риски / почему не факт: заметка №2 сильно тяготеет к FSS и multi-instance benchmark logic, что не совпадает с текущим главным направлением проекта; интеграция SAM2 сама по себе дорогая и может увести команду в сторону.
- Стартовый способ проверки: не тренировать FSSAM/SANSA, а взять маленький hard-case subset и сравнить текущие pseudo masks против SAM2-refined proposals по manual review.

### Карточка 12

- Идея: выборочные occlusion / amodal ideas применять только под реальные checkout failure modes, например руки, пакеты, частично закрытые товары.
- Откуда взято: one-shot note; блоки про occlusion handling, amodal segmentation и refinement under occlusion.
- Fit: средний.
- Предварительная оценка: полезно как специализированная ветка после того, как появятся подтвержденные ошибки на таких случаях.
- Почему это может помочь именно нам: именно checkout-сцены могут содержать частичные перекрытия, но пока в проекте это не зафиксировано как главный подтвержденный bottleneck.
- Куда встраивается в проект: как отдельный hard-case analysis поверх `review_bundle` и, возможно, как auxiliary refinement branch в `segformer_semisup`.
- Главные риски / почему не факт: можно потратить много времени на модную тему, которая не является главным источником ошибок в текущем датасете.
- Стартовый способ проверки: сначала собрать небольшой failure taxonomy из false positives / false negatives и только потом решать, нужна ли отдельная occlusion-aware идея.

## Отложено / низкий fit / не брать в основной трек

### Карточка 13

- Идея: full switch на one-shot / few-shot mainline через FSSAM, SANSA, FS-SAM2 или FS-DINO.
- Откуда взято: one-shot note.
- Fit: низкий.
- Предварительная оценка: это не следующий шаг проекта, а смена класса задачи и стека.
- Почему это может помочь именно нам: максимум как источник отдельных инженерных приёмов вокруг promptable segmentation и foundation-model refinement.
- Куда встраивается в проект: только в отдельный long-range R&D track, не в текущий основной pipeline.
- Главные риски / почему не факт: текущий проект не организован как FSS benchmark; у нас уже есть сильный supervised track и semi-sup история, а не задача "reference image -> segment all".
- Стартовый способ проверки: не делать full migration; брать только отдельные transferable идеи, если появится узкий use case.

### Карточка 14

- Идея: GeCo-style counting-first или detect-segment-count unified branch.
- Откуда взято: one-shot note.
- Fit: низкий.
- Предварительная оценка: побочная ветка, не соответствующая текущему objective проекта.
- Почему это может помочь именно нам: разве что как future extension для multi-instance accounting, но сейчас это не основной критерий качества проекта.
- Куда встраивается в проект: только как distant R&D.
- Главные риски / почему не факт: уводит в counting/detection formulation вместо прямого улучшения текущей segmentation quality.
- Стартовый способ проверки: не запускать без отдельного product reason.

### Карточка 15

- Идея: SemiVL / CLIP-language guidance как существенная ось улучшения.
- Откуда взято: semi-supervised note.
- Fit: низкий.
- Предварительная оценка: для бинарной сегментации `product vs background` это выглядит слабо связанным с текущим bottleneck.
- Почему это может помочь именно нам: максимум как auxiliary idea для richer prompts, если проект перейдет к open-vocabulary or product-category aware setting.
- Куда встраивается в проект: сейчас никуда осмысленно.
- Главные риски / почему не факт: легко получить complexity без реальной пользы.
- Стартовый способ проверки: не приоритизировать.

### Карточка 16

- Идея: любые точные обещания вида `+X% mIoU` из внешних заметок считать основой roadmap.
- Откуда взято: обе заметки, особенно semi-supervised note.
- Fit: низкий.
- Предварительная оценка: это не идея для внедрения, а источник потенциальной ошибки в приоритизации.
- Почему это может помочь именно нам: полезно только как анти-пример; проект должен опираться на artifact-grounded decisions, а не на чужие обещания переноса SOTA.
- Куда встраивается в проект: в правила чтения этого файла и в дисциплину последующих агентов.
- Главные риски / почему не факт: benchmark transfer на binary retail segmentation почти никогда не линеен.
- Стартовый способ проверки: каждый paper claim переводить в минимальную ablation-гипотезу без численных обещаний.

### Карточка 17

- Идея: тащить весь one-shot note как список "что модно в 2025–2026".
- Откуда взято: one-shot note целиком.
- Fit: низкий.
- Предварительная оценка: note полезен как обзор направлений, но слишком легко уводит в benchmark chasing.
- Почему это может помочь именно нам: только как secondary inspiration для proposal/refinement/occlusion ideas.
- Куда встраивается в проект: как вспомогательный источник, а не как план миграции.
- Главные риски / почему не факт: там много материалов про zero-shot, multi-instance prompting, few-shot memory matching и counting, что не совпадает с главным рабочим треком проекта.
- Стартовый способ проверки: использовать note только выборочно, через вопрос "есть ли прямой hook в `supervised_v4`, `segformer_semisup`, `review_bundle` или `dinov2_research`?"

## Анти-ловушки

- Не считать никакие ожидаемые приросты из внешних заметок фактами проекта. Это гипотезы до тех пор, пока нет локальных артефактов.
- Не путать few-shot / one-shot segmentation с текущим основным форматом проекта. Эти материалы можно использовать как источник идей для refinement, prompts, occlusion handling и foundation-model tooling, но не как прямой roadmap замены `supervised_v4`.
- Не переносить `SAM2`, `FSSAM`, `SANSA`, `FS-SAM2`, `FS-DINO` в mainline без отдельного scoped validation plan, который объясняет: зачем это лучше текущего supervised/semi-sup стека, какой минимальный эксперимент это проверит и какой compute budget допустим.
- Не повторять старую ошибку "широкая приемка pseudo-labels = хорошо". Для этого проекта уже подтверждено обратное.
- Не использовать `dinov2_research` как аргумент за смену основного pipeline. Пока это полезная proxy-research ветка, а не production-default.
- Если следующая работа касается semi-sup, сначала смотреть на pseudo-label quality governance, calibration и source filtering, а уже потом на более тяжёлые architectural rewrites.
