# Alpha Master: кредитный скоринг

Финальное решение задачи кредитного скоринга Альфа-Банка.

Public ROC-AUC: `0.785021`.

Финальный файл для отправки: `submission.csv`.

## Главная идея

Данные рассматриваются не только как обычная таблица, а как история кредитов клиента:

```text
client id -> кредит 1 -> кредит 2 -> ... -> кредит N -> вероятность дефолта
```

Основной прирост дала нейросеть по последовательности кредитов:

```text
категориальные признаки -> embeddings -> вектор кредита -> BiGRU -> pooling -> MLP -> prediction
```

Финальный результат получен ансамблем нескольких моделей:

- Alfa-style GRU по последовательности кредитов;
- несколько GRU-моделей с разными seed;
- TCN+GRU;
- GRU после masked-field pretraining;
- небольшой CNN-сигнал по порядку `id`;
- финальный conservative blend.

## Что лежит в репозитории

Основные файлы финального решения:

```text
neural.py              подготовка sequence-датасета из parquet
alfabank_gru.py        основная GRU-модель по истории кредитов
masked_pretrain.py     self-supervised pretraining на восстановление признаков
id_target_cnn.py       CNN по локальным target-rate признакам вокруг id
final_stacking.py      сборка финального локального ансамбля
conservative_blends.py финальный осторожный blend
```

Вспомогательные модули, которые нужны финальным скриптам:

```text
baseline.py            общие пути к данным и artifacts
blend.py               rank-normalization и compact submission writer
blend_new_methods.py   загрузка validation-предсказаний и id-prior
gru_sequence.py        ranking loss для GRU
```

Документация и запуск:

```text
SOLUTION.md              краткое описание финального решения
HISTORY.md               история экспериментов и метрик
docs/structure.md        структура проекта
scripts/check_inputs.sh  проверка входных файлов
scripts/run_final.sh     документированный порядок запуска
```

## Данные

В корне проекта должны лежать файлы соревнования:

```text
train_data.parquet
test_data.parquet
train_target.csv
sample_submission (1).csv
```

Эти файлы не хранятся в git, потому что они большие. Также в git не входят:

```text
artifacts/
neural_artifacts/
submission.csv
*.pt
*.npy
*.npz
```

## Установка

CPU-зависимости:

```bash
python -m pip install -r requirements.txt
```

Для нейросетевых запусков нужен PyTorch с CUDA:

```bash
python -m pip install -r requirements-neural.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Подготовка sequence-данных

```bash
python neural.py prepare --max-len 64 --partitions 32
```

После этого создается `neural_artifacts/` с sequence-shard файлами и `metadata.json`.

## Обучение моделей

Одна Alfa-style GRU:

```bash
python alfabank_gru.py all \
  --artifact-dir neural_artifacts \
  --run-name alfa_gru_seed777 \
  --seed 777
```

TCN+GRU:

```bash
python alfabank_gru.py all \
  --artifact-dir neural_artifacts \
  --architecture tcn_gru \
  --run-name alfa_tcn64_seed4242 \
  --seed 4242
```

Masked-field pretraining:

```bash
python masked_pretrain.py \
  --artifact-dir neural_artifacts \
  --output-name alfa_masked_pretrained.pt
```

Fine-tuning после pretraining:

```bash
python alfabank_gru.py all \
  --artifact-dir neural_artifacts \
  --pretrained-path neural_artifacts/alfa_masked_pretrained.pt \
  --run-name alfa_pretrained64_seed9001 \
  --seed 9001
```

ID Target CNN:

```bash
python id_target_cnn.py all --run-name id_target_cnn_local_seed111 --seed 111
```

## Финальный ансамбль

Когда нужные validation-предсказания и submission-файлы уже лежат в `artifacts/`, запускается:

```bash
python final_stacking.py
python conservative_blends.py
```

Выбранный финальный файл:

```text
artifacts/submission_conservative_70_30.csv
```

Он копируется в:

```text
submission.csv
```

Финальный blend:

```text
70% final_stacking
30% Alfa GRU multiseed ensemble
```

## Важное про воспроизводимость

Полное переобучение финального решения долгое и требует GPU. Кроме того, `final_stacking.py` ожидает, что промежуточные предсказания уже есть в `artifacts/`.

Поэтому `scripts/run_final.sh` — это не магическая кнопка “быстро получить финальный скор”, а документированный порядок запуска основных этапов. История того, какие модели участвовали в финальном ансамбле, записана в `HISTORY.md`.

## Валидация

Во всех основных экспериментах использовалось стабильное разбиение:

```text
id % 10 == 0  -> validation
id % 10 != 0  -> train
```

Метрика: ROC-AUC.

Итоговый public ROC-AUC:

```text
0.785021
```
