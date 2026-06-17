# Alpha Master: Credit Scoring

Решение задачи кредитного скоринга по истории кредитов клиента.

Финальный public ROC-AUC: `0.785021`.

Финальный файл для отправки: `submission.csv`.

## Идея решения

Исходные данные рассматриваются не просто как плоская таблица, а как последовательность кредитных событий клиента:

```text
client id -> credit 1 -> credit 2 -> ... -> credit N -> probability of default
```

Основной прирост дала нейросеть Alfa-style GRU:

```text
categorical fields -> field embeddings -> credit vector -> BiGRU -> pooling -> MLP -> prediction
```

Финальный результат получен не одной моделью, а ансамблем:

- Alfa-style GRU по последовательности кредитов;
- несколько seed одной GRU-архитектуры;
- TCN+GRU вариант;
- GRU после masked-field pretraining;
- небольшой CNN-сигнал по порядку `id`;
- финальный conservative blend.

## Основные файлы

```text
neural.py              подготовка sequence-датасета из parquet
alfabank_gru.py        основная GRU-модель по истории кредитов
masked_pretrain.py     self-supervised pretraining на восстановление признаков
id_target_cnn.py       CNN по локальным target-rate признакам вокруг id
final_stacking.py      сборка финального локального ансамбля
conservative_blends.py финальный осторожный blend для submission
```

Вспомогательные файлы:

```text
baseline.py            пути к данным и старый baseline-код
blend.py               rank-percentile и запись compact submission
blend_new_methods.py   загрузка validation-предсказаний и id-prior
gru_sequence.py        ranking loss для GRU
```

Документация:

```text
SOLUTION.md            краткое описание финального решения
HISTORY.md             история экспериментов и метрик
```

## Данные

В корне проекта должны лежать файлы соревнования:

```text
train_data.parquet
test_data.parquet
train_target.csv
sample_submission (1).csv
```

Эти файлы не хранятся в git, потому что они большие.

## Установка

Для CPU-части:

```bash
python -m pip install -r requirements.txt
```

Для нейросетевых экспериментов нужен PyTorch с CUDA. На GPU-сервере обычно достаточно:

```bash
python -m pip install -r requirements-neural.txt
```

Если PyTorch не установлен:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Подготовка последовательностей

```bash
python neural.py prepare --max-len 64 --partitions 32
```

Скрипт создаст директорию `neural_artifacts/` с `.npy` shard-файлами:

```text
train_sequences/
test_sequences/
metadata.json
```

## Обучение основной GRU

Пример запуска одной Alfa-style GRU:

```bash
python alfabank_gru.py all \
  --artifact-dir neural_artifacts \
  --run-name alfa_gru_seed777 \
  --seed 777
```

Для TCN+GRU:

```bash
python alfabank_gru.py all \
  --artifact-dir neural_artifacts \
  --architecture tcn_gru \
  --run-name alfa_tcn64_seed4242 \
  --seed 4242
```

## Masked Pretraining

Предобучение на восстановление скрытых признаков:

```bash
python masked_pretrain.py \
  --artifact-dir neural_artifacts \
  --output-name alfa_masked_pretrained.pt
```

Fine-tuning GRU с предобученными весами:

```bash
python alfabank_gru.py all \
  --artifact-dir neural_artifacts \
  --pretrained-path neural_artifacts/alfa_masked_pretrained.pt \
  --run-name alfa_pretrained64_seed9001 \
  --seed 9001
```

## ID Target CNN

Дополнительная модель, которая использует локальные target-rate признаки вокруг `id`:

```bash
python id_target_cnn.py all --run-name id_target_cnn_local_seed111 --seed 111
```

## Финальный ансамбль

Сборка локального stacking:

```bash
python final_stacking.py
```

Сборка conservative blend:

```bash
python conservative_blends.py
```

Финальный выбранный файл:

```text
artifacts/submission_conservative_70_30.csv
```

Для отправки он копируется в:

```text
submission.csv
```

## Валидация

Во всех основных экспериментах используется одно и то же разбиение:

```text
id % 10 == 0  -> validation
id % 10 != 0  -> train
```

Метрика: ROC-AUC.

## Итог

Финальный submission является blend-ом:

```text
70% final_stacking
30% Alfa GRU multiseed ensemble
```

Public leaderboard ROC-AUC:

```text
0.785021
```
