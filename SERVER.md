# P100 neural pipeline

Рекомендуемый образ: Ubuntu 22.04 Machine Learning.

Проверка GPU:

```bash
nvidia-smi
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
```

Сначала используйте PyTorch, уже установленный в ML-образе. Остальные
зависимости:

```bash
python3 -m pip install -r requirements-neural.txt
```

Если `import torch` не работает, установите совместимую CUDA-сборку:

```bash
python3 -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

В каталоге проекта должны находиться:

- `neural.py`;
- `train_data.parquet`;
- `test_data.parquet`;
- `train_target.csv`;
- `sample_submission (1).csv`.

Полный запуск:

```bash
python3 neural.py all
```

Надёжнее запускать этапы отдельно:

```bash
python3 neural.py prepare
python3 neural.py train
python3 neural.py predict
```

Чтобы процесс продолжил работу после закрытия SSH:

```bash
tmux new -s scoring
python3 neural.py train 2>&1 | tee neural_train.log
```

Отсоединение от `tmux`: `Ctrl+B`, затем `D`.

Возврат:

```bash
tmux attach -t scoring
```

Результат:

```text
neural_artifacts/submission_transformer.csv
neural_artifacts/transformer_validation.npz
```

Для дополнительных seed:

```bash
python3 neural.py train --seed 137
python3 neural.py predict --seed 137
```

Если возникнет CUDA out of memory:

```bash
python3 neural.py train --batch-size 256
```

Hybrid Transformer с полной историей и advanced-агрегатами:

```bash
python3 hybrid.py prepare
python3 hybrid.py train --seed 42
python3 hybrid.py predict --seed 42
```
