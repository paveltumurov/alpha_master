# Credit scoring baseline

Потоковый baseline для компьютера с 8 ГБ RAM:

1. Кредитная история агрегируется до одной строки на `id`.
2. Для каждого исходного признака считаются `mean` и `max`.
3. LightGBM обучается с валидацией по стабильному хешу `id`.
4. Дополнительно измеряется ROC-AUC на последних 10% `id`.

Установка:

```powershell
python -m pip install -r requirements.txt
```

Полный запуск:

```powershell
python baseline.py all
```

При первом запуске исходные строки распределяются по 32 дисковым
корзинам по остатку от деления `id`. Затем каждая небольшая корзина
агрегируется независимо. Промежуточные данные кэшируются в `artifacts/`.

Этапы можно запускать отдельно:

```powershell
python baseline.py aggregate
python baseline.py train
python baseline.py predict
```

Улучшенная версия с признаками последнего кредита, последних трёх
продуктов, разбросом ключевых полей, сводками платежей и просрочек:

```powershell
python enhanced.py all
```

Она сохраняет отдельные файлы `metrics_enhanced.json`,
`enhanced_lgbm.txt` и `submission_enhanced.csv`, не перезаписывая baseline.

После обучения обеих моделей можно подобрать rank-ансамбль на том же
holdout и создать `submission_blend.csv`:

```powershell
python blend.py
```

Третья версия добавляет точные частоты категорий и платежных состояний:

```powershell
python advanced.py all
```

Ансамбль baseline, enhanced и advanced:

```powershell
python blend_advanced.py
```

Нейросетевая модель последовательности кредитов для сервера с NVIDIA P100
описана в `SERVER.md` и запускается через `neural.py`.

Результаты сохраняются в `artifacts/`:

- `metrics.json` — локальные метрики;
- `baseline_lgbm.txt` — модель;
- `submission_baseline.csv` — файл для отправки.

Вероятности в submission записываются с точностью до 18 знаков после
запятой. Конечные и ведущие нули не записываются, чтобы CSV оставался
меньше ограничения в 25 МБ.
