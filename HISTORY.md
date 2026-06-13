# Project History

## Stage 1: Baseline

- Disk-backed hash partitioning for 8 GB RAM.
- Per-client aggregation: loan count, mean, and maximum.
- LightGBM with validation on `id % 10 == 0`.
- Local ROC-AUC: `0.738537`.
- Public leaderboard ROC-AUC: `0.736305`.

## Stage 2: Enhanced Features

- Last credit product features.
- Mean of the three most recent products.
- Standard deviations for core credit fields.
- Payment-history and overdue summaries.
- Local ROC-AUC: `0.756112`.
- Baseline/enhanced rank blend: `0.756139`.

## Stage 3: Advanced Categorical Features

- Exact category frequencies for credit type, status, currency, and holder type.
- Utilization-bin histograms.
- Payment-state frequencies.
- Feature selection by enhanced-model gain to fit into limited RAM.
- Local ROC-AUC: `0.764295`.
- Advanced/enhanced rank blend: `0.764637`.

## Stage 4: Neural Sequence Model

- Transformer over the last 32 credit products.
- Categorical embeddings for all source fields.
- CUDA FP16 training intended for an NVIDIA Tesla P100 16 GB.
- Sharded memory-mapped datasets for a 28 GB RAM server.
- Validation predictions are saved for blending with tree models.

## Validation Convention

The main validation fold contains clients satisfying `id % 10 == 0`.
This convention is kept unchanged across experiments so model changes can
be compared directly.
