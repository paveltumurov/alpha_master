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
- Transformer validation ROC-AUC: `0.776521`.
- Transformer/advanced rank blend: `0.778598`.
- Best blend weights: 75% Transformer and 25% advanced LightGBM.

## Stage 5: Deep Learning Ensemble

- Hybrid Transformer with the full credit history and 321 aggregate features.
- Hybrid validation ROC-AUC: `0.773656`.
- Three sequence Transformer seeds: `42`, `137`, and `2026`.
- Three-seed rank ensemble ROC-AUC: `0.781449`.
- Final seed/hybrid/advanced ensemble ROC-AUC: `0.782113`.

## Stage 6: Sequential Feature Engineering

- Previous-credit values and last/first deltas.
- Trends, total variation, and change counts over `rn`.
- Payment-state transitions and dispersion.
- Loans since the latest serious overdue event.
- Utilization and overdue aggregates by credit type.
- Local one-shard A/B: `0.737239` advanced vs `0.737711` combined.
- Full-data LightGBM ROC-AUC: `0.767786` at iteration `1877`.
- Engineered ensemble ROC-AUC: `0.782217`.
- Engineered ensemble public leaderboard ROC-AUC: `0.778326`.
- Final weights: 75% three-seed Transformer, 10% hybrid, and 15%
  engineered LightGBM.

## Stage 7: Alternative Models

- GPU CatBoost on 473 engineered aggregate features: `0.763973`.
- CatBoost received zero ensemble weight because its predictions were too
  correlated with engineered LightGBM.
- Bidirectional GRU with attention pooling and pairwise ranking loss.
- GRU validation ROC-AUC: `0.777897`.
- Transformer/GRU/hybrid/engineered ensemble ROC-AUC: `0.782957`.
- Final GRU ensemble weight: 27.5%.

## Stage 8: ID Time Prior

- A centered rolling target-rate prior over the ordered `id` space.
- Validation labels are excluded while constructing the validation prior.
- Window width: `200000`; Bayesian smoothing strength: `20`.
- Standalone prior ROC-AUC: `0.528191`.
- Final ensemble ROC-AUC: `0.783245`.
- Final time-prior weight: 3.4%.

## Stage 9: ID Target CNN

- Dilated 1D-CNN trained only on multi-scale target-rate sequences ordered by
  `id`; no credit-history features are used.
- Each training object's own target is removed from its input.
- The complete validation fold is excluded from validation inputs.
- Standalone CNN ROC-AUC: `0.533723`.
- CNN/time-prior/main-model ensemble ROC-AUC: `0.783308`.
- Public leaderboard ROC-AUC: `0.779497`.
- Final weights: 2.3% CNN, 1.7% time prior, and 96% main ensemble.

## Validation Convention

The main validation fold contains clients satisfying `id % 10 == 0`.
This convention is kept unchanged across experiments so model changes can
be compared directly.
