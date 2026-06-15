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

## Stage 10: Alfa-Style Field-Embedding GRU

- A separate adaptive embedding layer for every credit-history field.
- Concatenated field embeddings projected to a shared event representation.
- Bidirectional GRU with attention, mean, max, and final-state pooling.
- Additional mean and max pooling over the input event embeddings.
- OneCycle learning-rate schedule and pairwise ranking loss.
- Standalone validation ROC-AUC: `0.782730`.
- Final ensemble validation ROC-AUC: `0.785597`.
- Public leaderboard ROC-AUC: `0.781945`.
- Final weights: 45.66% Alfa-style GRU, 50.52% previous model ensemble,
  2.26% target CNN, and 1.56% time prior.

## Stage 11: Alfa GRU Multi-Seed Ensemble

- Additional Alfa-style GRU models with seeds `137` and `2026`.
- Standalone validation ROC-AUC: `0.783373` for seed `137` and `0.783530`
  for seed `2026`.
- Three-seed Alfa GRU and previous-model ensemble ROC-AUC: `0.787455`.
- Public leaderboard ROC-AUC: `0.783512`; leaderboard position: 12.
- Final weights: 21.41% seed `777`, 28.04% seed `137`, 27.53% seed `2026`,
  19.24% previous ensemble, 2.57% target CNN, and 1.21% time prior.

## Stage 12: Long Context, TCN, and Self-Supervised Pretraining

- Alfa-style GRU over the last 64 credits with per-epoch snapshots.
- A parallel dilated TCN branch for local transitions between credits.
- Masked-field pretraining on unlabeled train and test credit histories.
- Fine-tuning of the pretrained field embeddings and recurrent backbone.
- Additional target CNN models with narrow and wide `id` neighborhoods.
- GRU-64 standalone ROC-AUC: `0.782686`.
- TCN+GRU standalone ROC-AUC: `0.782910`.
- Pretrained GRU-64 standalone ROC-AUC: `0.783683`.
- Final constrained stacking ROC-AUC: `0.788513`.
- Final stack: 24.97% previous ensemble, 10.23% GRU-64 epoch 9,
  34.27% TCN+GRU, 26.69% pretrained GRU-64, and 3.83% local target CNN.

## Validation Convention

The main validation fold contains clients satisfying `id % 10 == 0`.
This convention is kept unchanged across experiments so model changes can
be compared directly.
