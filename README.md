# Alpha Master Credit Scoring

Final solution for the Alfa Bank credit scoring competition.

Public leaderboard ROC-AUC: `0.785021`.

Final submission file: `submission.csv`.

## Method

The key idea is to treat each client as a sequence of credit events, not as one flat row.

```text
client id -> credit 1 -> credit 2 -> ... -> credit N -> default probability
```

The strongest single family of models is an Alfa-style GRU:

```text
categorical fields -> field embeddings -> credit vector -> BiGRU -> pooling -> MLP -> prediction
```

The final result is an ensemble of several sequence models and auxiliary signals:

- Alfa-style GRU over credit histories;
- multi-seed GRU ensemble;
- TCN+GRU sequence model;
- GRU initialized from masked-field pretraining;
- local CNN signal over ordered `id`;
- conservative final blend.

## Repository Layout

Main final-solution scripts:

```text
neural.py              builds sequence datasets from parquet files
alfabank_gru.py        main field-embedding GRU model
masked_pretrain.py     self-supervised masked-field pretraining
id_target_cnn.py       CNN over local target-rate features around id
final_stacking.py      final local model stacking
conservative_blends.py final conservative blend
```

Required helper modules:

```text
baseline.py            shared paths and baseline utilities
blend.py               rank normalization and compact CSV writer
blend_new_methods.py   validation loading and id-prior utilities
gru_sequence.py        ranking loss used by GRU training
```

Project helpers:

```text
scripts/check_inputs.sh  checks that competition files exist
scripts/run_final.sh     documents the final reproducible pipeline
docs/structure.md        short project structure notes
SOLUTION.md              final solution summary
HISTORY.md               experiment history and scores
```

## Input Data

Put the competition files in the repository root:

```text
train_data.parquet
test_data.parquet
train_target.csv
sample_submission (1).csv
```

Large input files, model checkpoints, caches, and submissions are intentionally not tracked by git.

## Installation

CPU dependencies:

```bash
python -m pip install -r requirements.txt
```

Neural experiments require PyTorch with CUDA. On a GPU server:

```bash
python -m pip install -r requirements-neural.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Pipeline

Check inputs:

```bash
bash scripts/check_inputs.sh
```

Prepare sequence shards:

```bash
python neural.py prepare --max-len 64 --partitions 32
```

Train one Alfa-style GRU:

```bash
python alfabank_gru.py all \
  --artifact-dir neural_artifacts \
  --run-name alfa_gru_seed777 \
  --seed 777
```

Train TCN+GRU:

```bash
python alfabank_gru.py all \
  --artifact-dir neural_artifacts \
  --architecture tcn_gru \
  --run-name alfa_tcn64_seed4242 \
  --seed 4242
```

Run masked-field pretraining:

```bash
python masked_pretrain.py \
  --artifact-dir neural_artifacts \
  --output-name alfa_masked_pretrained.pt
```

Fine-tune from pretrained weights:

```bash
python alfabank_gru.py all \
  --artifact-dir neural_artifacts \
  --pretrained-path neural_artifacts/alfa_masked_pretrained.pt \
  --run-name alfa_pretrained64_seed9001 \
  --seed 9001
```

Train the id-based CNN:

```bash
python id_target_cnn.py all --run-name id_target_cnn_local_seed111 --seed 111
```

Build the final local stack:

```bash
python final_stacking.py
```

Build conservative blends:

```bash
python conservative_blends.py
```

The selected final file is:

```text
artifacts/submission_conservative_70_30.csv
```

It is copied to:

```text
submission.csv
```

## One-Command Entrypoint

The repository also contains a documented wrapper:

```bash
bash scripts/run_final.sh
```

It shows the intended end-to-end order. Full retraining is expensive and assumes that all intermediate model predictions listed in `HISTORY.md` are regenerated or already present in `artifacts/`.

## Validation

The main validation split is stable across experiments:

```text
id % 10 == 0  -> validation
id % 10 != 0  -> train
```

Metric: ROC-AUC.

## Final Blend

The submitted file is a conservative blend:

```text
70% final_stacking
30% Alfa GRU multiseed ensemble
```

Public leaderboard ROC-AUC:

```text
0.785021
```
