# Final Solution

## File To Submit

Submit `submission.csv`.

It is copied from:

```text
artifacts/submission_conservative_70_30.csv
```

Validation checks:

- rows: `900000`
- columns: `id, flag`
- `id` order matches `sample_submission`
- predictions are finite and within `[0, 1]`
- file size: `24.766927 MB`

Public leaderboard score for this file: `0.785021`.

## Method Summary

The final prediction is a conservative blend:

- `70%` final local stacking model
- `30%` public-tested Alfa GRU multiseed ensemble

The final local stacking model combines:

- Alfa-style GRU models with separate field embeddings
- multi-seed GRU ensemble
- 64-step credit-history GRU snapshots
- TCN+GRU sequence model
- masked-field pretrained GRU-64
- target-based CNN over ordered `id`
- smoothed target prior over ordered `id`
- engineered LightGBM and earlier neural sequence models

The blend was chosen because it had strong local ROC-AUC and transferred well to
the public leaderboard.

## Reproducibility Notes

The main scripts for the final stages are:

```text
alfabank_gru.py
masked_pretrain.py
id_target_cnn.py
final_stacking.py
conservative_blends.py
```

The project history with local and public scores is recorded in `HISTORY.md`.
