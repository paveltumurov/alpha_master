from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from baseline import ARTIFACTS, SAMPLE_SUBMISSION
from blend import rank_percentile, write_compact_submission


WEIGHTS = {
    "transformer42": 0.147625,
    "transformer137": 0.130375,
    "transformer2026": 0.272,
    "hybrid": 0.059,
    "engineered": 0.116,
    "gru": 0.275,
}

VALIDATION_PATHS = {
    "transformer42": ARTIFACTS / "transformer_validation.npz",
    "transformer137": ARTIFACTS / "transformer_validation_seed137.npz",
    "transformer2026": ARTIFACTS / "transformer_validation_seed2026.npz",
    "hybrid": ARTIFACTS / "hybrid_validation_seed42.npz",
    "engineered": ARTIFACTS / "engineered_validation.npz",
    "gru": ARTIFACTS / "gru_validation_seed314.npz",
}

SUBMISSION_PATHS = {
    "transformer42": ARTIFACTS / "submission_transformer.csv",
    "transformer137": ARTIFACTS / "submission_transformer_seed137.csv",
    "transformer2026": ARTIFACTS / "submission_transformer_seed2026.csv",
    "hybrid": ARTIFACTS / "submission_hybrid_seed42.csv",
    "engineered": ARTIFACTS / "submission_engineered.csv",
    "gru": ARTIFACTS / "submission_gru_seed314.csv",
}


def load_validation(path):
    data = np.load(path)
    id_key = "id" if "id" in data else "ids"
    target_key = "target" if "target" in data else "targets"
    prediction_key = "prediction" if "prediction" in data else "predictions"
    order = np.argsort(data[id_key], kind="mergesort")
    return (
        data[id_key][order],
        data[target_key][order],
        data[prediction_key][order],
    )


def main() -> None:
    validation_ids = None
    validation_targets = None
    validation_blend = None
    for name, path in VALIDATION_PATHS.items():
        ids, targets, prediction = load_validation(path)
        if validation_ids is None:
            validation_ids = ids
            validation_targets = targets
            validation_blend = np.zeros(ids.size, dtype=np.float64)
        elif not np.array_equal(validation_ids, ids):
            raise ValueError(f"Validation ids differ for {name}")
        elif not np.array_equal(validation_targets, targets):
            raise ValueError(f"Validation targets differ for {name}")
        validation_blend += WEIGHTS[name] * rank_percentile(prediction)

    auc = roc_auc_score(validation_targets, validation_blend)
    print(f"validation ROC-AUC={auc:.9f}")
    print(f"weights={WEIGHTS}")

    sample_ids = pl.read_csv(
        SAMPLE_SUBMISSION,
        schema_overrides={"id": pl.Int32},
    )["id"].to_numpy()
    test_blend = np.zeros(sample_ids.size, dtype=np.float64)
    for name, path in SUBMISSION_PATHS.items():
        submission = pl.read_csv(path)
        if not np.array_equal(submission["id"].to_numpy(), sample_ids):
            raise ValueError(f"Submission order differs for {name}")
        test_blend += WEIGHTS[name] * rank_percentile(
            submission["flag"].to_numpy()
        )

    output = ARTIFACTS / "submission_gru_blend.csv"
    write_compact_submission(sample_ids, test_blend, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
