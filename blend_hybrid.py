from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from baseline import ARTIFACTS, SAMPLE_SUBMISSION
from blend import rank_percentile, validation_prediction, write_compact_submission


def load_validation(path):
    data = np.load(path)
    order = np.argsort(data["id"], kind="mergesort")
    return data["id"][order], data["target"][order], data["prediction"][order]


def main() -> None:
    neural_ids, targets, neural = load_validation(
        ARTIFACTS / "transformer_validation.npz"
    )
    hybrid_ids, hybrid_targets, hybrid = load_validation(
        ARTIFACTS / "hybrid_validation_seed42.npz"
    )
    advanced_ids, advanced = validation_prediction(
        ARTIFACTS / "train_features_advanced.parquet",
        ARTIFACTS / "advanced_lgbm.txt",
    )
    if not np.array_equal(neural_ids, hybrid_ids):
        raise ValueError("Neural and hybrid ids differ")
    if not np.array_equal(neural_ids, advanced_ids):
        raise ValueError("Neural and advanced ids differ")
    if not np.array_equal(targets, hybrid_targets):
        raise ValueError("Validation targets differ")

    ranks = {
        "neural": rank_percentile(neural),
        "hybrid": rank_percentile(hybrid),
        "advanced": rank_percentile(advanced),
    }
    results = []
    for neural_weight in np.arange(0.40, 0.951, 0.025):
        remaining = 1.0 - neural_weight
        for hybrid_weight in np.arange(0.0, remaining + 0.001, 0.025):
            advanced_weight = remaining - hybrid_weight
            prediction = (
                neural_weight * ranks["neural"]
                + hybrid_weight * ranks["hybrid"]
                + advanced_weight * ranks["advanced"]
            )
            results.append(
                (
                    roc_auc_score(targets, prediction),
                    float(neural_weight),
                    float(hybrid_weight),
                    float(advanced_weight),
                )
            )
    results.sort(reverse=True)
    for auc, neural_weight, hybrid_weight, advanced_weight in results[:10]:
        print(
            f"auc={auc:.8f} neural={neural_weight:.3f} "
            f"hybrid={hybrid_weight:.3f} advanced={advanced_weight:.3f}"
        )

    best_auc, neural_weight, hybrid_weight, advanced_weight = results[0]
    sample_ids = pl.read_csv(
        SAMPLE_SUBMISSION, schema_overrides={"id": pl.Int32}
    )["id"].to_numpy()
    submissions = {
        "neural": ARTIFACTS / "submission_transformer.csv",
        "hybrid": ARTIFACTS / "submission_hybrid_seed42.csv",
        "advanced": ARTIFACTS / "submission_advanced.csv",
    }
    test_ranks = {}
    for name, path in submissions.items():
        frame = pl.read_csv(path)
        if not np.array_equal(frame["id"].to_numpy(), sample_ids):
            raise ValueError(f"Submission order differs for {name}")
        test_ranks[name] = rank_percentile(frame["flag"].to_numpy())

    prediction = (
        neural_weight * test_ranks["neural"]
        + hybrid_weight * test_ranks["hybrid"]
        + advanced_weight * test_ranks["advanced"]
    )
    output = ARTIFACTS / "submission_hybrid_blend.csv"
    write_compact_submission(sample_ids, prediction, output)
    print(f"Saved {output} with validation AUC {best_auc:.8f}")


if __name__ == "__main__":
    main()
