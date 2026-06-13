from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from baseline import ARTIFACTS, SAMPLE_SUBMISSION
from blend import rank_percentile, validation_prediction, write_compact_submission


def sorted_neural_validation() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(ARTIFACTS / "transformer_validation.npz")
    order = np.argsort(data["id"], kind="mergesort")
    return (
        data["id"][order],
        data["target"][order],
        data["prediction"][order],
    )


def main() -> None:
    neural_ids, targets, neural_prediction = sorted_neural_validation()
    advanced_ids, advanced_prediction = validation_prediction(
        ARTIFACTS / "train_features_advanced.parquet",
        ARTIFACTS / "advanced_lgbm.txt",
    )
    enhanced_ids, enhanced_prediction = validation_prediction(
        ARTIFACTS / "train_features_enhanced.parquet",
        ARTIFACTS / "enhanced_lgbm.txt",
    )
    if not np.array_equal(neural_ids, advanced_ids):
        raise ValueError("Neural and advanced validation ids differ")
    if not np.array_equal(neural_ids, enhanced_ids):
        raise ValueError("Neural and enhanced validation ids differ")

    validation_ranks = {
        "neural": rank_percentile(neural_prediction),
        "advanced": rank_percentile(advanced_prediction),
        "enhanced": rank_percentile(enhanced_prediction),
    }
    results: list[tuple[float, float, float, float]] = []
    for neural_weight in np.arange(0.40, 1.001, 0.05):
        remaining = 1.0 - neural_weight
        for advanced_weight in np.arange(0.0, remaining + 0.001, 0.05):
            enhanced_weight = remaining - advanced_weight
            prediction = (
                neural_weight * validation_ranks["neural"]
                + advanced_weight * validation_ranks["advanced"]
                + enhanced_weight * validation_ranks["enhanced"]
            )
            auc = roc_auc_score(targets, prediction)
            results.append(
                (
                    auc,
                    float(neural_weight),
                    float(advanced_weight),
                    float(enhanced_weight),
                )
            )

    results.sort(reverse=True)
    for auc, neural_weight, advanced_weight, enhanced_weight in results[:10]:
        print(
            f"auc={auc:.8f} neural={neural_weight:.2f} "
            f"advanced={advanced_weight:.2f} enhanced={enhanced_weight:.2f}"
        )

    best_auc, neural_weight, advanced_weight, enhanced_weight = results[0]
    sample_ids = pl.read_csv(
        SAMPLE_SUBMISSION, schema_overrides={"id": pl.Int32}
    )["id"].to_numpy()
    submissions = {
        "neural": ARTIFACTS / "submission_transformer.csv",
        "advanced": ARTIFACTS / "submission_advanced.csv",
        "enhanced": ARTIFACTS / "submission_enhanced.csv",
    }
    test_ranks: dict[str, np.ndarray] = {}
    for name, path in submissions.items():
        submission = pl.read_csv(path)
        if not np.array_equal(submission["id"].to_numpy(), sample_ids):
            raise ValueError(f"Submission order differs for {name}")
        test_ranks[name] = rank_percentile(submission["flag"].to_numpy())

    prediction = (
        neural_weight * test_ranks["neural"]
        + advanced_weight * test_ranks["advanced"]
        + enhanced_weight * test_ranks["enhanced"]
    )
    output = ARTIFACTS / "submission_neural_blend.csv"
    write_compact_submission(sample_ids, prediction, output)
    print(f"Saved {output} with validation AUC {best_auc:.8f}")


if __name__ == "__main__":
    main()
