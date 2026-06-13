from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from baseline import ARTIFACTS, SAMPLE_SUBMISSION, TRAIN_TARGET
from blend import rank_percentile, validation_prediction, write_compact_submission


MODELS = {
    "baseline": (
        ARTIFACTS / "train_features.parquet",
        ARTIFACTS / "baseline_lgbm.txt",
        ARTIFACTS / "submission_baseline.csv",
    ),
    "enhanced": (
        ARTIFACTS / "train_features_enhanced.parquet",
        ARTIFACTS / "enhanced_lgbm.txt",
        ARTIFACTS / "submission_enhanced.csv",
    ),
    "advanced": (
        ARTIFACTS / "train_features_advanced.parquet",
        ARTIFACTS / "advanced_lgbm.txt",
        ARTIFACTS / "submission_advanced.csv",
    ),
}


def main() -> None:
    validation_ids: np.ndarray | None = None
    validation_ranks: dict[str, np.ndarray] = {}
    for name, (features_path, model_path, _) in MODELS.items():
        ids, prediction = validation_prediction(features_path, model_path)
        if validation_ids is None:
            validation_ids = ids
        elif not np.array_equal(validation_ids, ids):
            raise ValueError(f"Validation ids differ for {name}")
        validation_ranks[name] = rank_percentile(prediction)

    target = (
        pl.read_csv(
            TRAIN_TARGET,
            schema_overrides={"id": pl.Int32, "flag": pl.UInt8},
        )
        .filter(pl.col("id") % 10 == 0)
        .sort("id")
    )
    y = target["flag"].to_numpy()

    results: list[tuple[float, float, float, float]] = []
    for advanced_weight in np.arange(0.60, 1.001, 0.05):
        remaining = 1.0 - advanced_weight
        for baseline_weight in np.arange(0.0, remaining + 0.001, 0.05):
            enhanced_weight = remaining - baseline_weight
            prediction = (
                advanced_weight * validation_ranks["advanced"]
                + enhanced_weight * validation_ranks["enhanced"]
                + baseline_weight * validation_ranks["baseline"]
            )
            auc = roc_auc_score(y, prediction)
            results.append(
                (
                    auc,
                    float(advanced_weight),
                    float(enhanced_weight),
                    float(baseline_weight),
                )
            )

    results.sort(reverse=True)
    for auc, advanced_weight, enhanced_weight, baseline_weight in results[:10]:
        print(
            f"auc={auc:.8f} advanced={advanced_weight:.2f} "
            f"enhanced={enhanced_weight:.2f} baseline={baseline_weight:.2f}"
        )

    best_auc, advanced_weight, enhanced_weight, baseline_weight = results[0]
    sample_ids = pl.read_csv(
        SAMPLE_SUBMISSION, schema_overrides={"id": pl.Int32}
    )["id"].to_numpy()
    submission_ranks: dict[str, np.ndarray] = {}
    for name, (_, _, submission_path) in MODELS.items():
        submission = pl.read_csv(submission_path)
        if not np.array_equal(submission["id"].to_numpy(), sample_ids):
            raise ValueError(f"Submission order differs for {name}")
        submission_ranks[name] = rank_percentile(submission["flag"].to_numpy())

    prediction = (
        advanced_weight * submission_ranks["advanced"]
        + enhanced_weight * submission_ranks["enhanced"]
        + baseline_weight * submission_ranks["baseline"]
    )
    output = ARTIFACTS / "submission_advanced_blend.csv"
    write_compact_submission(sample_ids, prediction, output)
    print(f"Saved {output} with validation AUC {best_auc:.8f}")


if __name__ == "__main__":
    main()
