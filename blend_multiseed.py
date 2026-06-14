from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from baseline import ARTIFACTS, SAMPLE_SUBMISSION
from blend import rank_percentile, validation_prediction, write_compact_submission


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
    ids42, targets, prediction42 = load_validation(
        ARTIFACTS / "transformer_validation.npz"
    )
    ids137, targets137, prediction137 = load_validation(
        ARTIFACTS / "transformer_validation_seed137.npz"
    )
    ids2026, targets2026, prediction2026 = load_validation(
        ARTIFACTS / "transformer_validation_seed2026.npz"
    )
    hybrid_ids, hybrid_targets, hybrid_prediction = load_validation(
        ARTIFACTS / "hybrid_validation_seed42.npz"
    )
    engineered_ids, engineered_targets, engineered_prediction = load_validation(
        ARTIFACTS / "engineered_validation.npz"
    )
    advanced_ids, advanced_prediction = validation_prediction(
        ARTIFACTS / "train_features_advanced.parquet",
        ARTIFACTS / "advanced_lgbm.txt",
    )
    if not (
        np.array_equal(ids42, ids137)
        and np.array_equal(ids42, ids2026)
        and np.array_equal(ids42, hybrid_ids)
        and np.array_equal(ids42, engineered_ids)
        and np.array_equal(ids42, advanced_ids)
    ):
        raise ValueError("Validation ids differ")
    if not (
        np.array_equal(targets, targets137)
        and np.array_equal(targets, targets2026)
        and np.array_equal(targets, hybrid_targets)
        and np.array_equal(targets, engineered_targets)
    ):
        raise ValueError("Validation targets differ")

    rank42 = rank_percentile(prediction42)
    rank137 = rank_percentile(prediction137)
    rank2026 = rank_percentile(prediction2026)
    hybrid_rank = rank_percentile(hybrid_prediction)
    advanced_rank = rank_percentile(advanced_prediction)
    engineered_rank = rank_percentile(engineered_prediction)

    seed_results = []
    for weight42 in np.arange(0.0, 1.001, 0.025):
        remaining = 1.0 - weight42
        for weight137 in np.arange(0.0, remaining + 0.001, 0.025):
            weight2026 = remaining - weight137
            seed_rank = (
                weight42 * rank42
                + weight137 * rank137
                + weight2026 * rank2026
            )
            seed_results.append(
                (
                    roc_auc_score(targets, seed_rank),
                    float(weight42),
                    float(weight137),
                    float(weight2026),
                    seed_rank,
                )
            )
    seed_results.sort(key=lambda item: item[0], reverse=True)
    seed_auc, weight42, weight137, weight2026, seed_rank = seed_results[0]
    print(
        f"best seed blend auc={seed_auc:.8f} "
        f"seed42={weight42:.3f} seed137={weight137:.3f} "
        f"seed2026={weight2026:.3f}"
    )

    results = []
    for seed_weight in np.arange(0.40, 0.951, 0.025):
        non_seed_weight = 1.0 - seed_weight
        for hybrid_weight in np.arange(0.0, non_seed_weight + 0.001, 0.025):
            tree_weight = non_seed_weight - hybrid_weight
            for advanced_weight in np.arange(0.0, tree_weight + 0.001, 0.025):
                engineered_weight = tree_weight - advanced_weight
                prediction = (
                    seed_weight * seed_rank
                    + hybrid_weight * hybrid_rank
                    + advanced_weight * advanced_rank
                    + engineered_weight * engineered_rank
                )
                results.append(
                    (
                        roc_auc_score(targets, prediction),
                        float(seed_weight),
                        float(hybrid_weight),
                        float(advanced_weight),
                        float(engineered_weight),
                    )
                )
    results.sort(reverse=True)
    for (
        auc,
        seed_weight,
        hybrid_weight,
        advanced_weight,
        engineered_weight,
    ) in results[:10]:
        print(
            f"auc={auc:.8f} seeds={seed_weight:.3f} "
            f"hybrid={hybrid_weight:.3f} advanced={advanced_weight:.3f} "
            f"engineered={engineered_weight:.3f}"
        )

    sample_ids = pl.read_csv(
        SAMPLE_SUBMISSION, schema_overrides={"id": pl.Int32}
    )["id"].to_numpy()
    paths = {
        "seed42": ARTIFACTS / "submission_transformer.csv",
        "seed137": ARTIFACTS / "submission_transformer_seed137.csv",
        "seed2026": ARTIFACTS / "submission_transformer_seed2026.csv",
        "hybrid": ARTIFACTS / "submission_hybrid_seed42.csv",
        "advanced": ARTIFACTS / "submission_advanced.csv",
        "engineered": ARTIFACTS / "submission_engineered.csv",
    }
    test_ranks = {}
    for name, path in paths.items():
        frame = pl.read_csv(path)
        if not np.array_equal(frame["id"].to_numpy(), sample_ids):
            raise ValueError(f"Submission order differs for {name}")
        test_ranks[name] = rank_percentile(frame["flag"].to_numpy())

    seed_test_rank = (
        weight42 * test_ranks["seed42"]
        + weight137 * test_ranks["seed137"]
        + weight2026 * test_ranks["seed2026"]
    )
    (
        best_auc,
        seed_weight,
        hybrid_weight,
        advanced_weight,
        engineered_weight,
    ) = results[0]
    prediction = (
        seed_weight * seed_test_rank
        + hybrid_weight * test_ranks["hybrid"]
        + advanced_weight * test_ranks["advanced"]
        + engineered_weight * test_ranks["engineered"]
    )
    output = ARTIFACTS / "submission_multiseed_blend.csv"
    write_compact_submission(sample_ids, prediction, output)
    print(f"Saved {output} with validation AUC {best_auc:.8f}")


if __name__ == "__main__":
    main()
