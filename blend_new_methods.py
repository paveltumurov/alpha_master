from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from baseline import ARTIFACTS, SAMPLE_SUBMISSION, TRAIN_TARGET
from blend import rank_percentile, write_compact_submission


TIME_WINDOW = 200_000
TIME_SMOOTHING = 20.0
CNN_VALIDATION_PATH = ARTIFACTS / "id_target_cnn_validation_seed5150.npz"
CNN_SUBMISSION_PATH = ARTIFACTS / "submission_id_target_cnn_seed5150.csv"
ALFA_GRU_WEIGHTS = {
    777: 0.2141,
    137: 0.2804,
    2026: 0.2753,
}
FINAL_TIME_WEIGHT = 0.0121
FINAL_CNN_WEIGHT = 0.0257

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


def id_time_prior(
    train_ids: np.ndarray,
    train_targets: np.ndarray,
    query_ids: np.ndarray,
    excluded_ids: np.ndarray | None = None,
) -> np.ndarray:
    max_id = int(max(train_ids.max(), query_ids.max())) + 1
    target_sums = np.zeros(max_id, dtype=np.float64)
    target_counts = np.zeros(max_id, dtype=np.int32)

    included = np.ones(train_ids.size, dtype=bool)
    if excluded_ids is not None:
        included &= ~np.isin(train_ids, excluded_ids, assume_unique=False)
    included_ids = train_ids[included]
    target_sums[included_ids] = train_targets[included]
    target_counts[included_ids] = 1

    global_rate = float(train_targets[included].mean())
    sum_prefix = np.concatenate(([0.0], np.cumsum(target_sums)))
    count_prefix = np.concatenate(([0], np.cumsum(target_counts)))
    half_window = TIME_WINDOW // 2
    left = np.maximum(query_ids.astype(np.int64) - half_window, 0)
    right = np.minimum(query_ids.astype(np.int64) + half_window + 1, max_id)
    window_sums = sum_prefix[right] - sum_prefix[left]
    window_counts = count_prefix[right] - count_prefix[left]
    return (window_sums + TIME_SMOOTHING * global_rate) / (
        window_counts + TIME_SMOOTHING
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

    base_auc = roc_auc_score(validation_targets, validation_blend)
    print(f"base validation ROC-AUC={base_auc:.9f}")
    print(f"weights={WEIGHTS}")

    target = pl.read_csv(
        TRAIN_TARGET,
        schema_overrides={"id": pl.Int32, "flag": pl.UInt8},
    )
    train_ids = target["id"].to_numpy()
    train_targets = target["flag"].to_numpy()
    validation_time_rank = rank_percentile(
        id_time_prior(
            train_ids,
            train_targets,
            validation_ids,
            excluded_ids=validation_ids,
        )
    )
    time_auc = roc_auc_score(validation_targets, validation_time_rank)
    cnn_ids, cnn_targets, cnn_prediction = load_validation(
        CNN_VALIDATION_PATH
    )
    if not np.array_equal(validation_ids, cnn_ids):
        raise ValueError("CNN validation ids differ")
    if not np.array_equal(validation_targets, cnn_targets):
        raise ValueError("CNN validation targets differ")
    cnn_rank = rank_percentile(cnn_prediction)
    cnn_auc = roc_auc_score(validation_targets, cnn_rank)

    alfa_validation_blend = np.zeros(validation_ids.size, dtype=np.float64)
    alfa_aucs = {}
    for seed, weight in ALFA_GRU_WEIGHTS.items():
        alfa_ids, alfa_targets, alfa_prediction = load_validation(
            ARTIFACTS / f"alfa_gru_validation_seed{seed}.npz"
        )
        if not np.array_equal(validation_ids, alfa_ids):
            raise ValueError(f"Alfa GRU validation ids differ for seed {seed}")
        if not np.array_equal(validation_targets, alfa_targets):
            raise ValueError(
                f"Alfa GRU validation targets differ for seed {seed}"
            )
        alfa_rank = rank_percentile(alfa_prediction)
        alfa_aucs[seed] = roc_auc_score(validation_targets, alfa_rank)
        alfa_validation_blend += weight * alfa_rank
    main_weight = (
        1.0
        - FINAL_TIME_WEIGHT
        - FINAL_CNN_WEIGHT
        - sum(ALFA_GRU_WEIGHTS.values())
    )
    final_validation_prediction = (
        main_weight * validation_blend
        + FINAL_TIME_WEIGHT * validation_time_rank
        + FINAL_CNN_WEIGHT * cnn_rank
        + alfa_validation_blend
    )
    best_auc = roc_auc_score(
        validation_targets,
        final_validation_prediction,
    )
    print(f"id time prior ROC-AUC={time_auc:.9f}")
    print(f"id target CNN ROC-AUC={cnn_auc:.9f}")
    print(f"Alfa-style GRU ROC-AUCs={alfa_aucs}")
    print(
        f"final blend ROC-AUC={best_auc:.9f}, "
        f"main_weight={main_weight:.4f}, "
        f"time_weight={FINAL_TIME_WEIGHT:.4f}, "
        f"cnn_weight={FINAL_CNN_WEIGHT:.4f}, "
        f"alfa_gru_weights={ALFA_GRU_WEIGHTS}"
    )

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

    test_time_rank = rank_percentile(
        id_time_prior(train_ids, train_targets, sample_ids)
    )
    cnn_submission = pl.read_csv(CNN_SUBMISSION_PATH)
    if not np.array_equal(cnn_submission["id"].to_numpy(), sample_ids):
        raise ValueError("CNN submission order differs from sample")
    test_cnn_rank = rank_percentile(cnn_submission["flag"].to_numpy())
    test_alfa_blend = np.zeros(sample_ids.size, dtype=np.float64)
    for seed, weight in ALFA_GRU_WEIGHTS.items():
        alfa_submission = pl.read_csv(
            ARTIFACTS / f"submission_alfa_gru_seed{seed}.csv"
        )
        if not np.array_equal(alfa_submission["id"].to_numpy(), sample_ids):
            raise ValueError(
                f"Alfa GRU submission order differs for seed {seed}"
            )
        test_alfa_blend += weight * rank_percentile(
            alfa_submission["flag"].to_numpy()
        )
    final_prediction = (
        main_weight * test_blend
        + FINAL_TIME_WEIGHT * test_time_rank
        + FINAL_CNN_WEIGHT * test_cnn_rank
        + test_alfa_blend
    )
    output = ARTIFACTS / "submission_alfa_gru_multiseed_blend.csv"
    write_compact_submission(sample_ids, final_prediction, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
