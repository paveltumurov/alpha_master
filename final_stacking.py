from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from baseline import ARTIFACTS, SAMPLE_SUBMISSION, TRAIN_TARGET
from blend import rank_percentile, write_compact_submission
from blend_new_methods import (
    ALFA_GRU_WEIGHTS,
    CNN_VALIDATION_PATH,
    FINAL_CNN_WEIGHT,
    FINAL_TIME_WEIGHT,
    VALIDATION_PATHS,
    WEIGHTS,
    id_time_prior,
    load_validation,
)


STACK_WEIGHTS = {
    "current": 0.249725,
    "gru64_epoch9": 0.102310,
    "tcn64": 0.342734,
    "pretrained64": 0.266927,
    "cnn_local": 0.038304,
}

EXTRA_VALIDATION_PATHS = {
    "gru64_epoch9": ARTIFACTS
    / "alfa_gru64_seed42_epoch9_validation.npz",
    "tcn64": ARTIFACTS / "alfa_tcn64_seed4242_validation.npz",
    "pretrained64": ARTIFACTS
    / "alfa_pretrained64_seed9001_validation.npz",
    "cnn_local": ARTIFACTS
    / "id_target_cnn_local_seed111_validation.npz",
}

SUBMISSION_PATHS = {
    "current": ARTIFACTS / "submission_alfa_gru_multiseed_blend.csv",
    "gru64_epoch9": ARTIFACTS
    / "submission_alfa_gru64_seed42_epoch9.csv",
    "tcn64": ARTIFACTS / "submission_alfa_tcn64_seed4242.csv",
    "pretrained64": ARTIFACTS
    / "submission_alfa_pretrained64_seed9001.csv",
    "cnn_local": ARTIFACTS
    / "submission_id_target_cnn_local_seed111.csv",
}


def current_validation() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validation_ids = None
    validation_targets = None
    base = None
    for name, path in VALIDATION_PATHS.items():
        ids, targets, prediction = load_validation(path)
        if validation_ids is None:
            validation_ids = ids
            validation_targets = targets
            base = np.zeros(ids.size, dtype=np.float64)
        base += WEIGHTS[name] * rank_percentile(prediction)

    target = pl.read_csv(
        TRAIN_TARGET,
        schema_overrides={"id": pl.Int32, "flag": pl.UInt8},
    )
    time_rank = rank_percentile(
        id_time_prior(
            target["id"].to_numpy(),
            target["flag"].to_numpy(),
            validation_ids,
            excluded_ids=validation_ids,
        )
    )
    _, _, cnn_prediction = load_validation(CNN_VALIDATION_PATH)
    cnn_rank = rank_percentile(cnn_prediction)
    main_weight = (
        1.0
        - FINAL_TIME_WEIGHT
        - FINAL_CNN_WEIGHT
        - sum(ALFA_GRU_WEIGHTS.values())
    )
    prediction = (
        main_weight * base
        + FINAL_TIME_WEIGHT * time_rank
        + FINAL_CNN_WEIGHT * cnn_rank
    )
    for seed, weight in ALFA_GRU_WEIGHTS.items():
        _, _, alfa_prediction = load_validation(
            ARTIFACTS / f"alfa_gru_validation_seed{seed}.npz"
        )
        prediction += weight * rank_percentile(alfa_prediction)
    return validation_ids, validation_targets, prediction


def main() -> None:
    ids, targets, current_prediction = current_validation()
    validation = STACK_WEIGHTS["current"] * current_prediction
    for name, path in EXTRA_VALIDATION_PATHS.items():
        extra_ids, extra_targets, prediction = load_validation(path)
        if not np.array_equal(ids, extra_ids):
            raise ValueError(f"Validation ids differ for {name}")
        if not np.array_equal(targets, extra_targets):
            raise ValueError(f"Validation targets differ for {name}")
        validation += STACK_WEIGHTS[name] * rank_percentile(prediction)
    auc = roc_auc_score(targets, validation)
    print(f"final validation ROC-AUC={auc:.9f}")
    print(f"weights={STACK_WEIGHTS}")

    sample_ids = pl.read_csv(
        SAMPLE_SUBMISSION,
        schema_overrides={"id": pl.Int32},
    )["id"].to_numpy()
    test_prediction = np.zeros(sample_ids.size, dtype=np.float64)
    for name, path in SUBMISSION_PATHS.items():
        submission = pl.read_csv(path)
        if not np.array_equal(submission["id"].to_numpy(), sample_ids):
            raise ValueError(f"Submission order differs for {name}")
        prediction = submission["flag"].to_numpy()
        if name != "current":
            prediction = rank_percentile(prediction)
        test_prediction += STACK_WEIGHTS[name] * prediction

    output = ARTIFACTS / "submission_final_stacking.csv"
    write_compact_submission(sample_ids, test_prediction, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
