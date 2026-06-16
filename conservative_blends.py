from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from baseline import ARTIFACTS, SAMPLE_SUBMISSION
from blend import rank_percentile, write_compact_submission
from blend_new_methods import load_validation
from final_stacking import (
    EXTRA_VALIDATION_PATHS,
    STACK_WEIGHTS,
    current_validation,
)


CURRENT_PUBLIC = "alfa_multiseed_public_0_783512"
FINAL_STACK = "final_stacking_local_0_788513"

BLENDS = {
    "submission_conservative_50_50.csv": 0.50,
    "submission_conservative_60_40.csv": 0.60,
    "submission_conservative_70_30.csv": 0.70,
    "submission_conservative_80_20.csv": 0.80,
    "submission_conservative_90_10.csv": 0.90,
}


def validation_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids, targets, current = current_validation()
    final = STACK_WEIGHTS["current"] * current
    for name, path in EXTRA_VALIDATION_PATHS.items():
        extra_ids, extra_targets, prediction = load_validation(path)
        if not np.array_equal(ids, extra_ids):
            raise ValueError(f"Validation ids differ for {name}")
        if not np.array_equal(targets, extra_targets):
            raise ValueError(f"Validation targets differ for {name}")
        final += STACK_WEIGHTS[name] * rank_percentile(prediction)
    return targets, current, final


def main() -> None:
    targets, current_validation_prediction, final_validation_prediction = (
        validation_predictions()
    )
    print(
        f"{CURRENT_PUBLIC}: "
        f"{roc_auc_score(targets, current_validation_prediction):.9f}"
    )
    print(
        f"{FINAL_STACK}: "
        f"{roc_auc_score(targets, final_validation_prediction):.9f}"
    )
    sample_ids = pl.read_csv(
        SAMPLE_SUBMISSION,
        schema_overrides={"id": pl.Int32},
    )["id"].to_numpy()
    current_submission = pl.read_csv(
        ARTIFACTS / "submission_alfa_gru_multiseed_blend.csv"
    )
    final_submission = pl.read_csv(ARTIFACTS / "submission_final_stacking.csv")
    if not np.array_equal(current_submission["id"].to_numpy(), sample_ids):
        raise ValueError("Current submission order differs from sample")
    if not np.array_equal(final_submission["id"].to_numpy(), sample_ids):
        raise ValueError("Final submission order differs from sample")

    current_test = current_submission["flag"].to_numpy()
    final_test = final_submission["flag"].to_numpy()
    for output_name, final_weight in BLENDS.items():
        validation_prediction = (
            final_weight * final_validation_prediction
            + (1.0 - final_weight) * current_validation_prediction
        )
        test_prediction = (
            final_weight * final_test + (1.0 - final_weight) * current_test
        )
        auc = roc_auc_score(targets, validation_prediction)
        output = ARTIFACTS / output_name
        write_compact_submission(sample_ids, test_prediction, output)
        print(
            f"{output.name}: final_weight={final_weight:.2f} "
            f"validation ROC-AUC={auc:.9f}"
        )


if __name__ == "__main__":
    main()
