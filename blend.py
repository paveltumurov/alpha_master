from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from baseline import ARTIFACTS, SAMPLE_SUBMISSION, TRAIN_TARGET


def rank_percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks / max(values.size - 1, 1)


def validation_prediction(
    features_path: Path, model_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    model = lgb.Booster(model_file=str(model_path))
    frame = (
        pl.scan_parquet(features_path)
        .filter(pl.col("id") % 10 == 0)
        .select(["id", *model.feature_name()])
        .sort("id")
        .collect(engine="streaming")
    )
    ids = frame["id"].to_numpy()
    X = (
        frame.select(model.feature_name())
        .to_numpy()
        .astype(np.float32, copy=False)
    )
    prediction = model.predict(X, num_iteration=model.best_iteration)
    return ids, prediction


def write_compact_submission(
    ids: np.ndarray, prediction: np.ndarray, output_path: Path
) -> None:
    temporary = output_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("id,flag\n")
        for row_id, value in zip(ids, prediction, strict=True):
            formatted = f"{value:.18f}".rstrip("0").rstrip(".")
            if formatted.startswith("0."):
                formatted = formatted[1:]
            stream.write(f"{row_id},{formatted}\n")
    temporary.replace(output_path)


def main() -> None:
    baseline_ids, baseline_prediction = validation_prediction(
        ARTIFACTS / "train_features.parquet",
        ARTIFACTS / "baseline_lgbm.txt",
    )
    enhanced_ids, enhanced_prediction = validation_prediction(
        ARTIFACTS / "train_features_enhanced.parquet",
        ARTIFACTS / "enhanced_lgbm.txt",
    )
    if not np.array_equal(baseline_ids, enhanced_ids):
        raise ValueError("Validation ids differ between models")

    target = (
        pl.read_csv(
            TRAIN_TARGET,
            schema_overrides={"id": pl.Int32, "flag": pl.UInt8},
        )
        .filter(pl.col("id") % 10 == 0)
        .sort("id")
    )
    y = target["flag"].to_numpy()
    baseline_rank = rank_percentile(baseline_prediction)
    enhanced_rank = rank_percentile(enhanced_prediction)

    best_auc = -1.0
    best_enhanced_weight = 1.0
    for enhanced_weight in np.linspace(0.5, 1.0, 11):
        prediction = (
            enhanced_weight * enhanced_rank
            + (1.0 - enhanced_weight) * baseline_rank
        )
        auc = roc_auc_score(y, prediction)
        print(f"enhanced_weight={enhanced_weight:.2f} auc={auc:.8f}")
        if auc > best_auc:
            best_auc = auc
            best_enhanced_weight = float(enhanced_weight)

    baseline_submission = pl.read_csv(ARTIFACTS / "submission_baseline.csv")
    enhanced_submission = pl.read_csv(ARTIFACTS / "submission_enhanced.csv")
    sample_ids = pl.read_csv(
        SAMPLE_SUBMISSION, schema_overrides={"id": pl.Int32}
    )["id"].to_numpy()
    if not np.array_equal(baseline_submission["id"].to_numpy(), sample_ids):
        raise ValueError("Baseline submission order differs from sample")
    if not np.array_equal(enhanced_submission["id"].to_numpy(), sample_ids):
        raise ValueError("Enhanced submission order differs from sample")

    blended = (
        best_enhanced_weight
        * rank_percentile(enhanced_submission["flag"].to_numpy())
        + (1.0 - best_enhanced_weight)
        * rank_percentile(baseline_submission["flag"].to_numpy())
    )
    output = ARTIFACTS / "submission_blend.csv"
    write_compact_submission(sample_ids, blended, output)
    print(
        f"Saved {output} with validation AUC {best_auc:.8f} "
        f"and enhanced weight {best_enhanced_weight:.2f}"
    )


if __name__ == "__main__":
    main()
