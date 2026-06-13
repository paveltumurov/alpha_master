from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from baseline import (
    ARTIFACTS,
    SAMPLE_SUBMISSION,
    TEST_DATA,
    TRAIN_DATA,
    TRAIN_TARGET,
    feature_columns,
    partition_parquet,
)


CORE_COLUMNS = [
    "pre_since_opened",
    "pre_since_confirmed",
    "pre_pterm",
    "pre_fterm",
    "pre_till_pclose",
    "pre_till_fclose",
    "pre_loans_credit_limit",
    "pre_loans_next_pay_summ",
    "pre_loans_outstanding",
    "pre_loans_total_overdue",
    "pre_loans_max_overdue_sum",
    "pre_loans_credit_cost_rate",
    "pre_loans5",
    "pre_loans530",
    "pre_loans3060",
    "pre_loans6090",
    "pre_loans90",
    "pre_util",
    "pre_over2limit",
    "pre_maxover2limit",
]

OVERDUE_ZERO_COLUMNS = [
    "is_zero_loans5",
    "is_zero_loans530",
    "is_zero_loans3060",
    "is_zero_loans6090",
    "is_zero_loans90",
]

PAYMENT_COLUMNS = [f"enc_paym_{month}" for month in range(25)]


def enhanced_expressions(columns: list[str]) -> list[pl.Expr]:
    expressions: list[pl.Expr] = [
        pl.len().cast(pl.UInt8).alias("loan_count"),
        pl.col("rn").max().cast(pl.UInt8).alias("rn_max"),
    ]

    expressions.extend(
        pl.col(column).mean().cast(pl.Float32).alias(f"{column}__mean")
        for column in columns
    )
    expressions.extend(
        pl.col(column).max().cast(pl.Int16).alias(f"{column}__max")
        for column in columns
    )

    # The latest product is often more informative than an all-history average.
    expressions.extend(
        pl.col(column)
        .sort_by("rn")
        .last()
        .cast(pl.Int16)
        .alias(f"{column}__last")
        for column in columns
    )

    # Keep dispersion and recent-window features on the compact, high-value subset.
    expressions.extend(
        pl.col(column).std().fill_null(0).cast(pl.Float32).alias(f"{column}__std")
        for column in CORE_COLUMNS
    )
    expressions.extend(
        pl.col(column)
        .sort_by("rn", descending=True)
        .head(3)
        .mean()
        .cast(pl.Float32)
        .alias(f"{column}__recent3_mean")
        for column in CORE_COLUMNS
    )

    payment_row_mean = pl.mean_horizontal(PAYMENT_COLUMNS)
    payment_row_max = pl.max_horizontal(PAYMENT_COLUMNS)
    payment_recent_mean = pl.mean_horizontal(PAYMENT_COLUMNS[:6])
    expressions.extend(
        [
            payment_row_mean.mean().cast(pl.Float32).alias("payment_level_mean"),
            payment_row_mean.max().cast(pl.Float32).alias("payment_level_max"),
            payment_row_max.mean().cast(pl.Float32).alias("payment_peak_mean"),
            payment_row_max.max().cast(pl.Int16).alias("payment_peak_max"),
            payment_recent_mean.mean().cast(pl.Float32).alias("payment_recent6_mean"),
            (
                payment_recent_mean.mean() - payment_row_mean.mean()
            ).cast(pl.Float32).alias("payment_recent_vs_history"),
        ]
    )

    serious_overdue = pl.any_horizontal(
        [pl.col(column) == 0 for column in OVERDUE_ZERO_COLUMNS[2:]]
    )
    any_overdue = pl.any_horizontal(
        [pl.col(column) == 0 for column in OVERDUE_ZERO_COLUMNS]
    )
    expressions.extend(
        [
            any_overdue.mean().cast(pl.Float32).alias("any_overdue_share"),
            any_overdue.max().cast(pl.UInt8).alias("any_overdue_ever"),
            serious_overdue.mean().cast(pl.Float32).alias("serious_overdue_share"),
            serious_overdue.max().cast(pl.UInt8).alias("serious_overdue_ever"),
            serious_overdue
            .sort_by("rn")
            .last()
            .cast(pl.UInt8)
            .alias("serious_overdue_last"),
        ]
    )
    return expressions


def aggregate_enhanced(
    source: Path,
    destination: Path,
    partition_dir: Path,
    batch_size: int,
    partition_count: int,
) -> None:
    columns = feature_columns(source)
    partitions = partition_parquet(
        source, partition_dir, batch_size, partition_count
    )
    destination.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for number, partition in enumerate(partitions, start=1):
            aggregated = (
                pl.scan_parquet(partition)
                .group_by("id")
                .agg(enhanced_expressions(columns))
                .with_columns(pl.col("id").cast(pl.Int32))
                .collect(engine="streaming")
            )
            arrow = aggregated.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(
                    destination,
                    arrow.schema,
                    compression="zstd",
                    use_dictionary=False,
                )
            writer.write_table(arrow)
            print(f"{source.name}: enhanced partition {number}/{partition_count}")
            del aggregated, arrow
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    print(f"Saved enhanced features to {destination}")


def load_train(
    features_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    features = pl.read_parquet(features_path)
    target = pl.read_csv(
        TRAIN_TARGET,
        schema_overrides={"id": pl.Int32, "flag": pl.UInt8},
    )
    frame = target.join(features, on="id", how="inner", validate="1:1").sort("id")
    if frame.height != target.height:
        raise ValueError("Not every train id received an aggregate row")
    feature_names = [c for c in frame.columns if c not in {"id", "flag"}]
    ids = frame["id"].to_numpy()
    y = frame["flag"].to_numpy()
    X = frame.select(feature_names).to_numpy().astype(np.float32, copy=False)
    del frame, features, target
    gc.collect()
    return ids, y, X, feature_names


def train_enhanced(features_path: Path, model_path: Path) -> dict[str, float | int]:
    ids, y, X, feature_names = load_train(features_path)
    valid_mask = ids % 10 == 0
    temporal_mask = ids >= np.quantile(ids, 0.9)
    train_idx = np.flatnonzero(~valid_mask)
    valid_idx = np.flatnonzero(valid_mask)

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=2200,
        learning_rate=0.025,
        num_leaves=47,
        max_depth=-1,
        min_child_samples=150,
        max_bin=127,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.2,
        reg_lambda=2.0,
        n_jobs=3,
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        X[train_idx],
        y[train_idx],
        eval_set=[(X[valid_idx], y[valid_idx])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(120), lgb.log_evaluation(50)],
        feature_name=feature_names,
    )
    random_prediction = model.predict_proba(
        X[valid_idx], num_iteration=model.best_iteration_
    )[:, 1]
    temporal_prediction = model.predict_proba(
        X[temporal_mask], num_iteration=model.best_iteration_
    )[:, 1]
    metrics: dict[str, float | int] = {
        "random_auc": float(roc_auc_score(y[valid_idx], random_prediction)),
        "temporal_auc": float(roc_auc_score(y[temporal_mask], temporal_prediction)),
        "best_iteration": int(model.best_iteration_),
        "feature_count": len(feature_names),
        "train_rows": int(train_idx.size),
        "valid_rows": int(valid_idx.size),
    }
    model.booster_.save_model(model_path)
    (ARTIFACTS / "metrics_enhanced.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def write_submission(
    features_path: Path, model_path: Path, output_path: Path
) -> None:
    model = lgb.Booster(model_file=str(model_path))
    features = pl.read_parquet(features_path).sort("id")
    ids = features["id"].to_numpy()
    X = (
        features.select(model.feature_name())
        .to_numpy()
        .astype(np.float32, copy=False)
    )
    prediction = model.predict(X, num_iteration=model.best_iteration)
    predicted = pl.DataFrame({"id": ids, "prediction": prediction})
    sample = pl.read_csv(SAMPLE_SUBMISSION, schema_overrides={"id": pl.Int32})
    submission = (
        sample.select("id")
        .join(predicted, on="id", how="left", validate="1:1")
        .rename({"prediction": "flag"})
    )
    if submission["flag"].null_count():
        raise ValueError("Submission contains ids without predictions")

    temporary = output_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("id,flag\n")
        for row_id, value in submission.iter_rows():
            formatted = f"{value:.18f}".rstrip("0").rstrip(".")
            if formatted.startswith("0."):
                formatted = formatted[1:]
            stream.write(f"{row_id},{formatted}\n")
    temporary.replace(output_path)
    print(f"Saved submission to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced credit scoring model")
    parser.add_argument(
        "stage",
        choices=("aggregate", "train", "predict", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--partitions", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_features = ARTIFACTS / "train_features_enhanced.parquet"
    test_features = ARTIFACTS / "test_features_enhanced.parquet"
    model_path = ARTIFACTS / "enhanced_lgbm.txt"
    output_path = ARTIFACTS / "submission_enhanced.csv"

    if args.stage in {"aggregate", "all"}:
        aggregate_enhanced(
            TRAIN_DATA,
            train_features,
            ARTIFACTS / "train_partitions",
            args.batch_size,
            args.partitions,
        )
        aggregate_enhanced(
            TEST_DATA,
            test_features,
            ARTIFACTS / "test_partitions",
            args.batch_size,
            args.partitions,
        )
    if args.stage in {"train", "all"}:
        train_enhanced(train_features, model_path)
    if args.stage in {"predict", "all"}:
        write_submission(test_features, model_path, output_path)


if __name__ == "__main__":
    main()
