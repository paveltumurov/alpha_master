from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
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
from enhanced import PAYMENT_COLUMNS, enhanced_expressions


CATEGORY_LEVELS = {
    "enc_loans_account_holder_type": range(7),
    "enc_loans_credit_status": range(7),
    "enc_loans_credit_type": range(8),
    "enc_loans_account_cur": range(4),
}


def advanced_expressions(columns: list[str]) -> list[pl.Expr]:
    expressions = enhanced_expressions(columns)

    for column, levels in CATEGORY_LEVELS.items():
        expressions.append(
            pl.col(column).n_unique().cast(pl.UInt8).alias(f"{column}__nunique")
        )
        expressions.extend(
            (pl.col(column) == level)
            .mean()
            .cast(pl.Float32)
            .alias(f"{column}__share_{level}")
            for level in levels
        )

    expressions.append(
        pl.col("pre_util").n_unique().cast(pl.UInt8).alias("pre_util__nunique")
    )
    expressions.extend(
        (pl.col("pre_util") == level)
        .mean()
        .cast(pl.Float32)
        .alias(f"pre_util__share_{level}")
        for level in range(20)
    )

    # Exact payment-state frequencies preserve information hidden by means.
    payment_state_counts = [
        pl.sum_horizontal(
            [(pl.col(column) == state).cast(pl.UInt8) for column in PAYMENT_COLUMNS]
        )
        for state in range(5)
    ]
    expressions.extend(
        count.mean().cast(pl.Float32).alias(f"payment_state_{state}__mean_count")
        for state, count in enumerate(payment_state_counts)
    )
    expressions.extend(
        count.sort_by("rn")
        .last()
        .cast(pl.UInt8)
        .alias(f"payment_state_{state}__last_count")
        for state, count in enumerate(payment_state_counts)
    )

    # Month 0..5 are treated as the most recent part of the encoded history.
    for month in range(6):
        column = f"enc_paym_{month}"
        expressions.extend(
            (pl.col(column) == state)
            .mean()
            .cast(pl.Float32)
            .alias(f"{column}__share_{state}")
            for state in range(5)
        )

    return expressions


def aggregate(
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
            frame = (
                pl.scan_parquet(partition)
                .group_by("id")
                .agg(advanced_expressions(columns))
                .with_columns(pl.col("id").cast(pl.Int32))
                .collect(engine="streaming")
            )
            table = frame.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(
                    destination,
                    table.schema,
                    compression="zstd",
                    use_dictionary=False,
                )
            writer.write_table(table)
            print(f"{source.name}: advanced partition {number}/{partition_count}")
            del frame, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    print(f"Saved advanced features to {destination}")


def load_train(
    features_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    features = pl.read_parquet(features_path)
    target = pl.read_csv(
        TRAIN_TARGET,
        schema_overrides={"id": pl.Int32, "flag": pl.UInt8},
    )
    frame = (
        target.join(features, on="id", how="inner", validate="1:1")
        .with_columns((pl.col("id") % 10 == 0).alias("__valid"))
        .sort(["__valid", "id"])
    )
    enhanced_model = lgb.Booster(model_file=str(ARTIFACTS / "enhanced_lgbm.txt"))
    enhanced_gain = sorted(
        zip(
            enhanced_model.feature_name(),
            enhanced_model.feature_importance(importance_type="gain"),
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    top_enhanced = [name for name, _ in enhanced_gain[:140]]
    enhanced_names = set(enhanced_model.feature_name())
    new_names = [
        column
        for column in features.columns
        if column != "id" and column not in enhanced_names
    ]
    names = top_enhanced + new_names
    print(
        f"Selected {len(top_enhanced)} enhanced + "
        f"{len(new_names)} frequency features"
    )
    train_size = frame["__valid"].arg_max()
    if train_size is None:
        raise ValueError("Validation split is empty")
    ids = frame["id"].to_numpy()
    y = frame["flag"].to_numpy()
    X = frame.select(names).to_numpy().astype(np.float32, copy=False)
    del frame, features, target
    gc.collect()
    return ids, y, X, names, int(train_size)


def train(features_path: Path, model_path: Path) -> dict[str, float | int]:
    ids, y, X, names, train_size = load_train(features_path)
    temporal_mask = ids >= np.quantile(ids, 0.9)

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=2600,
        learning_rate=0.025,
        num_leaves=63,
        min_child_samples=180,
        max_bin=127,
        subsample=0.85,
        colsample_bytree=0.72,
        reg_alpha=0.3,
        reg_lambda=2.5,
        n_jobs=3,
        random_state=137,
        verbosity=-1,
    )
    model.fit(
        X[:train_size],
        y[:train_size],
        eval_set=[(X[train_size:], y[train_size:])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(50)],
        feature_name=names,
    )
    valid_prediction = model.predict_proba(
        X[train_size:], num_iteration=model.best_iteration_
    )[:, 1]
    all_prediction = np.empty(X.shape[0], dtype=np.float32)
    for start in range(0, X.shape[0], 200_000):
        end = min(start + 200_000, X.shape[0])
        all_prediction[start:end] = model.predict_proba(
            X[start:end], num_iteration=model.best_iteration_
        )[:, 1]
    metrics: dict[str, float | int] = {
        "random_auc": float(
            roc_auc_score(y[train_size:], valid_prediction)
        ),
        "temporal_auc": float(
            roc_auc_score(y[temporal_mask], all_prediction[temporal_mask])
        ),
        "best_iteration": int(model.best_iteration_),
        "feature_count": len(names),
        "train_rows": train_size,
        "valid_rows": int(X.shape[0] - train_size),
    }
    model.booster_.save_model(model_path)
    (ARTIFACTS / "metrics_advanced.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def predict(features_path: Path, model_path: Path, output_path: Path) -> None:
    model = lgb.Booster(model_file=str(model_path))
    features = pl.read_parquet(features_path).sort("id")
    ids = features["id"].to_numpy()
    X = (
        features.select(model.feature_name())
        .to_numpy()
        .astype(np.float32, copy=False)
    )
    values = model.predict(X, num_iteration=model.best_iteration)
    predicted = pl.DataFrame({"id": ids, "prediction": values})
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
    parser = argparse.ArgumentParser(description="Advanced credit scoring model")
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
    train_features = ARTIFACTS / "train_features_advanced.parquet"
    test_features = ARTIFACTS / "test_features_advanced.parquet"
    model_path = ARTIFACTS / "advanced_lgbm.txt"
    output_path = ARTIFACTS / "submission_advanced.csv"

    if args.stage in {"aggregate", "all"}:
        aggregate(
            TRAIN_DATA,
            train_features,
            ARTIFACTS / "train_partitions",
            args.batch_size,
            args.partitions,
        )
        aggregate(
            TEST_DATA,
            test_features,
            ARTIFACTS / "test_partitions",
            args.batch_size,
            args.partitions,
        )
    if args.stage in {"train", "all"}:
        train(train_features, model_path)
    if args.stage in {"predict", "all"}:
        predict(test_features, model_path, output_path)


if __name__ == "__main__":
    main()
