from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from baseline import ARTIFACTS, SAMPLE_SUBMISSION, TEST_DATA, TRAIN_DATA, TRAIN_TARGET
from engineered_features import aggregate_dataset


def combine_features(
    advanced_path: Path,
    engineered_path: Path,
    destination: Path,
) -> None:
    combined = (
        pl.scan_parquet(advanced_path)
        .join(pl.scan_parquet(engineered_path), on="id", how="inner")
        .sort("id")
        .collect(engine="streaming")
    )
    combined.write_parquet(destination, compression="zstd")
    print(f"Saved {combined.shape} to {destination}")


def load_train(path: Path):
    features = pl.read_parquet(path)
    target = pl.read_csv(
        TRAIN_TARGET,
        schema_overrides={"id": pl.Int32, "flag": pl.UInt8},
    )
    frame = (
        target.join(features, on="id", how="inner", validate="1:1")
        .with_columns((pl.col("id") % 10 == 0).alias("__valid"))
        .sort(["__valid", "id"])
    )
    names = [
        column
        for column in frame.columns
        if column not in {"id", "flag", "__valid"}
    ]
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
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=63,
        min_child_samples=180,
        max_bin=127,
        colsample_bytree=0.72,
        reg_alpha=0.3,
        reg_lambda=2.5,
        n_jobs=10,
        force_col_wise=True,
        random_state=2026,
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
    prediction = model.predict_proba(
        X[train_size:],
        num_iteration=model.best_iteration_,
    )[:, 1]
    metrics: dict[str, float | int] = {
        "auc": float(roc_auc_score(y[train_size:], prediction)),
        "best_iteration": int(model.best_iteration_),
        "feature_count": len(names),
        "train_rows": train_size,
        "valid_rows": int(X.shape[0] - train_size),
    }
    model.booster_.save_model(model_path)
    (ARTIFACTS / "metrics_engineered.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
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
        raise ValueError("Submission contains missing predictions")

    temporary = output_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("id,flag\n")
        for row_id, value in submission.iter_rows():
            formatted = f"{value:.18f}".rstrip("0").rstrip(".")
            if formatted.startswith("0."):
                formatted = formatted[1:]
            stream.write(f"{row_id},{formatted}\n")
    temporary.replace(output_path)
    print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Engineered feature model")
    parser.add_argument(
        "stage",
        choices=("aggregate", "combine", "train", "predict", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--partitions", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_engineered = ARTIFACTS / "train_features_engineered.parquet"
    test_engineered = ARTIFACTS / "test_features_engineered.parquet"
    train_combined = ARTIFACTS / "train_features_combined.parquet"
    test_combined = ARTIFACTS / "test_features_combined.parquet"
    model_path = ARTIFACTS / "engineered_lgbm.txt"
    output_path = ARTIFACTS / "submission_engineered.csv"

    if args.stage in {"aggregate", "all"}:
        aggregate_dataset(
            TRAIN_DATA,
            ARTIFACTS / "train_partitions",
            train_engineered,
            args.partitions,
            args.batch_size,
        )
        aggregate_dataset(
            TEST_DATA,
            ARTIFACTS / "test_partitions",
            test_engineered,
            args.partitions,
            args.batch_size,
        )
    if args.stage in {"combine", "all"}:
        combine_features(
            ARTIFACTS / "train_features_advanced.parquet",
            train_engineered,
            train_combined,
        )
        combine_features(
            ARTIFACTS / "test_features_advanced.parquet",
            test_engineered,
            test_combined,
        )
    if args.stage in {"train", "all"}:
        train(train_combined, model_path)
    if args.stage in {"predict", "all"}:
        predict(test_combined, model_path, output_path)


if __name__ == "__main__":
    main()
