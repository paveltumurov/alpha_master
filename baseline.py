from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
TRAIN_DATA = ROOT / "train_data.parquet"
TEST_DATA = ROOT / "test_data.parquet"
TRAIN_TARGET = ROOT / "train_target.csv"
SAMPLE_SUBMISSION = ROOT / "sample_submission (1).csv"


def feature_columns(path: Path) -> list[str]:
    schema = pq.read_schema(path)
    return [name for name in schema.names if name not in {"id", "rn"}]


def partition_parquet(
    source: Path, partition_dir: Path, batch_size: int, partition_count: int
) -> list[Path]:
    marker = partition_dir / "_SUCCESS"
    paths = [
        partition_dir / f"part_{partition:02d}.parquet"
        for partition in range(partition_count)
    ]
    if marker.exists() and all(path.exists() for path in paths):
        print(f"Using cached partitions: {partition_dir}")
        return paths

    if partition_dir.exists():
        shutil.rmtree(partition_dir)
    partition_dir.mkdir(parents=True)

    parquet = pq.ParquetFile(source)
    writers: list[pq.ParquetWriter | None] = [None] * partition_count
    rows_read = 0
    try:
        for batch in parquet.iter_batches(batch_size=batch_size, use_threads=True):
            table = pa.Table.from_batches([batch])
            ids = table.column("id").to_numpy(zero_copy_only=False)
            partitions = ids % partition_count
            for partition in range(partition_count):
                indices = np.flatnonzero(partitions == partition)
                if indices.size == 0:
                    continue
                part_table = table.take(pa.array(indices))
                if writers[partition] is None:
                    writers[partition] = pq.ParquetWriter(
                        paths[partition],
                        part_table.schema,
                        compression="zstd",
                        use_dictionary=True,
                    )
                writers[partition].write_table(part_table)
            rows_read += batch.num_rows
            if rows_read % (batch_size * 10) < batch_size:
                print(f"{source.name}: partitioned {rows_read:,}/{parquet.metadata.num_rows:,}")
    finally:
        for writer in writers:
            if writer is not None:
                writer.close()

    marker.write_text("ok", encoding="ascii")
    return paths


def aggregate_parquet(
    source: Path,
    destination: Path,
    partition_dir: Path,
    batch_size: int,
    partition_count: int,
) -> None:
    columns = feature_columns(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partitions = partition_parquet(
        source, partition_dir, batch_size, partition_count
    )
    writer: pq.ParquetWriter | None = None
    destination.unlink(missing_ok=True)
    try:
        for number, partition in enumerate(partitions, start=1):
            expressions = [
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
            aggregated = (
                pl.scan_parquet(partition)
                .group_by("id")
                .agg(expressions)
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
            print(f"{source.name}: aggregated partition {number}/{partition_count}")
    finally:
        if writer is not None:
            writer.close()

    print(f"Saved features to {destination}")


def load_xy(features_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
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


def train_model(train_features: Path, model_path: Path) -> dict[str, float | int]:
    ids, y, X, feature_names = load_xy(train_features)

    # The official test ids are interleaved with train ids. A stable id hash
    # therefore mirrors the observed split better than a single chronological cut.
    random_valid = ids % 10 == 0
    temporal_border = np.quantile(ids, 0.9)
    temporal_valid = ids >= temporal_border

    train_idx = np.flatnonzero(~random_valid)
    valid_idx = np.flatnonzero(random_valid)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1500,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=3,
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        X[train_idx],
        y[train_idx],
        eval_set=[(X[valid_idx], y[valid_idx])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
        feature_name=feature_names,
    )

    random_prediction = model.predict_proba(
        X[valid_idx], num_iteration=model.best_iteration_
    )[:, 1]
    temporal_prediction = model.predict_proba(
        X[temporal_valid], num_iteration=model.best_iteration_
    )[:, 1]
    metrics: dict[str, float | int] = {
        "random_auc": float(roc_auc_score(y[valid_idx], random_prediction)),
        "temporal_auc": float(roc_auc_score(y[temporal_valid], temporal_prediction)),
        "best_iteration": int(model.best_iteration_),
        "train_rows": int(train_idx.size),
        "random_valid_rows": int(valid_idx.size),
        "temporal_valid_rows": int(temporal_valid.sum()),
        "feature_count": len(feature_names),
    }
    model.booster_.save_model(model_path)
    (ARTIFACTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def predict_submission(
    test_features: Path, model_path: Path, output_path: Path
) -> None:
    model = lgb.Booster(model_file=str(model_path))
    feature_names = model.feature_name()
    test = pl.read_parquet(test_features).sort("id")
    ids = test["id"].to_numpy()
    X = test.select(feature_names).to_numpy().astype(np.float32, copy=False)
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
    temporary_output = output_path.with_suffix(".tmp.csv")
    with temporary_output.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("id,flag\n")
        for row_id, value in submission.iter_rows():
            formatted = f"{value:.18f}".rstrip("0").rstrip(".")
            if formatted.startswith("0."):
                formatted = formatted[1:]
            elif formatted.startswith("-0."):
                formatted = f"-{formatted[2:]}"
            stream.write(f"{row_id},{formatted}\n")
    temporary_output.replace(output_path)
    print(f"Saved submission to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Memory-friendly credit scoring baseline")
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
    ARTIFACTS.mkdir(exist_ok=True)
    train_features = ARTIFACTS / "train_features.parquet"
    test_features = ARTIFACTS / "test_features.parquet"
    model_path = ARTIFACTS / "baseline_lgbm.txt"
    submission_path = ARTIFACTS / "submission_baseline.csv"

    if args.stage in {"aggregate", "all"}:
        aggregate_parquet(
            TRAIN_DATA,
            train_features,
            ARTIFACTS / "train_partitions",
            args.batch_size,
            args.partitions,
        )
        aggregate_parquet(
            TEST_DATA,
            test_features,
            ARTIFACTS / "test_partitions",
            args.batch_size,
            args.partitions,
        )
    if args.stage in {"train", "all"}:
        train_model(train_features, model_path)
    if args.stage in {"predict", "all"}:
        predict_submission(test_features, model_path, submission_path)


if __name__ == "__main__":
    main()
