from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

from baseline import ARTIFACTS, SAMPLE_SUBMISSION, TRAIN_TARGET


FEATURES = ARTIFACTS / "train_features_combined.parquet"
TEST_FEATURES = ARTIFACTS / "test_features_combined.parquet"
MODEL = ARTIFACTS / "engineered_catboost.cbm"
VALIDATION = ARTIFACTS / "catboost_validation.npz"
METRICS = ARTIFACTS / "metrics_catboost.json"
SUBMISSION = ARTIFACTS / "submission_catboost.csv"


def load_train(path: Path) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    int,
]:
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
    targets = frame["flag"].to_numpy()
    matrix = frame.select(names).to_numpy().astype(np.float32, copy=False)
    del frame, features, target
    gc.collect()
    return ids, targets, matrix, names, int(train_size)


def train(args: argparse.Namespace) -> None:
    ids, targets, matrix, names, train_size = load_train(FEATURES)
    train_pool = Pool(
        matrix[:train_size],
        targets[:train_size],
        feature_names=names,
    )
    valid_pool = Pool(
        matrix[train_size:],
        targets[train_size:],
        feature_names=names,
    )
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_strength=args.random_strength,
        bootstrap_type="Bayesian",
        bagging_temperature=args.bagging_temperature,
        border_count=args.border_count,
        random_seed=args.seed,
        task_type="GPU",
        devices="0",
        thread_count=args.threads,
        od_type="Iter",
        od_wait=args.patience,
        use_best_model=True,
        allow_writing_files=False,
        verbose=50,
    )
    model.fit(train_pool, eval_set=valid_pool)
    prediction = model.predict_proba(valid_pool)[:, 1]
    auc = float(roc_auc_score(targets[train_size:], prediction))
    np.savez_compressed(
        VALIDATION,
        id=ids[train_size:],
        target=targets[train_size:],
        prediction=prediction,
    )
    model.save_model(MODEL)
    metrics: dict[str, float | int] = {
        "auc": auc,
        "best_iteration": int(model.get_best_iteration()),
        "tree_count": int(model.tree_count_),
        "feature_count": len(names),
        "train_rows": train_size,
        "valid_rows": int(matrix.shape[0] - train_size),
    }
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def write_compact_submission(
    ids: np.ndarray,
    values: np.ndarray,
    destination: Path,
) -> None:
    temporary = destination.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("id,flag\n")
        for row_id, value in zip(ids, values):
            formatted = f"{value:.18f}".rstrip("0").rstrip(".")
            if formatted.startswith("0."):
                formatted = formatted[1:]
            stream.write(f"{row_id},{formatted}\n")
    temporary.replace(destination)


def predict(args: argparse.Namespace) -> None:
    model = CatBoostClassifier()
    model.load_model(MODEL)
    features = pl.read_parquet(TEST_FEATURES).sort("id")
    names = model.feature_names_
    ids = features["id"].to_numpy()
    matrix = features.select(names).to_numpy().astype(np.float32, copy=False)
    predictions = np.empty(matrix.shape[0], dtype=np.float64)
    for start in range(0, matrix.shape[0], args.predict_batch_size):
        stop = min(start + args.predict_batch_size, matrix.shape[0])
        predictions[start:stop] = model.predict_proba(matrix[start:stop])[:, 1]
        print(f"\rpredicted {stop:,}/{matrix.shape[0]:,}", end="", flush=True)
    print()

    predicted = pl.DataFrame({"id": ids, "prediction": predictions})
    sample = pl.read_csv(SAMPLE_SUBMISSION, schema_overrides={"id": pl.Int32})
    ordered = sample.select("id").join(
        predicted,
        on="id",
        how="left",
        validate="1:1",
    )
    if ordered["prediction"].null_count():
        raise ValueError("Submission contains missing predictions")
    write_compact_submission(
        ordered["id"].to_numpy(),
        ordered["prediction"].to_numpy(),
        SUBMISSION,
    )
    print(f"Saved {SUBMISSION}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU CatBoost experiment")
    parser.add_argument(
        "stage",
        choices=("train", "predict", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0)
    parser.add_argument("--random-strength", type=float, default=1.0)
    parser.add_argument("--bagging-temperature", type=float, default=1.0)
    parser.add_argument("--border-count", type=int, default=64)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--predict-batch-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"train", "all"}:
        train(args)
    if args.stage in {"predict", "all"}:
        predict(args)


if __name__ == "__main__":
    main()
