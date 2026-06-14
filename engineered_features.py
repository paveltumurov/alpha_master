from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from baseline import ARTIFACTS, TEST_DATA, TRAIN_DATA, partition_parquet


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

PAYMENT_COLUMNS = [f"enc_paym_{month}" for month in range(25)]
SERIOUS_ZERO_COLUMNS = [
    "is_zero_loans3060",
    "is_zero_loans6090",
    "is_zero_loans90",
]


def payment_state_count(state: int) -> pl.Expr:
    return pl.sum_horizontal(
        [(pl.col(column) == state).cast(pl.UInt8) for column in PAYMENT_COLUMNS]
    )


def row_expressions() -> list[pl.Expr]:
    payment_changes = pl.sum_horizontal(
        [
            (pl.col(PAYMENT_COLUMNS[index]) != pl.col(PAYMENT_COLUMNS[index + 1]))
            .cast(pl.UInt8)
            for index in range(len(PAYMENT_COLUMNS) - 1)
        ]
    )
    payment_bad_transitions = pl.sum_horizontal(
        [
            (
                (pl.col(PAYMENT_COLUMNS[index]) == 0)
                & (pl.col(PAYMENT_COLUMNS[index + 1]) != 0)
            ).cast(pl.UInt8)
            for index in range(len(PAYMENT_COLUMNS) - 1)
        ]
    )
    serious_overdue = pl.any_horizontal(
        [pl.col(column) == 0 for column in SERIOUS_ZERO_COLUMNS]
    )
    return [
        payment_changes.alias("__payment_changes"),
        payment_bad_transitions.alias("__payment_bad_transitions"),
        serious_overdue.cast(pl.UInt8).alias("__serious_overdue"),
        *[
            payment_state_count(state).alias(f"__payment_state_{state}")
            for state in range(5)
        ],
    ]


def group_expressions() -> list[pl.Expr]:
    expressions: list[pl.Expr] = []

    for column in CORE_COLUMNS:
        ordered = pl.col(column).sort_by("rn")
        ordered_diff = ordered.diff()
        previous = ordered.tail(2).first()
        first = ordered.first()
        last = ordered.last()
        expressions.extend(
            [
                previous.cast(pl.Int16).alias(f"{column}__previous"),
                (last - previous).cast(pl.Int16).alias(f"{column}__last_delta"),
                (last - first).cast(pl.Int16).alias(f"{column}__history_delta"),
                (
                    pl.cov("rn", column)
                    / pl.col("rn").var().replace(0, None)
                )
                .fill_null(0)
                .cast(pl.Float32)
                .alias(f"{column}__rn_slope"),
                ordered_diff
                .abs()
                .sum()
                .fill_null(0)
                .cast(pl.UInt16)
                .alias(f"{column}__total_variation"),
                (ordered_diff != 0)
                .sum()
                .cast(pl.UInt8)
                .alias(f"{column}__change_count"),
            ]
        )

    expressions.extend(
        [
            pl.col("__payment_changes")
            .mean()
            .cast(pl.Float32)
            .alias("payment_changes__mean"),
            pl.col("__payment_changes")
            .sort_by("rn")
            .last()
            .cast(pl.UInt8)
            .alias("payment_changes__last"),
            pl.col("__payment_bad_transitions")
            .mean()
            .cast(pl.Float32)
            .alias("payment_bad_transitions__mean"),
            pl.col("__payment_bad_transitions")
            .sort_by("rn")
            .last()
            .cast(pl.UInt8)
            .alias("payment_bad_transitions__last"),
            pl.col("__serious_overdue")
            .sort_by("rn")
            .diff()
            .abs()
            .sum()
            .fill_null(0)
            .cast(pl.UInt8)
            .alias("serious_overdue__transition_count"),
            (
                pl.col("rn").max()
                - pl.when(pl.col("__serious_overdue") == 1)
                .then(pl.col("rn"))
                .otherwise(None)
                .max()
            )
            .fill_null(pl.col("rn").max())
            .cast(pl.UInt8)
            .alias("loans_since_serious_overdue"),
        ]
    )

    for state in range(5):
        column = f"__payment_state_{state}"
        expressions.extend(
            [
                pl.col(column)
                .std()
                .fill_null(0)
                .cast(pl.Float32)
                .alias(f"payment_state_{state}__std"),
                (
                    pl.col(column).sort_by("rn").last()
                    - pl.col(column).sort_by("rn").first()
                )
                .cast(pl.Int16)
                .alias(f"payment_state_{state}__history_delta"),
            ]
        )

    for credit_type in range(8):
        type_filter = pl.col("enc_loans_credit_type") == credit_type
        expressions.extend(
            [
                pl.when(type_filter)
                .then(pl.col("pre_util"))
                .otherwise(None)
                .mean()
                .fill_null(-1)
                .cast(pl.Float32)
                .alias(f"credit_type_{credit_type}__util_mean"),
                pl.when(type_filter)
                .then(pl.col("__serious_overdue"))
                .otherwise(None)
                .mean()
                .fill_null(-1)
                .cast(pl.Float32)
                .alias(f"credit_type_{credit_type}__serious_overdue_share"),
            ]
        )

    return expressions


def aggregate_partition(path: Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .with_columns(row_expressions())
        .group_by("id")
        .agg(group_expressions())
        .with_columns(pl.col("id").cast(pl.Int32))
        .collect(engine="streaming")
    )


def aggregate_dataset(
    source: Path,
    partition_dir: Path,
    destination: Path,
    partitions: int,
    batch_size: int,
) -> None:
    paths = partition_parquet(source, partition_dir, batch_size, partitions)
    destination.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for index, path in enumerate(paths, start=1):
            frame = aggregate_partition(path)
            table = frame.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(
                    destination,
                    table.schema,
                    compression="zstd",
                    use_dictionary=False,
                )
            writer.write_table(table)
            print(f"{source.name}: engineered partition {index}/{partitions}")
            del frame, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()


def smoke_test(partition: Path) -> None:
    frame = aggregate_partition(partition)
    feature_columns = [column for column in frame.columns if column != "id"]
    numeric = frame.select(feature_columns).to_numpy()
    print(
        {
            "rows": frame.height,
            "features": len(feature_columns),
            "nulls": int(frame.null_count().to_numpy().sum()),
            "nan": int(np.isnan(numeric).sum()),
            "inf": int(np.isinf(numeric).sum()),
            "duplicate_ids": frame.height - frame["id"].n_unique(),
        }
    )
    print(frame.select(["id", *feature_columns[:8]]).head(3))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Additional credit features")
    parser.add_argument(
        "stage",
        choices=("smoke", "aggregate"),
        nargs="?",
        default="smoke",
    )
    parser.add_argument("--partitions", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "smoke":
        smoke_test(ARTIFACTS / "train_partitions" / "part_00.parquet")
        return

    aggregate_dataset(
        TRAIN_DATA,
        ARTIFACTS / "train_partitions",
        ARTIFACTS / "train_features_engineered.parquet",
        args.partitions,
        args.batch_size,
    )
    aggregate_dataset(
        TEST_DATA,
        ARTIFACTS / "test_partitions",
        ARTIFACTS / "test_features_engineered.parquet",
        args.partitions,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
