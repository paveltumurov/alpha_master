from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
TRAIN_DATA = ROOT / "train_data.parquet"
TEST_DATA = ROOT / "test_data.parquet"
TRAIN_TARGET = ROOT / "train_target.csv"
SAMPLE_SUBMISSION = ROOT / "sample_submission (1).csv"
NEURAL_DIR = ROOT / "neural_artifacts"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def feature_columns(path: Path) -> list[str]:
    schema = pq.read_schema(path)
    return [name for name in schema.names if name not in {"id", "rn"}]


def partition_parquet(
    source: Path,
    destination: Path,
    partition_count: int,
    batch_size: int,
) -> list[Path]:
    marker = destination / "_SUCCESS"
    paths = [
        destination / f"part_{index:02d}.parquet"
        for index in range(partition_count)
    ]
    if marker.exists() and all(path.exists() for path in paths):
        print(f"Using cached partitions: {destination}")
        return paths

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    parquet = pq.ParquetFile(source)
    writers: list[pq.ParquetWriter | None] = [None] * partition_count
    rows_read = 0
    try:
        for batch in parquet.iter_batches(batch_size=batch_size, use_threads=True):
            table = pa.Table.from_batches([batch])
            ids = table.column("id").to_numpy(zero_copy_only=False)
            partition_ids = ids % partition_count
            for index in range(partition_count):
                selected = np.flatnonzero(partition_ids == index)
                if selected.size == 0:
                    continue
                part = table.take(pa.array(selected))
                if writers[index] is None:
                    writers[index] = pq.ParquetWriter(
                        paths[index],
                        part.schema,
                        compression="zstd",
                        use_dictionary=True,
                    )
                writers[index].write_table(part)
            rows_read += batch.num_rows
            print(
                f"\r{source.name}: {rows_read:,}/{parquet.metadata.num_rows:,}",
                end="",
                flush=True,
            )
    finally:
        for writer in writers:
            if writer is not None:
                writer.close()
    print()
    marker.write_text("ok", encoding="ascii")
    return paths


def target_lookup() -> np.ndarray:
    target = pl.read_csv(
        TRAIN_TARGET,
        schema_overrides={"id": pl.Int32, "flag": pl.UInt8},
    )
    lookup = np.full(int(target["id"].max()) + 1, 255, dtype=np.uint8)
    lookup[target["id"].to_numpy()] = target["flag"].to_numpy()
    return lookup


def convert_partition(
    source: Path,
    output_prefix: Path,
    columns: list[str],
    max_len: int,
    labels: np.ndarray | None,
) -> np.ndarray:
    frame = (
        pl.read_parquet(source, columns=["id", "rn", *columns])
        .sort(["id", "rn"])
    )
    ids_all = frame["id"].to_numpy()
    ids, starts, counts = np.unique(
        ids_all, return_index=True, return_counts=True
    )
    client_index = np.repeat(np.arange(ids.size), counts)
    relative_position = np.arange(ids_all.size) - np.repeat(starts, counts)
    first_kept = np.repeat(np.maximum(counts - max_len, 0), counts)
    keep = relative_position >= first_kept
    positions = relative_position[keep] - first_kept[keep]

    values = (
        frame.select(columns)
        .to_numpy()
        .astype(np.uint8, copy=False)
    )
    maxima = values.max(axis=0)
    sequences = np.zeros(
        (ids.size, max_len, len(columns)),
        dtype=np.uint8,
    )
    # Shift real categorical values by one; zero remains the padding token.
    sequences[client_index[keep], positions, :] = values[keep] + 1
    lengths = np.minimum(counts, max_len).astype(np.uint8)

    np.save(f"{output_prefix}_x.npy", sequences, allow_pickle=False)
    np.save(
        f"{output_prefix}_id.npy",
        ids.astype(np.int32),
        allow_pickle=False,
    )
    np.save(f"{output_prefix}_len.npy", lengths, allow_pickle=False)
    if labels is not None:
        y = labels[ids]
        if np.any(y == 255):
            raise ValueError(f"Missing targets in {source}")
        np.save(f"{output_prefix}_y.npy", y, allow_pickle=False)

    del frame, values, sequences
    gc.collect()
    return maxima


def prepare(args: argparse.Namespace) -> None:
    NEURAL_DIR.mkdir(exist_ok=True)
    columns = feature_columns(TRAIN_DATA)
    labels = target_lookup()
    global_maxima = np.zeros(len(columns), dtype=np.uint8)

    for split, source, split_labels in [
        ("train", TRAIN_DATA, labels),
        ("test", TEST_DATA, None),
    ]:
        partition_dir = NEURAL_DIR / f"{split}_partitions"
        sequence_dir = NEURAL_DIR / f"{split}_sequences"
        sequence_dir.mkdir(exist_ok=True)
        partitions = partition_parquet(
            source,
            partition_dir,
            args.partitions,
            args.read_batch_size,
        )
        for index, partition in enumerate(partitions):
            prefix = sequence_dir / f"shard_{index:02d}"
            expected = Path(f"{prefix}_x.npy")
            if expected.exists():
                values = np.load(expected, mmap_mode="r")
                maxima = values.max(axis=(0, 1))
                maxima = np.maximum(maxima.astype(np.int16) - 1, 0)
            else:
                maxima = convert_partition(
                    partition,
                    prefix,
                    columns,
                    args.max_len,
                    split_labels,
                )
            global_maxima = np.maximum(global_maxima, maxima)
            print(f"{split}: sequence shard {index + 1}/{args.partitions}")

    metadata = {
        "feature_names": columns,
        "cardinalities": (global_maxima.astype(int) + 2).tolist(),
        "max_len": args.max_len,
        "partitions": args.partitions,
    }
    (NEURAL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Saved metadata to {NEURAL_DIR / 'metadata.json'}")


class ArrayDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        lengths: np.ndarray,
        y: np.ndarray | None,
        indices: np.ndarray,
    ) -> None:
        self.x = x
        self.lengths = lengths
        self.y = y
        self.indices = indices

    def __len__(self) -> int:
        return self.indices.size

    def __getitem__(self, index: int):
        row = int(self.indices[index])
        sequence = torch.from_numpy(np.array(self.x[row], copy=True)).long()
        length = torch.tensor(int(self.lengths[row]), dtype=torch.long)
        if self.y is None:
            return sequence, length
        target = torch.tensor(float(self.y[row]), dtype=torch.float32)
        return sequence, length, target


class CreditTransformer(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        max_len: int,
        d_model: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        offsets = np.cumsum([0, *cardinalities[:-1]], dtype=np.int64)
        self.register_buffer(
            "feature_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=True,
        )
        self.embedding = nn.EmbeddingBag(
            sum(cardinalities),
            d_model,
            mode="sum",
            include_last_offset=False,
        )
        self.position = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, feature_count = x.shape
        global_indices = x + self.feature_offsets[None, None, :]
        flat_indices = global_indices.reshape(-1)
        offsets = torch.arange(
            0,
            flat_indices.numel(),
            feature_count,
            device=x.device,
        )
        weights = (x != 0).reshape(-1).to(self.embedding.weight.dtype)
        hidden = self.embedding(
            flat_indices,
            offsets,
            per_sample_weights=weights,
        ).reshape(batch_size, sequence_length, -1)
        hidden = hidden / math.sqrt(feature_count)
        positions = torch.arange(sequence_length, device=x.device)
        hidden = hidden + self.position(positions)[None, :, :]

        padding_mask = (
            torch.arange(sequence_length, device=x.device)[None, :]
            >= lengths[:, None]
        )
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        hidden = self.norm(hidden)

        valid = (~padding_mask).unsqueeze(-1)
        mean_pool = (hidden * valid).sum(dim=1) / lengths.clamp_min(1)[:, None]
        max_pool = hidden.masked_fill(~valid, -1e4).max(dim=1).values
        last_index = (lengths - 1).clamp_min(0)
        last_pool = hidden[
            torch.arange(batch_size, device=x.device),
            last_index,
        ]
        return self.head(
            torch.cat([last_pool, mean_pool, max_pool], dim=1)
        ).squeeze(1)


def load_metadata() -> dict:
    return json.loads((NEURAL_DIR / "metadata.json").read_text(encoding="utf-8"))


def make_model(metadata: dict, args: argparse.Namespace) -> CreditTransformer:
    return CreditTransformer(
        cardinalities=metadata["cardinalities"],
        max_len=metadata["max_len"],
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
    )


def shard_paths(split: str, count: int) -> list[Path]:
    root = NEURAL_DIR / f"{split}_sequences"
    return [root / f"shard_{index:02d}" for index in range(count)]


@torch.inference_mode()
def validate(
    model: nn.Module,
    paths: list[Path],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_ids: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    for prefix in paths:
        x = np.load(f"{prefix}_x.npy", mmap_mode="r")
        ids = np.load(f"{prefix}_id.npy", mmap_mode="r")
        lengths = np.load(f"{prefix}_len.npy", mmap_mode="r")
        y = np.load(f"{prefix}_y.npy", mmap_mode="r")
        indices = np.flatnonzero(ids % 10 == 0)
        all_ids.append(np.asarray(ids[indices]))
        loader = DataLoader(
            ArrayDataset(x, lengths, y, indices),
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=False,
        )
        for sequences, batch_lengths, targets in loader:
            sequences = sequences.to(device, non_blocking=True)
            batch_lengths = batch_lengths.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(sequences, batch_lengths)
            all_predictions.append(
                torch.sigmoid(logits).float().cpu().numpy()
            )
            all_targets.append(targets.numpy())
        del loader, x, ids, lengths, y
        gc.collect()
    validation_ids = np.concatenate(all_ids)
    targets = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)
    return (
        float(roc_auc_score(targets, predictions)),
        validation_ids,
        targets,
        predictions,
    )


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check nvidia-smi and PyTorch")
    seed_everything(args.seed)
    metadata = load_metadata()
    paths = shard_paths("train", metadata["partitions"])
    device = torch.device("cuda")
    model = make_model(metadata, args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.pos_weight, device=device)
    )
    best_auc = -1.0
    patience_left = args.patience
    checkpoint_path = NEURAL_DIR / f"transformer_seed{args.seed}.pt"
    validation_path = NEURAL_DIR / f"transformer_validation_seed{args.seed}.npz"

    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled_paths = paths.copy()
        random.shuffle(shuffled_paths)
        running_loss = 0.0
        examples = 0
        for shard_number, prefix in enumerate(shuffled_paths, start=1):
            x = np.load(f"{prefix}_x.npy", mmap_mode="r")
            ids = np.load(f"{prefix}_id.npy", mmap_mode="r")
            lengths = np.load(f"{prefix}_len.npy", mmap_mode="r")
            y = np.load(f"{prefix}_y.npy", mmap_mode="r")
            indices = np.flatnonzero(ids % 10 != 0)
            dataset = ArrayDataset(x, lengths, y, indices)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True,
                persistent_workers=False,
                drop_last=True,
            )
            for sequences, batch_lengths, targets in loader:
                sequences = sequences.to(device, non_blocking=True)
                batch_lengths = batch_lengths.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(sequences, batch_lengths)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                running_loss += float(loss.detach()) * targets.size(0)
                examples += targets.size(0)
            del loader, dataset, x, ids, lengths, y
            gc.collect()
            print(
                f"\repoch {epoch}: shard {shard_number}/{len(paths)} "
                f"loss={running_loss / max(examples, 1):.5f}",
                end="",
                flush=True,
            )
        print()

        auc, validation_ids, validation_targets, validation_predictions = validate(
            model, paths, device, args
        )
        print(f"epoch {epoch}: validation ROC-AUC={auc:.8f}")
        if auc > best_auc:
            best_auc = auc
            patience_left = args.patience
            torch.save(
                {
                    "model": model.state_dict(),
                    "auc": auc,
                    "args": vars(args),
                    "metadata": metadata,
                },
                checkpoint_path,
            )
            np.savez(
                validation_path,
                id=validation_ids,
                target=validation_targets,
                prediction=validation_predictions,
            )
            print("Saved new best checkpoint")
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping")
                break

    print(f"Best validation ROC-AUC: {best_auc:.8f}")


@torch.inference_mode()
def predict(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check nvidia-smi and PyTorch")
    device = torch.device("cuda")
    checkpoint = torch.load(
        NEURAL_DIR / f"transformer_seed{args.seed}.pt",
        map_location=device,
        weights_only=False,
    )
    metadata = checkpoint["metadata"]
    saved_args = argparse.Namespace(**checkpoint["args"])
    model = make_model(metadata, saved_args).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    prediction_by_id: dict[int, float] = {}
    for shard_number, prefix in enumerate(
        shard_paths("test", metadata["partitions"]),
        start=1,
    ):
        x = np.load(f"{prefix}_x.npy", mmap_mode="r")
        ids = np.load(f"{prefix}_id.npy", mmap_mode="r")
        lengths = np.load(f"{prefix}_len.npy", mmap_mode="r")
        indices = np.arange(ids.size)
        loader = DataLoader(
            ArrayDataset(x, lengths, None, indices),
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=False,
        )
        offset = 0
        for sequences, batch_lengths in loader:
            sequences = sequences.to(device, non_blocking=True)
            batch_lengths = batch_lengths.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(sequences, batch_lengths)
            values = torch.sigmoid(logits).float().cpu().numpy()
            batch_ids = ids[offset : offset + values.size]
            prediction_by_id.update(
                zip(batch_ids.astype(int).tolist(), values.astype(float).tolist())
            )
            offset += values.size
        del loader, x, ids, lengths
        gc.collect()
        print(f"test prediction shard {shard_number}/{metadata['partitions']}")

    output = NEURAL_DIR / f"submission_transformer_seed{args.seed}.csv"
    with (
        open(SAMPLE_SUBMISSION, encoding="utf-8-sig", newline="") as source,
        output.open("w", encoding="ascii", newline="\n") as destination,
    ):
        reader = csv.DictReader(source)
        destination.write("id,flag\n")
        count = 0
        for row in reader:
            row_id = int(row["id"])
            value = prediction_by_id[row_id]
            formatted = f"{value:.18f}".rstrip("0").rstrip(".")
            if formatted.startswith("0."):
                formatted = formatted[1:]
            destination.write(f"{row_id},{formatted}\n")
            count += 1
    if count != 900_000:
        raise ValueError(f"Expected 900000 predictions, got {count}")
    print(f"Saved {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Credit history Transformer")
    parser.add_argument(
        "stage",
        choices=("prepare", "train", "predict", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--partitions", type=int, default=32)
    parser.add_argument("--read-batch-size", type=int, default=100_000)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"prepare", "all"}:
        prepare(args)
    if args.stage in {"train", "all"}:
        train(args)
    if args.stage in {"predict", "all"}:
        predict(args)


if __name__ == "__main__":
    main()
