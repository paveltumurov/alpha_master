from __future__ import annotations

import argparse
import gc
import json
import math
import random
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from neural import (
    ROOT,
    SAMPLE_SUBMISSION,
    TEST_DATA,
    TRAIN_DATA,
    TRAIN_TARGET,
    convert_partition,
    feature_columns,
    partition_parquet,
    target_lookup,
)


HYBRID_DIR = ROOT / "hybrid_artifacts"
ADVANCED_TRAIN = ROOT / "train_features_advanced.parquet"
ADVANCED_TEST = ROOT / "test_features_advanced.parquet"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_split(
    split: str,
    source: Path,
    aggregate_path: Path,
    labels: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, int]:
    partition_dir = HYBRID_DIR / f"{split}_partitions"
    sequence_dir = HYBRID_DIR / f"{split}_sequences"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    partitions = partition_parquet(
        source,
        partition_dir,
        args.partitions,
        args.read_batch_size,
    )

    aggregate = pl.read_parquet(aggregate_path).sort("id")
    aggregate_columns = [column for column in aggregate.columns if column != "id"]
    total_sum = np.zeros(len(aggregate_columns), dtype=np.float64)
    total_square_sum = np.zeros(len(aggregate_columns), dtype=np.float64)
    total_rows = 0
    maxima = np.zeros(len(feature_columns(source)), dtype=np.uint8)

    for index, partition in enumerate(partitions):
        prefix = sequence_dir / f"shard_{index:02d}"
        sequence_path = Path(f"{prefix}_x.npy")
        if not sequence_path.exists():
            shard_maxima = convert_partition(
                partition,
                prefix,
                feature_columns(source),
                args.max_len,
                labels,
            )
        else:
            values = np.load(sequence_path, mmap_mode="r")
            shard_maxima = np.maximum(
                values.max(axis=(0, 1)).astype(np.int16) - 1,
                0,
            ).astype(np.uint8)
        maxima = np.maximum(maxima, shard_maxima)

        ids = np.load(f"{prefix}_id.npy", mmap_mode="r")
        aggregate_shard = (
            aggregate.filter(pl.col("id") % args.partitions == index)
            .sort("id")
        )
        aggregate_ids = aggregate_shard["id"].to_numpy()
        if not np.array_equal(ids, aggregate_ids):
            raise ValueError(f"Aggregate ids differ in {split} shard {index}")
        matrix = (
            aggregate_shard.select(aggregate_columns)
            .to_numpy()
            .astype(np.float32, copy=False)
        )
        np.save(f"{prefix}_agg.npy", matrix, allow_pickle=False)
        if split == "train":
            total_sum += matrix.sum(axis=0, dtype=np.float64)
            total_square_sum += np.square(matrix, dtype=np.float64).sum(
                axis=0,
                dtype=np.float64,
            )
            total_rows += matrix.shape[0]
        print(f"{split}: hybrid shard {index + 1}/{args.partitions}")

    del aggregate
    gc.collect()
    return maxima, total_sum, total_square_sum, total_rows, aggregate_columns


def prepare(args: argparse.Namespace) -> None:
    HYBRID_DIR.mkdir(exist_ok=True)
    if not ADVANCED_TRAIN.exists() or not ADVANCED_TEST.exists():
        raise FileNotFoundError(
            "Upload train_features_advanced.parquet and "
            "test_features_advanced.parquet next to hybrid.py"
        )

    labels = target_lookup()
    train_result = prepare_split(
        "train",
        TRAIN_DATA,
        ADVANCED_TRAIN,
        labels,
        args,
    )
    test_result = prepare_split(
        "test",
        TEST_DATA,
        ADVANCED_TEST,
        None,
        args,
    )
    train_maxima, total_sum, total_square_sum, total_rows, aggregate_columns = (
        train_result
    )
    test_maxima = test_result[0]
    mean = total_sum / total_rows
    variance = np.maximum(total_square_sum / total_rows - mean**2, 1e-6)
    std = np.sqrt(variance)
    metadata = {
        "sequence_features": feature_columns(TRAIN_DATA),
        "cardinalities": (
            np.maximum(train_maxima, test_maxima).astype(int) + 2
        ).tolist(),
        "aggregate_features": aggregate_columns,
        "aggregate_mean": mean.tolist(),
        "aggregate_std": std.tolist(),
        "max_len": args.max_len,
        "partitions": args.partitions,
    }
    (HYBRID_DIR / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    print(f"Saved metadata to {HYBRID_DIR / 'metadata.json'}")


class HybridDataset(Dataset):
    def __init__(
        self,
        sequence: np.ndarray,
        aggregate: np.ndarray,
        lengths: np.ndarray,
        target: np.ndarray | None,
        indices: np.ndarray,
    ) -> None:
        self.sequence = sequence
        self.aggregate = aggregate
        self.lengths = lengths
        self.target = target
        self.indices = indices

    def __len__(self) -> int:
        return self.indices.size

    def __getitem__(self, index: int):
        row = int(self.indices[index])
        sequence = torch.from_numpy(
            np.array(self.sequence[row], copy=True)
        ).long()
        aggregate = torch.from_numpy(
            np.array(self.aggregate[row], copy=True)
        ).float()
        length = torch.tensor(int(self.lengths[row]), dtype=torch.long)
        if self.target is None:
            return sequence, aggregate, length
        target = torch.tensor(float(self.target[row]), dtype=torch.float32)
        return sequence, aggregate, length, target


class HybridTransformer(nn.Module):
    def __init__(
        self,
        metadata: dict,
        d_model: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        cardinalities = metadata["cardinalities"]
        offsets = np.cumsum([0, *cardinalities[:-1]], dtype=np.int64)
        self.register_buffer(
            "feature_offsets",
            torch.tensor(offsets, dtype=torch.long),
        )
        self.register_buffer(
            "aggregate_mean",
            torch.tensor(metadata["aggregate_mean"], dtype=torch.float32),
        )
        self.register_buffer(
            "aggregate_std",
            torch.tensor(metadata["aggregate_std"], dtype=torch.float32),
        )
        self.embedding = nn.EmbeddingBag(
            sum(cardinalities),
            d_model,
            mode="sum",
            include_last_offset=False,
        )
        self.position = nn.Embedding(metadata["max_len"], d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
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
        self.sequence_norm = nn.LayerNorm(d_model)
        aggregate_count = len(metadata["aggregate_features"])
        self.aggregate_encoder = nn.Sequential(
            nn.Linear(aggregate_count, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 1),
        )

    def sequence_features(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
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
        padding_mask = positions[None, :] >= lengths[:, None]
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        hidden = self.sequence_norm(hidden)

        valid = (~padding_mask).unsqueeze(-1)
        mean_pool = (hidden * valid).sum(dim=1) / lengths.clamp_min(1)[:, None]
        max_pool = hidden.masked_fill(~valid, -1e4).max(dim=1).values
        last_pool = hidden[
            torch.arange(batch_size, device=x.device),
            (lengths - 1).clamp_min(0),
        ]
        return torch.cat([last_pool, mean_pool, max_pool], dim=1)

    def forward(
        self,
        sequence: torch.Tensor,
        aggregate: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        sequence_vector = self.sequence_features(sequence, lengths)
        aggregate = (aggregate - self.aggregate_mean) / self.aggregate_std
        aggregate = torch.clamp(aggregate, -10.0, 10.0)
        aggregate_vector = self.aggregate_encoder(aggregate)
        return self.head(
            torch.cat([sequence_vector, aggregate_vector], dim=1)
        ).squeeze(1)


def load_metadata() -> dict:
    return json.loads((HYBRID_DIR / "metadata.json").read_text())


def make_model(metadata: dict, args: argparse.Namespace) -> HybridTransformer:
    return HybridTransformer(
        metadata,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
    )


def shard_prefixes(split: str, partitions: int) -> list[Path]:
    root = HYBRID_DIR / f"{split}_sequences"
    return [root / f"shard_{index:02d}" for index in range(partitions)]


def make_loader(
    prefix: Path,
    indices: np.ndarray,
    batch_size: int,
    workers: int,
    shuffle: bool,
    target_required: bool,
) -> DataLoader:
    sequence = np.load(f"{prefix}_x.npy", mmap_mode="r")
    aggregate = np.load(f"{prefix}_agg.npy", mmap_mode="r")
    lengths = np.load(f"{prefix}_len.npy", mmap_mode="r")
    target = (
        np.load(f"{prefix}_y.npy", mmap_mode="r")
        if target_required
        else None
    )
    dataset = HybridDataset(
        sequence,
        aggregate,
        lengths,
        target,
        indices,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=shuffle,
    )


@torch.inference_mode()
def validate(
    model: nn.Module,
    prefixes: list[Path],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ids_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    for prefix in prefixes:
        ids = np.load(f"{prefix}_id.npy", mmap_mode="r")
        indices = np.flatnonzero(ids % 10 == args.fold)
        loader = make_loader(
            prefix,
            indices,
            args.batch_size * 2,
            args.workers,
            False,
            True,
        )
        ids_all.append(np.asarray(ids[indices]))
        for sequence, aggregate, lengths, target in loader:
            sequence = sequence.to(device, non_blocking=True)
            aggregate = aggregate.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(sequence, aggregate, lengths)
            predictions_all.append(torch.sigmoid(logits).float().cpu().numpy())
            targets_all.append(target.numpy())
        del loader
        gc.collect()
    ids = np.concatenate(ids_all)
    targets = np.concatenate(targets_all)
    predictions = np.concatenate(predictions_all)
    return float(roc_auc_score(targets, predictions)), ids, targets, predictions


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    seed_everything(args.seed)
    metadata = load_metadata()
    prefixes = shard_prefixes("train", metadata["partitions"])
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
    checkpoint_path = HYBRID_DIR / f"hybrid_seed{args.seed}.pt"
    validation_path = HYBRID_DIR / f"hybrid_validation_seed{args.seed}.npz"
    best_auc = -1.0
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = prefixes.copy()
        random.shuffle(shuffled)
        running_loss = 0.0
        examples = 0
        for shard_number, prefix in enumerate(shuffled, start=1):
            ids = np.load(f"{prefix}_id.npy", mmap_mode="r")
            indices = np.flatnonzero(ids % 10 != args.fold)
            loader = make_loader(
                prefix,
                indices,
                args.batch_size,
                args.workers,
                True,
                True,
            )
            for sequence, aggregate, lengths, target in loader:
                sequence = sequence.to(device, non_blocking=True)
                aggregate = aggregate.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(sequence, aggregate, lengths)
                    loss = criterion(logits, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                running_loss += float(loss.detach()) * target.size(0)
                examples += target.size(0)
            del loader
            gc.collect()
            print(
                f"\repoch {epoch}: shard {shard_number}/{len(prefixes)} "
                f"loss={running_loss / max(examples, 1):.5f}",
                end="",
                flush=True,
            )
        print()

        auc, ids, targets, predictions = validate(
            model, prefixes, device, args
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
                id=ids,
                target=targets,
                prediction=predictions,
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
        raise RuntimeError("CUDA is unavailable")
    checkpoint_path = HYBRID_DIR / f"hybrid_seed{args.seed}.pt"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cuda",
        weights_only=False,
    )
    saved_args = argparse.Namespace(**checkpoint["args"])
    metadata = checkpoint["metadata"]
    model = make_model(metadata, saved_args).cuda()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    prediction_by_id: dict[int, float] = {}

    for number, prefix in enumerate(
        shard_prefixes("test", metadata["partitions"]),
        start=1,
    ):
        ids = np.load(f"{prefix}_id.npy", mmap_mode="r")
        indices = np.arange(ids.size)
        loader = make_loader(
            prefix,
            indices,
            args.batch_size * 2,
            args.workers,
            False,
            False,
        )
        offset = 0
        for sequence, aggregate, lengths in loader:
            sequence = sequence.cuda(non_blocking=True)
            aggregate = aggregate.cuda(non_blocking=True)
            lengths = lengths.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(sequence, aggregate, lengths)
            values = torch.sigmoid(logits).float().cpu().numpy()
            batch_ids = ids[offset : offset + values.size]
            prediction_by_id.update(
                zip(batch_ids.astype(int).tolist(), values.astype(float).tolist())
            )
            offset += values.size
        del loader
        gc.collect()
        print(f"test shard {number}/{metadata['partitions']}")

    sample = pl.read_csv(SAMPLE_SUBMISSION, schema_overrides={"id": pl.Int32})
    output = HYBRID_DIR / f"submission_hybrid_seed{args.seed}.csv"
    with output.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("id,flag\n")
        for row_id in sample["id"].to_numpy():
            value = prediction_by_id[int(row_id)]
            formatted = f"{value:.18f}".rstrip("0").rstrip(".")
            if formatted.startswith("0."):
                formatted = formatted[1:]
            stream.write(f"{row_id},{formatted}\n")
    print(f"Saved {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid credit Transformer")
    parser.add_argument(
        "stage",
        choices=("prepare", "train", "predict", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--partitions", type=int, default=32)
    parser.add_argument("--read-batch-size", type=int, default=100_000)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--pos-weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
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
