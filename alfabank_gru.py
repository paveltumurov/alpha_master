from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from gru_sequence import ranking_loss
from neural import (
    NEURAL_DIR,
    ROOT,
    SAMPLE_SUBMISSION,
    ArrayDataset,
    seed_everything,
)


def embedding_dimension(cardinality: int, maximum: int) -> int:
    return min(maximum, max(2, int(round(1.6 * cardinality**0.56))))


class FieldEmbeddings(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        max_embedding_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dimensions = [
            embedding_dimension(cardinality, max_embedding_dim)
            for cardinality in cardinalities
        ]
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, dimension, padding_idx=0)
                for cardinality, dimension in zip(
                    cardinalities,
                    self.dimensions,
                    strict=True,
                )
            ]
        )
        concatenated_dim = sum(self.dimensions)
        self.projection = nn.Sequential(
            nn.Linear(concatenated_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = [
            embedding(x[:, :, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        return self.projection(torch.cat(embedded, dim=-1))


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels * 2,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GLU(dim=1),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AlfaCreditGRU(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        max_len: int,
        max_embedding_dim: int,
        input_dim: int,
        hidden_size: int,
        layers: int,
        dropout: float,
        architecture: str,
        tcn_channels: int,
    ) -> None:
        super().__init__()
        self.fields = FieldEmbeddings(
            cardinalities,
            max_embedding_dim,
            input_dim,
            dropout,
        )
        self.position = nn.Embedding(max_len, input_dim)
        self.tcn = None
        if architecture == "tcn_gru":
            self.tcn = nn.Sequential(
                nn.Conv1d(input_dim, tcn_channels, kernel_size=1),
                *[
                    TemporalResidualBlock(tcn_channels, dilation, dropout)
                    for dilation in (1, 2, 4, 8, 16)
                ],
            )
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
        )
        recurrent_dim = hidden_size * 2
        self.output_norm = nn.LayerNorm(recurrent_dim)
        self.attention = nn.Sequential(
            nn.Linear(recurrent_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )
        pooled_dim = input_dim * 2 + recurrent_dim * 4
        if self.tcn is not None:
            pooled_dim += tcn_channels * 2
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, recurrent_dim * 2),
            nn.LayerNorm(recurrent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(recurrent_dim * 2, recurrent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(recurrent_dim, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        positions = torch.arange(sequence_length, device=x.device)
        embedded = self.fields(x)
        embedded = embedded + self.position(positions)[None, :, :]

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, state = self.gru(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=sequence_length,
        )
        output = self.output_norm(output)

        device_lengths = lengths.to(x.device)
        valid = positions[None, :] < device_lengths[:, None]
        valid_float = valid.unsqueeze(-1)
        denominator = device_lengths.clamp_min(1).unsqueeze(1)

        input_mean = (embedded * valid_float).sum(dim=1) / denominator
        input_max = embedded.masked_fill(
            ~valid_float,
            -1e4,
        ).max(dim=1).values
        tcn_pools: list[torch.Tensor] = []
        if self.tcn is not None:
            temporal = self.tcn(embedded.transpose(1, 2)).transpose(1, 2)
            temporal_mean = (temporal * valid_float).sum(dim=1) / denominator
            temporal_max = temporal.masked_fill(
                ~valid_float,
                -1e4,
            ).max(dim=1).values
            tcn_pools = [temporal_mean, temporal_max]

        attention_score = self.attention(output).squeeze(-1)
        attention_score = attention_score.masked_fill(~valid, -1e4)
        attention_weight = torch.softmax(attention_score, dim=1)
        attention_pool = (output * attention_weight.unsqueeze(-1)).sum(dim=1)
        output_mean = (output * valid_float).sum(dim=1) / denominator
        output_max = output.masked_fill(
            ~valid_float,
            -1e4,
        ).max(dim=1).values
        final_pool = torch.cat([state[-2], state[-1]], dim=1)

        pooled = torch.cat(
            [
                input_mean,
                input_max,
                attention_pool,
                output_mean,
                output_max,
                final_pool,
                *tcn_pools,
            ],
            dim=1,
        )
        return self.head(pooled).squeeze(1)


def make_model(metadata: dict, args: argparse.Namespace) -> AlfaCreditGRU:
    return AlfaCreditGRU(
        cardinalities=metadata["cardinalities"],
        max_len=metadata["max_len"],
        max_embedding_dim=args.max_embedding_dim,
        input_dim=args.input_dim,
        hidden_size=args.hidden_size,
        layers=args.layers,
        dropout=args.dropout,
        architecture=getattr(args, "architecture", "gru"),
        tcn_channels=getattr(args, "tcn_channels", 128),
    )


def experiment_dir(args: argparse.Namespace) -> Path:
    return ROOT / args.artifact_dir


def load_experiment_metadata(args: argparse.Namespace) -> dict:
    return json.loads(
        (experiment_dir(args) / "metadata.json").read_text(encoding="utf-8")
    )


def experiment_shard_paths(
    split: str,
    count: int,
    args: argparse.Namespace,
) -> list[Path]:
    root = experiment_dir(args) / f"{split}_sequences"
    return [root / f"shard_{index:02d}" for index in range(count)]


def experiment_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    return f"alfa_gru_seed{args.seed}"


def training_steps(paths: list[Path], batch_size: int) -> int:
    total = 0
    for prefix in paths:
        ids = np.load(f"{prefix}_id.npy", mmap_mode="r")
        total += int((ids % 10 != 0).sum()) // batch_size
        del ids
    return total


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
        targets = np.load(f"{prefix}_y.npy", mmap_mode="r")
        indices = np.flatnonzero(ids % 10 == 0)
        all_ids.append(np.asarray(ids[indices]))
        loader = DataLoader(
            ArrayDataset(x, lengths, targets, indices),
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        for sequences, batch_lengths, batch_targets in loader:
            sequences = sequences.to(device, non_blocking=True)
            batch_lengths = batch_lengths.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(sequences, batch_lengths)
            all_predictions.append(torch.sigmoid(logits).float().cpu().numpy())
            all_targets.append(batch_targets.numpy())
        del loader, x, ids, lengths, targets
        gc.collect()
    validation_ids = np.concatenate(all_ids)
    validation_targets = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)
    auc = float(roc_auc_score(validation_targets, predictions))
    return auc, validation_ids, validation_targets, predictions


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    seed_everything(args.seed)
    metadata = load_experiment_metadata(args)
    paths = experiment_shard_paths(
        "train",
        metadata["partitions"],
        args,
    )
    device = torch.device("cuda")
    model = make_model(metadata, args).to(device)
    if args.pretrained_path:
        pretrained = torch.load(
            args.pretrained_path,
            map_location=device,
            weights_only=False,
        )
        state = pretrained.get("model", pretrained)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"Loaded pretrained backbone from {args.pretrained_path}; "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.max_learning_rate,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = training_steps(paths, args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_learning_rate,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=args.warmup_fraction,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )
    scaler = torch.amp.GradScaler("cuda")
    classification_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.pos_weight, device=device)
    )
    output_dir = experiment_dir(args)
    run_name = experiment_name(args)
    checkpoint_path = output_dir / f"{run_name}.pt"
    validation_path = output_dir / f"{run_name}_validation.npz"
    best_auc = -1.0
    patience_left = args.patience

    print(
        f"features={len(metadata['cardinalities'])}, "
        f"embedding_dims={model.fields.dimensions}, "
        f"steps_per_epoch={steps_per_epoch}"
    )
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
            targets = np.load(f"{prefix}_y.npy", mmap_mode="r")
            indices = np.flatnonzero(ids % 10 != 0)
            loader = DataLoader(
                ArrayDataset(x, lengths, targets, indices),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=True,
            )
            for sequences, batch_lengths, batch_targets in loader:
                sequences = sequences.to(device, non_blocking=True)
                batch_lengths = batch_lengths.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(sequences, batch_lengths)
                    loss = classification_loss(logits, batch_targets)
                    loss = loss + args.rank_weight * ranking_loss(
                        logits,
                        batch_targets,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                running_loss += float(loss.detach()) * batch_targets.size(0)
                examples += batch_targets.size(0)
            del loader, x, ids, lengths, targets
            gc.collect()
            print(
                f"\repoch {epoch}: shard {shard_number}/{len(paths)} "
                f"loss={running_loss / max(examples, 1):.5f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}",
                end="",
                flush=True,
            )
        print()

        auc, ids, targets, predictions = validate(model, paths, device, args)
        print(f"epoch {epoch}: validation ROC-AUC={auc:.8f}")
        if args.save_snapshots:
            np.savez(
                output_dir / f"{run_name}_epoch{epoch}_validation.npz",
                id=ids,
                target=targets,
                prediction=predictions,
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "auc": auc,
                    "epoch": epoch,
                    "args": vars(args),
                    "metadata": metadata,
                },
                output_dir / f"{run_name}_epoch{epoch}.pt",
            )
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
    output_dir = experiment_dir(args)
    run_name = experiment_name(args)
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else output_dir / f"{run_name}.pt"
    )
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
        experiment_shard_paths("test", metadata["partitions"], saved_args),
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
        )
        offset = 0
        for sequences, batch_lengths in loader:
            sequences = sequences.cuda(non_blocking=True)
            batch_lengths = batch_lengths.cuda(non_blocking=True)
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
        print(f"test shard {number}/{metadata['partitions']}")

    output_name = args.output_name or run_name
    output = output_dir / f"submission_{output_name}.csv"
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
    parser = argparse.ArgumentParser(
        description="Alfa-style field-embedding BiGRU"
    )
    parser.add_argument(
        "stage",
        choices=("train", "predict", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-embedding-dim", type=int, default=16)
    parser.add_argument("--input-dim", type=int, default=192)
    parser.add_argument("--hidden-size", type=int, default=160)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument(
        "--architecture",
        choices=("gru", "tcn_gru"),
        default="gru",
    )
    parser.add_argument("--tcn-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-learning-rate", type=float, default=8e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--pos-weight", type=float, default=4.0)
    parser.add_argument("--rank-weight", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--artifact-dir", default=NEURAL_DIR.name)
    parser.add_argument("--run-name")
    parser.add_argument("--save-snapshots", action="store_true")
    parser.add_argument("--pretrained-path")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--output-name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"train", "all"}:
        train(args)
    if args.stage in {"predict", "all"}:
        predict(args)


if __name__ == "__main__":
    main()
