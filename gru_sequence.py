from __future__ import annotations

import argparse
import csv
import gc
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from neural import (
    NEURAL_DIR,
    SAMPLE_SUBMISSION,
    ArrayDataset,
    load_metadata,
    seed_everything,
    shard_paths,
)


class CreditGRU(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        max_len: int,
        d_model: int,
        hidden_size: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        offsets = np.cumsum([0, *cardinalities[:-1]], dtype=np.int64)
        self.register_buffer(
            "feature_offsets",
            torch.tensor(offsets, dtype=torch.long),
        )
        self.embedding = nn.EmbeddingBag(
            sum(cardinalities),
            d_model,
            mode="sum",
            include_last_offset=False,
        )
        self.position = nn.Embedding(max_len, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
        )
        recurrent_size = hidden_size * 2
        self.output_norm = nn.LayerNorm(recurrent_size)
        self.attention = nn.Sequential(
            nn.Linear(recurrent_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )
        self.head = nn.Sequential(
            nn.Linear(recurrent_size * 4, recurrent_size),
            nn.LayerNorm(recurrent_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(recurrent_size, 1),
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
        hidden = self.input_norm(hidden + self.position(positions)[None, :, :])

        packed = pack_padded_sequence(
            hidden,
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
        attention_score = self.attention(output).squeeze(-1)
        attention_score = attention_score.masked_fill(~valid, -1e4)
        attention_weight = torch.softmax(attention_score, dim=1)
        attention_pool = (output * attention_weight.unsqueeze(-1)).sum(dim=1)
        mean_pool = (output * valid.unsqueeze(-1)).sum(dim=1)
        mean_pool = mean_pool / device_lengths.clamp_min(1).unsqueeze(1)
        max_pool = output.masked_fill(~valid.unsqueeze(-1), -1e4).max(dim=1).values
        final_pool = torch.cat([state[-2], state[-1]], dim=1)
        features = torch.cat(
            [attention_pool, mean_pool, max_pool, final_pool],
            dim=1,
        )
        return self.head(features).squeeze(1)


def make_model(metadata: dict, args: argparse.Namespace) -> CreditGRU:
    return CreditGRU(
        cardinalities=metadata["cardinalities"],
        max_len=metadata["max_len"],
        d_model=args.d_model,
        hidden_size=args.hidden_size,
        layers=args.layers,
        dropout=args.dropout,
    )


def ranking_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    positive = logits[targets > 0.5]
    negative = logits[targets <= 0.5]
    if positive.numel() == 0 or negative.numel() == 0:
        return logits.new_zeros(())
    differences = positive[:, None] - negative[None, :]
    return torch.nn.functional.softplus(-differences).mean()


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
    classification_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.pos_weight, device=device)
    )
    checkpoint_path = NEURAL_DIR / f"gru_seed{args.seed}.pt"
    validation_path = NEURAL_DIR / f"gru_validation_seed{args.seed}.npz"
    best_auc = -1.0
    patience_left = args.patience

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
                running_loss += float(loss.detach()) * batch_targets.size(0)
                examples += batch_targets.size(0)
            del loader, x, ids, lengths, targets
            gc.collect()
            print(
                f"\repoch {epoch}: shard {shard_number}/{len(paths)} "
                f"loss={running_loss / max(examples, 1):.5f}",
                end="",
                flush=True,
            )
        print()
        auc, ids, targets, predictions = validate(model, paths, device, args)
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
    checkpoint = torch.load(
        NEURAL_DIR / f"gru_seed{args.seed}.pt",
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
        )
        offset = 0
        for sequences, batch_lengths in loader:
            sequences = sequences.cuda(non_blocking=True)
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

    output = NEURAL_DIR / f"submission_gru_seed{args.seed}.csv"
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
    parser = argparse.ArgumentParser(description="Credit history BiGRU")
    parser.add_argument(
        "stage",
        choices=("train", "predict", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--pos-weight", type=float, default=4.0)
    parser.add_argument("--rank-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=314)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"train", "all"}:
        train(args)
    if args.stage in {"predict", "all"}:
        predict(args)


if __name__ == "__main__":
    main()
