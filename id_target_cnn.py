from __future__ import annotations

import argparse
import random

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from baseline import ARTIFACTS, SAMPLE_SUBMISSION, TRAIN_TARGET


BIN_WIDTHS = (64, 256, 1024, 4096, 16384)
BIN_COUNT = 129
HALF_BINS = BIN_COUNT // 2


def run_name(args: argparse.Namespace) -> str:
    return args.run_name or f"id_target_cnn_seed{args.seed}"


def write_compact_submission(
    ids: np.ndarray,
    prediction: np.ndarray,
    output_path,
) -> None:
    temporary = output_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("id,flag\n")
        for row_id, value in zip(ids, prediction, strict=True):
            formatted = f"{value:.18f}".rstrip("0").rstrip(".")
            if formatted.startswith("0."):
                formatted = formatted[1:]
            stream.write(f"{row_id},{formatted}\n")
    temporary.replace(output_path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_targets() -> tuple[np.ndarray, np.ndarray]:
    target = pl.read_csv(
        TRAIN_TARGET,
        schema_overrides={"id": pl.Int32, "flag": pl.UInt8},
    ).sort("id")
    return target["id"].to_numpy(), target["flag"].to_numpy()


def build_prefixes(
    train_ids: np.ndarray,
    targets: np.ndarray,
    max_id: int,
    excluded: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    included = np.ones(train_ids.size, dtype=bool)
    if excluded is not None:
        included &= ~excluded
    sums = np.zeros(max_id + 1, dtype=np.float32)
    counts = np.zeros(max_id + 1, dtype=np.int32)
    sums[train_ids[included]] = targets[included]
    counts[train_ids[included]] = 1
    base_rate = float(targets[included].mean())
    return (
        np.concatenate(([0.0], np.cumsum(sums, dtype=np.float64))),
        np.concatenate(([0], np.cumsum(counts, dtype=np.int64))),
        base_rate,
    )


def make_features(
    query_ids: np.ndarray,
    sum_prefix: np.ndarray,
    count_prefix: np.ndarray,
    base_rate: float,
    own_targets: np.ndarray | None = None,
) -> np.ndarray:
    max_position = sum_prefix.size - 1
    offsets = np.arange(-HALF_BINS, HALF_BINS + 1, dtype=np.int64)
    channels: list[np.ndarray] = []
    query = query_ids.astype(np.int64, copy=False)[:, None]

    for width in BIN_WIDTHS:
        left = query + offsets[None, :] * width - width // 2
        right = left + width
        left = np.clip(left, 0, max_position)
        right = np.clip(right, 0, max_position)
        sums = sum_prefix[right] - sum_prefix[left]
        counts = count_prefix[right] - count_prefix[left]

        if own_targets is not None:
            center = HALF_BINS
            sums[:, center] -= own_targets
            counts[:, center] -= 1

        smoothing = 5.0
        rates = (sums + smoothing * base_rate) / (counts + smoothing)
        density = np.log1p(counts) / np.log1p(width)
        channels.append((rates - base_rate).astype(np.float32))
        channels.append(density.astype(np.float32))

    return np.stack(channels, axis=1)


class ResidualBlock(nn.Module):
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


class IdTargetCNN(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        input_channels = len(BIN_WIDTHS) * 2
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(channels, dilation, dropout)
                for dilation in (1, 2, 4, 8, 16, 1, 2, 4)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(channels * 3, channels * 2),
            nn.LayerNorm(channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.blocks(self.stem(x))
        center = hidden[:, :, HALF_BINS]
        mean = hidden.mean(dim=2)
        maximum = hidden.amax(dim=2)
        return self.head(torch.cat([center, mean, maximum], dim=1)).squeeze(1)


@torch.inference_mode()
def predict_ids(
    model: nn.Module,
    query_ids: np.ndarray,
    sum_prefix: np.ndarray,
    count_prefix: np.ndarray,
    base_rate: float,
    batch_size: int,
    own_targets: np.ndarray | None = None,
) -> np.ndarray:
    model.eval()
    predictions = np.empty(query_ids.size, dtype=np.float32)
    for start in range(0, query_ids.size, batch_size):
        end = min(start + batch_size, query_ids.size)
        own = None if own_targets is None else own_targets[start:end]
        features = make_features(
            query_ids[start:end],
            sum_prefix,
            count_prefix,
            base_rate,
            own,
        )
        tensor = torch.from_numpy(features).cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(tensor)
        predictions[start:end] = torch.sigmoid(logits).float().cpu().numpy()
    return predictions


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    seed_everything(args.seed)
    train_ids, targets = load_targets()
    validation_mask = train_ids % 10 == 0
    training_indices = np.flatnonzero(~validation_mask)
    validation_ids = train_ids[validation_mask]
    validation_targets = targets[validation_mask]
    max_id = max(
        int(train_ids.max()),
        int(
            pl.read_csv(
                SAMPLE_SUBMISSION,
                schema_overrides={"id": pl.Int32},
            )["id"].max()
        ),
    )
    sum_prefix, count_prefix, base_rate = build_prefixes(
        train_ids,
        targets,
        max_id,
        excluded=validation_mask,
    )

    model = IdTargetCNN(args.channels, args.dropout).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * args.steps_per_epoch,
    )
    scaler = torch.amp.GradScaler("cuda")
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.pos_weight, device="cuda")
    )
    name = run_name(args)
    checkpoint_path = ARTIFACTS / f"{name}.pt"
    validation_path = ARTIFACTS / f"{name}_validation.npz"
    best_auc = -1.0
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step in range(1, args.steps_per_epoch + 1):
            selected = np.random.choice(
                training_indices,
                size=args.batch_size,
                replace=False,
            )
            batch_ids = train_ids[selected]
            batch_targets = targets[selected]
            features = make_features(
                batch_ids,
                sum_prefix,
                count_prefix,
                base_rate,
                own_targets=batch_targets,
            )
            x = torch.from_numpy(features).cuda(non_blocking=True)
            y = torch.from_numpy(batch_targets.astype(np.float32)).cuda(
                non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = loss_function(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += float(loss.detach())
            if step % 50 == 0:
                print(
                    f"\repoch {epoch}: step {step}/{args.steps_per_epoch} "
                    f"loss={running_loss / step:.5f}",
                    end="",
                    flush=True,
                )
        print()

        validation_prediction = predict_ids(
            model,
            validation_ids,
            sum_prefix,
            count_prefix,
            base_rate,
            args.predict_batch_size,
        )
        auc = float(roc_auc_score(validation_targets, validation_prediction))
        print(f"epoch {epoch}: validation ROC-AUC={auc:.8f}")
        if auc > best_auc:
            best_auc = auc
            patience_left = args.patience
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "auc": auc,
                },
                checkpoint_path,
            )
            np.savez(
                validation_path,
                id=validation_ids,
                target=validation_targets,
                prediction=validation_prediction,
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
        ARTIFACTS / f"{run_name(args)}.pt",
        map_location="cuda",
        weights_only=False,
    )
    saved_args = argparse.Namespace(**checkpoint["args"])
    model = IdTargetCNN(saved_args.channels, saved_args.dropout).cuda()
    model.load_state_dict(checkpoint["model"])

    train_ids, targets = load_targets()
    sample_ids = pl.read_csv(
        SAMPLE_SUBMISSION,
        schema_overrides={"id": pl.Int32},
    )["id"].to_numpy()
    max_id = int(max(train_ids.max(), sample_ids.max()))
    sum_prefix, count_prefix, base_rate = build_prefixes(
        train_ids,
        targets,
        max_id,
    )
    prediction = predict_ids(
        model,
        sample_ids,
        sum_prefix,
        count_prefix,
        base_rate,
        args.predict_batch_size,
    )
    output = ARTIFACTS / f"submission_{run_name(args)}.csv"
    write_compact_submission(sample_ids, prediction, output)
    print(f"Saved {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN over target rates by id")
    parser.add_argument(
        "stage",
        choices=("train", "predict", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--predict-batch-size", type=int, default=4096)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--pos-weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=5150)
    parser.add_argument("--scale-multiplier", type=float, default=1.0)
    parser.add_argument("--run-name")
    return parser.parse_args()


def main() -> None:
    global BIN_WIDTHS
    args = parse_args()
    BIN_WIDTHS = tuple(
        max(4, int(round(width * args.scale_multiplier)))
        for width in BIN_WIDTHS
    )
    if args.stage in {"train", "all"}:
        train(args)
    if args.stage in {"predict", "all"}:
        predict(args)


if __name__ == "__main__":
    main()
