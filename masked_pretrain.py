from __future__ import annotations

import argparse
import gc
import random

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from alfabank_gru import (
    AlfaCreditGRU,
    experiment_dir,
    experiment_shard_paths,
    load_experiment_metadata,
    make_model,
)
from neural import ArrayDataset, seed_everything


class MaskedCreditModel(nn.Module):
    def __init__(
        self,
        backbone: AlfaCreditGRU,
        cardinalities: list[int],
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        recurrent_dim = hidden_size * 2
        self.decoders = nn.ModuleList(
            [
                nn.Linear(recurrent_dim, cardinality)
                for cardinality in cardinalities
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        sequence_length = x.size(1)
        positions = torch.arange(sequence_length, device=x.device)
        embedded = self.backbone.fields(x)
        embedded = embedded + self.backbone.position(positions)[None, :, :]
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.backbone.gru(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=sequence_length,
        )
        output = self.backbone.output_norm(output)
        masked_output = output[mask.any(dim=2)]
        masked_fields = mask[mask.any(dim=2)]
        return [
            decoder(masked_output[masked_fields[:, field]])
            for field, decoder in enumerate(self.decoders)
        ]


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    seed_everything(args.seed)
    metadata = load_experiment_metadata(args)
    backbone = make_model(metadata, args).cuda()
    model = MaskedCreditModel(
        backbone,
        metadata["cardinalities"],
        args.hidden_size,
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda")
    output = experiment_dir(args) / args.output_name

    paths = []
    for split in ("train", "test"):
        paths.extend(
            experiment_shard_paths(
                split,
                metadata["partitions"],
                args,
            )
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(paths)
        running_loss = 0.0
        batches = 0
        for number, prefix in enumerate(paths, start=1):
            x = np.load(f"{prefix}_x.npy", mmap_mode="r")
            lengths = np.load(f"{prefix}_len.npy", mmap_mode="r")
            indices = np.arange(lengths.size)
            loader = DataLoader(
                ArrayDataset(x, lengths, None, indices),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=True,
            )
            for sequences, batch_lengths in loader:
                sequences = sequences.cuda(non_blocking=True)
                batch_lengths = batch_lengths.cuda(non_blocking=True)
                positions = torch.arange(
                    sequences.size(1),
                    device=sequences.device,
                )
                valid = positions[None, :, None] < batch_lengths[:, None, None]
                mask = (
                    torch.rand(sequences.shape, device=sequences.device)
                    < args.mask_probability
                ) & valid
                original = sequences.clone()
                masked = sequences.masked_fill(mask, 0)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(masked, batch_lengths, mask)
                    loss_parts = []
                    event_mask = mask.any(dim=2)
                    field_mask = mask[event_mask]
                    original_events = original[event_mask]
                    for field, field_logits in enumerate(logits):
                        selected = field_mask[:, field]
                        if selected.any():
                            loss_parts.append(
                                nn.functional.cross_entropy(
                                    field_logits,
                                    original_events[selected, field],
                                )
                            )
                    loss = torch.stack(loss_parts).mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                running_loss += float(loss.detach())
                batches += 1
            del loader, x, lengths
            gc.collect()
            print(
                f"\repoch {epoch}: shard {number}/{len(paths)} "
                f"loss={running_loss / max(batches, 1):.5f}",
                end="",
                flush=True,
            )
        print()
        torch.save(
            {
                "model": model.backbone.state_dict(),
                "args": vars(args),
                "metadata": metadata,
                "epoch": epoch,
            },
            output,
        )
        print(f"Saved {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Masked-field pretraining for credit sequences"
    )
    parser.add_argument("--artifact-dir", default="hybrid_artifacts")
    parser.add_argument("--output-name", default="alfa_masked_pretrained.pt")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-embedding-dim", type=int, default=16)
    parser.add_argument("--input-dim", type=int, default=192)
    parser.add_argument("--hidden-size", type=int, default=160)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--architecture", default="gru")
    parser.add_argument("--tcn-channels", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--run-name")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
