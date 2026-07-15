"""
Standalone multi-node / multi-GPU PyTorch DDP worker.

This file is useful for:
- understanding the worker-side DDP code,
- testing with torchrun outside Kubeflow Pipelines,
- adapting the training logic into a custom training image.

The distributed_pipeline.py example uses Kubeflow TrainingClient v1.9, which
serializes a self-contained Python training function into PyTorchJob workers.
For that reason, the pipeline does not import this file at runtime. The logic is
kept intentionally similar.

Data modes:
1. Default: use sklearn.datasets.load_digits() locally on every worker.
2. Optional: pass --data-path to use a shared NPZ file containing:
   x_train, y_train, x_test, y_test

Required distributed environment variables are normally provided by torchrun or
Kubeflow Training Operator:
    RANK
    LOCAL_RANK
    WORLD_SIZE
    MASTER_ADDR
    MASTER_PORT
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


class DigitNet(torch.nn.Module):
    """Small classifier for 8x8 handwritten digit images."""

    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default="",
        help=(
            "Optional shared NPZ dataset path. "
            "When omitted, sklearn digits is loaded locally on every worker."
        ),
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/shared/model.pt",
        help="Output model path. Only global rank 0 writes this file.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Per-process batch size.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes per DDP process.",
    )
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    required = (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    )

    missing = [name for name in required if name not in os.environ]

    if missing:
        raise RuntimeError(
            "Missing distributed environment variables: "
            + ", ".join(missing)
            + ". Launch with torchrun or Kubeflow Training Operator."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "This script is intended for multi-GPU training."
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )

    return rank, local_rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def load_dataset(
    data_path: str,
    seed: int,
    test_size: float,
) -> tuple[TensorDataset, torch.Tensor, torch.Tensor]:
    if data_path:
        data = np.load(data_path)

        x_train = torch.from_numpy(data["x_train"]).float()
        y_train = torch.from_numpy(data["y_train"]).long()
        x_test = torch.from_numpy(data["x_test"]).float()
        y_test = torch.from_numpy(data["y_test"]).long()

    else:
        digits = load_digits()

        x = digits.images.astype(np.float32) / 16.0
        x = x[:, None, :, :]
        y = digits.target.astype(np.int64)

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )

        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).long()
        x_test = torch.from_numpy(x_test).float()
        y_test = torch.from_numpy(y_test).long()

    return TensorDataset(x_train, y_train), x_test, y_test


def evaluate_and_save(
    model: torch.nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
    model_path: str,
) -> None:
    model.eval()

    with torch.no_grad():
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        logits = model(x_test)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == y_test).float().mean().item()

    print(f"test_accuracy={accuracy:.4f}")

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scripted_model = torch.jit.script(model.cpu())
    scripted_model.save(str(output_path))

    print(f"model_saved={output_path}")


def train(args: argparse.Namespace) -> None:
    rank, local_rank, world_size, device = setup_distributed()

    try:
        if rank == 0:
            print("Starting distributed training")
            print(f"WORLD_SIZE={world_size}")
            print(f"MASTER_ADDR={os.environ['MASTER_ADDR']}")
            print(f"MASTER_PORT={os.environ['MASTER_PORT']}")

        print(
            f"[rank={rank}] "
            f"local_rank={local_rank}, "
            f"device={device}, "
            f"host={os.uname().nodename}"
        )

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        train_dataset, x_test, y_test = load_dataset(
            data_path=args.data_path,
            seed=args.seed,
            test_size=args.test_size,
        )

        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        model = DigitNet().to(device)

        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
        )

        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            model.train()

            total_loss = torch.tensor(0.0, device=device)
            total_batches = torch.tensor(0.0, device=device)

            for features, labels in train_loader:
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                logits = model(features)
                loss = loss_fn(logits, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.detach()
                total_batches += 1

            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_batches, op=dist.ReduceOp.SUM)

            average_loss = (total_loss / total_batches).item()

            if rank == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} "
                    f"loss={average_loss:.4f}"
                )

        dist.barrier()

        if rank == 0:
            evaluate_and_save(
                model=model.module,
                x_test=x_test,
                y_test=y_test,
                device=device,
                model_path=args.model_path,
            )

        dist.barrier()

    finally:
        cleanup_distributed()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
