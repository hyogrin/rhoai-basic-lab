import os

from kfp import dsl

PIPELINE_IMAGE = os.getenv(
    "PIPELINE_IMAGE",
    "image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/runtime-pytorch:pytorch",
)


@dsl.component(
    base_image=PIPELINE_IMAGE,
)
def distributed_train(
    dataset: dsl.Input[dsl.Dataset],
    model: dsl.Output[dsl.Model],
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.05,
    world_size: int = 2,
) -> float:
    """Run CPU-only PyTorch DDP with multiple local worker processes.

    This is intentionally a very small demo:
      - one KFP task/pod
      - torch.distributed backend=gloo
      - `world_size` local worker processes launched by torchrun
      - rank 0 writes a TorchScript model artifact

    It demonstrates distributed data-parallel training without requiring
    MPIJob, Horovod, a custom training image, or additional pip installs.
    """
    import os
    import subprocess
    import sys
    import tempfile
    import textwrap

    if world_size < 2:
        raise ValueError("world_size must be >= 2 for this distributed demo.")

    os.makedirs(os.path.dirname(model.path), exist_ok=True)

    worker_source = r"""
import argparse
import json
import os

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


class DigitNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()

    # torchrun sets RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, and MASTER_PORT.
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.manual_seed(42)

    data = np.load(args.data)
    x_train = torch.from_numpy(data["x_train"]).float()
    y_train = torch.from_numpy(data["y_train"]).long()
    x_test = torch.from_numpy(data["x_test"]).float()
    y_test = torch.from_numpy(data["y_test"]).long()

    train_dataset = TensorDataset(x_train, y_train)
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
    )

    raw_model = DigitNet()
    ddp_model = DDP(raw_model)
    optimizer = torch.optim.SGD(
        ddp_model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        ddp_model.train()

        running_loss = 0.0
        batches = 0

        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = ddp_model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batches += 1

        local_loss = torch.tensor(
            running_loss / max(batches, 1),
            dtype=torch.float32,
        )
        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
        average_loss = float(local_loss.item() / world_size)

        if rank == 0:
            print(
                f"epoch={epoch + 1}/{args.epochs} "
                f"world_size={world_size} loss={average_loss:.4f}"
            )

    # Only rank 0 evaluates and writes the model artifact.
    if rank == 0:
        ddp_model.module.eval()
        with torch.no_grad():
            predictions = ddp_model.module(x_test).argmax(dim=1)
            accuracy = float((predictions == y_test).float().mean().item())

        os.makedirs(os.path.dirname(args.model), exist_ok=True)

        # TorchScript keeps serving.py independent from the training class.
        scripted = torch.jit.script(ddp_model.module.cpu())
        scripted.save(args.model)

        with open(args.metrics, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "accuracy": accuracy,
                    "world_size": world_size,
                    "epochs": args.epochs,
                },
                handle,
            )

        print(f"accuracy={accuracy:.4f}")
        print(f"model={args.model}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
"""

    with tempfile.TemporaryDirectory(prefix="kfp-ddp-") as work_dir:
        worker_path = os.path.join(work_dir, "ddp_worker.py")
        metrics_path = os.path.join(work_dir, "metrics.json")

        with open(worker_path, "w", encoding="utf-8") as handle:
            handle.write(textwrap.dedent(worker_source))

        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={world_size}",
            worker_path,
            "--data",
            dataset.path,
            "--model",
            model.path,
            "--metrics",
            metrics_path,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--learning-rate",
            str(learning_rate),
        ]

        print("Launching:", " ".join(command))
        subprocess.run(command, check=True)

        import json

        with open(metrics_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)

    accuracy = float(metrics["accuracy"])

    model.metadata["framework"] = "pytorch"
    model.metadata["format"] = "torchscript"
    model.metadata["distributed_backend"] = "gloo"
    model.metadata["world_size"] = int(metrics["world_size"])
    model.metadata["epochs"] = int(metrics["epochs"])
    model.metadata["accuracy"] = accuracy

    return accuracy
