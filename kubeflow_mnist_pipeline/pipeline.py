"""Compile a two-task Kubeflow Pipeline.

Tasks
-----
1. prepare-data
2. distributed-train  -> CPU PyTorch DDP with 2 local worker processes

Run:
    python pipeline.py

Optional runtime image override:
    PIPELINE_IMAGE=quay.io/your/image:tag python pipeline.py

The selected image must already contain:
    Python 3.12
    PyTorch
    NumPy
    scikit-learn
    KFP 2.x

No packages are installed at task runtime.
"""

import argparse
import os

from kfp import compiler, dsl

from data import prepare_data
from train import distributed_train


@dsl.pipeline(
    name="mnist-style-cpu-ddp-demo",
    description="Two-task KFP pipeline: offline digits data prep + CPU PyTorch DDP training.",
)
def mnist_ddp_pipeline(
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.05,
    world_size: int = 2,
):
    data_task = prepare_data(
        seed=42,
        test_size=0.2,
    )
    data_task.set_caching_options(False)

    train_task = distributed_train(
        dataset=data_task.outputs["dataset"],
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        world_size=world_size,
    )
    train_task.set_caching_options(False)

    # Keep enough CPU available for two DDP worker processes.
    train_task.set_cpu_request("2")
    train_task.set_cpu_limit("4")
    train_task.set_memory_request("2Gi")
    train_task.set_memory_limit("4Gi")


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=os.path.join(script_dir, "mnist_ddp_pipeline.yaml"),
        help="Compiled KFP pipeline YAML path.",
    )
    args = parser.parse_args()

    compiler.Compiler().compile(
        pipeline_func=mnist_ddp_pipeline,
        package_path=args.output,
    )

    print(f"Compiled pipeline: {os.path.abspath(args.output)}")
    print(
        "Runtime image:",
        os.getenv(
            "PIPELINE_IMAGE",
            "image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/runtime-pytorch:pytorch",
        ),
    )


if __name__ == "__main__":
    main()
