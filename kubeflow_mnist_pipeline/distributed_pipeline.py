"""
Compile a two-task Kubeflow Pipeline that launches multi-node / multi-GPU
PyTorch DDP training through Kubeflow Trainer v2 (TrainJob).

Pipeline graph
--------------
1. prepare-data-config
2. launch-distributed-training
       |
       +--> TrainerClient.train(...)
              |
              +--> TrainJob  +  ClusterTrainingRuntime (torch-distributed)
                     +--> Worker pod(s) managed by Trainer v2
                           +--> torchrun per node
                           +--> NCCL + DistributedDataParallel

Prerequisites
-------------
- Kubeflow Trainer v2 must be enabled in the DataScienceCluster
- A ClusterTrainingRuntime named 'torch-distributed' must exist
  (pre-installed by RHOAI when Kubeflow Trainer is enabled)
- JobSet Operator must be installed from OLM
- GPU-capable nodes must be available in the cluster

Why task 1 outputs configuration instead of a large Dataset artifact
-------------------------------------------------------------------
A TrainJob creates separate worker pods that do not share the KFP task's
local artifact filesystem.

For this small offline demo, every training worker reconstructs the bundled
scikit-learn digits dataset locally using the same deterministic split config.
For real datasets, use S3/object storage, a shared PVC, or a data initializer.

Compile:
    python distributed_pipeline.py

Optional image override for the KFP launcher task:
    PIPELINE_IMAGE=<image-with-kfp> python distributed_pipeline.py
"""

import argparse
import os

from kfp import compiler, dsl


PIPELINE_IMAGE = os.getenv(
    "PIPELINE_IMAGE",
    "image-registry.openshift-image-registry.svc:5000/"
    "redhat-ods-applications/runtime-pytorch:pytorch",
)


@dsl.component(
    base_image=PIPELINE_IMAGE,
)
def prepare_data_config(
    seed: int = 42,
    test_size: float = 0.2,
) -> str:
    """Validate the bundled digit dataset and return deterministic split config."""
    import json

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()

    _, _, y_train, y_test = train_test_split(
        digits.images,
        digits.target,
        test_size=test_size,
        random_state=seed,
        stratify=digits.target,
    )

    config = {
        "seed": seed,
        "test_size": test_size,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "num_classes": 10,
        "input_shape": [1, 8, 8],
    }

    print(json.dumps(config, indent=2))
    return json.dumps(config)


@dsl.component(
    base_image=PIPELINE_IMAGE,
    packages_to_install=["kubernetes>=28.0.0"],
)
def launch_distributed_training(
    data_config_json: str,
    num_nodes: int = 2,
    gpus_per_node: int = 1,
    cpu_per_node: int = 4,
    memory_per_node: str = "8Gi",
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.05,
    runtime_name: str = "torch-distributed",
    model_pvc_name: str = "",
    model_path: str = "/mnt/shared/model.pt",
) -> str:
    """Create a TrainJob via Kubeflow Trainer v2, wait for it, return job name."""
    import json
    import time
    import uuid

    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config

    data_config = json.loads(data_config_json)
    job_name = "mnist-ddp-" + uuid.uuid4().hex[:8]

    seed = int(data_config["seed"])
    test_size = float(data_config["test_size"])

    import base64

    train_script = r"""
import os, sys, json
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

class DigitNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(64,128),
            torch.nn.ReLU(), torch.nn.Linear(128,10))
    def forward(self, x):
        return self.network(x)

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
cfg = json.loads(os.environ["TRAIN_CONFIG"])

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
dist.init_process_group(backend="nccl", init_method="env://")
try:
    print(f"[rank={rank}] local_rank={local_rank} world_size={world_size} "
          f"host={os.uname().nodename} device={device}")
    digits = load_digits()
    x = digits.images.astype(np.float32)/16.0
    x = x[:,None,:,:]
    y = digits.target.astype(np.int64)
    x_tr,x_te,y_tr,y_te = train_test_split(
        x,y,test_size=cfg["test_size"],random_state=cfg["seed"],stratify=y)
    x_tr=torch.from_numpy(x_tr).float(); y_tr=torch.from_numpy(y_tr).long()
    x_te=torch.from_numpy(x_te).float(); y_te=torch.from_numpy(y_te).long()
    ds = TensorDataset(x_tr,y_tr)
    sampler = DistributedSampler(ds,num_replicas=world_size,rank=rank,
                                 shuffle=True,seed=cfg["seed"])
    loader = DataLoader(ds,batch_size=cfg["batch_size"],sampler=sampler,
                        num_workers=0,pin_memory=True)
    torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
    model = DDP(DigitNet().to(device),device_ids=[local_rank],
                output_device=local_rank)
    opt = torch.optim.SGD(model.parameters(),lr=cfg["learning_rate"],momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    for ep in range(cfg["epochs"]):
        sampler.set_epoch(ep); model.train()
        tl=torch.tensor(0.0,device=device); tb=torch.tensor(0.0,device=device)
        for feat,lab in loader:
            feat=feat.to(device,non_blocking=True)
            lab=lab.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(model(feat),lab); loss.backward(); opt.step()
            tl+=loss.detach(); tb+=1
        dist.all_reduce(tl,op=dist.ReduceOp.SUM)
        dist.all_reduce(tb,op=dist.ReduceOp.SUM)
        if rank==0: print(f"epoch={ep+1}/{cfg['epochs']} loss={(tl/tb).item():.4f}")
    dist.barrier()
    if rank==0:
        model.module.eval()
        with torch.no_grad():
            acc=(model.module(x_te.to(device)).argmax(1)==y_te.to(device)).float().mean().item()
        print(f"test_accuracy={acc:.4f}")
        p=Path(cfg["model_path"]); p.parent.mkdir(parents=True,exist_ok=True)
        torch.jit.script(model.module.cpu()).save(str(p))
        print(f"model_saved={p}")
    dist.barrier()
finally:
    dist.destroy_process_group()
"""

    train_config = json.dumps({
        "seed": seed,
        "test_size": test_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "model_path": model_path,
    })

    script_b64 = base64.b64encode(train_script.strip().encode()).decode()

    trainjob_body = {
        "apiVersion": "trainer.kubeflow.org/v1alpha1",
        "kind": "TrainJob",
        "metadata": {"name": job_name},
        "spec": {
            "runtimeRef": {
                "name": runtime_name,
                "kind": "ClusterTrainingRuntime",
            },
            "trainer": {
                "numNodes": num_nodes,
                "env": [
                    {"name": "TRAIN_CONFIG", "value": train_config},
                ],
                "command": [
                    "bash", "-c",
                    f"echo {script_b64} | base64 -d > /tmp/_train.py && "
                    "python /tmp/_train.py",
                ],
                "resourcesPerNode": {
                    "requests": {
                        "nvidia.com/gpu": str(gpus_per_node),
                        "cpu": str(cpu_per_node),
                        "memory": memory_per_node,
                    },
                    "limits": {
                        "nvidia.com/gpu": str(gpus_per_node),
                    },
                },
            },
        },
    }

    pod_overrides = []

    if model_pvc_name:
        pod_overrides.append({
            "targetJobs": [{"name": "node"}],
            "spec": {
                "volumes": [{
                    "name": "model-storage",
                    "persistentVolumeClaim": {"claimName": model_pvc_name},
                }],
                "containers": [{
                    "name": "node",
                    "volumeMounts": [{
                        "name": "model-storage",
                        "mountPath": "/mnt/shared",
                    }],
                }],
            },
        })
    else:
        print(
            "WARNING: model_pvc_name is empty. "
            "The saved model will remain on the rank-0 worker's ephemeral filesystem."
        )

    if pod_overrides:
        trainjob_body["spec"]["podTemplateOverrides"] = pod_overrides

    k8s_config.load_incluster_config()
    custom_api = k8s_client.CustomObjectsApi()

    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
        namespace = f.read().strip()

    custom_api.create_namespaced_custom_object(
        group="trainer.kubeflow.org",
        version="v1alpha1",
        namespace=namespace,
        plural="trainjobs",
        body=trainjob_body,
    )
    print(f"Created TrainJob: {job_name}")

    timeout = 1800
    poll_interval = 15
    elapsed = 0

    while elapsed < timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval

        obj = custom_api.get_namespaced_custom_object(
            group="trainer.kubeflow.org",
            version="v1alpha1",
            namespace=namespace,
            plural="trainjobs",
            name=job_name,
        )

        conditions = obj.get("status", {}).get("conditions", [])
        for cond in conditions:
            ctype = cond.get("type", "")
            cstatus = cond.get("status", "")
            if ctype == "Complete" and cstatus == "True":
                print(f"TrainJob {job_name} completed successfully.")
                return job_name
            if ctype == "Failed" and cstatus == "True":
                msg = cond.get("message", "unknown reason")
                raise RuntimeError(f"TrainJob {job_name} failed: {msg}")

        print(f"Waiting... ({elapsed}s / {timeout}s)")

    raise TimeoutError(f"TrainJob {job_name} did not complete within {timeout}s")


@dsl.pipeline(
    name="mnist-multi-node-gpu-ddp",
    description=(
        "Two-task KFP pipeline that launches a Kubeflow Trainer v2 TrainJob "
        "for multi-node, multi-GPU PyTorch DDP."
    ),
)
def distributed_mnist_pipeline(
    seed: int = 42,
    test_size: float = 0.2,
    num_nodes: int = 2,
    gpus_per_node: int = 1,
    cpu_per_node: int = 4,
    memory_per_node: str = "8Gi",
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.05,
    runtime_name: str = "torch-distributed",
    model_pvc_name: str = "",
    model_path: str = "/mnt/shared/model.pt",
):
    data_task = prepare_data_config(
        seed=seed,
        test_size=test_size,
    )
    data_task.set_caching_options(False)

    train_task = launch_distributed_training(
        data_config_json=data_task.output,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        cpu_per_node=cpu_per_node,
        memory_per_node=memory_per_node,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        runtime_name=runtime_name,
        model_pvc_name=model_pvc_name,
        model_path=model_path,
    )
    train_task.set_caching_options(False)

    train_task.set_cpu_request("500m")
    train_task.set_cpu_limit("2")
    train_task.set_memory_request("512Mi")
    train_task.set_memory_limit("2Gi")


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=os.path.join(script_dir, "distributed_mnist_pipeline.yaml"),
        help="Compiled KFP pipeline YAML path.",
    )
    args = parser.parse_args()

    compiler.Compiler().compile(
        pipeline_func=distributed_mnist_pipeline,
        package_path=args.output,
    )

    print(f"Compiled pipeline: {os.path.abspath(args.output)}")
    print(f"Pipeline task image: {PIPELINE_IMAGE}")


if __name__ == "__main__":
    main()
