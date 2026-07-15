# Kubeflow MNIST-Style Distributed Training Demo

This directory contains two related examples:

1. **Simple CPU pipeline**
   - two KFP tasks
   - local CPU PyTorch DDP inside one training task pod
2. **Multi-node GPU pipeline**
   - two KFP tasks
   - the second task creates a Kubeflow Trainer v2 `TrainJob`
   - GPU training runs in separate worker pods managed by a `ClusterTrainingRuntime`
   - `torchrun`, NCCL, and PyTorch DDP

The dataset is `sklearn.datasets.load_digits()`, a small offline 0-9 handwritten
digit dataset. It is used instead of the full MNIST dataset so the demo does not
need `torchvision` or an external dataset download.

---

## Files

| File | Purpose |
|---|---|
| `data.py` | Data preparation component for the simple CPU pipeline |
| `train.py` | Single-pod CPU DDP component |
| `pipeline.py` | Compiles the simple two-task CPU pipeline |
| `distributed_train.py` | Standalone multi-node / multi-GPU DDP worker reference |
| `distributed_pipeline.py` | Compiles the two-task pipeline that launches a multi-node GPU `TrainJob` via Kubeflow Trainer v2 |
| `serving.py` | Loads the saved TorchScript model for local prediction or a small HTTP server |
| `rbac-trainjob.yaml` | RBAC Role/RoleBinding for the KFP service account to manage `TrainJob` resources |

---

# 1. Simple CPU pipeline

## Architecture

```text
KFP Pipeline

prepare-data
    |
    v
distributed-train
    |- local DDP process 0
    `- local DDP process 1
```

Compile:

```bash
python pipeline.py
```

Output:

```text
mnist_ddp_pipeline.yaml
```

Recommended parameters:

```text
epochs=5
batch_size=64
learning_rate=0.05
world_size=2
```

---

# 2. Multi-node GPU pipeline (Kubeflow Trainer v2)

## Prerequisites

The multi-node GPU pipeline uses **Kubeflow Trainer v2** (`TrainJob` API).
The following must be in place before running the pipeline:

1. **Red Hat OpenShift AI** with the Kubeflow Trainer component enabled in the
   `DataScienceCluster`.
2. **JobSet Operator** installed from OLM.
3. A **`ClusterTrainingRuntime`** named `torch-distributed` must exist
   (pre-installed by RHOAI when Kubeflow Trainer is enabled).
4. **NVIDIA GPU-capable nodes** with a working GPU Operator / device plugin.
5. **RBAC permissions** for the pipeline service account to create and inspect
   `TrainJob` resources.

Verify the runtime is available:

```bash
oc get clustertrainingruntime torch-distributed
```

Apply RBAC for the KFP service account (adjust the namespace and service
account name if different from `demo` / `pipeline-runner-dspa`):

```bash
oc apply -f rbac-trainjob.yaml
```

This grants the pipeline service account permission to create/get/list/watch
`TrainJob` resources and read `ClusterTrainingRuntime` resources.
See [section 13 (RBAC)](#13-rbac) for details.

> **Note:** This pipeline does **not** use the legacy Training Operator v1
> (`PyTorchJob`). If your cluster only has Training Operator v1, see the
> [Kubeflow migration guide](https://www.kubeflow.org/docs/components/trainer/operator-guides/migration/).

---

## Architecture

```text
KFP Pipeline

Task 1
prepare-data-config
        |
        v
Task 2
launch-distributed-training
        |
        | TrainerClient.train(runtime="torch-distributed", ...)
        v
Kubeflow Trainer v2
        |
        v
TrainJob  +  ClusterTrainingRuntime (torch-distributed)

Worker Pod / Node 0
  GPU 0 -> local rank 0
  GPU 1 -> local rank 1
  ...

Worker Pod / Node 1
  GPU 0 -> local rank 0
  GPU 1 -> local rank 1
  ...

All processes:
  torchrun (configured by the runtime)
  + NCCL
  + DistributedDataParallel
  + DistributedSampler
```

For example:

```text
num_nodes=2
gpus_per_node=4
```

results conceptually in:

```text
2 worker pods
x
4 GPUs requested per worker pod
=
8 DDP processes
WORLD_SIZE=8
```

The exact pod placement depends on the Kubernetes scheduler, available GPU nodes,
taints/tolerations, affinity rules, and any queue or gang-scheduling setup.

---

# 3. Why task 1 does not pass a large KFP Dataset artifact to TrainJob workers

A `TrainJob` is a separate Kubernetes workload created by the second KFP task.
Its worker pods do not automatically share the launcher task's local artifact
filesystem.

For this tiny demo:

1. task 1 validates the bundled scikit-learn digits dataset,
2. task 1 outputs deterministic split configuration,
3. every TrainJob worker reconstructs the same dataset locally,
4. `DistributedSampler` gives each DDP process a different training shard.

This avoids external downloads and shared dataset storage.

For a real dataset, replace this pattern with:

- S3-compatible object storage,
- a shared RWX PVC,
- a Kubeflow Trainer data initializer,
- a data cache / distributed data layer.

---

# 4. Requirements for the multi-node GPU pipeline

The cluster needs:

- Red Hat OpenShift AI (RHOAI) 3.2+ with Kubeflow Trainer v2 enabled
- JobSet Operator installed from OLM
- A `ClusterTrainingRuntime` named `torch-distributed`
- NVIDIA GPU-capable Kubernetes nodes
- A working NVIDIA device plugin / GPU Operator setup
- Network connectivity between training worker pods for NCCL
- RBAC permissions for the pipeline service account

The KFP launcher task image needs:

- Python 3.12
- KFP 2.x
- Kubernetes Python client (`kubernetes` package, installed at runtime by the component)

The training worker image is managed by the `ClusterTrainingRuntime` and
typically includes:

- Python 3.12
- PyTorch with CUDA support
- NumPy
- scikit-learn

---

# 5. Pipeline image

The KFP launcher task defaults to:

```text
image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/runtime-pytorch:pytorch
```

Your OpenShift AI installation may use a different image.

Override at compile time:

```bash
PIPELINE_IMAGE=<image-for-kfp-launcher-task> python distributed_pipeline.py
```

The training worker image is determined by the `ClusterTrainingRuntime`
(`torch-distributed`), not by the pipeline. To use a custom training image,
create a namespace-scoped `TrainingRuntime` or override the runtime
configuration.

---

# 6. Compile the multi-node GPU pipeline

Run:

```bash
python distributed_pipeline.py
```

Output:

```text
distributed_mnist_pipeline.yaml
```

Or:

```bash
python distributed_pipeline.py --output my_distributed_pipeline.yaml
```

Upload the compiled YAML to the Kubeflow / OpenShift AI Pipelines UI.

---

# 7. Recommended first multi-node run

Start small:

```text
num_nodes=2
gpus_per_node=1
cpu_per_node=4
memory_per_node=8Gi
epochs=5
batch_size=64
learning_rate=0.05
```

This gives:

```text
Worker Pod 0 -> 1 GPU -> rank 0
Worker Pod 1 -> 1 GPU -> rank 1

WORLD_SIZE=2
```

After this works, try:

```text
num_nodes=2
gpus_per_node=4
```

for:

```text
WORLD_SIZE=8
```

---

# 8. Persist the model with a shared PVC

Without a PVC, training can complete, but the saved model remains on the rank-0
worker pod's ephemeral filesystem.

For persistence, create a RWX PVC:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mnist-model-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 1Gi
```

Apply:

```bash
oc apply -f pvc.yaml
```

Run the pipeline with:

```text
model_pvc_name=mnist-model-pvc
model_path=/mnt/shared/model.pt
```

The Kubeflow SDK uses the `pvc://` URI scheme to automatically mount the PVC
on all training pods. Only global rank 0 saves the TorchScript model.

---

# 9. Main pipeline parameters

## Distributed topology

```text
num_nodes=2
gpus_per_node=1
```

Conceptually:

```text
WORLD_SIZE = num_nodes * gpus_per_node
```

Example:

```text
2 nodes x 4 GPUs = 8 DDP processes
```

## Worker resources

```text
cpu_per_node=4
memory_per_node=8Gi
```

GPU resources are requested via:

```text
gpus_per_node
```

mapped to `nvidia.com/gpu` in the `resources_per_node` dict.

## Training

```text
epochs=5
batch_size=64
learning_rate=0.05
```

`batch_size` is per DDP process.

Approximate global batch size:

```text
global batch size = batch_size x WORLD_SIZE
```

Example:

```text
64 x 8 = 512
```

Larger distributed runs can require learning-rate tuning.

## Runtime

```text
runtime_name=torch-distributed
```

The `ClusterTrainingRuntime` to use. RHOAI pre-installs `torch-distributed`.
List all available runtimes:

```bash
oc get clustertrainingruntime
```

---

# 10. What distributed_pipeline.py does

The KFP graph has exactly two tasks:

```text
prepare-data-config
        |
        v
launch-distributed-training
```

The second task uses the Kubernetes Python client to create a `TrainJob` CR
directly:

```python
from kubernetes import client as k8s_client, config as k8s_config

k8s_config.load_incluster_config()
custom_api = k8s_client.CustomObjectsApi()

custom_api.create_namespaced_custom_object(
    group="trainer.kubeflow.org",
    version="v1alpha1",
    namespace=namespace,
    plural="trainjobs",
    body={
        "apiVersion": "trainer.kubeflow.org/v1alpha1",
        "kind": "TrainJob",
        "metadata": {"name": job_name},
        "spec": {
            "runtimeRef": {
                "name": "torch-distributed",
                "kind": "ClusterTrainingRuntime",
            },
            "trainer": {
                "numNodes": num_nodes,
                "command": ["bash", "-c", "... inline training script ..."],
                "resourcesPerNode": {
                    "requests": {
                        "nvidia.com/gpu": str(gpus_per_node),
                        "cpu": str(cpu_per_node),
                        "memory": memory_per_node,
                    },
                },
            },
        },
    },
)
```

The training script is embedded directly in the `TrainJob` command via
`python -c`, with training parameters passed through the `TRAIN_CONFIG`
environment variable. This avoids needing ConfigMap creation permissions.

The launcher task then polls the `TrainJob` status until it completes or
fails.

The KFP launcher task itself does not request a GPU.

The `TrainJob` worker pods (managed by the `ClusterTrainingRuntime`) request
the GPUs.

---

# 11. distributed_train.py

`distributed_train.py` is the standalone worker-side equivalent.

Use it for:

- learning how DDP environment variables work,
- direct `torchrun` testing,
- custom training-image development,
- NCCL debugging independent of KFP.

The pipeline does not import this file at runtime. The training code is
embedded inline in the `TrainJob` command.

The logic is intentionally similar.

---

## Single-node multi-GPU test

One node with four GPUs:

```bash
torchrun \
  --standalone \
  --nproc-per-node=4 \
  distributed_train.py \
  --model-path /mnt/shared/model.pt
```

Result:

```text
LOCAL_RANK=0 -> GPU 0
LOCAL_RANK=1 -> GPU 1
LOCAL_RANK=2 -> GPU 2
LOCAL_RANK=3 -> GPU 3

WORLD_SIZE=4
```

---

## Manual two-node example

Assume:

```text
Node 0 IP: 10.0.0.10
2 nodes
4 GPUs per node
```

Node 0:

```bash
torchrun \
  --nnodes=2 \
  --nproc-per-node=4 \
  --node-rank=0 \
  --master-addr=10.0.0.10 \
  --master-port=29500 \
  distributed_train.py \
  --model-path /mnt/shared/model.pt
```

Node 1:

```bash
torchrun \
  --nnodes=2 \
  --nproc-per-node=4 \
  --node-rank=1 \
  --master-addr=10.0.0.10 \
  --master-port=29500 \
  distributed_train.py \
  --model-path /mnt/shared/model.pt
```

Topology:

```text
Node 0
  rank 0 -> GPU 0
  rank 1 -> GPU 1
  rank 2 -> GPU 2
  rank 3 -> GPU 3

Node 1
  rank 4 -> GPU 0
  rank 5 -> GPU 1
  rank 6 -> GPU 2
  rank 7 -> GPU 3

WORLD_SIZE=8
```

---

# 12. Optional shared NPZ dataset for distributed_train.py

The standalone worker can also load a shared NPZ file containing:

```text
x_train
y_train
x_test
y_test
```

Example:

```bash
torchrun \
  --standalone \
  --nproc-per-node=4 \
  distributed_train.py \
  --data-path /mnt/shared/digits.npz \
  --model-path /mnt/shared/model.pt
```

When `--data-path` is omitted, each worker loads the built-in scikit-learn
digits dataset locally with the same deterministic split.

---

# 13. RBAC

The service account used by the `launch-distributed-training` KFP task
(typically `pipeline-runner-dspa`) must be allowed to interact with Kubeflow
Trainer v2 resources.

Required permissions:

```text
create/get/list/watch/delete  TrainJobs          (trainer.kubeflow.org)
get/list                      TrainingRuntimes   (trainer.kubeflow.org)
get/list                      ClusterTrainingRuntimes (trainer.kubeflow.org, cluster-scoped)
```

Apply the provided RBAC manifest (edit namespace/service account if needed):

```bash
oc apply -f rbac-trainjob.yaml
```

Verify:

```bash
oc auth can-i create trainjobs.trainer.kubeflow.org \
  --as=system:serviceaccount:demo:pipeline-runner-dspa -n demo
```

A `403 Forbidden` error in the launcher task means the pipeline service
account does not have the required permissions.

For full RHOAI guidance, see
[Configuring User Permissions for SDK Access](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/working_with_distributed_workloads/running-kubeflow-trainerv2_distributed-workloads).

---

# 14. Test the saved model with serving.py

One prediction:

```bash
python serving.py \
  --model /mnt/shared/model.pt \
  --index 7
```

Start the HTTP server:

```bash
python serving.py \
  --model /mnt/shared/model.pt \
  --serve \
  --port 8080
```

Health check:

```bash
curl http://localhost:8080/health
```

---

# 15. Troubleshooting

## CUDA is not available

Check:

- the training image (managed by the `ClusterTrainingRuntime`) contains
  CUDA-enabled PyTorch,
- GPUs are allocatable,
- the NVIDIA device plugin / GPU Operator is healthy,
- the TrainJob worker pod actually received a GPU.

## NCCL initialization fails

Check:

- pod-to-pod network connectivity,
- NetworkPolicies and firewalls,
- RDMA configuration when applicable,
- NCCL network interface selection,
- that all worker pods started successfully.

Useful debug environment variables:

```text
NCCL_DEBUG=INFO
TORCH_DISTRIBUTED_DEBUG=DETAIL
```

## TrainJob pods remain Pending

Check:

- available GPU count,
- node selectors,
- taints and tolerations,
- resource quotas,
- scheduler / Kueue configuration,
- PVC topology and access mode.

## PVC mount fails across nodes

Use a storage backend that supports the required multi-node access pattern.
A shared model PVC commonly needs `ReadWriteMany`.

## The KFP task cannot create TrainJob

Check the pipeline task service account RBAC.
See [section 13 (RBAC)](#13-rbac).

## ClusterTrainingRuntime not found

Verify that Kubeflow Trainer v2 is enabled in the `DataScienceCluster` and
the JobSet Operator is installed:

```bash
oc get clustertrainingruntime
oc get crd trainjobs.trainer.kubeflow.org
```

---

# 16. Summary

Use:

```text
pipeline.py
```

for the smallest CPU-only KFP demo (no Training Operator or Trainer required).

Use:

```text
distributed_pipeline.py
```

for the KFP -> Kubeflow Trainer v2 -> multi-node GPU DDP demo.

Use:

```text
distributed_train.py
```

to understand or manually test worker-side PyTorch DDP.
