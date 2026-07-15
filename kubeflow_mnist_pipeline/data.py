import os

from kfp import dsl

# Override at compile time if your cluster uses a different runtime image:
#   PIPELINE_IMAGE=quay.io/...:tag python pipeline.py
PIPELINE_IMAGE = os.getenv(
    "PIPELINE_IMAGE",
    "image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/runtime-pytorch:pytorch",
)


@dsl.component(
    base_image=PIPELINE_IMAGE,
)
def prepare_data(
    dataset: dsl.Output[dsl.Dataset],
    seed: int = 42,
    test_size: float = 0.2,
) -> int:
    """Prepare a small 0-9 handwritten-digit dataset without network access.

    The component uses sklearn.datasets.load_digits() because it is bundled with
    scikit-learn and therefore does not require torchvision or an external
    dataset download. The pipeline remains MNIST-style while being fully
    offline/self-contained.
    """
    import os

    import numpy as np
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()

    # Shape: [N, 1, 8, 8], values normalized to [0, 1].
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

    os.makedirs(os.path.dirname(dataset.path), exist_ok=True)

    # Pass an open file handle so NumPy does not silently append ".npz" to the
    # KFP artifact path.
    with open(dataset.path, "wb") as handle:
        np.savez_compressed(
            handle,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )

    dataset.metadata["framework"] = "scikit-learn"
    dataset.metadata["dataset"] = "digits"
    dataset.metadata["num_classes"] = 10
    dataset.metadata["train_samples"] = int(len(x_train))
    dataset.metadata["test_samples"] = int(len(x_test))
    dataset.metadata["input_shape"] = [1, 8, 8]

    print(
        f"Prepared dataset: train={len(x_train)}, test={len(x_test)}, "
        f"path={dataset.path}"
    )
    return int(len(x_train))
