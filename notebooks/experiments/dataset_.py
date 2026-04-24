from pathlib import Path

import numpy as np
import toml
import torch
from scipy.spatial import Delaunay
from sklearn.datasets import make_blobs
from torch.utils.data import DataLoader, TensorDataset

script_dir = Path(__file__).parent
config = toml.load(script_dir / 'settings.toml')['figures']


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_equidistant_centers(
    n_classes: int, n_features: int, scale: float
) -> np.ndarray:
    assert n_features >= n_classes - 1, (
        f'Need n_features >= n_classes-1 for equidistant centers, '
        f'got {n_features} < {n_classes - 1}'
    )
    # Centered identity rows are equidistant (distance sqrt(2)); SVD gives
    # the n_classes-1 non-trivial directions as an orthonormal embedding.
    basis = np.eye(n_classes) - 1.0 / n_classes
    U, S, _ = np.linalg.svd(basis, full_matrices=False)
    coords = U[:, : n_classes - 1] * S[: n_classes - 1] * scale
    centers = np.zeros((n_classes, n_features))
    centers[:, : n_classes - 1] = coords
    return centers


def make_equidistant_blobs_dataset(
    n_samples: int,
    n_features: int,
    n_classes: int,
    cluster_std: float,
    center_scale: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    result = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=make_equidistant_centers(n_classes, n_features, scale=center_scale),
        cluster_std=cluster_std,
        random_state=42,
    )
    assert len(result) == 2
    X, y = result

    return X, y


def make_blobs_dataset(
    n_samples: int, n_features: int, n_classes: int, cluster_std: float
) -> tuple[np.ndarray, np.ndarray]:
    result = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_classes,
        cluster_std=cluster_std,
        random_state=42,
    )
    assert len(result) == 2
    X, y = result

    return X, y


def make_geometric_dataset(
    n_samples: int, n_features: int, n_classes: int, cluster_std: float
) -> tuple[np.ndarray, np.ndarray]:
    # Make a set of n_classes points in n_features dimensions.
    corners = np.random.randn(n_classes + n_features, n_features)

    # Make a Delauney triangulation of the corners to get simplices.
    tri = Delaunay(corners)

    # Sample points uniformly from the simplices.
    points = []
    classes = []
    for idx, simplex in enumerate(tri.simplices):
        simplex_corners = corners[simplex]
        for _ in range(n_samples // len(tri.simplices)):
            weights = np.random.dirichlet(np.ones(len(simplex)))
            point = np.dot(weights, simplex_corners)
            points.append(point)
            classes.append(simplex)

    return np.array(points), np.array(classes)


make_dataset = make_equidistant_blobs_dataset


def make_dataloader(
    n_samples: int,
    n_features: int,
    n_classes: int,
    cluster_std: float,
    center_scale: float,
    device: torch.device,
    batch_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, DataLoader]:

    X, y = make_dataset(n_samples, n_features, n_classes, cluster_std, center_scale)

    X_tensor = torch.Tensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)

    # batch_size=None means full-batch training.
    dataloader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size or X_tensor.shape[0],
        shuffle=True,
    )

    return X_tensor, y_tensor, dataloader
