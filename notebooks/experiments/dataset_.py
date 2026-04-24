from pathlib import Path

import numpy as np
import toml
import torch
from scipy.spatial import Delaunay
from sentence_transformers.util import get_device_name
from sklearn.datasets import make_blobs
from torch.utils.data import DataLoader, TensorDataset

script_dir = Path(__file__).parent
config = toml.load(script_dir / 'settings.toml')['figures']


def get_device() -> torch.device:
    return torch.device(get_device_name())


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


make_dataset = make_geometric_dataset


def make_dataloader(
    n_samples: int,
    n_features: int,
    n_classes: int,
    cluster_std: float,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, DataLoader]:

    X, y = make_dataset(n_samples, n_features, n_classes, cluster_std)

    X_tensor = torch.Tensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)

    dataloader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    return X_tensor, y_tensor, dataloader
