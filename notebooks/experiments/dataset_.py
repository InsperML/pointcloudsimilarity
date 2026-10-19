import torch
from torch.utils.data import DataLoader, TensorDataset

from pointcloudsimilarity.datasets import equidistant_blobs


def make_dataloader(
    n_samples: int,
    n_features: int,
    n_classes: int,
    cluster_std: float,
    center_scale: float,
    device: torch.device,
    batch_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, DataLoader]:

    X, y = equidistant_blobs(
        n_samples, n_features, n_classes, cluster_std, center_scale, random_state=42
    )

    X_tensor = torch.Tensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)

    # batch_size=None means full-batch training.
    dataloader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size or X_tensor.shape[0],
        shuffle=True,
    )

    return X_tensor, y_tensor, dataloader
