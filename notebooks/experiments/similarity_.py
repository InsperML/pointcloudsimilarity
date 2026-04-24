import numpy as np
import torch
from tqdm import tqdm

from pointcloudsimilarity.similarities import TASSimilarityTorch


def _sweep_model_similarity(
    X: torch.Tensor,
    Y: torch.Tensor,
) -> tuple[list[int], list[np.ndarray]]:
    ks = list(range(1, X.shape[0] - 1))
    similarities_nngs = []
    for k in tqdm(ks, disable=True):
        metric = TASSimilarityTorch(k=k, batch_size=100, normalize=True)
        sim = metric(X, Y)
        similarities_nngs.append(sim)
    return ks, similarities_nngs


def _compute_similarity(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    X_tensor: torch.Tensor,
) -> list[np.ndarray]:
    with torch.no_grad():
        _, z1 = model1(X_tensor)
        _, z2 = model2(X_tensor)
        _, similarities_nngs = _sweep_model_similarity(z1, z2)
        return similarities_nngs


class SimilarityTracker:
    def __init__(
        self,
        model1: torch.nn.Module,
        model2: torch.nn.Module,
        X_tensor: torch.Tensor,
    ) -> None:
        self.model1 = model1
        self.model2 = model2
        self.X_tensor = X_tensor
        self.similarities = []

    def add_similarity(self) -> None:
        similarity = _compute_similarity(self.model1, self.model2, self.X_tensor)
        self.similarities.append(similarity)
