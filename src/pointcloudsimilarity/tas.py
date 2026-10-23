import numpy as np

from .core_similarity import Similarity, resolve_device
from .tas_core.normalization import normalize_tas, resolve_k
from .tas_core.tas_cpu import mean_neighborhood_similarity_from_points
from .tas_core.tas_torch import compute_tas_pytorch_batched, compute_tas_spectrum_torch


def tas_spectrum(
    pc1,
    pc2,
    ks,
    normalize=True,
    only_intersection=False,
    device=None,
    batch_size=2000,
):
    """
    Topological Alignment Spectrum: TAS at every neighborhood size in ks.

    Equivalent to calling TASSimilarityTorch(k=k)(pc1, pc2) for each k, but
    runs a single nearest-neighbor search for the largest k.

    Parameters:
    pc1 (np.ndarray or torch.Tensor): First point cloud of shape (N, D1).
    pc2 (np.ndarray or torch.Tensor): Second point cloud of shape (N, D2).
    ks (iterable of int or float): Neighborhood sizes, or fractions of N.
    normalize (bool): Apply hypergeometric normalization.
    only_intersection (bool): Use intersection size instead of Jaccard index.
    device (str): 'cuda', 'cpu', or 'mps'. If None, auto-detects.
    batch_size (int): Size of GPU chunks. Decrease if OOM.

    Returns:
    np.ndarray: TAS value for each k, aligned with ks.
    """
    ks = list(ks)
    n_samples = pc1.shape[0]
    raw = compute_tas_spectrum_torch(
        pc1,
        pc2,
        ks,
        batch_size=batch_size,
        device=resolve_device(device),
        only_intersection=only_intersection,
    )
    if not normalize:
        return raw
    return np.array(
        [
            normalize_tas(r, n_samples, resolve_k(k, n_samples), only_intersection)
            for r, k in zip(raw, ks)
        ]
    )


class TASSimilarityCPU(Similarity):
    def __init__(
        self,
        k=0.2,
        self_is_neighbor=False,
        metric='minkowski',
        n_jobs=1,
        normalize=True,
    ):
        """
        Initialize the TAS similarity measure.

        Parameters:
        k (int or float): Number of neighbors or fraction of points to consider as neighbors.
        self_is_neighbor (bool): Whether to include the point itself as its neighbor.
        metric (str): Distance metric to use for nearest neighbors.
        n_jobs (int): Number of parallel jobs to run for nearest neighbors computation.
        """
        self.k = k
        self.self_is_neighbor = self_is_neighbor
        self.metric = metric
        self.n_jobs = n_jobs
        self.normalize = normalize

    def __call__(self, pc1, pc2, **kwargs):
        """
        Compute the TAS similarity between two point clouds.

        Parameters:
        pc1 (np.ndarray): First point cloud of shape (N, D1).
        pc2 (np.ndarray): Second point cloud of shape (N, D2).
        (pc1 and pc2 must have the same number of points, but can have different dimensionality)

        Returns:
        float: TAS similarity between the two point clouds.
        """
        k = resolve_k(self.k, pc1.shape[0])
        tas = mean_neighborhood_similarity_from_points(
            pc1,
            pc2,
            k=k,
            n_jobs=self.n_jobs,
            metric=self.metric,
        )
        if self.normalize:
            tas = normalize_tas(tas, pc1.shape[0], k)

        return tas


class TASSimilarityTorch(Similarity):
    def __init__(
        self,
        k=0.2,
        normalize=True,
        device=None,
        batch_size=2000,
        only_intersection=False,
    ):
        """
        Optimized TAS class.

        Parameters:
        k (int or float): Number of neighbors or fraction.
        normalize (bool): Apply hypergeometric normalization.
        device (str): 'cuda', 'cpu', or 'mps'. If None, auto-detects.
        batch_size (int): Size of GPU chunks. Decrease if OOM.
        """
        self.k = k
        self.normalize = normalize
        self.batch_size = batch_size
        self.only_intersection = only_intersection

        self.device = resolve_device(device)

    def __call__(self, pc1, pc2, **kwargs):
        """
        Compute TAS similarity.
        """
        n_samples = pc1.shape[0]
        k = resolve_k(self.k, n_samples)

        # Calculate raw TAS using the PyTorch engine
        tas_raw = compute_tas_pytorch_batched(
            pc1,
            pc2,
            k=k,
            batch_size=self.batch_size,
            device=self.device,
            only_intersection=self.only_intersection,
        )

        if self.normalize:
            return normalize_tas(
                tas_raw, n_samples, k, only_intersection=self.only_intersection
            )

        return tas_raw

