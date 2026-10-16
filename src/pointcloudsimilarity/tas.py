import torch

from .core_similarity import Similarity
from .tas_core.normalization import normalize_tas, resolve_k
from .tas_core.tas_cpu import mean_neighborhood_similarity_from_points
from .tas_core.tas_faiss import TASFAISS
from .tas_core.tas_torch import compute_tas_pytorch_batched


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
        Initialize the NNGS similarity measure.

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
        Compute the NNGS similarity between two point clouds.

        Parameters:
        pc1 (np.ndarray): First point cloud of shape (N, D1).
        pc2 (np.ndarray): Second point cloud of shape (N, D2).
        (pc1 and pc2 must have the same number of points, but can have different dimensionality)

        Returns:
        float: NNGS similarity between the two point clouds.
        """
        k = resolve_k(self.k, pc1.shape[0])
        nngs = mean_neighborhood_similarity_from_points(
            pc1,
            pc2,
            k=k,
            n_jobs=self.n_jobs,
            metric=self.metric,
        )
        if self.normalize:
            nngs = normalize_tas(nngs, pc1.shape[0], k)

        return nngs


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

        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

    def __call__(self, pc1, pc2, **kwargs):
        """
        Compute TAS similarity.
        """
        n_samples = pc1.shape[0]
        k = resolve_k(self.k, n_samples)

        # Calculate raw NNGS using the PyTorch engine
        nngs_raw = compute_tas_pytorch_batched(
            pc1,
            pc2,
            k=k,
            batch_size=self.batch_size,
            device=self.device,
            only_intersection=self.only_intersection,
        )

        if self.normalize:
            return normalize_tas(
                nngs_raw, n_samples, k, only_intersection=self.only_intersection
            )

        return nngs_raw


class TASSimilarityFaiss(Similarity):
    def __init__(
        self,
        k=5,
        approximate=False,
        metric='euclidean',
        batch_size=5000,
        gpu=True,
        normalize=True,
    ):
        """
        TAS Similarity using FAISS backend.

        Parameters:
        k (int or float): Number of neighbors or fraction.
        approximate (bool): Use approximate search (IVF) or exact (Flat).
        metric (str): 'euclidean' or 'cosine'.
        batch_size (int): Chunk size for Jaccard calculation.
        gpu (bool): Use GPU if available.
        normalize (bool): Apply hypergeometric normalization.
        """
        self.k = k
        self.approximate = approximate
        self.metric = metric
        self.batch_size = batch_size
        self.gpu = gpu
        self.normalize = normalize
        self.faiss_calculator = TASFAISS(
            approximate=self.approximate,
            metric=self.metric,
            gpu=self.gpu,
        )

    def __call__(self, pc1, pc2, **kwargs):
        """
        Compute TAS similarity using FAISS.
        """
        n_samples = pc1.shape[0]
        k = resolve_k(self.k, n_samples)

        # Rebuild the indexes on every call: the same object may be reused
        # on different point clouds.
        self.faiss_calculator.fit(pc1, pc2)
        tas_raw = self.faiss_calculator(
            k=k,
            batch_size=self.batch_size,
        )

        if self.normalize:
            return normalize_tas(tas_raw, n_samples, k)

        return tas_raw
