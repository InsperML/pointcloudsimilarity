

import torch
from .core_similarity import Similarity
from .nngs_core.nngs_cpu import mean_neighborhood_similarity_from_points, hypergeometric_bound, normalize_nngs
from .nngs_core.nngs_torch import compute_nngs_pytorch_batched
from .nngs_core.nngs_faiss import compute_nngs_faiss



class NNGSSimilarityCPU(Similarity):
    def __init__(self, k=0.2, self_is_neighbor=False, metric='minkowski', n_jobs=1, normalize=True):
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
        nngs = mean_neighborhood_similarity_from_points(
            pc1,
            pc2,
            k=self.k,
            n_jobs=self.n_jobs,
            metric=self.metric,
        )
        if self.normalize:
            nngs = normalize_nngs(nngs, pc1.shape[0], self.k)

        return nngs

class NNGSSimilarityTorch(Similarity):
    def __init__(self, k=0.2, normalize=True, device=None, batch_size=2000):
        """
        Optimized NNGS class.
        
        Parameters:
        k (int or float): Number of neighbors or fraction.
        normalize (bool): Apply hypergeometric normalization.
        device (str): 'cuda', 'cpu', or 'mps'. If None, auto-detects.
        batch_size (int): Size of GPU chunks. Decrease if OOM.
        """
        self.k = k
        self.normalize = normalize
        self.batch_size = batch_size
        
        if device is None:
            if torch.cuda.is_available(): self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): self.device = 'mps'
            else: self.device = 'cpu'
        else:
            self.device = device

    def __call__(self, pc1, pc2, **kwargs):
        """
        Compute NNGS similarity.
        """
        # Calculate raw NNGS using the PyTorch engine
        nngs_raw = compute_nngs_pytorch_batched(
            pc1, 
            pc2, 
            k=self.k, 
            batch_size=self.batch_size, 
            device=self.device
        )
        
        if self.normalize:
            # Recalculate k if it was a float, for the bound formula
            N = pc1.shape[0]
            k_val = int(self.k * N) if isinstance(self.k, float) else self.k
            
            # Apply Normalization
            min_bound = hypergeometric_bound(N, k_val)
            
            # Prevent div by zero if bound is 1 (weird edge case)
            if min_bound >= 1.0:
                return 0.0
            
            nngs_normalized = (nngs_raw - min_bound) / (1 - min_bound)
            return nngs_normalized

        return nngs_raw


class NNGSSimilarityFaiss(Similarity):
    def __init__(self, k=5, approximate=False, metric='euclidean', batch_size=5000, gpu=True, normalize=True):
        """
        NNGS Similarity using FAISS backend.
        
        Parameters:
        k (int): Number of neighbors.
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

    def __call__(self, pc1, pc2, **kwargs):
        """
        Compute NNGS similarity using FAISS.
        """
        nngs_raw = compute_nngs_faiss(
            pc1,
            pc2,
            k=self.k,
            approximate=self.approximate,
            metric=self.metric,
            batch_size=self.batch_size,
            gpu=self.gpu
        )
        
        if self.normalize:
            N = pc1.shape[0]
            k_val = self.k
            
            min_bound = hypergeometric_bound(N, k_val)
            
            if min_bound >= 1.0:
                return 0.0
            
            nngs_normalized = (nngs_raw - min_bound) / (1 - min_bound)
            return nngs_normalized

        return nngs_raw