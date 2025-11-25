
import numpy as np
import torch
from .core_similarity import Similarity

# --- Helper: The Mathematical Bound ---
def hypergeometric_bound(n, k):
    """
    Expected Jaccard similarity between two random subsets of size k 
    drawn from a pool of size n.
    """
    if n <= 1 or k >= n: 
        return 0.0 # Edge cases
    # Derived from Hypergeometric distribution mean for intersection size
    # E[|A n B|] = k^2 / n
    # E[J] approx E[Int] / (2k - E[Int])
    return k / (2 * (n - 1) - k)

# --- Optimized Engine ---
def compute_nngs_pytorch_batched(X, Y, k, batch_size=5000, device='cuda'):
    """
    Computes NNGS using PyTorch with:
    1. GPU Acceleration
    2. Batched execution (prevents OOM on large N)
    3. Vectorized Intersection (no Python sets)
    """
    # 1. Setup
    n_samples = X.shape[0]
    
    # Handle fractional k (e.g., 0.2 -> 20% of N)
    if isinstance(k, float):
        k = int(k * n_samples)
    
    # Validation
    if k >= n_samples:
        k = n_samples - 1
    
    if k < 1:
        k = 1
    
    # Search for k+1 because the point itself is index 0
    search_k = k + 1

    # Move to GPU/Device
    # Assuming inputs are numpy or torch, convert to float32 tensor
    if not isinstance(X, torch.Tensor): X = torch.tensor(X)
    if not isinstance(Y, torch.Tensor): Y = torch.tensor(Y)
    
    X = X.float().to(device)
    Y = Y.float().to(device)

    # Normalize for Cosine Similarity equivalence
    # (Euclidean distance on normalized vectors preserves rank of Cosine)
    X_sq_norms = (X ** 2).sum(dim=1, keepdim=True) 
    Y_sq_norms = (Y ** 2).sum(dim=1, keepdim=True)

    total_intersection = 0.0
    
    # 2. Batched Processing
    # We iterate through the dataset in chunks to calculate neighbors and similarity
    # This keeps VRAM usage constant regardless of N.
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        
        # --- A. Neighbors for X (Batch) ---
        X_batch = X[start:end]
        X_batch_sq = X_sq_norms[start:end] # (B, 1)
        
        # dist_sq = ||x||^2 + ||other||^2 - 2<x, other>
        # We broadcast: (B, 1) + (1, N) - (B, N)
        # Note: We use X as both query and key for the graph on X
        
        # 1. Dot Product (Heavy lifting, uses Tensor Cores)
        dot_x = torch.mm(X_batch, X.t()) 
        
        # 2. Expand Euclidean Distance
        # dist_x[i, j] = ||x_i||^2 + ||x_j||^2 - 2(x_i . x_j)
        dist_x = X_batch_sq + X_sq_norms.t() - 2 * dot_x
        
        # 3. TopK (Smallest distance is best)
        # largest=False because we want MINIMUM distance
        _, idx_x = torch.topk(dist_x, k=search_k, dim=1, largest=False)
        idx_x = idx_x[:, 1:] 

        # --- B. Neighbors for Y (Batch) ---
        Y_batch = Y[start:end]
        Y_batch_sq = Y_sq_norms[start:end]
        
        dot_y = torch.mm(Y_batch, Y.t())
        dist_y = Y_batch_sq + Y_sq_norms.t() - 2 * dot_y
        
        _, idx_y = torch.topk(dist_y, k=search_k, dim=1, largest=False)
        idx_y = idx_y[:, 1:]

        # --- C. Vectorized Jaccard (Same as before) ---
        matches = (idx_x.unsqueeze(2) == idx_y.unsqueeze(1))
        batch_intersections = matches.sum(dim=(1, 2)).float()
        unions = (2 * k) - batch_intersections
        jaccards = batch_intersections / unions.clamp(min=1e-8)
        
        total_intersection += jaccards.sum().item()

        # Clean memory
        del dot_x, dist_x, idx_x, dot_y, dist_y, idx_y, matches
    
    return total_intersection / n_samples



class NNGSSimilarity(Similarity):
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

# # NNGS, as described in "Measuring similarity between embedding spaces using induced neighborhood graphs" (https://arxiv.org/abs/2411.08687)

# import numpy as np
# from sklearn.neighbors import kneighbors_graph
# from .core_similarity import Similarity

# class NNGSSimilarity(Similarity):
#     def __init__(self, k=0.2, self_is_neighbor=False, metric='minkowski', n_jobs=1, normalize=True):
#         """
#         Initialize the NNGS similarity measure.

#         Parameters:
#         k (int or float): Number of neighbors or fraction of points to consider as neighbors.
#         self_is_neighbor (bool): Whether to include the point itself as its neighbor.
#         metric (str): Distance metric to use for nearest neighbors.
#         n_jobs (int): Number of parallel jobs to run for nearest neighbors computation.
#         """
#         self.k = k
#         self.self_is_neighbor = self_is_neighbor
#         self.metric = metric
#         self.n_jobs = n_jobs
#         self.normalize = normalize

#     def __call__(self, pc1, pc2, **kwargs):
#         """
#         Compute the NNGS similarity between two point clouds.

#         Parameters:
#         pc1 (np.ndarray): First point cloud of shape (N, D1).
#         pc2 (np.ndarray): Second point cloud of shape (N, D2).
#         (pc1 and pc2 must have the same number of points, but can have different dimensionality)

#         Returns:
#         float: NNGS similarity between the two point clouds.
#         """
#         nngs = mean_neighborhood_similarity_from_points(
#             pc1,
#             pc2,
#             k=self.k,
#             n_jobs=self.n_jobs,
#             metric=self.metric,
#         )
#         if self.normalize:
#             nngs = normalize_nngs(nngs, pc1.shape[0], self.k)

#         return nngs

# def hypergeometric_bound(n, k):
#         min_bound = k /(2*(n-1)-k)
#         return min_bound

# def normalize_nngs(similarity, n_points, k):
#     """
#     Normalize NNGS similarity using hypergeometric bound.

#     Parameters:
#     similarity (float): Raw NNGS similarity.
#     n_points (int): Number of points in the point clouds.
#     k (int): Number of neighbors used in NNGS.

#     Returns:
#     float: Normalized NNGS similarity.
#     """
#     min_bound = hypergeometric_bound(n_points, k)
#     normalized_similarity = (similarity - min_bound) / (1 - min_bound)
#     return normalized_similarity

# def nearest_neighbors(
#     x,
#     k,
#     self_is_neighbor=False,
#     metric='minkowski',
#     n_jobs=1,
# ):
#     if isinstance(k, float):
#         k = int(k * x.shape[0])
#     G = kneighbors_graph(
#         x,
#         k,
#         mode='connectivity',
#         metric=metric,
#         include_self=self_is_neighbor,
#         n_jobs=n_jobs,
#     )
#     A = []
#     for i in range(G.shape[0]):
#         A.append(G.getrow(i).nonzero()[1])

#     A = np.vstack(A)
#     return A


# def compute_jaccard_similarity(sx, sy):
#     """
#     Compute Jaccard similarity between two sets of indices.
#     """
#     return len(sx.intersection(sy)) / len(sx.union(sy))


# def mean_neighborhood_similarity_from_neighborhood(nx, ny):
#     num_points = nx.shape[0]
#     inter = 0
#     for i in range(num_points):
#         sx = set(nx[i])
#         sy = set(ny[i])
#         inter += compute_jaccard_similarity(sx, sy)
#     inter /= num_points
#     return inter


# def mean_neighborhood_similarity_from_points(
#     X,
#     Y,
#     k,
#     n_jobs=1,
#     metric='minkowski',
# ):
#     """
#     This is $NNGS(X, Y, k)$
#     """
#     nx = nearest_neighbors(X, k=k, n_jobs=n_jobs, metric=metric)
#     ny = nearest_neighbors(Y, k=k, n_jobs=n_jobs, metric=metric)
#     return mean_neighborhood_similarity_from_neighborhood(nx, ny)




