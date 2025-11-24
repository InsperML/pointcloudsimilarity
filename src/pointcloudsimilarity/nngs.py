# NNGS, as described in "Measuring similarity between embedding spaces using induced neighborhood graphs" (https://arxiv.org/abs/2411.08687)

import numpy as np
from sklearn.neighbors import kneighbors_graph
from .core_similarity import Similarity

class NNGSSimilarity(Similarity):
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

def hypergeometric_bound(n, k):
        min_bound = k /(2*(n-1)-k)
        return min_bound

def normalize_nngs(similarity, n_points, k):
    """
    Normalize NNGS similarity using hypergeometric bound.

    Parameters:
    similarity (float): Raw NNGS similarity.
    n_points (int): Number of points in the point clouds.
    k (int): Number of neighbors used in NNGS.

    Returns:
    float: Normalized NNGS similarity.
    """
    min_bound = hypergeometric_bound(n_points, k)
    normalized_similarity = (similarity - min_bound) / (1 - min_bound)
    return normalized_similarity

def nearest_neighbors(
    x,
    k,
    self_is_neighbor=False,
    metric='minkowski',
    n_jobs=1,
):
    if isinstance(k, float):
        k = int(k * x.shape[0])
    G = kneighbors_graph(
        x,
        k,
        mode='connectivity',
        metric=metric,
        include_self=self_is_neighbor,
        n_jobs=n_jobs,
    )
    A = []
    for i in range(G.shape[0]):
        A.append(G.getrow(i).nonzero()[1])

    A = np.vstack(A)
    return A


def compute_jaccard_similarity(sx, sy):
    """
    Compute Jaccard similarity between two sets of indices.
    """
    return len(sx.intersection(sy)) / len(sx.union(sy))


def mean_neighborhood_similarity_from_neighborhood(nx, ny):
    num_points = nx.shape[0]
    inter = 0
    for i in range(num_points):
        sx = set(nx[i])
        sy = set(ny[i])
        inter += compute_jaccard_similarity(sx, sy)
    inter /= num_points
    return inter


def mean_neighborhood_similarity_from_points(
    X,
    Y,
    k,
    n_jobs=1,
    metric='minkowski',
):
    """
    This is $NNGS(X, Y, k)$
    """
    nx = nearest_neighbors(X, k=k, n_jobs=n_jobs, metric=metric)
    ny = nearest_neighbors(Y, k=k, n_jobs=n_jobs, metric=metric)
    return mean_neighborhood_similarity_from_neighborhood(nx, ny)


