# NNGS, as described in "Measuring similarity between embedding spaces using induced neighborhood graphs" (https://arxiv.org/abs/2411.08687)

import numpy as np
import scipy.stats as stats
from sklearn.neighbors import kneighbors_graph


def hypergeometric_bound(n, k, only_intersection=False):
    if n < 1 or k > n:
        return 0.0  # Edge cases
    # Derived from Hypergeometric distribution mean for intersection size
    # E[|A n B|] = k^2 / n
    # E[J] approx E[Int] / (2k - E[Int])
    if only_intersection:
        return k**2 / n
    else:
        return k / (2 * n - k)


def normalize_tas(similarity, n_points, k):
    """
    Normalize TAS similarity using hypergeometric bound.

    Parameters:
    similarity (float): Raw TAS similarity.
    n_points (int): Number of points in the point clouds.
    k (int): Number of neighbors used in NNGS.

    Returns:
    float: Normalized TAS similarity.
    """
    min_bound = hypergeometric_bound(n_points - 1, k)
    normalized_similarity = (similarity - min_bound) / (1 - min_bound)
    # normalized_similarity = stats.hypergeom.pmf(similarity * k, n_points, k, n_points-1)
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
