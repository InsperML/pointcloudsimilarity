"""
Shared neighborhood-size and normalization logic for all TAS backends.

Every backend must resolve k and normalize the same way, otherwise the
CPU and PyTorch implementations report different values for the
same point clouds.
"""


def resolve_k(k, n_samples):
    """
    Number of neighbors per point.

    Parameters:
    k (int or float): Number of neighbors, or (float) fraction of n_samples.
    n_samples (int): Number of points in each point cloud.

    Returns:
    int: Number of neighbors, clamped to [1, n_samples - 1] since a point
    is never its own neighbor.
    """
    if isinstance(k, float):
        k = int(k * n_samples)
    return max(1, min(int(k), n_samples - 1))


def hypergeometric_bound(n, k, only_intersection=False):
    """
    Expected neighborhood overlap between two random k-subsets drawn from a
    pool of n candidates.

    Parameters:
    n (int): Pool size (number of candidate neighbors, i.e. n_samples - 1).
    k (int): Neighborhood size.
    only_intersection (bool): If True, return the expected intersection size
        instead of the expected Jaccard index.
    """
    # Derived from Hypergeometric distribution mean for intersection size
    # E[|A n B|] = k^2 / n
    # E[J] approx E[Int] / (2k - E[Int])
    if only_intersection:
        return k**2 / n
    return k / (2 * n - k)


def normalize_tas(similarity, n_points, k, only_intersection=False):
    """
    Normalize raw TAS so that 0 is the random baseline and 1 is a perfect match.

    Parameters:
    similarity (float): Raw TAS (mean Jaccard, or mean intersection size if
        only_intersection).
    n_points (int): Number of points in the point clouds.
    k (int): Number of neighbors actually used (see resolve_k).
    only_intersection (bool): Whether similarity is a mean intersection size.

    Returns:
    float: Normalized TAS similarity.
    """
    n_candidates = n_points - 1  # a point is never its own neighbor
    if k >= n_candidates:
        # Every neighborhood holds every other point, so all match trivially.
        return 1.0
    min_bound = hypergeometric_bound(n_candidates, k, only_intersection)
    max_bound = k if only_intersection else 1.0
    return (similarity - min_bound) / (max_bound - min_bound)
