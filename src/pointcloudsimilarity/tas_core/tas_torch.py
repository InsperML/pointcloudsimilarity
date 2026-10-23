import numpy as np
import torch

from .normalization import resolve_k


def neighborhood_overlaps(idx_x, idx_y, n_samples, k, only_intersection=False):
    """
    Per-point overlap between two batches of neighbor index rows.

    Parameters:
    idx_x, idx_y (torch.Tensor): Neighbor indices in [0, n_samples) of shape
        (B, k).
    n_samples (int): Number of points the indices refer to.
    k (int): Number of neighbors per row.
    only_intersection (bool): Return intersection sizes instead of Jaccard indices.

    Returns:
    torch.Tensor: Shape (B,) float tensor.
    """
    # Mark each row's Y neighbors, then count the X neighbors that are marked.
    # Memory is O(B * N) rather than O(B * k^2) for comparing all pairs.
    in_y = torch.zeros((idx_y.shape[0], n_samples), dtype=torch.bool, device=idx_y.device)
    in_y.scatter_(1, idx_y, True)
    intersections = in_y.gather(1, idx_x).sum(dim=1).float()
    if only_intersection:
        # Raw intersection size, normalized against k later
        return intersections
    return intersections / ((2 * k) - intersections)


def _nearest_neighbors(batch, batch_sq_norms, points, sq_norms, k, start):
    """
    Indices of the k nearest points for rows start..start+B of points, sorted by
    distance, so the first k' columns are the k'-neighborhood for any k' <= k.
    """
    # dist[i, j] = ||b_i||^2 + ||p_j||^2 - 2(b_i . p_j)
    # Broadcast: (B, 1) + (1, N) - (B, N); the matmul uses Tensor Cores.
    dist = batch_sq_norms + sq_norms.t() - 2 * torch.mm(batch, points.t())
    # Exclude each point by its index. Assuming it comes back first fails for
    # duplicate points, and when rounding makes another point look closer.
    rows = torch.arange(dist.shape[0], device=dist.device)
    dist[rows, rows + start] = float('inf')
    _, idx = torch.topk(dist, k=k, dim=1, largest=False, sorted=True)
    return idx


def compute_jaccards_torch(
    X,
    Y,
    k,
    batch_size=5000,
    device='cuda',
    only_intersection=False,
):
    """
    Per-point neighborhood overlap between X and Y, computed with PyTorch:
    1. GPU Acceleration
    2. Batched execution (prevents OOM on large N)
    3. Vectorized Intersection (no Python sets)

    Returns:
    np.ndarray: Shape (N,) Jaccard indices (or intersection sizes if
    only_intersection).
    """
    n_samples = X.shape[0]

    # Handle fractional k (e.g., 0.2 -> 20% of N) and clamp to [1, N-1]
    k = resolve_k(k, n_samples)

    # Assuming inputs are numpy or torch, convert to float32 tensor
    X = torch.as_tensor(X).float().to(device)
    Y = torch.as_tensor(Y).float().to(device)

    # Precompute squared norms for ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    X_sq_norms = (X**2).sum(dim=1, keepdim=True)
    Y_sq_norms = (Y**2).sum(dim=1, keepdim=True)

    # Iterate in chunks so VRAM usage stays constant regardless of N.
    all_jaccards = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        idx_x = _nearest_neighbors(X[start:end], X_sq_norms[start:end], X, X_sq_norms, k, start)
        idx_y = _nearest_neighbors(Y[start:end], Y_sq_norms[start:end], Y, Y_sq_norms, k, start)
        jaccards = neighborhood_overlaps(idx_x, idx_y, n_samples, k, only_intersection)
        all_jaccards.append(jaccards.cpu())

    return torch.cat(all_jaccards, dim=0).numpy()


def compute_tas_pytorch_batched(
    X,
    Y,
    k,
    batch_size=5000,
    device='cuda',
    only_intersection=False,
):
    """
    Raw TAS (mean neighborhood overlap) between X and Y. See compute_jaccards_torch.
    """
    jaccards = compute_jaccards_torch(
        X,
        Y,
        k,
        batch_size=batch_size,
        device=device,
        only_intersection=only_intersection,
    )
    return float(jaccards.mean(dtype='float64'))


def compute_tas_spectrum_torch(
    X,
    Y,
    ks,
    batch_size=2000,
    device='cuda',
    only_intersection=False,
):
    """
    Raw TAS for every k in ks from a single neighbor search.

    The k-neighborhoods for every k are prefixes of the max(ks)-neighborhood,
    so one top-k search serves the whole spectrum. A neighbor is in both
    k-neighborhoods iff max(its rank in X, its rank in Y) < k, so a cumulative
    histogram of that max-rank gives the intersection size for every k at once.

    Returns:
    np.ndarray: Shape (len(ks),) raw TAS values (mean Jaccard, or mean
    intersection size if only_intersection), aligned with ks.
    """
    n_samples = X.shape[0]
    ks = np.array([resolve_k(k, n_samples) for k in ks])
    if ks.size == 0:
        raise ValueError('ks must contain at least one neighborhood size.')
    max_k = int(ks.max())

    X = torch.as_tensor(X).float().to(device)
    Y = torch.as_tensor(Y).float().to(device)
    X_sq_norms = (X**2).sum(dim=1, keepdim=True)
    Y_sq_norms = (Y**2).sum(dim=1, keepdim=True)

    k_range = torch.arange(1, max_k + 1, device=device)  # (K,)
    total = torch.zeros(max_k, dtype=torch.float64, device=device)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        idx_x = _nearest_neighbors(
            X[start:end], X_sq_norms[start:end], X, X_sq_norms, max_k, start
        )
        idx_y = _nearest_neighbors(
            Y[start:end], Y_sq_norms[start:end], Y, Y_sq_norms, max_k, start
        )
        B = idx_x.shape[0]

        # rank_y[b, j]: position of point j in row b's Y neighbors, max_k if absent
        rank_y = torch.full((B, n_samples), max_k, dtype=torch.long, device=device)
        rank_y.scatter_(1, idx_y, (k_range - 1).expand(B, -1))
        # The X neighbor at position p is in both k-neighborhoods iff max_rank < k
        max_rank = torch.maximum(k_range - 1, rank_y.gather(1, idx_x))  # (B, K)

        hist = torch.zeros((B, max_k + 1), device=device)
        hist.scatter_add_(1, max_rank, torch.ones_like(max_rank, dtype=hist.dtype))
        intersections = hist[:, :max_k].cumsum(dim=1)  # [:, k-1] -> size k

        if only_intersection:
            overlaps = intersections
        else:
            overlaps = intersections / ((2 * k_range) - intersections)
        total += overlaps.sum(dim=0, dtype=torch.float64)

    return (total / n_samples).cpu().numpy()[ks - 1]
