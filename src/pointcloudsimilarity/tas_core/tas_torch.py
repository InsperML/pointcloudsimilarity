import torch

from .normalization import resolve_k


def neighborhood_overlaps(idx_x, idx_y, k, only_intersection=False):
    """
    Per-point overlap between two batches of neighbor index rows.

    Parameters:
    idx_x, idx_y (torch.Tensor): Neighbor indices of shape (B, k).
    k (int): Number of neighbors per row.
    only_intersection (bool): Return intersection sizes instead of Jaccard indices.

    Returns:
    torch.Tensor: Shape (B,) float tensor.
    """
    # Broadcasting Trick: (B, k, 1) == (B, 1, k) -> (B, k, k)
    matches = idx_x.unsqueeze(2) == idx_y.unsqueeze(1)
    intersections = matches.sum(dim=(1, 2)).float()
    if only_intersection:
        # Raw intersection size, normalized against k later
        return intersections
    return intersections / ((2 * k) - intersections)


def _nearest_neighbors(batch, batch_sq_norms, points, sq_norms, k):
    # dist[i, j] = ||b_i||^2 + ||p_j||^2 - 2(b_i . p_j)
    # Broadcast: (B, 1) + (1, N) - (B, N); the matmul uses Tensor Cores.
    dist = batch_sq_norms + sq_norms.t() - 2 * torch.mm(batch, points.t())
    # Search for k + 1 because the point itself is index 0
    _, idx = torch.topk(dist, k=k + 1, dim=1, largest=False)
    return idx[:, 1:]


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
        idx_x = _nearest_neighbors(X[start:end], X_sq_norms[start:end], X, X_sq_norms, k)
        idx_y = _nearest_neighbors(Y[start:end], Y_sq_norms[start:end], Y, Y_sq_norms, k)
        jaccards = neighborhood_overlaps(idx_x, idx_y, k, only_intersection)
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
