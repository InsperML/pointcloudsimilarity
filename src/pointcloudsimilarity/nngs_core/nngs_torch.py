import torch


# --- Helper: The Mathematical Bound ---
def hypergeometric_bound(n, k, only_intersection=False):
    """
    Expected Jaccard similarity between two random subsets of size k 
    drawn from a pool of size n.
    """
    if n <= 1 or k >= n:
        return 0.0  # Edge cases
    # Derived from Hypergeometric distribution mean for intersection size
    # E[|A n B|] = k^2 / n
    # E[J] approx E[Int] / (2k - E[Int])
    if only_intersection:
        return (k**2)/(n**2) 
    else:
        return k / (2 * (n - 1) - k)


# --- Optimized Engine ---
def compute_nngs_pytorch_batched(
    X,
    Y,
    k,
    batch_size=5000,
    device='cuda',
    only_intersection=False,
):
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

    # --- Precompute Squared Norms for the Expansion Trick ---
    # Shape: (N, 1)
    # This is necessary for ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    X_sq_norms = (X**2).sum(dim=1, keepdim=True)
    Y_sq_norms = (Y**2).sum(dim=1, keepdim=True)

    total_intersection = 0.0

    # 2. Batched Processing
    # We iterate through the dataset in chunks to calculate neighbors and similarity
    # This keeps VRAM usage constant regardless of N.

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)

        # --- A. Neighbors for X (Batch) ---
        X_batch = X[start:end]
        X_batch_sq = X_sq_norms[start:end]  # (B, 1)

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
        if only_intersection:
            unions = X_batch.shape[0]
        else:
            unions = ((2 * k) - batch_intersections).clamp(min=1e-8)
    
        jaccards = batch_intersections / unions

        total_intersection += jaccards.sum().item()

        # Clean memory
        del dot_x, dist_x, idx_x, dot_y, dist_y, idx_y, matches

    return total_intersection / n_samples
