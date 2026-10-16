import torch

from .core_similarity import Similarity, resolve_device


class CKASimilarity(Similarity):
    def __init__(
        self,
        kernel='linear',
        sigma=None,
        device=None,
        scale_by_dim=False,
        scale_by_alpha: None | float = None,
        initial_quantile=0.5,
    ):
        """
        PyTorch-Optimized CKA Similarity.

        Parameters:
        kernel (str): 'linear' or 'rbf'.
        sigma (float): Bandwidth for RBF, shared by both point clouds. If None,
            each cloud gets its own bandwidth: the initial_quantile of its pairwise
            distances, times scale_by_alpha.
        device (str): 'cuda', 'cpu', or 'mps'. Auto-detected if None.
        """
        assert kernel in ['linear', 'rbf'], "Kernel must be 'linear' or 'rbf'"
        self.kernel = kernel
        self.sigma = sigma
        self.scale_by_dim = scale_by_dim
        self.scale_by_alpha = scale_by_alpha
        self.initial_quantile = initial_quantile

        self.device = resolve_device(device)

    def __call__(self, pc1, pc2, **kwargs):
        label = kwargs.get('label', None)

        # Move to GPU/Tensor
        if not isinstance(pc1, torch.Tensor):
            pc1 = torch.tensor(pc1)
        if not isinstance(pc2, torch.Tensor):
            pc2 = torch.tensor(pc2)

        X = pc1.float().to(self.device)
        Y = pc2.float().to(self.device)

        # Center the Features (Important for Linear CKA optimization)
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

        if self.kernel == 'linear':
            return linear_cka_optimized(X, Y)

        elif self.kernel == 'rbf':
            # RBF CKA requires the full N*N kernel matrix
            # Note: This WILL OOM on large N (e.g., N > 30k on 24GB VRAM)
            # This limitation is part of your argument against CKA.

            # Estimate Sigma if needed (using a subset to save time)
            sigma_x = sigma_y = self.sigma
            if self.sigma is None:
                # Each cloud's bandwidth comes from its own pairwise distances, so
                # the result is symmetric and invariant to scaling either cloud.
                # Kornblith et al. use alpha in {0.8, 0.4, 0.2} times the median.
                subset_size = min(2000, X.shape[0])
                idx = torch.randperm(X.shape[0])[:subset_size]
                sigma_x = estimate_sigma_torch(X[idx], q=self.initial_quantile)
                sigma_y = estimate_sigma_torch(Y[idx], q=self.initial_quantile)
                if label is not None:
                    print(f'Label: {label}, Estimated sigmas: {sigma_x}, {sigma_y}')
                if self.scale_by_alpha is not None:
                    sigma_x *= self.scale_by_alpha
                    sigma_y *= self.scale_by_alpha

            return rbf_cka_gpu(X, Y, sigma_x, self.scale_by_dim, sigma_y=sigma_y)


def linear_cka_optimized(X, Y, threshold=1e-16):
    """
    Computes Linear CKA using the Inner Product trick.
    Complexity: O(N * D^2) instead of O(N^2).
    Mathematically equivalent to standard CKA but avoids N*N matrix.
    """
    # HSIC(X, Y) = ||Y^T X||_F^2 / (N-1)^2  (assuming centered X, Y)
    # CKA = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))

    # 1. Numerator: ||Y^T X||_F^2
    # Matrix multiplication (D2, N) @ (N, D1) -> (D2, D1)
    inter_corr = torch.mm(Y.t(), X)
    numerator = torch.norm(inter_corr) ** 2

    # 2. Denominator terms
    # ||X^T X||_F^2
    self_corr_X = torch.mm(X.t(), X)
    term_X = torch.norm(self_corr_X) ** 2

    # ||Y^T Y||_F^2
    self_corr_Y = torch.mm(Y.t(), Y)
    term_Y = torch.norm(self_corr_Y) ** 2

    # 3. Combine
    denom = torch.sqrt(term_X * term_Y)

    if denom < threshold:
        return 0.0

    return (numerator / denom).item()


def rbf_cka_gpu(X, Y, sigma, scale_by_dim=False, threshold=1e-16, sigma_y=None):
    """
    Computes RBF CKA on GPU.
    Complexity: O(N^2).
    Requires full Gram Matrix computation.

    sigma is the bandwidth for X; sigma_y the one for Y (defaults to sigma).
    """
    sigma_x = sigma
    sigma_y = sigma if sigma_y is None else sigma_y
    if scale_by_dim:
        sigma_x = sigma_x * X.shape[1]
        sigma_y = sigma_y * Y.shape[1]

    # 1. Compute Gram Matrices (Kernel Matrices), minus 1.
    # Double centering removes constants, so K - 1 centers to the same matrix as
    # K, and expm1 keeps the precision that exp loses when sigma is large and
    # every entry of K is close to 1.
    # torch.cdist computes euclidean distance
    K = torch.expm1(
        -torch.cdist(X, X, compute_mode='donot_use_mm_for_euclid_dist').pow(2)
        / (2 * sigma_x**2)
    )
    L = torch.expm1(
        -torch.cdist(Y, Y, compute_mode='donot_use_mm_for_euclid_dist').pow(2)
        / (2 * sigma_y**2)
    )

    # 2. Center the Kernels (Double Centering: HKH)
    # CKA does not change when a kernel is rescaled; scaling each to a peak of 1
    # keeps the sums of squares below from underflowing for large sigma.
    K_c = _scale_to_unit_peak(center_gram_matrix(K))
    L_c = _scale_to_unit_peak(center_gram_matrix(L))

    # 3. Compute CKA
    # Trace(K_c @ L_c) is technically sum(elementwise_mult(K_c, L_c.T))
    # Since Kernels are symmetric, K_c @ L_c == sum(K_c * L_c)
    numerator = torch.sum(K_c * L_c)

    denom_x = torch.sqrt(torch.sum(K_c * K_c))
    denom_y = torch.sqrt(torch.sum(L_c * L_c))

    denom = denom_x * denom_y

    if denom < threshold:
        return linear_cka_optimized(X, Y)

    return (numerator / denom).item()


def center_gram_matrix(K):
    """
    Applies double centering to a kernel matrix K.
    K_c = K - row_mean - col_mean + grand_mean
    """
    row_means = K.mean(dim=1, keepdim=True)  # (N, 1)
    col_means = K.mean(dim=0, keepdim=True)  # (1, N)
    return K - row_means - col_means + K.mean()


def _scale_to_unit_peak(M):
    peak = M.abs().max()
    return M / peak if peak > 0 else M


def estimate_sigma_torch(data, q=0.5):
    """
    Estimates sigma using the median distance heuristic on GPU.
    """
    # Calculate pairwise distances (Squared Euclidean)
    # Note: data should be a subset if N is huge
    dists = torch.pdist(data).pow(2)

    # Median
    median_dist = dists.quantile(q)

    # Sigma = sqrt(median / 2) * alpha?
    # Standard heuristic is often just sqrt(median) or similar.
    # Your original code used: sigma = median_dist * alpha (on SQRT dists usually)
    # Let's match your logic:
    # Your old code: pairwise_distances(metric='sqeuclidean') -> median -> * alpha
    # rbf_kernel: exp(-dist_sq / (2*sigma^2))

    sigma = torch.sqrt(median_dist)  # This is a standard heuristic
    return sigma.item()
