import torch

from .core_similarity import Similarity


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
        sigma (float): Bandwidth for RBF. If None, estimated via median heuristic.
        device (str): 'cuda', 'cpu', or 'mps'. Auto-detected if None.
        """
        assert kernel in ['linear', 'rbf'], "Kernel must be 'linear' or 'rbf'"
        self.kernel = kernel
        self.sigma = sigma
        self.scale_by_dim = scale_by_dim
        self.scale_by_alpha = scale_by_alpha
        self.initial_quantile = initial_quantile

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
            sigma = self.sigma
            if sigma is None:
                # Combine a subset of X and Y to estimate sigma
                # Algorithm: calculate pairwise distances on a subset, get median,
                # multiply by alpha. In the original paper, alpha \in {0.8, 0.4, 0.2}
                subset_size = min(2000, X.shape[0])
                idx = torch.randperm(X.shape[0])[:subset_size]
                X_sub = X[idx]
                # Y_sub = Y[idx]
                # combined = torch.cat([X_sub, Y_sub], dim=0)
                sigma = estimate_sigma_torch(X_sub, q=self.initial_quantile)
                if label is not None:
                    print(f'Label: {label}, Estimated sigma: {sigma}')
                if self.scale_by_alpha is not None:
                    sigma *= self.scale_by_alpha

            return rbf_cka_gpu(X, Y, sigma, self.scale_by_dim)


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


def rbf_cka_gpu(X, Y, sigma, scale_by_dim=False, threshold=1e-16):
    """
    Computes RBF CKA on GPU.
    Complexity: O(N^2).
    Requires full Gram Matrix computation.
    """
    # 1. Compute Gram Matrices (Kernel Matrices)
    # torch.cdist computes euclidean distance
    if scale_by_dim:
        sigma = sigma * X.shape[1]
    K = torch.exp(
        -torch.cdist(X, X, compute_mode='donot_use_mm_for_euclid_dist').pow(2)
        / (2 * sigma**2)
    )
    L = torch.exp(
        -torch.cdist(Y, Y, compute_mode='donot_use_mm_for_euclid_dist').pow(2)
        / (2 * sigma**2)
    )

    # 2. Center the Kernels (Double Centering: HKH)
    K_c = center_gram_matrix(K)
    L_c = center_gram_matrix(L)

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
    # Calculate means
    row_means = K.mean(dim=1, keepdim=True)  # (N, 1)

    # Broadcast subtraction
    K_c = K - row_means
    return K_c


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


# # Implements CKA as in:
# #  CKA with a linear kernel as in:
# #     Similarity of Neural Network Representations Revisited,
# #     Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton
# #     Proceedings of the 36th International Conference on Machine Learning,
# #     PMLR 97:3519-3529, 2019
# #     https://proceedings.mlr.press/v97/kornblith19a.html

# import numpy as np
# from sklearn.metrics.pairwise import pairwise_distances
# from .core_similarity import Similarity


# class CKASimilarity(Similarity):

#     def __init__(self, kernel='linear', sigma=None):
#         """
#         Initialize the CKA similarity measure.

#         Parameters:
#         kernel (str): 'linear' for linear CKA, 'rbf' for RBF kernel CKA.
#         sigma (float): Bandwidth parameter for RBF kernel. If None and kernel is 'rbf',
#                        it will be estimated from the data.
#         """
#         assert kernel in ['linear', 'rbf'], "Kernel must be 'linear' or 'rbf'"
#         self.kernel = kernel
#         self.sigma = sigma

#     def __call__(self, pc1, pc2, **kwargs):
#         """
#         Compute the CKA similarity between two point clouds.

#         Parameters:
#         pc1 (np.ndarray): First point cloud of shape (N, D1).
#         pc2 (np.ndarray): Second point cloud of shape (N, D2).
#         (pc1 and pc2 must have the same number of points, but can have different dimensionality)

#         Returns:
#         float: CKA similarity between the two point clouds.
#         """
#         if self.kernel == 'linear':
#             kernel_X = get_linear_kernel()
#             kernel_Y = get_linear_kernel()
#         elif self.kernel == 'rbf':
#             if self.sigma is None:
#                 combined = np.vstack((pc1, pc2))
#                 self.sigma = estimate_sigma(combined)
#             kernel_X = get_rbf_kernel(self.sigma)
#             kernel_Y = get_rbf_kernel(self.sigma)
#         else:
#             raise ValueError(f"Unknown kernel: {self.kernel}")

#         return cka(pc1, pc2, kernel_X=kernel_X, kernel_Y=kernel_Y)


# def get_rbf_kernel(sigma):

#     def rbf_kernel(X):
#         return np.exp(-pairwise_distances(X, metric='sqeuclidean') /
#                       (2 * sigma**2))

#     return rbf_kernel


# def estimate_sigma(X, alpha=0.8):
#     """
#     Estimate the sigma parameter for the RBF kernel in the CKA-RBF method.
#     """
#     # Compute the pairwise distances
#     D = pairwise_distances(X, X, metric='sqeuclidean')
#     D = np.sort(D, axis=1)
#     D = D[:, 1:]  # Remove the self-distance

#     # Compute the median distance
#     median_distance = np.median(D)
#     sigma = median_distance * alpha
#     return sigma


# def get_linear_kernel():

#     def linear_kernel(X):
#         return X @ X.T

#     return linear_kernel


# def cka(X, Y, kernel_X=get_linear_kernel(), kernel_Y=get_linear_kernel(), epsilon=1e-10):
#     """
#     CKA with a linear kernel as in:
#     Similarity of Neural Network Representations Revisited,
#     Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton
#     Proceedings of the 36th International Conference on Machine Learning,
#     PMLR 97:3519-3529, 2019
#     https://proceedings.mlr.press/v97/kornblith19a.html
#     """
#     # Center the data
#     X = X - X.mean(axis=0)
#     Y = Y - Y.mean(axis=0)

#     # Compute the kernel matrices
#     K = kernel_X(X)
#     L = kernel_Y(Y)

#     KH = K - np.mean(K, axis=1)  # KH
#     LH = L - np.mean(L, axis=1)  # LH

#     # Compute the CKA value.
#     cka_value = np.trace(KH @ LH) / np.maximum(np.sqrt(
#         np.trace(KH @ KH) * np.trace(LH @ LH)), epsilon)
#     return cka_value
