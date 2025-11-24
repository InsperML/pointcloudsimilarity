# Implements CKA as in:
#  CKA with a linear kernel as in:
#     Similarity of Neural Network Representations Revisited,
#     Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton
#     Proceedings of the 36th International Conference on Machine Learning,
#     PMLR 97:3519-3529, 2019
#     https://proceedings.mlr.press/v97/kornblith19a.html

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from .core_similarity import Similarity


class CKASimilarity(Similarity):

    def __init__(self, kernel='linear', sigma=None):
        """
        Initialize the CKA similarity measure.

        Parameters:
        kernel (str): 'linear' for linear CKA, 'rbf' for RBF kernel CKA.
        sigma (float): Bandwidth parameter for RBF kernel. If None and kernel is 'rbf',
                       it will be estimated from the data.
        """
        assert kernel in ['linear', 'rbf'], "Kernel must be 'linear' or 'rbf'"
        self.kernel = kernel
        self.sigma = sigma

    def __call__(self, pc1, pc2, **kwargs):
        """
        Compute the CKA similarity between two point clouds.

        Parameters:
        pc1 (np.ndarray): First point cloud of shape (N, D1).
        pc2 (np.ndarray): Second point cloud of shape (N, D2).
        (pc1 and pc2 must have the same number of points, but can have different dimensionality)

        Returns:
        float: CKA similarity between the two point clouds.
        """
        if self.kernel == 'linear':
            kernel_X = get_linear_kernel()
            kernel_Y = get_linear_kernel()
        elif self.kernel == 'rbf':
            if self.sigma is None:
                combined = np.vstack((pc1, pc2))
                self.sigma = estimate_sigma(combined)
            kernel_X = get_rbf_kernel(self.sigma)
            kernel_Y = get_rbf_kernel(self.sigma)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

        return cka(pc1, pc2, kernel_X=kernel_X, kernel_Y=kernel_Y)


def get_rbf_kernel(sigma):

    def rbf_kernel(X):
        return np.exp(-pairwise_distances(X, metric='sqeuclidean') /
                      (2 * sigma**2))

    return rbf_kernel


def estimate_sigma(X, alpha=0.8):
    """
    Estimate the sigma parameter for the RBF kernel in the CKA-RBF method.
    """
    # Compute the pairwise distances
    D = pairwise_distances(X, X, metric='sqeuclidean')
    D = np.sort(D, axis=1)
    D = D[:, 1:]  # Remove the self-distance

    # Compute the median distance
    median_distance = np.median(D)
    sigma = median_distance * alpha
    return sigma


def get_linear_kernel():

    def linear_kernel(X):
        return X @ X.T

    return linear_kernel


def cka(X, Y, kernel_X=get_linear_kernel(), kernel_Y=get_linear_kernel(), epsilon=1e-10):
    """
    CKA with a linear kernel as in:
    Similarity of Neural Network Representations Revisited,
    Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton
    Proceedings of the 36th International Conference on Machine Learning,
    PMLR 97:3519-3529, 2019
    https://proceedings.mlr.press/v97/kornblith19a.html
    """
    # Center the data
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Compute the kernel matrices
    K = kernel_X(X)
    L = kernel_Y(Y)

    KH = K - np.mean(K, axis=1)  # KH
    LH = L - np.mean(L, axis=1)  # LH

    # Compute the CKA value.
    cka_value = np.trace(KH @ LH) / np.maximum(np.sqrt(
        np.trace(KH @ KH) * np.trace(LH @ LH)), epsilon)
    return cka_value