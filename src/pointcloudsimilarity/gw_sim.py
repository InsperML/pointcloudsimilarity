# This implements Gromov-Wasserstein Distances for Point Cloud Similarity

import numpy as np
from .core_similarity import Similarity

import ot
from scipy.spatial.distance import cdist


class GWSimilarity(Similarity):
    def __call__(self, pc1, pc2, **kwargs):
        """
        Compute the Gromov-Wasserstein similarity between two point clouds.

        Parameters:
        pc1 (np.ndarray): First point cloud of shape (N, D1).
        pc2 (np.ndarray): Second point cloud of shape (N, D2).
        (pc1 and pc2 must have the same number of points, but can have different dimensionality)

        Returns:
        float: Gromov-Wasserstein similarity between the two point clouds.
        """
        return gw_sim(pc1, pc2)

def gw_sim(X1, X2, epsilon=1e-10):
    """Compute the Gromov-Wasserstein similarity between two point clouds X1 and X2.

    Parameters:
    X1 (np.ndarray): First point cloud of shape (N, D1).
    X2 (np.ndarray): Second point cloud of shape (N, D2).
    (X1 and X2 must have the same number of points, but can have different dimensionality)

    Returns:
    float: Gromov-Wasserstein similarity (1/GW-distance) between the two point clouds.
    """

    n_points = X1.shape[0]
    
    # --- 1. Compute the pairwise distance matrices within each space ---
    # Gromov-Wasserstein compares these distance matrices.
    C1 = cdist(X1, X1, metric='euclidean')
    C2 = cdist(X2, X2, metric='euclidean')

    # Normalize the distance matrices (common practice)
    C1 /= C1.max()
    C2 /= C2.max()

    # --- 3. Define the distributions on the points (uniform in this case) ---
    p = ot.unif(n_points)
    q = ot.unif(n_points)

    # --- 4. Compute the Gromov-Wasserstein distance ---
    _,  log = ot.gromov.gromov_wasserstein(C1, C2, p, q, loss_fun='square_loss', log=True)

    return -log['gw_dist']  # Return similarity as inverse of distance