# I got this code from here - https://github.com/google/svcca/
# The repo has been archived, so I'm including the code here directly.

from .core_similarity import Similarity
from .legacy.pwcca import compute_pwcca

import numpy as np

class PWCCASimilarity(Similarity):
    def __call__(self, pc1, pc2, **kwargs):
        """
        Compute the PWCCA similarity between two point clouds.

        Parameters:
        pc1 (np.ndarray): First point cloud of shape (N, D1).
        pc2 (np.ndarray): Second point cloud of shape (N, D2).
        (pc1 and pc2 must have the same number of points, but can have different dimensionality)

        Returns:
        float: PWCCA similarity between the two point clouds.
        """
        sim, _, _ = compute_pwcca(pc1.T, pc2.T, epsilon=1e-10)
        return sim


