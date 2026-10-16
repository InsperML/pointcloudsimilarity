# I got this code from here - https://github.com/google/svcca/
# The repo has been archived, so I'm including the code here directly.

import warnings

import numpy as np

from .core_similarity import Similarity
from .legacy.pwcca import compute_pwcca


class PWCCASimilarity(Similarity):
    def __init__(self, symmetric=False):
        """
        Initialize the PWCCA similarity measure.

        Parameters:
        symmetric (bool): If True, compute symmetric PWCCA by averaging
                          PWCCA(pc1, pc2) and PWCCA(pc2, pc1).
        """
        self.symmetric = symmetric

    def __call__(self, pc1, pc2, **kwargs):
        """
        Compute the PWCCA similarity between two point clouds.

        Parameters:
        pc1 (np.ndarray): First point cloud of shape (N, D1).
        pc2 (np.ndarray): Second point cloud of shape (N, D2).
        (pc1 and pc2 must have the same number of points, but can have different dimensionality)

        Returns:
        float or None: PWCCA similarity between the two point clouds, or None
        (with a RuntimeWarning) when PWCCA is undefined for these inputs.
        """
        pc1 = np.asarray(pc1, dtype=np.float64)
        pc2 = np.asarray(pc2, dtype=np.float64)
        if not (np.isfinite(pc1).all() and np.isfinite(pc2).all()):
            raise ValueError('PWCCA inputs must not contain NaN or inf.')

        n_points = pc1.shape[0]
        if max(pc1.shape[1], pc2.shape[1]) >= n_points:
            _warn_undefined(
                f'CCA needs more points ({n_points}) than dimensions '
                f'({pc1.shape[1]}, {pc2.shape[1]})'
            )
            return None
        if not (pc1.std(axis=0).any() and pc2.std(axis=0).any()):
            _warn_undefined('a point cloud has zero variance')
            return None

        try:
            sim, _, _ = compute_pwcca(pc1.T, pc2.T, epsilon=1e-10)
            if self.symmetric:
                sim_rev, _, _ = compute_pwcca(pc2.T, pc1.T, epsilon=1e-10)
                sim = 0.5 * (sim + sim_rev)
        except np.linalg.LinAlgError as e:
            _warn_undefined(f'numerical failure ({e})')
            return None
        return sim


def _warn_undefined(reason):
    warnings.warn(
        f'PWCCA is undefined for these inputs: {reason}. Returning None.',
        RuntimeWarning,
        stacklevel=3,
    )
