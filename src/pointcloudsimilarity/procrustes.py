# Procrustes Similarity

from .core_similarity import Similarity
import numpy as np
from scipy.spatial import procrustes

class ProcrustesSimilarity(Similarity):
    def __call__(self, pc1, pc2, **kwargs):
        """
        Compute the Procrustes distance between two point clouds.

        Parameters:
        pc1 (np.ndarray): First point cloud of shape (N, D).
        pc2 (np.ndarray): Second point cloud of shape (N, D).
        (pc1 and pc2 must have the same number of points and same dimensionality)

        Returns:
        float: Procrustes distance between the two point clouds.
        """
        _, _, disparity = procrustes(pc1, pc2)
        return 1-disparity
