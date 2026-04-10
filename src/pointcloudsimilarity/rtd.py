# This is RTD as described in:
# Representation Topology Divergence: a Method for Comparing Neural
# Network Representations
# Serguei Barannikov, Ilya Trofimov, Nikita Balabin, Evgeny Burnaev
# ICML 2022
# https://arxiv.org/pdf/2201.00058


import numpy as np
from .core_similarity import Similarity
try:
    import rtd
    HAS_RTD = True
except ImportError as e:
    logging.error("RTD is not installed. Please install it to use this feature.")
    HAS_RTD = False
    
import logging


class RTDSimilarity(Similarity):
    def __init__(self):
        """
        Initialize the RTD similarity measure.

        """

    def __call__(self, pc1, pc2, **kwargs):
        """
        Compute the RTD similarity between two point clouds.

        Parameters:
        pc1 (np.ndarray): First point cloud of shape (N, D1).
        pc2 (np.ndarray): Second point cloud of shape (N, D2).
        (pc1 and pc2 must have the same number of points, but can have different dimensionality)

        Returns:
        float: RTD similarity between the two point clouds.
        """
        if not HAS_RTD:
            return 0.0  # or raise an exception, depending on how you want to handle this case
        return 1/(1+rtd.rtd(pc1, pc2))

