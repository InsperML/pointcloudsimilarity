# This is GULP as described in:
# GULP: a prediction-based metric between representations
# Boix-Adrserà et al.
# Neurips 2022
# https://arxiv.org/pdf/2210.06545


import numpy as np
from .core_similarity import Similarity
import logging

class GULPSimilarity(Similarity):
    def __init__(self, lambda_=1.0, centralize_data=True):
        """
        Initialize the GULP similarity measure.

        Parameters:
        lambda_ (float): Regularization parameter.
        centralize_data (bool): Whether to centralize the data before computation.
        """
        self.lambda_ = lambda_
        self.centralize_data = centralize_data

    def __call__(self, pc1, pc2, **kwargs):
        """
        Compute the GULP similarity between two point clouds.

        Parameters:
        pc1 (np.ndarray): First point cloud of shape (N, D1).
        pc2 (np.ndarray): Second point cloud of shape (N, D2).
        (pc1 and pc2 must have the same number of points, but can have different dimensionality)

        Returns:
        float: GULP similarity between the two point clouds.
        """
        return gulp_sim(pc1, pc2, lambda_=self.lambda_,
                        centralize_data=self.centralize_data)

def gulp(X, Y, lambda_=1.0, centralize_data=True):
    """This is GULP as described in:
    GULP: a prediction-based metric between representations
    Boix-Adrserà et al.
    Neurips 2022
    https://arxiv.org/pdf/2210.06545
    """

    def empirical_cross_variance(X, Y):
        return X.T @ Y / X.shape[0]

    if centralize_data:
        # Center the data
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)

        # Compute the squared Frobenius norms
        norm1 = np.linalg.norm(X, axis=1, keepdims=True)
        norm2 = np.linalg.norm(Y, axis=1, keepdims=True)

        # Divide by the norms
        X = X / norm1
        Y = Y / norm2

    sigma_x = empirical_cross_variance(X, X)
    sigma_y = empirical_cross_variance(Y, Y)
    sigma_xy = empirical_cross_variance(X, Y)

    sigma_x_reg_inv = np.linalg.inv(sigma_x +
                                    lambda_ * np.eye(sigma_x.shape[0]))
    sigma_y_reg_inv = np.linalg.inv(sigma_y +
                                    lambda_ * np.eye(sigma_y.shape[0]))

    a1 = sigma_x_reg_inv @ sigma_x
    term1 = np.trace(a1 @ a1)

    a2 = sigma_y_reg_inv @ sigma_y
    term2 = np.trace(a2 @ a2)

    a3 = sigma_x_reg_inv @ sigma_xy @ sigma_y_reg_inv @ sigma_xy.T
    term3 = -2 * np.trace(a3)

    dsquare = term1 + term2 + term3

    return dsquare


def gulp_sim(*args, **kwargs):
    return 1-gulp(*args, **kwargs)