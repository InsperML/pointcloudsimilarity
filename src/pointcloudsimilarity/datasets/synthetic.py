"""
Synthetic point clouds with known geometric relations.

The paired generators return (X, Y): the same points before and after a
controlled change. All generators take scikit-learn's random_state: None draws
from NumPy's global random state, in the order the notebooks drew before these
functions moved here, so np.random.seed() reproduces their data exactly; an int
or np.random.RandomState leaves the global state untouched.
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.utils import check_random_state

__all__ = [
    'equidistant_blobs',
    'jittered_blobs',
    'mix_with_noise',
    'noise_pair',
    'power_distortion',
    'shuffled_blobs',
    'simplex_centers',
]


def jittered_blobs(
    n_samples,
    n_features,
    centers,
    cluster_std,
    noise_std,
    random_state=None,
):
    """
    Gaussian blobs X, and Y with every point moved by Gaussian noise.

    Local jitter: small neighborhoods change, the clusters stay where they are.

    Parameters:
    n_samples (int): Number of points.
    n_features (int): Dimensionality, ignored if centers is an array.
    centers (int or np.ndarray): Number of blobs, or their centers of shape
        (n_centers, n_features).
    cluster_std (float): Standard deviation of each blob.
    noise_std (float): Standard deviation of the noise added to get Y.
    random_state (None, int or np.random.RandomState): See the module docstring.

    Returns:
    tuple of np.ndarray: X and Y, each of shape (n_samples, n_features).
    """
    rng = check_random_state(random_state)
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=rng,
    )
    return X, X + rng.normal(scale=noise_std, size=X.shape)


def shuffled_blobs(n_samples, centers, cluster_std, random_state=None):
    """
    Gaussian blobs X, and Y with the blobs moved to each other's centers.

    Every point keeps its offset from its blob's center, so each blob's internal
    neighborhoods are unchanged while the blobs trade places.

    Parameters:
    n_samples (int): Number of points, a multiple of the number of centers.
    centers (np.ndarray): Blob centers of shape (n_centers, n_features).
    cluster_std (float): Standard deviation of each blob.
    random_state (None, int or np.random.RandomState): See the module docstring.

    Returns:
    tuple of np.ndarray: X and Y, each of shape (n_samples, n_features), holding
    the blobs in the order of centers.
    """
    rng = check_random_state(random_state)
    centers = np.asarray(centers)
    n_centers, n_features = centers.shape
    per_center, remainder = divmod(n_samples, n_centers)
    if remainder:
        raise ValueError(
            f'n_samples ({n_samples}) must be a multiple of the number of '
            f'centers ({n_centers}).'
        )
    shuffled = rng.permutation(centers)
    offsets = [
        rng.normal(scale=cluster_std, size=(per_center, n_features))
        for _ in range(n_centers)
    ]
    X = np.vstack([offset + center for offset, center in zip(offsets, centers)])
    Y = np.vstack([offset + center for offset, center in zip(offsets, shuffled)])
    return X, Y


def power_distortion(
    n_samples,
    n_features,
    strength,
    exponent=2.0,
    random_state=None,
):
    """
    Standard normal X, and Y = X + strength * |X|**exponent * X / ||X||.

    Each coordinate moves away from zero by a power of its magnitude, so points
    far out are stretched much more than points near the center.

    Parameters:
    n_samples (int): Number of points.
    n_features (int): Dimensionality.
    strength (float): Scale of the displacement.
    exponent (float): Power of each coordinate's magnitude.
    random_state (None, int or np.random.RandomState): See the module docstring.

    Returns:
    tuple of np.ndarray: X and Y, each of shape (n_samples, n_features).
    """
    rng = check_random_state(random_state)
    X = rng.normal(size=(n_samples, n_features))
    phi = np.abs(X) ** exponent
    Y = X + phi * X / np.linalg.norm(X, axis=1, keepdims=True) * strength
    return X, Y


def noise_pair(n_samples, n_features, random_state=None):
    """
    Two independent standard normal point clouds: the chance-level baseline.

    Parameters:
    n_samples (int): Number of points.
    n_features (int): Dimensionality.
    random_state (None, int or np.random.RandomState): See the module docstring.

    Returns:
    tuple of np.ndarray: X and Y, each of shape (n_samples, n_features).
    """
    rng = check_random_state(random_state)
    X = rng.normal(size=(n_samples, n_features))
    Y = rng.normal(size=(n_samples, n_features))
    return X, Y


def mix_with_noise(X, noise_level, random_state=None):
    """
    Interpolate a point cloud toward independent standard normal noise.

    Parameters:
    X (np.ndarray): Point cloud of shape (N, D).
    noise_level (float): 0 returns X, 1 returns pure noise.
    random_state (None, int or np.random.RandomState): See the module docstring.

    Returns:
    np.ndarray: (1 - noise_level) * X + noise_level * noise, of shape (N, D).
    """
    rng = check_random_state(random_state)
    return (1 - noise_level) * X + noise_level * rng.standard_normal(X.shape)


def simplex_centers(n_classes, n_features, scale=1.0):
    """
    Vertices of a regular simplex: n_classes points all sqrt(2) * scale apart.

    Parameters:
    n_classes (int): Number of points.
    n_features (int): Dimensionality, at least n_classes - 1. The coordinates
        beyond the first n_classes - 1 are zero.
    scale (float): Scale of the simplex.

    Returns:
    np.ndarray: Centers of shape (n_classes, n_features).
    """
    if n_features < n_classes - 1:
        raise ValueError(
            f'Need n_features >= n_classes - 1 for equidistant centers, '
            f'got {n_features} < {n_classes - 1}.'
        )
    # Centered identity rows are equidistant (distance sqrt(2)); SVD gives
    # the n_classes-1 non-trivial directions as an orthonormal embedding.
    basis = np.eye(n_classes) - 1.0 / n_classes
    U, S, _ = np.linalg.svd(basis, full_matrices=False)
    coords = U[:, : n_classes - 1] * S[: n_classes - 1] * scale
    centers = np.zeros((n_classes, n_features))
    centers[:, : n_classes - 1] = coords
    return centers


def equidistant_blobs(
    n_samples,
    n_features,
    n_classes,
    cluster_std,
    center_scale=10.0,
    random_state=None,
):
    """
    Labelled Gaussian blobs around simplex_centers, so all classes are equally apart.

    Parameters:
    n_samples (int): Number of points.
    n_features (int): Dimensionality, at least n_classes - 1.
    n_classes (int): Number of blobs.
    cluster_std (float): Standard deviation of each blob.
    center_scale (float): Scale of the simplex the centers lie on.
    random_state (None, int or np.random.RandomState): See the module docstring.

    Returns:
    tuple of np.ndarray: Points X of shape (n_samples, n_features) and their
    integer labels y of shape (n_samples,).
    """
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=simplex_centers(n_classes, n_features, scale=center_scale),
        cluster_std=cluster_std,
        random_state=random_state,
    )
    return X, y
