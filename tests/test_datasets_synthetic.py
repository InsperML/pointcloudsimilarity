"""
pointcloudsimilarity.datasets.synthetic against the notebook code it replaced.

The _v1_* functions are copied from v1.0, changed only in formatting. With
random_state=None the generators must draw exactly what that code drew from
NumPy's global random state, and leave it in the same state, so the paper
notebooks keep their data.
"""

import subprocess
import sys

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from pointcloudsimilarity import datasets

SEEDS = [0, 1, 1234]
# Like ijcnn/01: four centers at corners of the hypercube.
CENTERS = np.sign(np.random.default_rng(0).normal(0, 5, size=(4, 10)))


# Verbatim from v1.0 notebooks/ijcnn/01-toy_case_sweeps.py (blobs_with_noise is
# also in 00-downsampling.py).
def _v1_blobs_with_noise(N, D, centers, blob_std, noise_std):
    X, _ = make_blobs(
        n_samples=N,
        n_features=D,
        centers=centers,
        cluster_std=blob_std,
    )
    noise = np.random.normal(scale=noise_std, size=(N, D))
    Y = X + noise
    return X, Y


def _v1_shuffled_blobs(N, D, centers, blob_std, random_state=None):
    if not random_state:
        random_state = np.random.randint(0, 10000)

    if type(centers) == int:  # noqa: E721
        cx = np.random.normal(0, 5, size=(centers, D))
        n_centers = centers
    else:
        cx = centers
        n_centers = centers.shape[0]

    np.random.seed(random_state)
    cy = np.random.permutation(cx)
    samples_per_center = N // n_centers
    assert samples_per_center * n_centers == N, (
        'N must be divisible by number of centers'
    )

    X = []
    Y = []

    for cx_, cy_ in zip(cx, cy):
        blob = np.random.normal(loc=0, scale=blob_std, size=(samples_per_center, D))
        X.append(blob + cx_)
        Y.append(blob + cy_)

    X = np.vstack(X)
    Y = np.vstack(Y)
    return X, Y


def _v1_distortion(N, D, distortion_strength, distortion_coefficient=2.0):
    X = np.random.normal(size=(N, D))
    phi = np.abs(X) ** distortion_coefficient
    Y = X + phi * X / np.linalg.norm(X, axis=1, keepdims=True) * distortion_strength
    return X, Y


# Verbatim from v1.0 notebooks/experiments/dataset_.py.
def _v1_make_equidistant_centers(
    n_classes: int, n_features: int, scale: float
) -> np.ndarray:
    assert n_features >= n_classes - 1, (
        f'Need n_features >= n_classes-1 for equidistant centers, '
        f'got {n_features} < {n_classes - 1}'
    )
    # Centered identity rows are equidistant (distance sqrt(2)); SVD gives
    # the n_classes-1 non-trivial directions as an orthonormal embedding.
    basis = np.eye(n_classes) - 1.0 / n_classes
    U, S, _ = np.linalg.svd(basis, full_matrices=False)
    coords = U[:, : n_classes - 1] * S[: n_classes - 1] * scale
    centers = np.zeros((n_classes, n_features))
    centers[:, : n_classes - 1] = coords
    return centers


def _v1_make_equidistant_blobs_dataset(
    n_samples: int,
    n_features: int,
    n_classes: int,
    cluster_std: float,
    center_scale: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    result = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=_v1_make_equidistant_centers(n_classes, n_features, scale=center_scale),
        cluster_std=cluster_std,
        random_state=42,
    )
    assert len(result) == 2
    X, y = result

    return X, y


def _assert_arrays_equal(expected, actual):
    assert len(actual) == len(expected)
    for e, a in zip(expected, actual):
        np.testing.assert_array_equal(a, e)


def _assert_same_state(expected, actual):
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


def _assert_same_global_draws(old, new, seed):
    np.random.seed(seed)
    expected = old()
    expected_state = np.random.get_state()
    np.random.seed(seed)
    actual = new()
    _assert_arrays_equal(expected, actual)
    _assert_same_state(expected_state, np.random.get_state())


# The configurations of ijcnn/00-downsampling.py and ijcnn/01-toy_case_sweeps.py.
@pytest.mark.parametrize('seed', SEEDS)
@pytest.mark.parametrize(
    'N, D, centers, blob_std, noise_std',
    [(1000, 100, 5, 0.5, 0.5), (500, 10, CENTERS, 0.1, 0.1)],
)
def test_jittered_blobs_matches_v1(seed, N, D, centers, blob_std, noise_std):
    _assert_same_global_draws(
        lambda: _v1_blobs_with_noise(N, D, centers, blob_std, noise_std),
        lambda: datasets.jittered_blobs(N, D, centers, blob_std, noise_std),
        seed,
    )


def _first_draw_moves_every_blob(seed, n_centers):
    order = np.random.RandomState(seed).permutation(n_centers)
    return not (order == np.arange(n_centers)).any()


def _destinations(X, Y, centers):
    """The index of the center each blob of shuffled_blobs moved to."""
    shifts = (Y - X).reshape(len(centers), -1, centers.shape[1])[:, 0]
    return [
        int(np.argmin(np.linalg.norm(centers - c, axis=1))) for c in shifts + centers
    ]


# Seed 1 is ijcnn/01's. With these seeds the first draw already moves every blob.
@pytest.mark.parametrize('seed', [1, 2, 9])
def test_shuffled_blobs_matches_v1_after_reseeding(seed):
    assert _first_draw_moves_every_blob(seed, len(CENTERS))
    # v1.0 reseeded the global state with random_state. Callers that need its
    # exact data, like ijcnn/01, now call np.random.seed() themselves.
    expected = _v1_shuffled_blobs(500, 10, CENTERS, 0.1, random_state=seed)
    expected_state = np.random.get_state()
    np.random.seed(seed)
    actual = datasets.shuffled_blobs(500, CENTERS, 0.1)
    _assert_arrays_equal(expected, actual)
    _assert_same_state(expected_state, np.random.get_state())


# With these seeds the first draw leaves a blob at its own center, as v1.0 did.
@pytest.mark.parametrize('seed', [3, 4, 7])
def test_shuffled_blobs_redraws_until_every_blob_moves(seed):
    assert not _first_draw_moves_every_blob(seed, len(CENTERS))
    _, v1_Y = _v1_shuffled_blobs(500, 10, CENTERS, 0.1, random_state=seed)
    np.random.seed(seed)
    X, Y = datasets.shuffled_blobs(500, CENTERS, 0.1)
    assert not np.array_equal(Y, v1_Y)
    assert all(d != i for i, d in enumerate(_destinations(X, Y, CENTERS)))


@pytest.mark.parametrize('n_centers', [2, 3, 5])
def test_shuffled_blobs_never_leaves_a_blob_in_place(n_centers):
    centers = np.random.default_rng(n_centers).normal(0, 5, size=(n_centers, 4))
    for seed in range(100):
        X, Y = datasets.shuffled_blobs(10 * n_centers, centers, 0.1, random_state=seed)
        destinations = _destinations(X, Y, centers)
        assert sorted(destinations) == list(range(n_centers))
        assert all(d != i for i, d in enumerate(destinations))


def test_shuffled_blobs_needs_two_centers():
    with pytest.raises(ValueError, match='at least 2 centers'):
        datasets.shuffled_blobs(10, CENTERS[:1], 0.1)


@pytest.mark.parametrize('seed', SEEDS)
@pytest.mark.parametrize('strength, exponent', [(50, 40), (1.0, 2.0)])
def test_power_distortion_matches_v1(seed, strength, exponent):
    _assert_same_global_draws(
        lambda: _v1_distortion(300, 10, strength, exponent),
        lambda: datasets.power_distortion(300, 10, strength, exponent),
        seed,
    )


@pytest.mark.parametrize('seed', SEEDS)
def test_noise_pair_matches_v1(seed):
    # Inline in v1.0 ijcnn/01-toy_case_sweeps.py.
    N, D = 500, 10
    _assert_same_global_draws(
        lambda: (np.random.normal(size=(N, D)), np.random.normal(size=(N, D))),
        lambda: datasets.noise_pair(N, D),
        seed,
    )


@pytest.mark.parametrize('seed', SEEDS)
@pytest.mark.parametrize('alpha', [0.0, 0.3, 1.0])
def test_mix_with_noise_matches_v1(seed, alpha):
    # Inline in v1.0 notebooks/01-whitenoise.py.
    X, _ = make_blobs(n_samples=300, n_features=20, centers=3, random_state=42)
    N, D = X.shape
    _assert_same_global_draws(
        lambda: ((1 - alpha) * X + alpha * np.random.randn(N, D),),
        lambda: (datasets.mix_with_noise(X, alpha),),
        seed,
    )


@pytest.mark.parametrize(
    'n_classes, n_features, scale', [(5, 5, 1.0), (3, 10, 10.0), (2, 1, 0.5)]
)
def test_simplex_centers_match_v1(n_classes, n_features, scale):
    np.testing.assert_array_equal(
        datasets.simplex_centers(n_classes, n_features, scale),
        _v1_make_equidistant_centers(n_classes, n_features, scale),
    )


def test_equidistant_blobs_match_v1():
    # The configuration in v1.0 notebooks/experiments/experiments.toml.
    _assert_arrays_equal(
        _v1_make_equidistant_blobs_dataset(400, 5, 5, 2, 1.0),
        datasets.equidistant_blobs(400, 5, 5, 2, center_scale=1.0, random_state=42),
    )


GENERATORS = {
    'jittered_blobs': lambda rs: datasets.jittered_blobs(
        200, 5, 3, 0.1, 0.1, random_state=rs
    ),
    'shuffled_blobs': lambda rs: datasets.shuffled_blobs(
        200, CENTERS, 0.1, random_state=rs
    ),
    'power_distortion': lambda rs: datasets.power_distortion(
        200, 5, 1.0, random_state=rs
    ),
    'noise_pair': lambda rs: datasets.noise_pair(200, 5, random_state=rs),
    'mix_with_noise': lambda rs: (
        datasets.mix_with_noise(np.ones((200, 5)), 0.5, random_state=rs),
    ),
    'equidistant_blobs': lambda rs: datasets.equidistant_blobs(
        200, 5, 4, 0.5, random_state=rs
    ),
}


@pytest.mark.parametrize('name', GENERATORS)
def test_int_random_state_is_reproducible_and_leaves_global_state(name):
    np.random.seed(0)
    before = np.random.get_state()
    first = GENERATORS[name](3)
    _assert_same_state(before, np.random.get_state())
    _assert_arrays_equal(first, GENERATORS[name](3))


def test_shuffled_blobs_moves_whole_blobs():
    X, Y = datasets.shuffled_blobs(200, CENTERS, 0.1, random_state=0)
    shifts = (Y - X).reshape(len(CENTERS), -1, CENTERS.shape[1])
    # Every point of a blob moves by the same vector, from one center to another.
    np.testing.assert_allclose(shifts, shifts[:, :1].repeat(shifts.shape[1], axis=1))
    destinations = shifts[:, 0] + CENTERS
    assert sorted(map(tuple, destinations.round(9))) == sorted(map(tuple, CENTERS))


def test_shuffled_blobs_needs_a_multiple_of_the_centers():
    with pytest.raises(ValueError, match='multiple'):
        datasets.shuffled_blobs(201, CENTERS, 0.1)


@pytest.mark.parametrize('n_classes, scale', [(2, 1.0), (5, 3.0), (10, 0.5)])
def test_simplex_centers_are_equidistant(n_classes, scale):
    centers = datasets.simplex_centers(n_classes, n_classes + 2, scale)
    distances = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    off_diagonal = distances[~np.eye(n_classes, dtype=bool)]
    np.testing.assert_allclose(off_diagonal, np.sqrt(2) * scale)


def test_simplex_centers_need_enough_features():
    with pytest.raises(ValueError, match='n_features'):
        datasets.simplex_centers(5, 3)


def test_mix_with_noise_endpoints():
    X = np.arange(12.0).reshape(4, 3)
    np.testing.assert_array_equal(datasets.mix_with_noise(X, 0.0, random_state=0), X)
    noise = np.random.RandomState(0).standard_normal(X.shape)
    np.testing.assert_array_equal(
        datasets.mix_with_noise(X, 1.0, random_state=0), noise
    )


def test_import_loads_no_heavy_dependencies():
    heavy = ['datasets', 'gensim', 'tensorflow', 'transformers']
    code = (
        'import sys, pointcloudsimilarity.datasets; '
        f'sys.exit(sorted(m for m in {heavy!r} if m in sys.modules) or 0)'
    )
    result = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
