import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from pointcloudsimilarity.similarities import (
    TASSimilarityCPU,
    TASSimilarityFaiss,
    TASSimilarityTorch,
)

N, D = 300, 8


def _backends():
    backends = [
        pytest.param(lambda **kw: TASSimilarityCPU(**kw), id='cpu'),
        pytest.param(lambda **kw: TASSimilarityTorch(device='cpu', **kw), id='torch'),
    ]
    try:
        import faiss  # noqa: F401

        backends.append(
            pytest.param(lambda **kw: TASSimilarityFaiss(gpu=False, **kw), id='faiss')
        )
    except ImportError:
        pass
    return backends


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def clouds(rng):
    X = rng.standard_normal((N, D))
    Y = X + 0.3 * rng.standard_normal((N, D))
    return X, Y


@pytest.mark.parametrize('k', [10, 0.1])
@pytest.mark.parametrize('normalize', [False, True])
def test_backends_agree(clouds, k, normalize):
    X, Y = clouds
    reference = TASSimilarityCPU(k=k, normalize=normalize)(X, Y)
    for backend in _backends():
        result = backend.values[0](k=k, normalize=normalize)(X, Y)
        assert result == pytest.approx(reference, abs=1e-3), backend.id


@pytest.mark.parametrize('make', _backends())
@pytest.mark.parametrize('k', [10, 0.1])
def test_random_baseline_is_zero(make, rng, k):
    X = rng.standard_normal((N, D))
    sim = make(k=k)
    scores = [sim(X, rng.standard_normal((N, D))) for _ in range(20)]
    assert np.mean(scores) == pytest.approx(0.0, abs=0.02)


@pytest.mark.parametrize('make', _backends())
def test_identical_clouds_score_one(make, clouds):
    X, _ = clouds
    assert make(k=10)(X, X) == pytest.approx(1.0)


@pytest.mark.parametrize('make', _backends())
def test_object_reuse_on_new_data(make, clouds, rng):
    X, Y = clouds
    Z = rng.standard_normal((N, D))
    sim = make(k=10)
    sim(X, Y)
    assert sim(X, Z) == pytest.approx(make(k=10)(X, Z))


@pytest.mark.parametrize('make', _backends())
def test_different_dimensionality(make, clouds, rng):
    X, _ = clouds
    Y = X @ rng.standard_normal((D, 3))
    result = make(k=10)(X, Y)
    assert -1.0 <= result <= 1.0


@pytest.mark.parametrize('make', _backends())
def test_k_covering_all_points_is_trivially_one(make, clouds, rng):
    X, _ = clouds
    Z = rng.standard_normal((N, D))
    assert make(k=N)(X, Z) == pytest.approx(1.0)


@pytest.mark.parametrize('only_intersection', [False, True])
def test_torch_batch_size_invariance(clouds, only_intersection):
    X, Y = clouds
    results = [
        TASSimilarityTorch(
            k=10, device='cpu', batch_size=bs, only_intersection=only_intersection
        )(X, Y)
        for bs in [N, 100, 77]
    ]
    assert results == pytest.approx([results[0]] * len(results))


def test_torch_only_intersection_bounds(clouds, rng):
    X, _ = clouds
    sim = TASSimilarityTorch(k=10, device='cpu', only_intersection=True)
    assert sim(X, X) == pytest.approx(1.0)
    scores = [sim(X, rng.standard_normal((N, D))) for _ in range(20)]
    assert np.mean(scores) == pytest.approx(0.0, abs=0.02)


def test_neighborhood_overlaps_matches_sets(rng):
    from pointcloudsimilarity.tas_core.tas_torch import neighborhood_overlaps

    n, k = 50, 6
    idx_x = np.stack([rng.choice(n, k, replace=False) for _ in range(20)])
    idx_y = np.stack([rng.choice(n, k, replace=False) for _ in range(20)])
    idx_x[0, -1] = idx_y[0, -1] = -1  # FAISS "no neighbor found" must not match
    expected = [len(set(a) & set(b) - {-1}) for a, b in zip(idx_x, idx_y)]
    result = neighborhood_overlaps(
        torch.from_numpy(idx_x), torch.from_numpy(idx_y), n, k, only_intersection=True
    )
    assert result.tolist() == expected


def _reference_tas(X, Y, k):
    # Brute force, excluding each point from its own neighborhood by index.
    def neighbors(Z):
        d = ((Z[:, None] - Z[None]) ** 2).sum(-1)
        np.fill_diagonal(d, np.inf)
        return np.argsort(d, axis=1, kind='stable')[:, :k]

    inter = np.array([len(set(a) & set(b)) for a, b in zip(neighbors(X), neighbors(Y))])
    m = len(X) - 1
    raw, base = np.mean(inter / (2 * k - inter)), k / (2 * m - k)
    return (raw - base) / (1 - base)


@pytest.fixture
def clouds_with_duplicates(rng):
    # 15 pairs of exact duplicates (as with repeated sentences), placed far from
    # the other points and from each other.
    pairs = 30.0 * rng.standard_normal((15, D))
    X = np.vstack([rng.standard_normal((N - 30, D)), pairs, pairs])
    return X, X + 1e-4 * rng.standard_normal(X.shape)


# With exact duplicates a k-neighborhood is not unique if the k-th and (k+1)-th
# neighbors tie, so compare at k=1 on isolated pairs, where there are no ties:
# a duplicated point's neighbor must be its twin, never itself.
@pytest.mark.parametrize('make', _backends())
def test_duplicate_points(make, clouds_with_duplicates):
    X, Y = clouds_with_duplicates
    assert make(k=1)(X, Y) == pytest.approx(_reference_tas(X, Y, 1))


def test_spectrum_duplicate_points(clouds_with_duplicates):
    from pointcloudsimilarity.similarities import tas_spectrum

    X, Y = clouds_with_duplicates
    assert tas_spectrum(X, Y, [1], device='cpu') == pytest.approx([_reference_tas(X, Y, 1)])


def test_torch_neighbors_never_include_self(clouds_with_duplicates):
    from pointcloudsimilarity.tas_core.tas_torch import _nearest_neighbors

    X = torch.tensor(clouds_with_duplicates[0], dtype=torch.float32)
    sq = (X**2).sum(dim=1, keepdim=True)
    for start in (0, 100):  # also as a later batch
        idx = _nearest_neighbors(X[start:], sq[start:], X, sq, 10, start)
        own = torch.arange(start, len(X)).unsqueeze(1)
        assert not (idx == own).any()


def test_faiss_drop_self():
    from pointcloudsimilarity.tas_core.tas_faiss import _drop_self

    rows = np.array([[0, 5, 6], [7, 1, 8], [3, 4, 9]])  # row 2 lacks its own index
    assert _drop_self(rows, 2).tolist() == [[5, 6], [7, 8], [3, 4]]


def test_faiss_defaults_to_cpu_without_gpu():
    pytest.importorskip('faiss')
    code = (
        'import numpy as np; from pointcloudsimilarity.similarities import TASSimilarityFaiss; '
        'X = np.random.default_rng(0).standard_normal((100, 4)); '
        'assert TASSimilarityFaiss(k=5)(X, X) == 1.0'
    )
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
    assert subprocess.run([sys.executable, '-c', code], env=env, check=False).returncode == 0
