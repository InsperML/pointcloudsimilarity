import numpy as np
import pytest

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
