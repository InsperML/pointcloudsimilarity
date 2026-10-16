import numpy as np
import pytest

from pointcloudsimilarity.similarities import TASSimilarityTorch, tas_spectrum

N, D = 300, 8


@pytest.fixture
def clouds():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, D))
    Y = X[:, :5] + 0.3 * rng.standard_normal((N, 5))
    return X, Y


@pytest.mark.parametrize('normalize', [True, False])
@pytest.mark.parametrize('only_intersection', [False, True])
def test_matches_per_k_similarity(clouds, normalize, only_intersection):
    X, Y = clouds
    ks = [1, 2, 5, 17, 0.1, 100, 0.5, N - 2, N - 1]
    spectrum = tas_spectrum(
        X, Y, ks, normalize=normalize, only_intersection=only_intersection, device='cpu'
    )
    expected = [
        TASSimilarityTorch(
            k=k, normalize=normalize, only_intersection=only_intersection, device='cpu'
        )(X, Y)
        for k in ks
    ]
    assert spectrum == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_order_and_duplicates_preserved(clouds):
    X, Y = clouds
    ks = [40, 3, 40, 10]
    spectrum = tas_spectrum(X, Y, ks, device='cpu')
    assert spectrum[0] == spectrum[2]
    assert spectrum == pytest.approx(tas_spectrum(X, Y, [40, 3, 10], device='cpu')[[0, 1, 0, 2]])


def test_batch_size_invariance(clouds):
    X, Y = clouds
    ks = np.arange(1, N - 1, 7)
    reference = tas_spectrum(X, Y, ks, device='cpu', batch_size=N)
    for bs in [100, 77]:
        assert tas_spectrum(X, Y, ks, device='cpu', batch_size=bs) == pytest.approx(reference)


def test_identical_clouds_and_trivial_k(clouds):
    X, _ = clouds
    assert tas_spectrum(X, X, [1, 10, 0.3], device='cpu') == pytest.approx([1.0, 1.0, 1.0])
    rng = np.random.default_rng(1)
    assert tas_spectrum(X, rng.standard_normal((N, D)), [N], device='cpu')[0] == 1.0


def test_empty_ks_raises(clouds):
    X, Y = clouds
    with pytest.raises(ValueError, match='ks'):
        tas_spectrum(X, Y, [], device='cpu')
