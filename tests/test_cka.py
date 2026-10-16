import numpy as np
import pytest
from scipy.spatial.distance import pdist

from pointcloudsimilarity import cka


def test_cka_rbf_sigma_small():
    pc1 = np.random.randn(500, 30)
    pc2 = pc1 + 0.05 * np.random.randn(500, 30)
    sim = cka.CKASimilarity(kernel='rbf', sigma=1e-10)
    similarity = sim(pc1, pc2)

    assert similarity is not None
    assert similarity > 0.8


def test_cka_rbf_sigma_large():
    pc1 = np.random.randn(5, 3)
    pc2 = pc1 + 0.05 * np.random.randn(5, 3)
    sim = cka.CKASimilarity(kernel='rbf', sigma=1e10)
    similarity = sim(pc1, pc2)

    assert similarity is not None
    assert similarity > 0.8


def _textbook_rbf_cka(X, Y, sigma_x, sigma_y):
    # Kornblith et al.: HSIC(K, L) / sqrt(HSIC(K, K) HSIC(L, L)) with H K H centering.
    def centered_kernel(Z, sigma):
        sq = ((Z[:, None] - Z[None]) ** 2).sum(-1)
        H = np.eye(len(Z)) - 1.0 / len(Z)
        return H @ np.exp(-sq / (2 * sigma**2)) @ H

    K, L = centered_kernel(X, sigma_x), centered_kernel(Y, sigma_y)
    return (K * L).sum() / np.sqrt((K * K).sum() * (L * L).sum())


@pytest.fixture
def structured_clouds():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((150, 6))
    Y = np.tanh(X @ rng.standard_normal((6, 6))) + 0.3 * rng.standard_normal((150, 6))
    return X, Y


@pytest.mark.parametrize('sigma', [0.5, 1.0, 3.0, 10.0, 100.0])
def test_cka_rbf_matches_textbook(structured_clouds, sigma):
    X, Y = structured_clouds
    sim = cka.CKASimilarity(kernel='rbf', sigma=sigma, device='cpu')
    assert sim(X, Y) == pytest.approx(_textbook_rbf_cka(X, Y, sigma, sigma), abs=1e-4)


def test_cka_rbf_auto_sigma_is_per_cloud(structured_clouds):
    X, Y = structured_clouds
    Y = 5.0 * Y
    sigma_x, sigma_y = (np.sqrt(np.median(pdist(Z) ** 2)) for Z in (X, Y))
    expected = _textbook_rbf_cka(X, Y, sigma_x, sigma_y)
    sim = cka.CKASimilarity(kernel='rbf', device='cpu')
    assert sim(X, Y) == pytest.approx(expected, abs=1e-4)
    assert sim(Y, X) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize('alpha', [1e3, 1e5])
def test_cka_rbf_large_bandwidth_tends_to_linear(structured_clouds, alpha):
    X, Y = structured_clouds
    linear = cka.CKASimilarity(kernel='linear', device='cpu')(X, Y)
    rbf = cka.CKASimilarity(kernel='rbf', scale_by_alpha=alpha, device='cpu')(X, Y)
    assert rbf == pytest.approx(linear, abs=1e-3)
