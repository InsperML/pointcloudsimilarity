import numpy as np
import pytest
import torch

from pointcloudsimilarity import cka


def test_cka_rbf_sigma_small():
    rng = np.random.default_rng(0)
    pc1 = rng.standard_normal((500, 30))
    pc2 = pc1 + 0.05 * rng.standard_normal((500, 30))
    sim = cka.CKASimilarity(kernel='rbf', sigma=1e-10)
    similarity = sim(pc1, pc2)

    assert similarity is not None
    assert similarity > 0.8


def test_cka_rbf_sigma_large():
    rng = np.random.default_rng(0)
    pc1 = rng.standard_normal((5, 3))
    pc2 = pc1 + 0.05 * rng.standard_normal((5, 3))
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


def _reference_rbf_cka(X, Y, threshold):
    # Kornblith et al.'s reference code (google-research/representation_similarity,
    # Demo.ipynb): cka(gram_rbf(X, threshold), gram_rbf(Y, threshold)).
    def gram_rbf(x):
        dot_products = x.dot(x.T)
        sq_norms = np.diag(dot_products)
        sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
        sq_median_distance = np.median(sq_distances)
        return np.exp(-sq_distances / (2 * threshold**2 * sq_median_distance))

    def center_gram(gram):
        means = np.mean(gram, 0)
        means -= np.mean(means) / 2
        return gram - means[:, None] - means[None, :]

    gx, gy = center_gram(gram_rbf(X)), center_gram(gram_rbf(Y))
    return gx.ravel().dot(gy.ravel()) / (np.linalg.norm(gx) * np.linalg.norm(gy))


@pytest.mark.parametrize('threshold', [0.2, 0.4, 0.8, 1.0])
@pytest.mark.parametrize('y_scale', [1.0, 10.0])
def test_cka_rbf_auto_sigma_matches_reference(structured_clouds, threshold, y_scale):
    X, Y = structured_clouds
    Y = y_scale * Y
    expected = _reference_rbf_cka(X, Y, threshold)
    sim = cka.CKASimilarity(kernel='rbf', scale_by_alpha=threshold, device='cpu')
    assert sim(X, Y) == pytest.approx(expected, abs=1e-5)
    assert sim(Y, X) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize('n', [7, 8, 1000])
@pytest.mark.parametrize('q', [0.0, 0.3, 0.5, 1.0])
def test_quantile_matches_numpy(n, q):
    values = np.random.default_rng(n).standard_normal(n)
    assert cka._quantile(torch.tensor(values), q) == pytest.approx(np.quantile(values, q))


@pytest.mark.parametrize('alpha', [1e3, 1e5])
def test_cka_rbf_large_bandwidth_tends_to_linear(structured_clouds, alpha):
    X, Y = structured_clouds
    linear = cka.CKASimilarity(kernel='linear', device='cpu')(X, Y)
    rbf = cka.CKASimilarity(kernel='rbf', scale_by_alpha=alpha, device='cpu')(X, Y)
    assert rbf == pytest.approx(linear, abs=1e-3)
