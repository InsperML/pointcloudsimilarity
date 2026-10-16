"""Properties every metric should have, checked on the same point clouds."""

import warnings

import numpy as np
import pytest

from pointcloudsimilarity.similarities import (
    CKASimilarity,
    GULPSimilarity,
    GWSimilarity,
    ProcrustesSimilarity,
    PWCCASimilarity,
    TASSimilarityCPU,
    TASSimilarityTorch,
)

METRICS = {
    'cka_linear': lambda: CKASimilarity(device='cpu'),
    'cka_rbf_auto': lambda: CKASimilarity(kernel='rbf', device='cpu'),
    'cka_rbf_alpha': lambda: CKASimilarity(kernel='rbf', scale_by_alpha=0.4, device='cpu'),
    'gulp': GULPSimilarity,
    'procrustes': ProcrustesSimilarity,
    'gw': GWSimilarity,
    'pwcca': PWCCASimilarity,
    'pwcca_symmetric': lambda: PWCCASimilarity(symmetric=True),
    'tas_cpu': lambda: TASSimilarityCPU(k=10),
    'tas_torch': lambda: TASSimilarityTorch(k=10, device='cpu'),
}
# PWCCA weights by X's canonical directions; symmetric=True averages both orders.
ASYMMETRIC = {'pwcca'}
# GW compares raw distance matrices, so it depends on each cloud's scale.
SCALE_DEPENDENT = {'gw'}


@pytest.fixture
def clouds():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 6))
    Y = np.tanh(X @ rng.standard_normal((6, 6))) + 0.3 * rng.standard_normal((200, 6))
    return X, Y


def _score(name, X, Y):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return METRICS[name]()(X, Y)


@pytest.mark.parametrize('name', METRICS)
def test_identical_clouds_score_one(name, clouds):
    X, _ = clouds
    assert _score(name, X, X) == pytest.approx(1.0, abs=1e-5)


# Known bug: the legacy PWCCA weights CCA directions with uncentered activations,
# so shifting the first cloud changes the score.
PWCCA_TRANSLATION = pytest.mark.xfail(strict=True, reason='PWCCA uses uncentered weights')


@pytest.mark.parametrize(
    'name',
    [pytest.param(n, marks=PWCCA_TRANSLATION) if n.startswith('pwcca') else n for n in METRICS],
)
def test_translation_invariance(name, clouds):
    X, Y = clouds
    assert _score(name, X + 3.0, Y - 2.0) == pytest.approx(_score(name, X, Y), abs=1e-4)


@pytest.mark.parametrize('name', sorted(set(METRICS) - ASYMMETRIC))
def test_symmetry(name, clouds):
    X, Y = clouds
    assert _score(name, Y, X) == pytest.approx(_score(name, X, Y), abs=1e-4)


@pytest.mark.parametrize('name', sorted(set(METRICS) - SCALE_DEPENDENT))
def test_scale_invariance(name, clouds):
    X, Y = clouds
    assert _score(name, X, 7.0 * Y) == pytest.approx(_score(name, X, Y), abs=1e-4)
