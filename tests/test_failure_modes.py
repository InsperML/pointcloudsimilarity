import subprocess
import sys

import numpy as np
import pytest

from pointcloudsimilarity import gw_sim, pwcca, rtd
from pointcloudsimilarity.similarities import available_metrics, compute_similarity


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_import_does_not_load_tensorflow():
    code = (
        'import sys, pointcloudsimilarity.similarities; '
        "sys.exit('tensorflow' in sys.modules)"
    )
    assert subprocess.run([sys.executable, '-c', code], check=False).returncode == 0


def test_gw_runs(rng):
    X = rng.standard_normal((20, 3))
    assert 0.0 < gw_sim.GWSimilarity()(X, X + 0.01 * rng.standard_normal((20, 3))) <= 1.0


@pytest.mark.skipif(rtd.rtd is not None, reason='rtd is installed')
def test_rtd_missing_raises():
    with pytest.raises(ImportError, match='rtd'):
        rtd.RTDSimilarity()


@pytest.mark.parametrize('symmetric', [False, True])
def test_pwcca_different_dimensionality(rng, symmetric):
    X = rng.standard_normal((200, 5))
    Y = X[:, :3] + 0.01 * rng.standard_normal((200, 3))
    sim = pwcca.PWCCASimilarity(symmetric=symmetric)
    assert sim(X, Y) > 0.9
    assert sim(Y, X) > 0.9


def test_pwcca_more_dimensions_than_points_warns(rng):
    X = rng.standard_normal((10, 20))
    with pytest.warns(RuntimeWarning, match='more points'):
        assert pwcca.PWCCASimilarity()(X, X) is None


def test_pwcca_zero_variance_warns(rng):
    with pytest.warns(RuntimeWarning, match='zero variance'):
        assert pwcca.PWCCASimilarity()(np.ones((50, 5)), rng.standard_normal((50, 5))) is None


def test_pwcca_nan_raises(rng):
    X = rng.standard_normal((50, 5))
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match='NaN'):
        pwcca.PWCCASimilarity()(X, X)


@pytest.mark.parametrize('method', ['tas_torch', 'nngs_torch'])
def test_compute_similarity_tas_names(rng, method):
    X = rng.standard_normal((50, 4))
    assert compute_similarity(X, X, method=method, k=5, device='cpu') == pytest.approx(1.0)


def test_compute_similarity_unknown_method_lists_available(rng):
    with pytest.raises(ValueError, match='tas_torch'):
        compute_similarity(np.zeros((5, 2)), np.zeros((5, 2)), method='nngs')
    assert 'tas_cpu' in available_metrics
