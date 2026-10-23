import numpy as np
import pytest

from pointcloudsimilarity.similarities import (
    TASSimilarityCPU,
    TASSimilarityTorch,
)


@pytest.mark.parametrize('N', [100, 1000])
def test_tas_benchmark_cpu(benchmark, N):
    rng = np.random.default_rng(0)
    D = 20
    k = 30

    pc1 = rng.standard_normal((N, D))
    pc2 = pc1 + 0.01 * rng.standard_normal((N, D))
    sim = TASSimilarityCPU(k=k)
    result = benchmark(sim, pc1, pc2)
    assert 0.0 <= result <= 1.0


@pytest.mark.parametrize('N', [100, 1000])
def test_tas_benchmark_torch(benchmark, N):
    rng = np.random.default_rng(0)
    D = 20
    k = 30

    pc1 = rng.standard_normal((N, D))
    pc2 = pc1 + 0.01 * rng.standard_normal((N, D))
    sim = TASSimilarityTorch(k=k)
    result = benchmark(sim, pc1, pc2)
    assert 0.0 <= result <= 1.0
