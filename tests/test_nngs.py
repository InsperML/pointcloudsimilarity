
import numpy as np
import pytest

from pointcloudsimilarity.similarities import (NNGSSimilarityCPU,
                                               NNGSSimilarityFaiss,
                                               NNGSSimilarityTorch)


@pytest.mark.parametrize("N", [100, 1000, 10000])
def test_nngs_benchmark_faiss(benchmark, N):
    D = 20
    k = 30
    
    pc1 = np.random.randn(N, D)
    pc2 = pc1 + 0.01 * np.random.randn(N, D)
    sim = NNGSSimilarityFaiss(k=k)
    result = benchmark(sim, pc1, pc2)
    assert 0.0 <= result <= 1.0

@pytest.mark.parametrize("N", [100, 1000, 10000])
def test_nngs_benchmark_cpu(benchmark, N):
    D = 20
    k = 30
    
    pc1 = np.random.randn(N, D)
    pc2 = pc1 + 0.01 * np.random.randn(N, D)
    sim = NNGSSimilarityCPU(k=k)
    result = benchmark(sim, pc1, pc2)
    assert 0.0 <= result <= 1.0

@pytest.mark.parametrize("N", [100, 1000, 10000])
def test_nngs_benchmark_torch(benchmark, N):
    
    D = 20
    k = 30
    
    pc1 = np.random.randn(N, D)
    pc2 = pc1 + 0.01 * np.random.randn(N, D)
    sim = NNGSSimilarityTorch(k=k)
    result = benchmark(sim, pc1, pc2)
    assert 0.0 <= result <= 1.0