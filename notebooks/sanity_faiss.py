
import numpy as np

from pointcloudsimilarity.similarities import (TASSimilarityCPU,
                                               TASSimilarityFaiss,
                                               TASSimilarityTorch)

def run():
    N = 10000
    D = 20
    k = 9000

    pc1 = np.random.randn(N, D)
    pc2 = pc1 + 0.01 * np.random.randn(N, D)
    sim = TASSimilarityFaiss(k=k, metric='cosine', normalize=True, approximate=True, gpu=True)
    sim2 = TASSimilarityTorch(k=k, device='cuda', normalize=True)
    result = sim(pc1, pc2)
    result2 = sim2(pc1, pc2)
    print(f"TAS Similarity (Faiss) for N={N}, D={D}, k={k}: {result}")
    print(f"NNGS Similarity (Torch) for N={N}, D={D}, k={k}: {result2}")
    
if __name__ == "__main__":
    run()
    
    