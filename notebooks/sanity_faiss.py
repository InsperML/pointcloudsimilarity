
import numpy as np

from pointcloudsimilarity.similarities import (NNGSSimilarityCPU,
                                               NNGSSimilarityFaiss,
                                               NNGSSimilarityTorch)

def run():
    N = 100
    D = 20
    k = 30

    pc1 = np.random.randn(N, D)
    pc2 = pc1 + 0.01 * np.random.randn(N, D)
    sim = NNGSSimilarityFaiss(k=k)
    result = sim(pc1, pc2)
    print(f"NNGS Similarity (Faiss) for N={N}, D={D}, k={k}: {result}")
    
if __name__ == "__main__":
    run()
    
    