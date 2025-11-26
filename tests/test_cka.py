from pointcloudsimilarity import cka, gulp, procrustes, pwcca, nngs, gw_sim

import numpy as np

def test_cka_rbf_sigma_small():
    pc1 = np.random.randn(500, 30)
    pc2 = pc1 + 0.05 * np.random.randn(500, 30)
    sim = cka.CKASimilarity(kernel='rbf', sigma=1e-10)
    similarity = sim(pc1, pc2)
    
    
    assert similarity > 0.8
    
def test_cka_rbf_sigma_large():
    pc1 = np.random.randn(5, 3)
    pc2 = pc1 + 0.05 * np.random.randn(5, 3)
    sim = cka.CKASimilarity(kernel='rbf', sigma=1e10)
    similarity = sim(pc1, pc2)
    
    
    assert similarity > 0.8
    