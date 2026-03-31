from pointcloudsimilarity import cka, gulp, procrustes, pwcca, gw_sim, tas

import numpy as np

def test_procrustes_1():
    pc1 = np.random.randn(10, 3)
    pc2 = pc1 + 1.0
    sim = procrustes.ProcrustesSimilarity()
    dist = sim(pc1, pc2)
    assert dist >=0.9

def test_procrustes_2():
    pc1 = np.random.randn(100, 20)
    pc2 = np.random.randn(100, 20)
    sim = procrustes.ProcrustesSimilarity()
    dist = sim(pc1, pc2)
    assert dist < 0.2
    
def test_cka_linear_1():
    pc1 = np.random.randn(10, 5)
    pc2 = pc1 + 0.01 * np.random.randn(10, 5)
    sim = cka.CKASimilarity(kernel='linear')
    similarity = sim(pc1, pc2)
    assert similarity > 0.9
    
def test_cka_linear_2():
    pc1 = np.random.randn(12, 3)
    pc2 = np.random.randn(12, 3)
    sim = cka.CKASimilarity(kernel='linear')
    similarity = sim(pc1, pc2)
    assert similarity < 0.9

def test_cka_rbf_1():
    pc1 = np.random.randn(15, 4)
    pc2 = pc1 + 0.05 * np.random.randn(15, 4)
    sim = cka.CKASimilarity(kernel='rbf')
    similarity = sim(pc1, pc2)
    assert similarity > 0.8
    
def test_cka_rbf_2():
    pc1 = np.random.randn(20, 6)
    pc2 = np.random.randn(20, 6)
    sim = cka.CKASimilarity(kernel='rbf')
    similarity = sim(pc1, pc2)
    assert similarity < 0.8
    
def test_gulp_1():
    pc1 = np.random.randn(20, 6)
    pc2 = pc1
    sim = gulp.GULPSimilarity(lambda_=0.1)
    similarity = sim(pc1, pc2)
    assert similarity > 0.5 
    
def test_gulp_2():
    pc1 = np.random.randn(25, 8)
    pc2 = np.random.randn(25, 8)
    sim = gulp.GULPSimilarity(lambda_=0.1)
    similarity = sim(pc1, pc2)
    assert similarity < 1e-5
    
def test_gulp_3():
    pc1 = np.random.randn(30, 10)
    pc2 = pc1 + 10 * np.random.randn(30, 10) + np.tanh(10*pc1)
    sim = gulp.GULPSimilarity(lambda_=0.1)
    similarity = sim(pc1, pc2)
    assert similarity < 1e-3
    

def test_pwcca_1():
    pc1 = np.random.randn(15, 5)
    pc2 = pc1 + 0.01 * np.random.randn(15, 5)
    sim = pwcca.PWCCASimilarity()
    similarity = sim(pc1, pc2)
    assert similarity > 0.9
    
def test_pwcca_2():
    pc1 = np.random.randn(18, 7)
    pc2 = np.random.randn(18, 7)
    sim = pwcca.PWCCASimilarity()
    similarity = sim(pc1, pc2)
    assert similarity < 0.9
    
def test_nngs_1():
    pc1 = np.random.randn(100, 4)
    pc2 = pc1 + 0.01 * np.random.randn(100, 4)
    sim = tas.TASSimilarityCPU(k=5)
    similarity = sim(pc1, pc2)
    assert similarity > 0.8
    
def test_nngs_2():
    pc1 = np.random.randn(100, 6)
    pc2 = np.random.randn(100, 6)
    sim = tas.TASSimilarityCPU(k=5)
    similarity = sim(pc1, pc2)
    assert similarity < 0.5
    
# def test_gw_1():
#     pc1 = np.random.randn(10, 3)
#     pc2 = pc1 + 0.01 * np.random.randn(10, 3)
#     sim = gw_sim.gw_sim(pc1, pc2)
#     assert sim > 50.0  # High similarity
    
# def test_gw_2():
#     pc1 = np.random.randn(10, 3)
#     pc2 = np.random.randn(10, 3)
#     sim = gw_sim.gw_sim(pc1, pc2)
#     assert sim < 50.0  # Low similarity