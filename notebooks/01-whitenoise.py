import pointcloudsimilarity.similarities as pcsim

from pointcloudsimilarity.similarities import (
    CKASimilarity,
    GULPSimilarity,
    ProcrustesSimilarity,
    GWSimilarity,
    NNGSSimilarity,
    PWCCASimilarity,
)

import numpy as np
from typing import DefaultDict
from tqdm import  tqdm
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

my_metrics = {
    "cka_linear": CKASimilarity(),
    "cka_rbf_autosigma": CKASimilarity(kernel='rbf'),
    "cka_rbf_sigma1": CKASimilarity(kernel='rbf', sigma=1.0),
    "cka_rbf_sigma01": CKASimilarity(kernel='rbf', sigma=0.1),
    "gulp": GULPSimilarity(),
    "procrustes": ProcrustesSimilarity(),
    "gw_sim": GWSimilarity(),
    "nngs_k80": NNGSSimilarity(k=80),
    "nngs_k40": NNGSSimilarity(k=40),
    "nngs_k10": NNGSSimilarity(k=2),
    "pwcca": PWCCASimilarity(),
}



def operate_on_point_cloud(X, metrics, c=None):
    similarities = DefaultDict(list)
    N, D = X.shape
    alpha_values = np.arange(0, 1.0, 0.02)
    for i, alpha in enumerate(alpha_values):
        print(f"Noise level: {alpha}")
        
        Y = (1-alpha) * X + alpha * np.random.randn(N, D)
        
        for method in tqdm(metrics.keys()):
            sim = my_metrics[method](X, Y)
            similarities[method].append(sim)
        
        if c is not None:
            score = silhouette_score(Y, c)
            similarities['silhouette'].append(score)
    return similarities, alpha_values

    


def blob_experiment(N, D, n_blobs=1, distance=2.0):
    N = 100  # number of points per blob
    D = 20   # dimensionality
    X, c = make_blobs(n_samples=N * n_blobs, n_features=D, centers=n_blobs, 
                      cluster_std=0.1, center_box=(-distance, distance), random_state=42)
    similarities, alpha_values = operate_on_point_cloud(X, my_metrics, c=c)
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    for method, sims in similarities.items():
        plt.plot(alpha_values, sims, marker='o', label=method)

    plt.title(f'Point Cloud Similarity under Increasing Noise\n(N={N}, D={D}, n_blobs={n_blobs}, distance={distance})')
    plt.xlabel('Noise Level')
    plt.ylabel('Similarity Score')
    plt.legend()
    plt.grid()
    plt.savefig(f'experiment_{N}_{D}_{n_blobs}_{distance}.png')



def main():
    blob_experiment(100, 20, n_blobs=3, distance=2.0)

if __name__ == "__main__":
    main()
    
