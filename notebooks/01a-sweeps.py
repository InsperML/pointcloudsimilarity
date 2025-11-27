import time
from typing import DefaultDict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from tqdm import tqdm

import pointcloudsimilarity.similarities as pcsim
from pointcloudsimilarity.similarities import (CKASimilarity, GULPSimilarity,
                                               GWSimilarity,
                                               NNGSSimilarityTorch,
                                               ProcrustesSimilarity,
                                               PWCCASimilarity)


def noisy_blobs(N, D, n_blobs=2, distance=20.0, cluster_std=0.1):

    centers = np.random.uniform(-distance, distance, size=(n_blobs, D))

    X, _ = make_blobs(n_samples=N * len(centers), 
                      n_features=centers.shape[1], 
                      centers=centers, 
                      cluster_std=cluster_std, 
                      random_state=42)

    # Noisy blobs
    Y = X + np.random.randn(*X.shape) * (cluster_std * 2.0)
    return X, Y

def shuffled_blobs(N, D, n_blobs=2, distance=20.0, cluster_std=0.1):

    centers = np.random.uniform(-distance, distance, size=(n_blobs, D))

    X, _ = make_blobs(n_samples=N * len(centers), 
                      n_features=centers.shape[1], 
                      centers=centers, 
                      cluster_std=cluster_std, 
                      random_state=42)


    perm = np.random.permutation(len(centers))
    centers_shuffled = centers[perm]
    
    Y, _ = make_blobs(n_samples=N * len(centers_shuffled), 
                      n_features=centers_shuffled.shape[1], 
                      centers=centers_shuffled, 
                      cluster_std=cluster_std, 
                      random_state=42)
    return X, Y
    
def sweep_model_similarity(X, Y):
    alphas = np.logspace(-1, 5, 50)
    similarities_cka = []
    for alpha in tqdm(alphas):
        metric = CKASimilarity(kernel='rbf', scale_by_alpha=alpha)
        #print(f"Calculating layerwise similarities using {metric.__class__.__name__}...")
        #t0 = time.perf_counter()
        sim = metric(X, Y)
        #t1 = time.perf_counter()
        #print(f"Time taken: {t1 - t0:.2f} seconds.")
        #print(f"Calculated layerwise similarities using {metric.__class__.__name__}.")
        similarities_cka.append(sim)

    ks = np.arange(1, X.shape[0]-1, 1)
    similarities_nngs = []
    for k in tqdm(ks):
        metric = NNGSSimilarityTorch(k=k, batch_size=X.shape[0], normalize=True)
        #print(f"Calculating layerwise similarities using {metric.__class__.__name__}...")
        #t0 = time.perf_counter()
        sim = metric(X, Y)
        #t1 = time.perf_counter()
        #print(f"Time taken: {t1 - t0:.2f} seconds.")
        #print(f"Calculated layerwise similarities using {metric.__class__.__name__}.")
        similarities_nngs.append(sim)

    return alphas, similarities_cka, ks, similarities_nngs


def blob_experiment(N, D, n_blobs=1, distance=2.0):
    X, Y = noisy_blobs(N, D, n_blobs=n_blobs, distance=distance)
    alphas, similarities_cka, ks, similarities_nngs = sweep_model_similarity(X, Y)
    
    X, Y = shuffled_blobs(N, D, n_blobs=n_blobs, distance=distance)
    alphas_shuf, similarities_cka_shuf, ks_shuf, similarities_nngs_shuf = sweep_model_similarity(X, Y)
     

    plt.figure(figsize=(8, 5))
    plt.subplot(2,1,1)
    plt.plot(alphas, np.array(similarities_cka), label='Noisy blobs')
    plt.plot(alphas_shuf, np.array(similarities_cka_shuf), label='Shuffled blobs')
    plt.xscale('log')
    plt.xlabel('Alpha (scaling factor for sigma)')
    plt.ylabel('RBF-CKA')
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
 

    plt.subplot(2,1,2)
    plt.plot(ks, np.array(similarities_nngs), label='Noisy blobs')  
    plt.plot(ks_shuf, np.array(similarities_nngs_shuf), label='Shuffled blobs')
    plt.xlabel('K (neighborhood size)')
    plt.ylabel('NNGS')
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
    
    plt.suptitle(f'Point Cloud Similarity Sweep\n(points_per_blob={N}, D={D}, n_blobs={n_blobs}, distance={distance})')
    plt.tight_layout()
    plt.savefig(figname:=f'sweep_experiment_{N}_{D}_{n_blobs}_{distance}.png')
    plt.close()



def main():
    blob_experiment(100, 20, n_blobs=5, distance=10.0)
    blob_experiment(200, 50, n_blobs=3, distance=5.0)
    blob_experiment(200, 100, n_blobs=3, distance=5.0)
    blob_experiment(100, 720, n_blobs=5, distance=5.0)
        
if __name__ == "__main__":
    main()
    
