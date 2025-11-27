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

from sklearn.datasets import load_digits

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP

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

    ks = np.arange(1, X.shape[0]-1, 10)
    similarities_nngs = []
    for k in tqdm(ks):
        metric = NNGSSimilarityTorch(k=k, batch_size=100, normalize=True)
        #print(f"Calculating layerwise similarities using {metric.__class__.__name__}...")
        #t0 = time.perf_counter()
        sim = metric(X, Y)
        #t1 = time.perf_counter()
        #print(f"Time taken: {t1 - t0:.2f} seconds.")
        #print(f"Calculated layerwise similarities using {metric.__class__.__name__}.")
        similarities_nngs.append(sim)

    return alphas, similarities_cka, ks, similarities_nngs


def dimension_reduction_experiment():
    X, _ = load_digits(return_X_y=True, n_class=2)
    print(X.shape)
    Y = PCA(n_components=2).fit_transform(X)
    print(Y.shape)
    alphas, similarities_cka, ks, similarities_nngs = sweep_model_similarity(X, Y)
    
    Y_umap = UMAP(n_components=2, n_neighbors=15).fit_transform(X)
    print(Y_umap.shape)
    alphas_umap, similarities_cka_umap, ks_umap, similarities_nngs_umap = sweep_model_similarity(X, Y_umap)    

    Y_tsne = TSNE(n_components=2, random_state=42, perplexity=15).fit_transform(X)
    alphas_shuf, similarities_cka_shuf, ks_shuf, similarities_nngs_shuf = sweep_model_similarity(X, Y_tsne)
     

    plt.figure(figsize=(8, 5))
    plt.subplot(2,1,1)
    plt.plot(alphas, np.array(similarities_cka), label='PCA')
    plt.plot(alphas_umap, np.array(similarities_cka_umap), label='UMAP')
    plt.plot(alphas_shuf, np.array(similarities_cka_shuf), label='t-SNE')
    plt.xscale('log')
    plt.xlabel('Alpha (scaling factor for sigma)')
    plt.ylabel('RBF-CKA')
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
 

    plt.subplot(2,1,2)
    plt.plot(ks, np.array(similarities_nngs), label='PCA')  
    plt.plot(ks_umap, np.array(similarities_nngs_umap), label='UMAP')
    plt.plot(ks_shuf, np.array(similarities_nngs_shuf), label='t-SNE')
    plt.xlabel('K (neighborhood size)')
    plt.ylabel('NNGS')
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
    
    plt.suptitle(f'Dimension Reduction Similarity Sweep\nDigits dataset (n_samples={X.shape[0]}, n_features={X.shape[1]})')
    plt.tight_layout()
    plt.savefig(figname:=f'dimension_reduction_experiment_{X.shape[0]}_{X.shape[1]}.png')
    plt.close()



def main():
    dimension_reduction_experiment()
        
if __name__ == "__main__":
    main()
    
