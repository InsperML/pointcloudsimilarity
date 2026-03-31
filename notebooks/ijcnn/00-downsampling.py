from pathlib import Path
from pprint import pprint
from typing import DefaultDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from pointcloudsimilarity.similarities import (
    CKASimilarity,
    GULPSimilarity,
    GWSimilarity,
    TASSimilarityTorch,
    ProcrustesSimilarity,
    PWCCASimilarity,
    RTDSimilarity
)

script_dir = Path(__file__).parent
config = toml.load(script_dir / 'settings.toml')['figures']


def blobs_with_noise(N, D, centers, blob_std, noise_std):
    X, _ = make_blobs(
        n_samples=N,
        n_features=D,
        centers=centers,
        cluster_std=blob_std,
    )
    noise = np.random.normal(scale=noise_std, size=(N, D))
    Y = X + noise
    return X, Y


def nngs_sweeep(X, Y):
    ks = np.arange(1, X.shape[0] - 1, 1)
    results = []
    for k in ks:
        nngs = TASSimilarityTorch(
            k=k,
            normalize=True,
            batch_size=200,
        )
        sim = nngs(X, Y)
        results.append(sim)
    return results, np.array(ks) / X.shape[0]

if __name__ == '__main__':
    X, Y = blobs_with_noise(
        N=1000,
        D=100,
        centers=5,
        blob_std=0.5,
        noise_std=0.5,
    )
    
    nngs_blobs, ks = nngs_sweeep(X, Y)
    nngs_blobs2, ks2 = nngs_sweeep(X[:500], Y[:500])
    nngs_blobs10, ks10 = nngs_sweeep(X[:100], Y[:100])

    plt.figure(figsize=(config['width'], config['height']//2))
    experiment1 = 'Local Jitter'
    experiment2 = 'Shuffled Centers'
    experiment3 = 'Random Noise'
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.plot(ks, nngs_blobs, label=f'n=1000', color=colors[0])
    plt.plot(ks2, nngs_blobs2, label=f'n=500', color=colors[1])
    plt.plot(ks10, nngs_blobs10, label=f'n=100', color=colors[2])
    #plt.xticks(ticks=np.arange(0, 1.1, 0.1), labels=np.round(np.linspace(0,1,11), 2))

    # plt.plot(ks, nngs_distortion, label="Distortion", alpha=0.7)

    # plt.plot(ks, np.array(ks)**2 / N**2, 'k:', label='Expected value')
    plt.xlabel('$\\alpha$')
    # plt.semilogy()
    plt.ylabel('TA$(X, Y, \\alpha)$')
    # plt.title("NNGS Similarity under Increasing Noise for Various k")

    # plt.legend(bbox_to_anchor=(0.5, -0.30), loc='upper center', ncol=4,
    #            fontsize='small')
    plt.legend(
        #bbox_to_anchor=(1.05, 0.5),  # place legend just outside the right edge
        #loc='center left',  # anchor the left side of the legend box
        ncol=3,  # usually one column looks better on the side
        # fontsize='small'
    )

    # plt.tight_layout()
    plt.ylim(-0.1, 1.1)
    plt.savefig(
        script_dir / config['output_dir'] / 'demo_downsampling.pdf',
        dpi=300,
        bbox_inches='tight',
    )
