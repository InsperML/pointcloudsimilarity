from pointcloudsimilarity.similarities import (
    CKASimilarity,
    GULPSimilarity,
    ProcrustesSimilarity,
    GWSimilarity,
    NNGSSimilarityTorch,
    PWCCASimilarity,
)

import toml
from pathlib import Path
import numpy as np
from typing import DefaultDict
from tqdm import tqdm
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from itertools import product

script_dir = Path(__file__).parent
config = toml.load(script_dir / "settings.toml")['figures']


N = 100
D = 10
#distribution, distribution_name = lambda : np.random.uniform(-1, 1, size=(N, D)), "uniform"
#distribution = lambda : np.random.exponential(scale=1.0, size=(N, D))
#distribution, distribution_name = lambda : np.random.normal(size=(N, D)), "normal"
distribution, distribution_name = lambda : np.random.standard_cauchy(size=(N, D)), "cauchy"
distribution, distribution_name = lambda : make_blobs(n_samples=N, n_features=D, centers=2,)[0], "blobs"

def sweep_distribution_alpha(distribution, distribution_name, N, D):
    X = distribution()

    ks = np.arange(1, N - 1, 10)
    alphas = [0.7]#np.linspace(0.0, 1.0, 10, endpoint=True)
    num_runs = 10
    results = np.zeros( (len(alphas), len(ks), num_runs) )

    for idx_alpha, idx_k, idx_run in tqdm(product(range(len(alphas)), range(len(ks)), range(num_runs))):
        nngs = NNGSSimilarityTorch(
            k=ks[idx_k],
            normalize=True,
        )
        
        phi = distribution()
        Y = alphas[idx_alpha] * X + (1-alphas[idx_alpha]) * phi
        sim = nngs(X, Y)
        results[idx_alpha, idx_k, idx_run] = sim
        

    plt.figure(figsize=(config['width'], config['height']))
    for idx_alpha in range(len(alphas)):
        sims = results[idx_alpha].mean(axis=1)
        sims_std = results[idx_alpha].std(axis=1)
        #plt.errorbar(ks, sims, yerr=sims_std, label=f"alpha={alpha:.2f}", alpha=0.7)
        plt.plot(ks, sims, label=f"alpha={alphas[idx_alpha] :.2f}", alpha=0.7)
        plt.fill_between(ks, 
                        np.array(sims) - np.array(sims_std),
                            np.array(sims) + np.array(sims_std), alpha=0.2)
    #plt.plot(ks, np.array(ks)**2 / N**2, 'k:', label='Expected value')
    plt.xlabel("$k$")
    plt.ylabel("$NNGS(X, Y, k)$")
    #plt.semilogy()
    plt.ylim(0,1)

    #plt.semilogx()

        #plt.title("NNGS Similarity under Increasing Noise for Various k")
    #plt.legend()
        
    plt.tight_layout()
    plt.savefig(script_dir / config['output_dir'] / f"whitenoise_sweep_k_{distribution_name}.png", dpi=300, bbox_inches='tight')

    plt.figure(figsize=(config['width'], config['height']))
    for idx_ks in range(len(ks)):
        sims = results[:,idx_ks].mean(axis=1)
        sims_std = results[:,idx_ks].std(axis=1)
        #plt.errorbar(ks, sims, yerr=sims_std, label=f"alpha={alpha:.2f}", alpha=0.7)
        plt.plot(alphas, sims, label=f"k={ks[idx_ks] :.2f}", alpha=0.7)
        plt.fill_between(alphas, 
                        np.array(sims) - np.array(sims_std),
                            np.array(sims) + np.array(sims_std), alpha=0.2)
    #plt.plot(ks, np.array(ks)**2 / N**2, 'k:', label='Expected value')
    plt.xlabel("$\\alpha$")
    plt.ylabel("$NNGS(X, Y, k)$")
        #plt.title("NNGS Similarity under Increasing Noise for Various k")
    plt.ylim(0,1)
    #plt.legend()
        
    plt.tight_layout()
    plt.savefig(script_dir / config['output_dir'] / f"whitenoise_sweep_alpha_{distribution_name}.png", dpi=300, bbox_inches='tight')
    
if __name__ == "__main__":
    N = 100
    D = 20
    for distribution, distribution_name in [
        (lambda : np.random.uniform(-1, 1, size=(N, D)), "uniform"),
        (lambda : np.random.exponential(scale=1.0, size=(N, D)), "exponential"),
        (lambda : np.random.normal(size=(N, D)), "normal"),
        (lambda : np.random.standard_cauchy(size=(N, D)), "cauchy"),
        (lambda : make_blobs(n_samples=N, n_features=D, centers=2,)[0], "blobs"),
    ]:
        sweep_distribution_alpha(distribution, distribution_name, N, D)


