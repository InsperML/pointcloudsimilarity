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

script_dir = Path(__file__).parent
config = toml.load(script_dir / "settings.toml")['figures']

distribution = lambda: np.random.uniform(-1, 1, size=(N, D))
#distribution = lambda : np.random.exponential(scale=1.0, size=(N, D))
distribution, distribution_name = lambda: np.random.normal(size=(N, D)
                                                           ), "normal"
distribution, distribution_name = lambda: np.random.standard_cauchy(size=(N, D)
                                                                    ), "cauchy"


def sweep_distribution(distribution, distribution_name, N, D):
    X = distribution()

    ks = np.arange(1, N - 1, 1)
    results = {}
    for alpha in [0.7]:#tqdm(np.linspace(0.0, 1.0, 10, endpoint=True)):
        nngs_sims = []
        nngs_sims_std = []
        for k in ks:
            nngs = NNGSSimilarityTorch(
                k=k,
                normalize=True,
            )

            this_sims = []
            for i in range(10):
                phi = distribution()
                Y = alpha * X + (1 - alpha) * phi
                sim = nngs(X, Y)
                this_sims.append(sim)

            sim = np.mean(this_sims)
            std_sim = np.std(this_sims)
            nngs_sims.append(sim)
            nngs_sims_std.append(std_sim)

        results[alpha] = (nngs_sims, nngs_sims_std)

    plt.figure(figsize=(config['width'], config['height']))
    for alpha in results.keys():
        sims, sims_std = results[alpha]
        #plt.errorbar(ks, sims, yerr=sims_std, label=f"alpha={alpha:.2f}", alpha=0.7)
        plt.plot(ks, sims, label=f"alpha={alpha:.2f}", alpha=0.7)
        plt.fill_between(ks,
                         np.array(sims) - np.array(sims_std),
                         np.array(sims) + np.array(sims_std),
                         alpha=0.2)
    #plt.plot(ks, np.array(ks)**2 / N**2, 'k:', label='Expected value')
    plt.xlabel("$k$")
    #plt.semilogy()
    plt.ylabel("$NNGS(X, Y, k)$")
    #plt.title("NNGS Similarity under Increasing Noise for Various k")
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        script_dir / config['output_dir'] /
        f"whitenoise_sweep_{distribution_name}.png",
        dpi=300,
        bbox_inches='tight',
    )


if __name__ == "__main__":
    N = 100
    D = 20
    sweep_distribution(lambda: np.random.uniform(-1, 1, size=(N, D)),
                       "uniform")
    sweep_distribution(lambda: np.random.exponential(scale=1.0, size=(N, D)),
                       "exponential")
    sweep_distribution(lambda: np.random.normal(size=(N, D)), "normal")
    sweep_distribution(lambda: np.random.standard_cauchy(size=(N, D)),
                       "cauchy")
    sweep_distribution(
        lambda: make_blobs(
            n_samples=N,
            n_features=D,
            centers=2,
        )[0], "blobs")
