from pathlib import Path
from pprint import pprint
from typing import DefaultDict

import lib_get_embeddings
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


def shuffled_blobs(N, D, centers, blob_std, random_state=None):
    if not random_state:
        random_state = np.random.randint(0, 10000)

    if type(centers) == int:
        cx = np.random.normal(0, 5, size=(centers, D))
        n_centers = centers
    else:
        cx = centers
        n_centers = centers.shape[0]

    np.random.seed(random_state)
    cy = np.random.permutation(cx)
    samples_per_center = N // n_centers
    assert samples_per_center * n_centers == N, (
        'N must be divisible by number of centers'
    )

    X = []
    Y = []

    for cx_, cy_ in zip(cx, cy):
        blob = np.random.normal(loc=0, scale=blob_std, size=(samples_per_center, D))
        X.append(blob + cx_)
        Y.append(blob + cy_)

    X = np.vstack(X)
    Y = np.vstack(Y)
    return X, Y


def distortion(N, D, distortion_strength, distortion_coefficient=2.0):
    X = np.random.normal(size=(N, D))
    phi = np.abs(X) ** distortion_coefficient
    Y = X + phi * X / np.linalg.norm(X, axis=1, keepdims=True) * distortion_strength
    return X, Y


def nngs_sweeep(X, Y):
    ks = np.arange(1, X.shape[0] - 1, 1)
    results = []
    for k in ks:
        nngs = TASSimilarityTorch(
            k=k,
            normalize=True,
        )
        sim = nngs(X, Y)
        results.append(sim)
    return results, ks


def rbf_cka_seep(X, Y):
    sigmas = np.logspace(-2, 2, 20)
    results = []
    for sigma in sigmas:
        cka = CKASimilarity(kernel='rbf', sigma=sigma)
        sim = cka(X, Y)
        results.append(sim)
    return results, sigmas


def run_all_sweeps(N, D):
    np.random.seed(1)

    centers = np.random.normal(0, 5, size=(4, D))
    centers = np.sign(centers)

    # Local Jitter
    X, Y = blobs_with_noise(N, D, centers, 0.1, 0.1)
    sim_blobs = lib_get_embeddings.calculate_all_similaritiees(
        X,
        Y,
        lib_get_embeddings.similarities,
    )
    nngs_blobs, _ = nngs_sweeep(X, Y)

    # X, Y = distortion(N, D, 50, 40)
    # sim_distortion = calculate_all_similaritiees(X, Y, similarities)
    # nngs_distortion, _ = nngs_sweeep(X, Y)

    # Shuffled Centers
    X, Y = shuffled_blobs(N, D, centers, 0.1, random_state=1)
    sim_shuffled = lib_get_embeddings.calculate_all_similaritiees(
        X,
        Y,
        lib_get_embeddings.similarities,
    )
    nngs_shuffled, ks = nngs_sweeep(X, Y)

    # Random Noise
    X, Y = np.random.normal(size=(N, D)), np.random.normal(size=(N, D))
    sim_noise = lib_get_embeddings.calculate_all_similaritiees(
        X,
        Y,
        lib_get_embeddings.similarities,
    )
    nngs_noise, ks = nngs_sweeep(X, Y)

    return (
        ks,
        nngs_blobs,
        nngs_shuffled,
        nngs_noise,
        sim_blobs,
        sim_shuffled,
        sim_noise,
    )


if __name__ == '__main__':
    (
        ks5,
        nngs_blobs5,
        nngs_shuffled5,
        nngs_noise5,
        sim_blobs5,
        sim_shuffled5,
        sim_noise5,
    ) = run_all_sweeps(N=500, D=10)
    (
        ks50,
        nngs_blobs50,
        nngs_shuffled50,
        nngs_noise50,
        sim_blobs50,
        sim_shuffled50,
        sim_noise50,
    ) = run_all_sweeps(N=500, D=50)
    (
        ks100,
        nngs_blobs100,
        nngs_shuffled100,
        nngs_noise100,
        sim_blobs100,
        sim_shuffled100,
        sim_noise100,
    ) = run_all_sweeps(N=500, D=100)
    (
        ks2000,
        nngs_blobs2000,
        nngs_shuffled2000,
        nngs_noise2000,
        sim_blobs2000,
        sim_shuffled2000,
        sim_noise2000,
    ) = run_all_sweeps(N=500, D=2000)

    plt.figure(figsize=(config['width'], config['height']))
    experiment1 = 'Local Jitter'
    experiment2 = 'Shuffled Centers'
    experiment3 = 'Random Noise'
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.plot(ks5, nngs_blobs5, color=colors[0], label=f'{experiment1}, D=10', alpha=0.7)
    plt.plot(
        ks50,
        nngs_blobs50,
        color=colors[0],
        linestyle='--',
        label=f'{experiment1}, D=50',
        alpha=0.7,
    )
    plt.plot(
        ks100,
        nngs_blobs100,
        color=colors[0],
        linestyle='-.',
        label=f'{experiment1}, D=100',
        alpha=0.7,
    )
    plt.plot(
        ks2000,
        nngs_blobs2000,
        color=colors[0],
        linestyle=':',
        label=f'{experiment1}, D=2000',
        alpha=0.7,
    )
    plt.plot(
        ks5,
        nngs_shuffled5,
        color=colors[1],
        label=f'{experiment2}, D=10',
        alpha=0.7,
    )
    plt.plot(
        ks50,
        nngs_shuffled50,
        color=colors[1],
        linestyle='--',
        label=f'{experiment2}, D=50',
        alpha=0.7,
    )
    plt.plot(
        ks100,
        nngs_shuffled100,
        color=colors[1],
        linestyle='-.',
        label=f'{experiment2}, D=100',
        alpha=0.7,
    )
    plt.plot(
        ks2000,
        nngs_shuffled2000,
        color=colors[1],
        linestyle=':',
        label=f'{experiment2}, D=2000',
        alpha=0.7,
    )
    plt.plot(
        ks5,
        nngs_noise5,
        color=colors[2],
        label=f'{experiment3}, D=10',
        alpha=0.7,
    )
    plt.plot(
        ks50,
        nngs_noise50,
        color=colors[2],
        linestyle='--',
        label=f'{experiment3}, D=50',
        alpha=0.7,
    )
    plt.plot(
        ks100,
        nngs_noise100,
        color=colors[2],
        linestyle='-.',
        label=f'{experiment3}, D=100',
        alpha=0.7,
    )
    plt.plot(
        ks2000,
        nngs_noise2000,
        color=colors[2],
        linestyle=':',
        label=f'{experiment3}, D=2000',
        alpha=0.7,
    )

    # plt.plot(ks, nngs_distortion, label="Distortion", alpha=0.7)

    # plt.plot(ks, np.array(ks)**2 / N**2, 'k:', label='Expected value')
    plt.xlabel('$k$')
    # plt.semilogy()
    plt.ylabel('TA$(X, Y, k)$')
    # plt.title("NNGS Similarity under Increasing Noise for Various k")

    # plt.legend(bbox_to_anchor=(0.5, -0.30), loc='upper center', ncol=4,
    #            fontsize='small')
    plt.legend(
        bbox_to_anchor=(1.05, 0.5),  # place legend just outside the right edge
        loc='center left',  # anchor the left side of the legend box
        ncol=1,  # usually one column looks better on the side
        # fontsize='small'
    )

    # plt.tight_layout()
    plt.ylim(-0.1, 1.1)
    plt.savefig(
        script_dir / config['output_dir'] / 'toy_distortion_types.pdf',
        dpi=300,
        bbox_inches='tight',
    )

    sim_blobs = {
        '10': sim_blobs5,
        '50': sim_blobs50,
        '100': sim_blobs100,
        '2000': sim_blobs2000,
    }

    sim_shuffled = {
        '10': sim_shuffled5,
        '50': sim_shuffled50,
        '100': sim_shuffled100,
        '2000': sim_shuffled2000,
    }

    sim_noise = {
        '10': sim_noise5,
        '50': sim_noise50,
        '100': sim_noise100,
        '2000': sim_noise2000,
    }

    print('Blobs:')
    df_sim_blobs = pd.DataFrame(sim_blobs)
    print(df_sim_blobs.round(2).to_latex(float_format='%.2f'))

    print('Shuffled:')
    df_sim_shuffled = pd.DataFrame(sim_shuffled)
    print(df_sim_shuffled.round(2).to_latex(float_format='%.2f'))

    print('Noise:')
    df_sim_noise = pd.DataFrame(sim_noise)
    print(df_sim_noise.round(2).to_latex(float_format='%.2f'))
