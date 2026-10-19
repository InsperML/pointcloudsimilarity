import gensim.downloader as api
from pathlib import Path
import torch
from pointcloudsimilarity.datasets import (aligned_glove_vectors,
                                           load_glove_twitter,
                                           mix_with_noise)
from pointcloudsimilarity.similarities import (CKASimilarity, GULPSimilarity,
                                               GWSimilarity,
                                               TASSimilarityTorch,
                                               tas_spectrum,
                                               ProcrustesSimilarity,
                                               PWCCASimilarity)
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# # # info = api.info()  # show info about available models/datasets
# print("Loading model 100")
# model100 = api.load("glove-twitter-100"
#                     )  # download the model and return as object ready for use
# print("Loading model 25")
# model25 = api.load("glove-twitter-25"
#                    )  # download the model and return as object ready for use
# print("Loading model 50")
# model50 = api.load("glove-twitter-50"
#                    )  # download the model and return as object ready for use
# print("Loading model 200")
# model200 = api.load("glove-twitter-200")  # download the model and return
# with open("glove_models.pkl", "wb") as f:
#     pickle.dump(
#         {
#             "model100": model100,
#             "model25": model25,
#             "model50": model50,
#             "model200": model200,
#         }, f)
# print("Models saved to glove_models.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Filled by 00-load_gensim_glove.py, or by the first run of this script.
GLOVE_CACHE = Path(__file__).parent / 'glove_models'


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

    ks = np.arange(1, X.shape[0] - 1, 20)
    ks = np.arange(1, 500, 10)
    similarities_nngs = list(tas_spectrum(X, Y, ks, batch_size=100))

    return alphas, similarities_cka, ks, similarities_nngs


def glove_extract():
    """Words 190000 to 191000 (by frequency in GloVe 100) in every GloVe model."""
    models = {dim: load_glove_twitter(dim, cache_dir=GLOVE_CACHE)
              for dim in (200, 100, 50, 25)}
    print(f"Models loaded from {GLOVE_CACHE}")
    return aligned_glove_vectors(models, start=190000, n_words=1000, reference=100)


def glove_experiment_noise():
    words, vectors = glove_extract()
    # Normalize each sample (row) to unit L2 norm
    alpha = 0.1
    X200 = torch.Tensor(vectors[200])
    X100 = torch.Tensor(vectors[100])
    X50 = torch.Tensor(vectors[50])
    X25 = torch.Tensor(vectors[25])
    #X = F.normalize(torch.tensor(v100), p=2, dim=1, eps=1e-12)

    alphas100, similarities_cka100, ks100, similarities100 = sweep_model_similarity(
        X200.detach().cpu().numpy(),
        X100.detach().cpu().numpy())
    alphas50, similarities_cka50, ks50, similarities50 = sweep_model_similarity(
        X200.detach().cpu().numpy(),
        X50.detach().cpu().numpy())
    alphas25, similarities_cka25, ks25, similarities25 = sweep_model_similarity(
        X200.detach().cpu().numpy(),
        X25.detach().cpu().numpy())

    plt.figure(figsize=(8, 5))
    plt.subplot(2, 1, 1)

    plt.plot(alphas100,
             np.array(similarities_cka100),
             label='GloVe 200 vs GloVe 100')
    plt.plot(alphas50,
             np.array(similarities_cka50),
             label='GloVe 200 vs GloVe 50')
    plt.plot(alphas25,
             np.array(similarities_cka25),
             label='GloVe 200 vs GloVe 25')
    plt.xscale('log')
    plt.xlabel('Alpha (scaling factor for sigma)')
    plt.ylabel('RBF-CKA')
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(ks100, np.array(similarities100), label='GloVe 200 vs GloVe 100')
    plt.plot(ks50, np.array(similarities50), label='GloVe 200 vs GloVe 50')
    plt.plot(ks25, np.array(similarities25), label='GloVe 200 vs GloVe 25')
    plt.xlabel('K (neighborhood size)')
    plt.ylabel('NNGS')
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()

    plt.suptitle(f'Model Similarity Sweep(n_samples={X200.shape[0]})')
    plt.tight_layout()
    plt.savefig(figname := f'glove_similarity_experiment_all_models.png')
    plt.close()


def glove_experiment_noise_sweep():
    words, vectors = glove_extract()
    # Normalize each sample (row) to unit L2 norm

    X = vectors[25]

    nngs = {}
    alphas = np.linspace(0, 1, 100)
    ks = [1, 3, 10, 30, 100, 300]
    for alpha in tqdm(alphas):
        for k in ks:
            Y = mix_with_noise(X, alpha)
            metric = TASSimilarityTorch(k=k, batch_size=100, normalize=True)
            sim = metric(X, Y)
            if k not in nngs:
                nngs[k] = []
            nngs[k].append(sim)

    plt.figure(figsize=(8, 5))
    for k in ks:
        plt.plot(alphas, np.array(nngs[k]), label=f'NNGS k={k}')
    plt.xlabel('Alpha')
    plt.ylabel('NNGS')
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(figname := f'glove_similarity_noise_sweep.png')
    plt.close()


def glove_experiment():
    words, vectors = glove_extract()
    # Normalize each sample (row) to unit L2 norm

    alphas, similarities_cka, ks, similarities = sweep_model_similarity(
        X.detach().cpu().numpy(),
        Y.detach().cpu().numpy())

    plt.figure(figsize=(8, 5))
    plt.subplot(2, 1, 1)

    plt.plot(alphas, np.array(similarities_cka), label='GloVe 25 vs GloVe 100')
    plt.xscale('log')
    plt.xlabel('Alpha (scaling factor for sigma)')
    plt.ylabel('RBF-CKA')
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(ks, np.array(similarities), label='GloVe 25 vs GloVe 100')
    plt.xlabel('K (neighborhood size)')
    plt.ylabel('NNGS')
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()

    plt.suptitle(f'Model Similarity Sweep(n_samples={X.shape[0]})')
    plt.tight_layout()
    plt.savefig(figname := f'glove_similarity_experiment_all_models.png')
    plt.close()


if __name__ == "__main__":
    glove_experiment_noise()
