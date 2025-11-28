import gensim.downloader as api
import pickle
import torch
from pointcloudsimilarity.similarities import (CKASimilarity, GULPSimilarity,
                                               GWSimilarity,
                                               NNGSSimilarityTorch,
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
    similarities_nngs = []
    for k in tqdm(ks):
        metric = NNGSSimilarityTorch(k=k, batch_size=100, normalize=True)
        #print(f"Calculating layerwise similarities using {metric.__class__.__name__}...")
        #t0 = time.perf_counter()
        sim = metric(torch.Tensor(X).cuda(), torch.Tensor(Y).cuda())
        #t1 = time.perf_counter()
        #print(f"Time taken: {t1 - t0:.2f} seconds.")
        #print(f"Calculated layerwise similarities using {metric.__class__.__name__}.")
        similarities_nngs.append(sim)

    return alphas, similarities_cka, ks, similarities_nngs


def glove_extract():
    with open("glove_models.pkl", "rb") as f:
        models = pickle.load(f)
    model200 = models["model200"]
    model100 = models["model100"]
    model50 = models["model50"]
    model25 = models["model25"]
    print("Models loaded from glove_models.pkl")

    model100.sort_by_descending_frequency()
    all_vocab = list(model100.key_to_index)
    #all_vocab = [ w for w in all_vocab if w.isalnum() ]
    all_vocab = [w for w in all_vocab if w.isascii()]

    for w in all_vocab[190000:191000]:
        v200 = model200[w]
        v100 = model100[w]
        v50 = model50[w]
        v25 = model25[w]
        yield w, v200, v100, v50, v25


def glove_experiment_noise():
    w, v200, v100, v50, v25 = zip(*glove_extract())
    # Normalize each sample (row) to unit L2 norm
    alpha = 0.1
    X200 = torch.Tensor(np.array(v200))
    X100 = torch.Tensor(np.array(v100))
    X50 = torch.Tensor(np.array(v50))
    X25 = torch.Tensor(np.array(v25))
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
    w, v200, v100, v50, v25 = zip(*glove_extract())
    # Normalize each sample (row) to unit L2 norm

    X = torch.tensor(v25)

    nngs = {}
    alphas = np.linspace(0, 1, 100)
    ks = [1, 3, 10, 30, 100, 300]
    for alpha in tqdm(alphas):
        for k in ks:
            Y = X * (1 - alpha) + torch.randn_like(X) * alpha
            metric = NNGSSimilarityTorch(k=k, batch_size=100, normalize=True)
            sim = metric(torch.Tensor(X).cuda(), torch.Tensor(Y).cuda())
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
    w, v200, v100, v50, v25 = zip(*glove_extract())
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
