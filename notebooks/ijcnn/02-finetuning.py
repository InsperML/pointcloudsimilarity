import time
from typing import DefaultDict
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import BitsAndBytesConfig
import lib_get_embeddings as lib_emb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import torch
import pandas as pd

import pointcloudsimilarity.similarities as pcsim
from pointcloudsimilarity.similarities import (CKASimilarity, GULPSimilarity,
                                               GWSimilarity,
                                               NNGSSimilarityTorch,
                                               ProcrustesSimilarity,
                                               PWCCASimilarity, RTDSimilarity)
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sentence_transformers import SentenceTransformer

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from pathlib import Path
import toml

try:
    # BitsAndBytesConfig appears in recent transformers versions
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None
    _HAS_BNB = False

import lib_get_embeddings

script_dir = Path(__file__).parent
config = toml.load(script_dir / "settings.toml")['figures']

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
    #ks = np.arange(1, 500, 1)
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


def get_dataset(dataset_name,
                split="train",
                column_X="text",
                column_Y="label",
                n_samples=1000):
    print(f"Loading dataset {dataset_name}...")
    ds = load_dataset(dataset_name, split=split, streaming=False)
    print(f"Downloaded dataset {dataset_name} with {len(ds)} samples.")
    texts = [article[column_X] for article in ds]
    ans = [article[column_Y] for article in ds]

    rng = np.random.default_rng(12345)
    print(f"Loaded {len(texts)} samples.")
    if len(texts) > n_samples:
        idx = rng.choice(len(texts), size=n_samples, replace=False)
        texts = [texts[i] for i in idx]
        ans = [ans[i] for i in idx]
    else:
        texts = texts[:n_samples]
        ans = ans[:n_samples]
    return texts, ans


def similarities_model_vs_finetuning(dataset='imdb',
                                     column_X="sentence",
                                     column_Y="label",
                                     n_samples=500,
                                     split="train",
                                     n_labels=2):
    N_SAMPLES = n_samples
    max_samples = N_SAMPLES
    print("Loading dataset...")
    #ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    texts, labels = get_dataset(dataset,
                                column_X=column_X,
                                column_Y=column_Y,
                                n_samples=N_SAMPLES,
                                split=split)
    dataset_name = dataset.split('/')[-1].replace('_', '')
    print(f"Loaded {len(texts)} samples.")

    opt_k = int(len(texts) / n_labels)
    print(opt_k)
    similarities = {
        'CKA Linear':
        CKASimilarity(kernel='linear'),
        'CKA RBF':
        CKASimilarity(kernel='rbf'),
        'GULP':
        GULPSimilarity(),
        'Procrustes':
        ProcrustesSimilarity(),
        'GW': GWSimilarity(),
        'PWCCA':
        PWCCASimilarity(),
        'RTD': RTDSimilarity(),
        'NNGS ($k=10$)':
        NNGSSimilarityTorch(k=10, normalize=True, batch_size=150),
        f'NNGS ($k={opt_k}$)':
        NNGSSimilarityTorch(k=opt_k, normalize=True, batch_size=150),
    }

    print("Getting embeddings...")
    emb_bert0 = lib_emb.get_bert_embeddings("google/multiberts-seed_0", texts)
    emb_bert1 = lib_emb.get_bert_embeddings("google/multiberts-seed_1", texts)

    # emb_bert0_ft_unsup0 = lib_emb.get_bert_embeddings(
    #     f"tiagoft/multiberts-seed_0_{dataset_name}_mlm_nsp_finetuned", texts)
    # emb_bert0_ft_unsup1 = lib_emb.get_bert_embeddings(
    #     f"tiagoft/multiberts-seed_1_{dataset_name}_mlm_nsp_finetuned", texts)

    emb_bert_finetuned0 = lib_emb.get_finetuned_bert_embeddings(
        f"tiagoft/multiberts-seed_0_{dataset_name}_finetuned",
        "google/multiberts-seed_0", texts)

    emb_bert_finetuned1 = lib_emb.get_finetuned_bert_embeddings(
        f"tiagoft/multiberts-seed_1_{dataset_name}_finetuned",
        "google/multiberts-seed_1", texts)

    emb_bert0 = F.normalize(emb_bert0, p=2, dim=1, eps=1e-12)
    emb_bert1 = F.normalize(emb_bert1, p=2, dim=1, eps=1e-12)
    # emb_bert0_ft_unsup0 = F.normalize(emb_bert0_ft_unsup0,
    #                                   p=2,
    #                                   dim=1,
    #                                   eps=1e-12)
    # emb_bert0_ft_unsup1 = F.normalize(emb_bert0_ft_unsup1,
    #                                   p=2,
    #                                   dim=1,
    #                                   eps=1e-12)
    emb_bert_finetuned0 = F.normalize(emb_bert_finetuned0,
                                      p=2,
                                      dim=1,
                                      eps=1e-12)
    emb_bert_finetuned1 = F.normalize(emb_bert_finetuned1,
                                      p=2,
                                      dim=1,
                                      eps=1e-12)

    alphas_bert_finetuned0, similarities_cka_bert_finetuned0, ks_bert_finetuned0, similarities_nngs_bert_finetuned0 = sweep_model_similarity(
        emb_bert0.detach().cpu().numpy(),
        emb_bert_finetuned0.detach().cpu().numpy(),
    )
    alphas_bert_inits, similarities_cka_bert_inits, ks_bert_inits, similarities_nngs_bert_inits = sweep_model_similarity(
        emb_bert0.detach().cpu().numpy(),
        emb_bert1.detach().cpu().numpy(),
    )

    alphas_bert_finetuned, similarities_cka_bert_finetuned, ks_bert_finetuned, similarities_nngs_bert_finetuned = sweep_model_similarity(
        emb_bert_finetuned0.detach().cpu().numpy(),
        emb_bert_finetuned1.detach().cpu().numpy(),
    )
    # alphas_bert_finetuned_unsup, similarities_cka_bert_finetuned_unsup, ks_bert_finetuned_unsup, similarities_nngs_bert_finetuned_unsup = sweep_model_similarity(
    #     emb_bert0_ft_unsup0.detach().cpu().numpy(),
    #     emb_bert0_ft_unsup1.detach().cpu().numpy(),
    # )

    plt.figure(figsize=(config['width'], config['height']))
    plt.plot(ks_bert_finetuned,
             similarities_nngs_bert_finetuned,
             label="FT vs. FT")
    plt.plot(ks_bert_inits,
             similarities_nngs_bert_inits,
             label="PT vs. PT",
             #linestyle='dashed',
             )
    plt.plot(ks_bert_finetuned0,
             similarities_nngs_bert_finetuned0,
             label="PT vs. FT")
    # plt.plot(ks_bert_finetuned_unsup,
    #          similarities_nngs_bert_finetuned_unsup,
    #          label="FT unsup. vs. FT unsup.",
    #          linestyle='dotted')
    plt.xlabel("$k$")
    plt.ylabel("$NNGS(X, Y, k)$")

    plt.ylim(0, 1)
    plt.legend(
        loc="upper center",  # position relative to bbox
        bbox_to_anchor=(0.5, -0.3),  # center it below the axes
        ncol=3,
        #fontsize="xx-small",  # number of columns
    )

    plt.tight_layout()
    plt.savefig(script_dir / config['output_dir'] /
                f"finetuning_{dataset_name}_{split}.pdf",
                dpi=300,
                bbox_inches='tight')

    sim_pt_pt = lib_get_embeddings.calculate_all_similaritiees(
        emb_bert0.detach().cpu().numpy(),
        emb_bert1.detach().cpu().numpy(), similarities)
    sim_pt_ft = lib_get_embeddings.calculate_all_similaritiees(
        emb_bert0.detach().cpu().numpy(),
        emb_bert_finetuned0.detach().cpu().numpy(), similarities)
    sim_ft_ft = lib_get_embeddings.calculate_all_similaritiees(
        emb_bert_finetuned0.detach().cpu().numpy(),
        emb_bert_finetuned1.detach().cpu().numpy(), similarities)
    # sim_ft_unsup_ft_unsup = lib_get_embeddings.calculate_all_similaritiees(
    #     emb_bert0_ft_unsup0.detach().cpu().numpy(),
    #     emb_bert0_ft_unsup1.detach().cpu().numpy(), similarities)

    df_pt_seeds = pd.DataFrame({
        'FT vs. FT':
        sim_ft_ft,
        'PT vs. PT':
        sim_pt_pt,
        'PT vs. FT':
        sim_pt_ft,

        # 'FT unsup. vs. FT unsup.':
        # sim_ft_unsup_ft_unsup,
    })
    print(df_pt_seeds.round(2).to_latex(float_format="%.2f"))


def main():

    similarities_model_vs_finetuning(
        dataset='sst2',
        column_X="sentence",
        column_Y="label",
        n_samples=2000,
        split="test",
    )
    similarities_model_vs_finetuning(
        dataset='imdb',
        column_X="text",
        column_Y="label",
        n_samples=2000,
        split="test",
    )
    similarities_model_vs_finetuning(
        dataset='fancyzhx/ag_news',
        column_X="text",
        column_Y="label",
        n_samples=2000,
        split="test",
        n_labels=4,
    )


if __name__ == "__main__":
    main()
