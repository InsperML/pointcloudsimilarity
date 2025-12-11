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

import pointcloudsimilarity.similarities as pcsim
from pointcloudsimilarity.similarities import (CKASimilarity, GULPSimilarity,
                                               GWSimilarity,
                                               NNGSSimilarityTorch,
                                               ProcrustesSimilarity,
                                               PWCCASimilarity)
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

    ks = np.arange(1, X.shape[0]-1, 20)
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



def get_dataset(dataset_name, split="train", column_X="text", column_Y = "label", n_samples=1000):
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

def similarities_model_vs_finetuning(dataset='sst2', column_X="sentence", column_Y = "label", n_samples=500, split="train"):
    N_SAMPLES = n_samples
    max_samples = N_SAMPLES
    print("Loading dataset...")
    #ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    texts, labels = get_dataset(dataset, column_X=column_X, column_Y=column_Y, n_samples=N_SAMPLES, split=split)
    print(f"Loaded {len(texts)} samples.")

    print("Getting embeddings...")
    emb_bert = lib_emb.get_bert_embeddings("google/multiberts-seed_0",
                                    texts)
    emb_bert1 = lib_emb.get_bert_embeddings("google/multiberts-seed_1",
                                    texts)
    
    emb_bert_finetuned0 = lib_emb.get_finetuned_bert_embeddings("tiagoft/multiberts-seed_0_sst2_finetuned", "google/multiberts-seed_0",
                                    texts)
    
    emb_bert_finetuned1 = lib_emb.get_finetuned_bert_embeddings("tiagoft/multiberts-seed_1_sst2_finetuned", "google/multiberts-seed_1",
                                    texts)
    emb_gpt0 = lib_emb.get_gptx_embeddings("EleutherAI/pythia-160m",
                                   1,
                                   texts,
                                   )
    emb_gpt1 = lib_emb.get_gptx_embeddings("EleutherAI/pythia-160m",
                                   2,
                                   texts,
                                   )
    emb_gpt0_ft = lib_emb.get_finetuned_gptx_embeddings("EleutherAI/pythia-160m",
                                                "tiagoft/pythia-160m-seed1_sst2_finetuned",
                                   texts,
                                   )
    emb_gpt1_ft = lib_emb.get_finetuned_gptx_embeddings("EleutherAI/pythia-160m",
                                                "tiagoft/pythia-160m-seed2_sst2_finetuned",
                                   texts,
                                   )


    # Normalize each sample (row) to unit L2 norm
    
    emb_bert = F.normalize(emb_bert, p=2, dim=1, eps=1e-12)
    emb_bert1 = F.normalize(emb_bert1, p=2, dim=1, eps=1e-12)
    emb_bert_finetuned0 = F.normalize(emb_bert_finetuned0, p=2, dim=1, eps=1e-12)
    emb_bert_finetuned1 = F.normalize(emb_bert_finetuned1, p=2, dim=1, eps=1e-12)
    emb_gpt0 = F.normalize(emb_gpt0, p=2, dim=1, eps=1e-12)
    emb_gpt1 = F.normalize(emb_gpt1, p=2, dim=1, eps=1e-12)
    emb_gpt0_ft = F.normalize(emb_gpt0_ft, p=2, dim=1, eps=1e-12)
    emb_gpt1_ft = F.normalize(emb_gpt1_ft, p=2, dim=1, eps=1e-12)

    alphas_bert_finetuned0, similarities_cka_bert_finetuned0, ks_bert_finetuned0, similarities_nngs_bert_finetuned0 = sweep_model_similarity(
        emb_bert.detach().cpu().numpy(),
        emb_bert_finetuned0.detach().cpu().numpy(),)
    alphas_bert_inits, similarities_cka_bert_inits, ks_bert_inits, similarities_nngs_bert_inits = sweep_model_similarity(
        emb_bert.detach().cpu().numpy(),
        emb_bert1.detach().cpu().numpy(),)
    
    alphas_bert_finetuned1, similarities_cka_bert_finetuned1, ks_bert_finetuned1, similarities_nngs_bert_finetuned1 = sweep_model_similarity(
        emb_bert.detach().cpu().numpy(),
        emb_bert_finetuned1.detach().cpu().numpy(),)
    alphas_bert_finetuned, similarities_cka_bert_finetuned, ks_bert_finetuned, similarities_nngs_bert_finetuned = sweep_model_similarity(
        emb_bert_finetuned0.detach().cpu().numpy(),
        emb_bert_finetuned1.detach().cpu().numpy(),)
    alphas_gpt0_ft, similarities_cka_gpt0_ft, ks_gpt0_ft, similarities_nngs_gpt0_ft = sweep_model_similarity(
        emb_gpt0.detach().cpu().numpy(),
        emb_gpt0_ft.detach().cpu().numpy(),)
    alphas_gpt1_ft, similarities_cka_gpt1_ft, ks_gpt1_ft, similarities_nngs_gpt1_ft = sweep_model_similarity(
        emb_gpt0.detach().cpu().numpy(),
        emb_gpt1_ft.detach().cpu().numpy(),)
    alphas_gpt_ft, similarities_cka_gpt_ft, ks_gpt_ft, similarities_nngs_gpt_ft = sweep_model_similarity(
        emb_gpt0_ft.detach().cpu().numpy(),
        emb_gpt1_ft.detach().cpu().numpy(),)
    alphas_gpt_bert_ft, similarities_cka_gpt_bert_ft, ks_gpt_bert_ft, similarities_nngs_gpt_bert_ft = sweep_model_similarity(
        emb_bert_finetuned0.detach().cpu().numpy(),
        emb_gpt0_ft.detach().cpu().numpy(),)
    alphas_gpt_inits, similarities_cka_gpt_inits, ks_gpt_inits, similarities_nngs_gpt_inits = sweep_model_similarity(
        emb_gpt0.detach().cpu().numpy(),
        emb_gpt1.detach().cpu().numpy(),)
    
    
    
    
    plt.figure(figsize=(config['width'], config['height']*2))
    plt.plot(ks_bert_finetuned, similarities_nngs_bert_finetuned, label="BERT Finetuned vs. Finetuned")
    plt.plot(ks_bert_finetuned0, similarities_nngs_bert_finetuned0, label="BERT Base vs. Finetuned")
    plt.plot(ks_bert_finetuned1, similarities_nngs_bert_finetuned1, label="BERT Base vs. Other Finetuned")
    plt.plot(ks_gpt_ft, similarities_nngs_gpt_ft, label="GPT, Finetuned vs. Finetuned")
    plt.plot(ks_gpt0_ft, similarities_nngs_gpt0_ft, label="GPT Base vs. Finetuned")
    plt.plot(ks_gpt1_ft, similarities_nngs_gpt1_ft, label="GPT Base vs. Other Finetuned")
    plt.plot(ks_gpt_bert_ft, similarities_nngs_gpt_bert_ft, label="BERT Finetuned vs. GPT Finetuned")
    plt.plot(ks_bert_inits, similarities_nngs_bert_inits, label="BERT Base vs. Other Base", linestyle='dashed')
    plt.plot(ks_gpt_inits, similarities_nngs_gpt_inits, label="GPT Base vs. Other Base", linestyle='dashed')
    plt.xlabel("$\\k$")
    plt.ylabel("$NNGS(X, Y, k)$")
        #plt.title("NNGS Similarity under Increasing Noise for Various k")
    plt.ylim(0, 1)
    plt.legend(loc="upper center",           # position relative to bbox
    bbox_to_anchor=(0.5, -0.3),   # center it below the axes
    ncol=2, fontsize="xx-small",                         # number of columns
)
        
    plt.tight_layout()
    plt.savefig(script_dir / config['output_dir'] / f"finetuning_{dataset.split('/')[-1]}_{split}.png", dpi=300, bbox_inches='tight')





def main():

    similarities_model_vs_finetuning(dataset='sst2', column_X="sentence", column_Y = "label", n_samples=500, split="train")
    similarities_model_vs_finetuning(dataset='sst2', column_X="sentence", column_Y = "label", n_samples=500, split="test")
    similarities_model_vs_finetuning(dataset='imdb', column_X="text", column_Y = "label", n_samples=250, split="test")
    similarities_model_vs_finetuning(dataset='fancyzhx/ag_news', column_X="text", column_Y = "label", n_samples=250, split="test")
    similarities_model_vs_finetuning(dataset='tiagoft/arxiv-cs-cl-balanced-sample-2025', column_X="summary", column_Y = "primary_category", n_samples=250)
if __name__ == "__main__":
    main()
