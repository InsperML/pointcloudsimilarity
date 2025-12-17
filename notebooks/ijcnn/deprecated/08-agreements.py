import time
from pathlib import Path
from typing import DefaultDict

import lib_get_embeddings as lib_emb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.datasets import load_digits, make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoTokenizer, BitsAndBytesConfig)
from umap import UMAP

import pointcloudsimilarity.similarities as pcsim
from pointcloudsimilarity.similarities import (CKASimilarity, GULPSimilarity,
                                               GWSimilarity,
                                               NNGSSimilarityTorch,
                                               ProcrustesSimilarity,
                                               PWCCASimilarity)

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


similarities = {
    'cka linear': CKASimilarity(kernel='linear'),
    'cka rbf': CKASimilarity(kernel='rbf'),
    'nngs (k=10)': NNGSSimilarityTorch(k=10, normalize=True),
    'nngs (k=125)': NNGSSimilarityTorch(k=125, normalize=True),
    'gulp': GULPSimilarity(),
    'procrustes': ProcrustesSimilarity(),
    #'gw': GWSimilarity(),
    'pwcca': PWCCASimilarity(),
}

def calculate_all_similaritiees(X, Y, similarities):
    results = {}
    for name, sim in similarities.items():
        results[name] = sim(X, Y)
    return results

def similarities_model_vs_finetuning(dataset='sst2',
                                     column_X="sentence",
                                     column_Y="label",
                                     n_samples=500,
                                     split="train",
                                     force_reload=False):
    N_SAMPLES = n_samples
    max_samples = N_SAMPLES
    print(dataset, split)
    fname = script_dir / config[
        'output_dir'] / f"similarities_model_vs_finetuning_{dataset}_{split}.csv"
    if fname.exists() and not force_reload:
        print(f"File {fname} already exists. Loading results from file.")
        df_outputs = pd.read_csv(fname)
        df_outputs['agreement_rate'] = df_outputs['agreement_rate'].apply(lambda x : x/(2*len(df_outputs)-x))
        
        df_corr = df_outputs[['agreement_rate', 'nngs_k_low',
                              'nngs_k_high', 'nngs_k_very_high', 'cka_linear']].corr(method='pearson')
        
        y = df_outputs['agreement_rate']
        x = df_outputs[['nngs_k_low', 'nngs_k_high', 'nngs_k_very_high', 'cka_linear']]
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(x, y)
        r2 = model.score(x, y)
        print(f"R^2 of linear regression combining all NNGS similarities: {r2:.4f}")
        print(f"Coefficients: {model.coef_}")
        
        import scipy.stats as stats
        r, p = stats.pearsonr(df_outputs['agreement_rate'], df_outputs['nngs_k_low'])
        print(f"Pearson correlation (NNGS low-k): r={r:.4f}, p={p:.4e}")
        r, p = stats.pearsonr(df_outputs['agreement_rate'], df_outputs['nngs_k_high'])
        print(f"Pearson correlation (NNGS high-k): r={r:.4f}, p={p:.4e}")
        r, p = stats.pearsonr(df_outputs['agreement_rate'], df_outputs['nngs_k_very_high'])
        print(f"Pearson correlation (NNGS very high-k): r={r:.4f}, p={p:.4e}")
        r, p = stats.pearsonr(df_outputs['agreement_rate'], df_outputs['cka_linear'])
        print(f"Pearson correlation (CKA linear): r={r:.4f}, p={p:.4e}")
        
        print(df_corr)
        plt.figure(figsize=(config['width'], config['width']))
        plt.scatter(df_outputs['agreement_rate'],
                    df_outputs['nngs_k_low'],
                    label='NNGS low-k',
                    alpha=0.7)
        plt.scatter(df_outputs['agreement_rate'],
                    df_outputs['nngs_k_high'],
                    label=f'NNGS high-k',
                    alpha=0.7)
        plt.scatter(df_outputs['agreement_rate'],
                    df_outputs['nngs_k_very_high'],
                    label=f'NNGS very high-k',
                    alpha=0.7)
        plt.scatter(df_outputs['agreement_rate'],
                    df_outputs['cka_linear'],
                    label=f'CKA linear',
                    alpha=0.7)
        plt.xlabel("Agreement Rate between Models")
        plt.ylabel("NNGS Similarity between Embeddings")
        plt.legend()
        plt.tight_layout()
        plt.savefig(script_dir / config['output_dir'] /
                    f"similarities_model_vs_finetuning_{dataset}_{split}.png",
                    dpi=300,
                    bbox_inches='tight')

    else:
        print("Loading dataset...")
        #ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        texts, labels = get_dataset(dataset,
                                    column_X=column_X,
                                    column_Y=column_Y,
                                    n_samples=N_SAMPLES,
                                    split=split)
        print(f"Loaded {len(texts)} samples.")
        #print(f"Labels: {labels}")
        actual_n_samples = len(texts)
        y_pred = []
        embeddings = []

        # Extract labels and embeddings
        n_models = 24
        for i in tqdm(range(n_models)):
            model = f'tiagoft/multiberts-seed_{i}_sst2_finetuned'
            base_model = f'google/multiberts-seed_{i}'

            y_pred_ = lib_emb.classify_with_finetuned_bert(
                model,
                base_model,
                texts,
            )
            embeddings_ = lib_emb.get_finetuned_bert_embeddings(
                model,
                base_model,
                texts,
            )
            y_pred.append(y_pred_)
            embeddings.append(F.normalize(embeddings_, p=2, dim=1, eps=1e-12))

        print(
            "Embeddings and classifcaittons extracted. Calculating agreement rates and similarities..."
        )

        metric_low = NNGSSimilarityTorch(k=3, batch_size=100, normalize=True)
        metric_high = NNGSSimilarityTorch(k=actual_n_samples // 2,
                                          batch_size=100,
                                          normalize=True)
        metric_very_high = NNGSSimilarityTorch(k=3 * actual_n_samples // 4,
                                                batch_size=100,
                                                normalize=True)
        cka = CKASimilarity(kernel='linear')
        
        all_outputs = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                y_pred0 = y_pred[i]
                y_pred1 = y_pred[j]
                agreement_rate = (
                    np.array(y_pred0) == np.array(y_pred1)).mean()

                emb0 = embeddings[i]
                emb1 = embeddings[j]

                sim_low = metric_low(emb0.detach().cpu(), emb1.detach().cpu())
                sim_high = metric_high(emb0.detach().cpu(),
                                       emb1.detach().cpu())
                sim_very_high = metric_very_high(emb0.detach().cpu(),
                                                 emb1.detach().cpu())
                output = {
                    'model_0': f'multiberts-seed_{i}',
                    'model_1': f'multiberts-seed_{j}',
                    'agreement_rate': agreement_rate,
                    'nngs_k_low': sim_low,
                    'nngs_k_high': sim_high,
                    'nngs_k_very_high': sim_very_high,
                    'cka_linear': cka(emb0.detach().cpu(),
                                      emb1.detach().cpu()), 
                }

                all_outputs.append(output)

        df_outputs = pd.DataFrame(all_outputs)
        df_outputs.to_csv(
            script_dir / config['output_dir'] /
            f"similarities_model_vs_finetuning_{dataset}_{split}.csv",
            index=False)
        df_corr = df_outputs[['agreement_rate', 'nngs_k_low',
                              'nngs_k_high', 'nngs_k_very_high', 'cka_linear']].corr(method='pearson')
        print(df_corr)


def main():
    force_reload = True
    # similarities_model_vs_finetuning(
    #     dataset='imdb',
    #     column_X="text",
    #     column_Y="label",
    #     n_samples=1000,
    #     split="test",
    #     force_reload=force_reload,
    # )
    # similarities_model_vs_finetuning(
    #     dataset='sst2',
    #     column_X="sentence",
    #     column_Y="label",
    #     n_samples=1000,
    #     split="validation",
    #     force_reload=force_reload,
    # )
    similarities_model_vs_finetuning(
        dataset='sst2',
        column_X="sentence",
        column_Y="label",
        n_samples=2000,
        split="test",
        force_reload=force_reload,
    )
    # similarities_model_vs_finetuning(
    #     dataset='sst2',
    #     column_X="sentence",
    #     column_Y="label",
    #     n_samples=1000,
    #     split="train",
    #     force_reload=force_reload,
    # )

if __name__ == "__main__":
    main()
