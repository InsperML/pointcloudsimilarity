import time
from typing import DefaultDict
from transformers import AutoModel, AutoTokenizer, AutoConfig, GPTNeoXConfig, GPTNeoXForCausalLM, AutoModelForSequenceClassification, GPTNeoXConfig, AutoModelForCausalLM
from datasets import load_dataset

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


def get_gptx_embeddings(base_model_name,
                        seed,
                        texts,
                        step=143000,
                        device='cuda'):
    print(f"Loading {base_model_name}...")
    config = AutoConfig.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        f"{base_model_name}-seed{seed}", revision=f"step{step}",
        config=config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    batch = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    print(f"Extracting GPT-X embeddings for {len(texts)} samples...")
    with torch.no_grad():
        out = model(**batch, output_hidden_states=True)
    
    mask = batch["attention_mask"].unsqueeze(-1)
    n_tokens = (mask.sum(dim=1) - 1).squeeze(-1)
    #print(n_tokens.shape, n_tokens)
    print(out.hidden_states[-1].shape)
    last = out.hidden_states[-1][torch.arange(out.hidden_states[-1].size(0)), n_tokens, :]
    print(last.shape)
    return last.cpu()
#    return (last * mask).sum(dim=1) / mask.sum(dim=1)

def get_finetuned_gptx_embeddings(base_model_name,
                                  finetuned_model_name,
                        texts,
                        device='cuda'):
    print(f"Loading {base_model_name}...")
    config = AutoConfig.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        finetuned_model_name,).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    batch = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    print(f"Extracting GPT-X embeddings for {len(texts)} samples...")
    with torch.no_grad():
        out = model(**batch, output_hidden_states=True)
    
    mask = batch["attention_mask"].unsqueeze(-1)
    n_tokens = (mask.sum(dim=1) - 1).squeeze(-1)
    #print(n_tokens.shape, n_tokens)
    print(out.hidden_states[-1].shape)
    last = out.hidden_states[-1][torch.arange(out.hidden_states[-1].size(0)), n_tokens, :]
    print(last.shape)
    return last.cpu()
#    return (last * mask).sum(dim=1) / mask.sum(dim=1)


def get_sbert_embeddings(model_name, texts, max_samples=2000):
    print(f"Loading {model_name}...")
    # Load model on GPU
    model = SentenceTransformer(
        model_name, device='cuda' if torch.cuda.is_available() else 'cpu')

    # SBERT handles batching and tokenization internally
    # normalize_embeddings=True is CRITICAL. SBERT is trained for Cosine Similarity.
    embeddings = model.encode(texts,
                              convert_to_tensor=True,
                              show_progress_bar=True,
                              normalize_embeddings=True)

    return embeddings


def get_gpt_embeddings(model_name, texts, max_samples=2000, batch_size=32):
    """
    Extracts embeddings from GPT-style (Decoder-only) models.
    Uses the 'Last Token' hidden state strategy.
    
    Args:
        model_name: e.g. 'gpt2', 'distilgpt2', 'gpt2-medium'
    """
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # GPT-2 does not have a pad token by default, so we use EOS
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embeddings = []

    print(f"Extracting GPT embeddings for {len(texts)} samples...")

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = tokenizer(batch_texts,
                               return_tensors="pt",
                               padding=True,
                               truncation=True,
                               max_length=128).to(device)

            # Forward pass (output_hidden_states=True is implicit in default return)
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state  # (Batch, Seq, Dim)

            # --- CRITICAL STEP: Extract the Last Token ---
            # Since sequences have different lengths and are padded,
            # the "last token" is not at index -1. It is at index seq_len - 1.
            # We use the attention mask to find the length.

            # attention_mask is 1 for real tokens, 0 for pad
            # sum(1) gives the length of real tokens
            # subtract 1 to get the index (0-based)
            sequence_lengths = inputs.attention_mask.sum(dim=1) - 1

            # Gather the vector at the last real token position for each batch item
            batch_embeddings = last_hidden_states[
                torch.arange(last_hidden_states.size(0)), sequence_lengths]

            embeddings.append(batch_embeddings.cpu())

    return torch.cat(embeddings, dim=0)


def get_bert_embeddings(model_name, texts, max_samples=2000):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []

    print(f"Extracting embeddings for {max_samples} samples...")
    # Iterate with small batch size to avoid VRAM issues
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts,
                               padding=True,
                               truncation=True,
                               max_length=512,
                               return_tensors="pt").to(device)

            outputs = model(**inputs)
            # Use [CLS] token (index 0) as the sentence representation
            cls_emb = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_emb.cpu())

    return torch.cat(embeddings, dim=0)



def get_finetuned_bert_embeddings(model_name,
                                  tokenizer_name,
                                  texts,
                                  max_samples=2000):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
        device)
    model.eval()

    embeddings = []

    print(f"Extracting embeddings for {max_samples} samples...")
    # Iterate with small batch size to avoid VRAM issues
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts,
                               padding=True,
                               truncation=True,
                               max_length=128,
                               return_tensors="pt").to(device)

            outputs = model(**inputs, output_hidden_states=True)
            # Use [CLS] token (index 0) as the sentence representation
            last_hidden_states = outputs.hidden_states[
                -1]  # Last layer hidden states
            cls_embedding = last_hidden_states[:, 0, :]  # [CLS] token
            embeddings.append(cls_embedding.cpu())

    return torch.cat(embeddings, dim=0)


def model_vs_model_experiment():
    N_SAMPLES = 500
    max_samples = N_SAMPLES
    # ds = load_dataset("wikimedia/wikipedia",
    #                   "20231101.en",
    #                   split="train",
    #                   streaming=True)
    ds = load_dataset("sst2", split="train", streaming=False)
    texts = []
    for i, article in enumerate(ds.take(N_SAMPLES * 10)):
        texts.append(article['sentence'][:200])
    #texts = [article['text'] for article in ds]
    # Randomly select N_SAMPLES texts
    rng = np.random.default_rng(1234)
    if len(texts) > N_SAMPLES:
        idx = rng.choice(len(texts), size=N_SAMPLES, replace=False)
        texts = [texts[i] for i in idx]
    else:
        texts = texts[:N_SAMPLES]

    # Extract Teacher (BERT) and Student (DistilBERT)
    # N=2000 is enough to see topology, but 5000 is better if you have time.

    emb_gpt0 = get_gptx_embeddings("EleutherAI/pythia-410m",
                                   1,
                                   texts,
                                   )
    emb_gpt1 = get_gptx_embeddings("EleutherAI/pythia-410m",
                                   2,
                                   texts,
                                   )
    emb_gpt0_ft = get_finetuned_gptx_embeddings("EleutherAI/pythia-410m",
                                                "tiagoft/pythia-410m-seed1_sst2_finetuned",
                                   texts,
                                   )
    emb_gpt1_ft = get_finetuned_gptx_embeddings("EleutherAI/pythia-410m",
                                                "tiagoft/pythia-410m-seed2_sst2_finetuned",
                                   texts,
                                   )
        


    # Normalize each sample (row) to unit L2 norm
    emb_gpt0 = F.normalize(emb_gpt0, p=2, dim=1, eps=1e-12)
    emb_gpt1 = F.normalize(emb_gpt1, p=2, dim=1, eps=1e-12)
    emb_gpt0_ft = F.normalize(emb_gpt0_ft, p=2, dim=1, eps=1e-12)
    emb_gpt1_ft = F.normalize(emb_gpt1_ft, p=2, dim=1, eps=1e-12)

    alphas_init, similarities_cka_init, ks_init, similarities_nngs_init = sweep_model_similarity(
        emb_gpt0.detach().cpu().numpy(),
        emb_gpt1.detach().cpu().numpy())
    alphas_ft, similarities_cka_ft, ks_ft, similarities_nngs_ft = sweep_model_similarity(
        emb_gpt0_ft.detach().cpu().numpy(),
        emb_gpt0.detach().cpu().numpy())
    alphas_ft2, similarities_cka_ft2, ks_ft2, similarities_nngs_ft2 = sweep_model_similarity(
        emb_gpt1_ft.detach().cpu().numpy(),
        emb_gpt0_ft.detach().cpu().numpy())


    fig = plt.figure(figsize=(8, 6))

    # Upper subplot: RBF-CKA
    ax1 = fig.add_subplot(2, 1, 1)
    l1, = ax1.plot(alphas_init,
                   np.array(similarities_cka_init),
                   label='Pre-trained inits')
    l2, = ax1.plot(alphas_ft,
                   np.array(similarities_cka_ft),
                   label='Fine-tuned vs. Pre-trained')
    l3, = ax1.plot(alphas_ft2,
                     np.array(similarities_cka_ft2),
                        label='Fine-tuned vs. Fine-tuned')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Alpha (scaling factor for sigma)')
    ax1.set_ylabel('RBF-CKA')
    ax1.set_ylim(0, 1.05)
    ax1.grid()

    # Lower subplot: NNGS
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(ks_init, np.array(similarities_nngs_init))
    ax2.plot(ks_ft, np.array(similarities_nngs_ft))
    ax2.plot(ks_ft2, np.array(similarities_nngs_ft2))
    
    ax2.set_xlabel('K (neighborhood size)')
    ax2.set_ylabel('NNGS')
    ax2.set_ylim(0, 1.05)
    ax2.grid()

    # Single legend below both subplots
    handles = [l1, l2, l3]
    labels = [h.get_label() for h in handles]

    fig.legend(handles,
               labels,
               loc='lower center',
               ncol=3,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f'Model Similarity Sweep between Pythia Fine-tunings\n(N={N_SAMPLES})'
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for the legend
    fig.savefig(
        figname := 'model_similarity_experiment_finetunings_pythia.png')
    plt.close()


def main():

    model_vs_model_experiment()


if __name__ == "__main__":
    main()
