import time
from typing import DefaultDict
from transformers import AutoModel, AutoTokenizer
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
                                               TASSimilarityTorch,
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

    ks = np.arange(1, X.shape[0]-1, 20)
    #ks = np.arange(1, 500, 1)
    similarities_nngs = []
    for k in tqdm(ks):
        metric = TASSimilarityTorch(k=k, batch_size=100, normalize=True)
        #print(f"Calculating layerwise similarities using {metric.__class__.__name__}...")
        #t0 = time.perf_counter()
        sim = metric(torch.Tensor(X).cuda(), torch.Tensor(Y).cuda())
        #t1 = time.perf_counter()
        #print(f"Time taken: {t1 - t0:.2f} seconds.")
        #print(f"Calculated layerwise similarities using {metric.__class__.__name__}.")
        similarities_nngs.append(sim)

    return alphas, similarities_cka, ks, similarities_nngs

def get_sbert_embeddings(model_name, texts, max_samples=2000):
    print(f"Loading {model_name}...")
    # Load model on GPU
    model = SentenceTransformer(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')
    


    
    # SBERT handles batching and tokenization internally
    # normalize_embeddings=True is CRITICAL. SBERT is trained for Cosine Similarity.
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True)
    
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
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            
            # Forward pass (output_hidden_states=True is implicit in default return)
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state # (Batch, Seq, Dim)
            
            # --- CRITICAL STEP: Extract the Last Token ---
            # Since sequences have different lengths and are padded, 
            # the "last token" is not at index -1. It is at index seq_len - 1.
            # We use the attention mask to find the length.
            
            # attention_mask is 1 for real tokens, 0 for pad
            # sum(1) gives the length of real tokens
            # subtract 1 to get the index (0-based)
            sequence_lengths = inputs.attention_mask.sum(dim=1) - 1
            
            # Gather the vector at the last real token position for each batch item
            batch_embeddings = last_hidden_states[torch.arange(last_hidden_states.size(0)), sequence_lengths]
            
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
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                               max_length=512, return_tensors="pt").to(device)
            
            outputs = model(**inputs)
            # Use [CLS] token (index 0) as the sentence representation
            cls_emb = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_emb.cpu())
            
    return torch.cat(embeddings, dim=0)


def model_vs_model_experiment():
    N_SAMPLES = 1000 
    max_samples = N_SAMPLES
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    texts = []
    for i, article in enumerate(ds.take(N_SAMPLES*10)):
        texts.append(article['text'])
    
    # Randomly select N_SAMPLES texts
    rng = np.random.default_rng(1234)
    if len(texts) > N_SAMPLES:
        idx = rng.choice(len(texts), size=N_SAMPLES, replace=False)
        texts = [texts[i] for i in idx]
    else:
        texts = texts[:N_SAMPLES]
      
    # Extract Teacher (BERT) and Student (DistilBERT)
    # N=2000 is enough to see topology, but 5000 is better if you have time.

    emb_teacher_bert = get_bert_embeddings("google/multiberts-seed_0", texts, max_samples=N_SAMPLES)
    emb_student_bert = get_bert_embeddings("google/multiberts-seed_2", texts, max_samples=N_SAMPLES)
    emb_teacher_gpt = get_gpt_embeddings("gpt2", texts, max_samples=N_SAMPLES)
    emb_student_gpt = get_gpt_embeddings("distilgpt2", texts, max_samples=N_SAMPLES)
    emb_teacher_sbert = get_sbert_embeddings("distiluse-base-multilingual-cased-v1", texts, max_samples=N_SAMPLES)
    emb_student_sbert = get_sbert_embeddings("distiluse-base-multilingual-cased-v2", texts, max_samples=N_SAMPLES)

    # Normalize each sample (row) to unit L2 norm
    emb_teacher_bert = F.normalize(emb_teacher_bert, p=2, dim=1, eps=1e-12)
    emb_student_bert = F.normalize(emb_student_bert, p=2, dim=1, eps=1e-12)
    emb_teacher_gpt = F.normalize(emb_teacher_gpt, p=2, dim=1, eps=1e-12)
    emb_student_gpt = F.normalize(emb_student_gpt, p=2, dim=1, eps=1e-12)
    emb_teacher_sbert = F.normalize(emb_teacher_sbert, p=2, dim=1, eps=1e-12)
    emb_student_sbert = F.normalize(emb_student_sbert, p=2, dim=1, eps=1e-12)


    print(f"Teacher Shape: {emb_teacher_bert.shape}")
    print(f"Student Shape: {emb_student_bert.shape}")
    alphas_bert, similarities_cka_bert, ks_bert, similarities_nngs_bert = sweep_model_similarity(emb_teacher_bert.detach().cpu().numpy(), emb_student_bert.detach().cpu().numpy())
    alphas_gpt, similarities_cka_gpt, ks_gpt, similarities_nngs_gpt = sweep_model_similarity(emb_teacher_gpt.detach().cpu().numpy(), emb_student_gpt.detach().cpu().numpy())
    alphas_sbert, similarities_cka_sbert, ks_sbert, similarities_nngs_sbert = sweep_model_similarity(emb_teacher_sbert.detach().cpu().numpy(), emb_student_sbert.detach().cpu().numpy())
     

    plt.figure(figsize=(8, 5))
    plt.subplot(2,1,1)

    plt.plot(alphas_bert, np.array(similarities_cka_bert), label='bert-base-uncased (two inits)')
    plt.plot(alphas_gpt, np.array(similarities_cka_gpt), label='GPT2 vs DistilGPT2')
    plt.plot(alphas_sbert, np.array(similarities_cka_sbert), label='SBERT-v1 vs SBERT-v2')
    plt.xscale('log')
    plt.xlabel('Alpha (scaling factor for sigma)')
    plt.ylabel('RBF-CKA')
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
 

    plt.subplot(2,1,2)
    plt.plot(ks_bert, np.array(similarities_nngs_bert), label='bert-base-uncased (two inits)')  
    plt.plot(ks_gpt, np.array(similarities_nngs_gpt), label='GPT2 vs DistilGPT2')
    plt.plot(ks_sbert, np.array(similarities_nngs_sbert), label='SBERT-v1 vs SBERT-v2')
    plt.xlabel('K (neighborhood size)')
    plt.ylabel('NNGS')
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
    
    plt.suptitle(f'Model Similarity Sweep(n_samples={emb_teacher_bert.shape[0]})')
    plt.tight_layout()
    plt.savefig(figname:=f'model_similarity_experiment_all_models.png')
    plt.close()



def main():

    model_vs_model_experiment()
        
if __name__ == "__main__":
    main()
    
