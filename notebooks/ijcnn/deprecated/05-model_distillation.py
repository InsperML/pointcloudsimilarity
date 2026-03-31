import time
from typing import DefaultDict
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import BitsAndBytesConfig

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
        metric = TASSimilarityTorch(k=k, batch_size=100, normalize=True)
        #print(f"Calculating layerwise similarities using {metric.__class__.__name__}...")
        #t0 = time.perf_counter()
        sim = metric(torch.Tensor(X).cuda(), torch.Tensor(Y).cuda())
        #t1 = time.perf_counter()
        #print(f"Time taken: {t1 - t0:.2f} seconds.")
        #print(f"Calculated layerwise similarities using {metric.__class__.__name__}.")
        similarities_nngs.append(sim)

    return alphas, similarities_cka, ks, similarities_nngs

def get_gpt_embeddings(model_name, texts, max_samples=2000, batch_size=32, quantize : str | None = None):
    """
    Extracts embeddings from GPT-style (Decoder-only) models.
    Uses the 'Last Token' hidden state strategy.
    
    Args:
        model_name: e.g. 'gpt2', 'distilgpt2', 'gpt2-medium'
    """
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if quantize is not None:
        if quantize == '8bits':
            # 8-bit quantization; set load_in_4bit=True for more compression
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,        # or load_in_4bit=True
                # llm_int8_threshold=6.0,
                # llm_int8_has_fp16_weight=True
            )
        elif quantize == '4bits':
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
            )
        else:
            print("Quantization mode ")
            raise Exception
        model = AutoModel.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto"
        )
        print("Model loaded with BitsAndBytes 8-bit quantization.")
    else:  
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


def get_bert_embeddings(model_name, texts, max_samples=2000, quantize : None | str = None):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if quantize is not None:
        if quantize == '8bits':
            # 8-bit quantization; set load_in_4bit=True for more compression
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,        # or load_in_4bit=True
                # llm_int8_threshold=6.0,
                # llm_int8_has_fp16_weight=True
            )
        elif quantize == '4bits':
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
            )
        else:
            print("Quantization mode ")
            raise Exception
        model = AutoModel.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto"
        )
        print("Model loaded with BitsAndBytes 8-bit quantization.")
    else:
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


def model_vs_quantized_model_similarities():
    N_SAMPLES = 1000
    max_samples = N_SAMPLES
    print("Loading dataset...")
    #ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    #ds = load_dataset("imdb", split="train", streaming=False)
    ds = load_dataset("tiagoft/arxiv-cs-cl-balanced-sample-2025", split="train", streaming=False)
    texts = []
    #for i, article in enumerate(ds.take(N_SAMPLES*10)):
    #    texts.append(article['texsummaryt'])
    texts = [article['summary'] for article in ds]
    # Randomly select N_SAMPLES texts
    rng = np.random.default_rng(1234)
    if len(texts) > N_SAMPLES:
        idx = rng.choice(len(texts), size=N_SAMPLES, replace=False)
        texts = [texts[i] for i in idx]
    else:
        texts = texts[:N_SAMPLES]

    print("Getting embeddings...")
    emb_bert0 = get_bert_embeddings("bert-base-uncased",
                                    texts,
                                    max_samples=N_SAMPLES)
    emb_bert_distill = get_bert_embeddings("distilbert-base-uncased",
                                    texts,
                                    max_samples=N_SAMPLES)

    emb_bert_simcse_unsup = get_bert_embeddings("princeton-nlp/unsup-simcse-bert-base-uncased",
                                         texts,
                                         max_samples=N_SAMPLES,)
    emb_bert_simcse_sup = get_bert_embeddings("princeton-nlp/sup-simcse-bert-base-uncased",
                                    texts,
                                    max_samples=N_SAMPLES)
 

    # Normalize each sample (row) to unit L2 norm
    emb_bert0 = F.normalize(emb_bert0, p=2, dim=1, eps=1e-12)
    emb_bert_distill = F.normalize(emb_bert_distill, p=2, dim=1, eps=1e-12)
    emb_bert_simcse_unsup = F.normalize(emb_bert_simcse_unsup, p=2, dim=1, eps=1e-12)
    emb_bert_simcse_sup = F.normalize(emb_bert_simcse_sup, p=2, dim=1, eps=1e-12)
    

    alphas_bert_distill, similarities_cka_bert_distill, ks_bert_distill, similarities_nngs_bert_distill = sweep_model_similarity(
        emb_bert0.detach().cpu().numpy(),
        emb_bert_distill.detach().cpu().numpy(),)
    alphas_sim_sup, similarities_cka_sim_sup, ks_sim_sup, similarities_nngs_sim_sup = sweep_model_similarity(
        emb_bert0.detach().cpu().numpy(),
        emb_bert_simcse_sup.detach().cpu().numpy())
    alphas_sim_unsup, similarities_cka_sim_unsup, ks_sim_unsup, similarities_nngs_sim_unsup = sweep_model_similarity(
        emb_bert0.detach().cpu().numpy(),
        emb_bert_simcse_unsup.detach().cpu().numpy())
    alphas_sim_sup_vs_unsup, similarities_cka_sim_sup_vs_unsup, ks_sim_sup_vs_unsup, similarities_nngs_sim_sup_vs_unsup = sweep_model_similarity(
        emb_bert_simcse_sup.detach().cpu().numpy(),
        emb_bert_simcse_unsup.detach().cpu().numpy())   
    
    
    
    plt.figure(figsize=(config['width'], config['height']))
    plt.plot(ks_bert_distill, similarities_nngs_bert_distill, label="Distilled")
    plt.plot(ks_sim_sup, similarities_nngs_sim_sup, label="SimCSE Supervised")
    plt.plot(ks_sim_unsup, similarities_nngs_sim_unsup, label="SimCSE Unsupervised")
    plt.plot(ks_sim_sup_vs_unsup, similarities_nngs_sim_sup_vs_unsup, label="SimCSE Sup vs Unsup")
    plt.xlabel("$\\k$")
    plt.ylabel("$NNGS(X, Y, k)$")
        #plt.title("NNGS Similarity under Increasing Noise for Various k")
    plt.ylim(0, 1)
    plt.legend(loc="upper center",           # position relative to bbox
    bbox_to_anchor=(0.5, -0.3),   # center it below the axes
    ncol=2, fontsize="x-small",                         # number of columns
)
        
    plt.tight_layout()
    plt.savefig(script_dir / config['output_dir'] / f"distillation.png", dpi=300)





def main():

    model_vs_quantized_model_similarities()

if __name__ == "__main__":
    main()
