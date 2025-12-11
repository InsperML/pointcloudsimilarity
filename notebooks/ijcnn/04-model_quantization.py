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
    emb_bert_large = get_bert_embeddings("bert-large-uncased",
                                    texts,
                                    max_samples=N_SAMPLES)
    emb_bert_large_8bits = get_bert_embeddings("bert-large-uncased",
                                    texts,
                                    max_samples=N_SAMPLES,
                                    quantize='8bits')
    emb_bert_large_4bits = get_bert_embeddings("bert-large-uncased",
                                            texts,
                                            max_samples=N_SAMPLES,
                                            quantize='4bits')
                                            
    
    emb_bert0 = get_bert_embeddings("bert-base-uncased",
                                    texts,
                                    max_samples=N_SAMPLES)
    emb_bert_8bits = get_bert_embeddings("bert-base-uncased",
                                    texts,
                                    max_samples=N_SAMPLES,
                                    quantize='8bits')
    emb_bert_4bits = get_bert_embeddings("bert-base-uncased",
                                         texts,
                                         max_samples=N_SAMPLES,
                                         quantize='4bits')
    emb_gpt = get_gpt_embeddings("gpt2",
                                    texts,
                                    max_samples=N_SAMPLES)
    emb_gpt_8bits = get_gpt_embeddings("gpt2",
                                    texts,
                                    max_samples=N_SAMPLES,
                                    quantize='8bits')
    emb_gpt_4bits = get_gpt_embeddings("gpt2",
                                         texts,
                                         max_samples=N_SAMPLES,
                                         quantize='4bits')
    emb_gpt_large = get_gpt_embeddings("gpt2-large",
                                    texts,
                                    max_samples=N_SAMPLES)
    emb_gpt_large_8bits = get_gpt_embeddings("gpt2-large",
                                    texts,
                                    max_samples=N_SAMPLES,
                                    quantize='8bits')
    emb_gpt_large_4bits = get_gpt_embeddings("gpt2-large",
                                         texts,
                                         max_samples=N_SAMPLES,
                                         quantize='4bits')


    
    # emb_bert_ft0 = get_finetuned_bert_embeddings(
    #     "tiagoft/multiberts-seed_0_sst2_finetuned",
    #     "google/multiberts-seed_0",
    #     texts,
    #     max_samples=N_SAMPLES)
    # emb_bert_ft1 = get_finetuned_bert_embeddings(
    #     "tiagoft/multiberts-seed_1_sst2_finetuned",
    #     "google/multiberts-seed_1",
    #     texts,
    #     max_samples=N_SAMPLES)
    # emb_bert_ft0_lora = get_finetuned_bert_embeddings(
    #     "tiagoft/multiberts-seed_0_sst2_finetuned_lora",
    #     "google/multiberts-seed_0",
    #     texts,
    #     max_samples=N_SAMPLES)
    # emb_bert_ft1_lora = get_finetuned_bert_embeddings(
    #     "tiagoft/multiberts-seed_1_sst2_finetuned_lora",
    #     "google/multiberts-seed_1",
    #     texts,
    #     max_samples=N_SAMPLES)

    # Normalize each sample (row) to unit L2 norm
    emb_bert0 = F.normalize(emb_bert0, p=2, dim=1, eps=1e-12)
    emb_bert_8bits = F.normalize(emb_bert_8bits, p=2, dim=1, eps=1e-12)
    emb_bert_4bits = F.normalize(emb_bert_4bits, p=2, dim=1, eps=1e-12)
    
    emb_gpt = F.normalize(emb_gpt, p=2, dim=1, eps=1e-12)
    emb_gpt_8bits = F.normalize(emb_gpt_8bits, p=2, dim=1, eps=1e-12)
    emb_gpt_4bits = F.normalize(emb_gpt_4bits, p=2, dim=1, eps=1e-12)
    
    emb_bert_large = F.normalize(emb_bert_large, p=2, dim=1, eps=1e-12)
    emb_bert_large_8bits = F.normalize(emb_bert_large_8bits, p=2, dim=1, eps=1e-12)
    emb_bert_large_4bits = F.normalize(emb_bert_large_4bits, p=2, dim=1, eps=1e-12)

    emb_gpt_large = F.normalize(emb_gpt_large, p=2, dim=1, eps=1e-12)
    emb_gpt_large_8bits = F.normalize(emb_gpt_large_8bits, p=2, dim=1, eps=1e-12)
    emb_gpt_large_4bits = F.normalize(emb_gpt_large_4bits, p=2, dim=1, eps=1e-12)

    # emb_bert_ft0 = F.normalize(emb_bert_ft0, p=2, dim=1, eps=1e-12)
    # emb_bert_ft1 = F.normalize(emb_bert_ft1, p=2, dim=1, eps=1e-12)
    # emb_bert_ft0_lora = F.normalize(emb_bert_ft0_lora, p=2, dim=1, eps=1e-12)
    # emb_bert_ft1_lora = F.normalize(emb_bert_ft1_lora, p=2, dim=1, eps=1e-12)

    alphas_bert8, similarities_cka_bert8, ks_bert8, similarities_nngs_bert8 = sweep_model_similarity(
        emb_bert0.detach().cpu().numpy(),
        emb_bert_8bits.detach().cpu().numpy(),)
    alphas_bert4, similarities_cka_bert4, ks_bert4, similarities_nngs_bert4 = sweep_model_similarity(
        emb_bert0.detach().cpu().numpy(),
        emb_bert_4bits.detach().cpu().numpy())
    alphas_gpt8, similarities_cka_gpt8, ks_gpt8, similarities_nngs_gpt8 = sweep_model_similarity(
        emb_gpt.detach().cpu().numpy(),
        emb_gpt_8bits.detach().cpu().numpy())
    alphas_gpt4, similarities_cka_gpt4, ks_gpt4, similarities_nngs_gpt4 = sweep_model_similarity(
        emb_gpt.detach().cpu().numpy(),
        emb_gpt_4bits.detach().cpu().numpy())   
    alphas_bertlarge8, similarities_cka_bertlarge8, ks_bertlarge8, similarities_nngs_bertlarge8 = sweep_model_similarity(
        emb_bert_large.detach().cpu().numpy(),
        emb_bert_large_8bits.detach().cpu().numpy())
    alphas_bertlarge4, similarities_cka_bertlarge4, ks_bertlarge4, similarities_nngs_bertlarge4 = sweep_model_similarity(
        emb_bert_large.detach().cpu().numpy(),
        emb_bert_large_4bits.detach().cpu().numpy())
    alphas_gptlarge8, similarities_cka_gptlarge8, ks_gptlarge8, similarities_nngs_gptlarge8 = sweep_model_similarity(
        emb_gpt_large.detach().cpu().numpy(),
        emb_gpt_large_8bits.detach().cpu().numpy())
    alphas_gptlarge4, similarities_cka_gptlarge4, ks_gptlarge4, similarities_nngs_gptlarge4 = sweep_model_similarity(
        emb_gpt_large.detach().cpu().numpy(),
        emb_gpt_large_4bits.detach().cpu().numpy())   
    
    
    plt.figure(figsize=(config['width'], config['height']))
    plt.plot(ks_bert8, similarities_nngs_bert8, label="BERT Base 8-bits")
    plt.plot(ks_bert4, similarities_nngs_bert4, label="BERT Base 4-bits")
    plt.plot(ks_gpt8, similarities_nngs_gpt8, label="GPT2 8-bits")
    plt.plot(ks_gpt4, similarities_nngs_gpt4, label="GPT2 4-bits")
    plt.plot(ks_bertlarge8, similarities_nngs_bertlarge8, label="BERT Large 8-bits")
    plt.plot(ks_bertlarge4, similarities_nngs_bertlarge4, label="BERT Large 4-bits")
    plt.plot(ks_gptlarge8, similarities_nngs_gptlarge8, label="GPT2 Large 8-bits")
    plt.plot(ks_gptlarge4, similarities_nngs_gptlarge4, label="GPT2 Large 4-bits")
    plt.xlabel("$\\k$")
    plt.ylabel("$NNGS(X, Y, k)$")
        #plt.title("NNGS Similarity under Increasing Noise for Various k")
    plt.ylim(0, 1)
    plt.legend(loc="upper center",           # position relative to bbox
    bbox_to_anchor=(0.5, -0.3),   # center it below the axes
    ncol=4, fontsize="x-small",                         # number of columns
)
        
    plt.tight_layout()
    plt.savefig(script_dir / config['output_dir'] / f"quantization.png", dpi=300)





def main():

    model_vs_quantized_model_similarities()

if __name__ == "__main__":
    main()
