import time
from typing import DefaultDict
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoConfig
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
                                               TASSimilarityFaiss,
                                               ProcrustesSimilarity,
                                               PWCCASimilarity,
                                               RTDSimilarity)
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sentence_transformers import SentenceTransformer

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from pathlib import Path
import toml


def get_gpt_embeddings(model_name,
                       texts,
                       max_samples=2000,
                       batch_size=32,
                       quantize: str | None = None,
                       device='cuda'):
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
                load_in_8bit=True,  # or load_in_4bit=True
                # llm_int8_threshold=6.0,
                # llm_int8_has_fp16_weight=True
            )
        elif quantize == '4bits':
            quant_config = BitsAndBytesConfig(load_in_4bit=True, )
        else:
            print("Quantization mode ")
            raise Exception
        model = AutoModel.from_pretrained(model_name,
                                          quantization_config=quant_config,
                                          device_map="auto")
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


def get_bert_embeddings(model_name,
                        texts,
                        quantize: None | str = None,
                        device='cuda'):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if quantize is not None:
        if quantize == '8bits':
            # 8-bit quantization; set load_in_4bit=True for more compression
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,  # or load_in_4bit=True
                # llm_int8_threshold=6.0,
                # llm_int8_has_fp16_weight=True
            )
        elif quantize == '4bits':
            quant_config = BitsAndBytesConfig(load_in_4bit=True, )
        else:
            print("Quantization mode ")
            raise Exception
        model = AutoModel.from_pretrained(model_name,
                                          quantization_config=quant_config,
                                          device_map="auto")
        print("Model loaded with BitsAndBytes quantization.")
    else:
        model = AutoModel.from_pretrained(model_name).to(device)

    model.eval()

    embeddings = []

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

def classify_with_finetuned_bert(model_name,
                                  tokenizer_name,
                                  texts,
                                  device='cuda',
                                  ):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
        device)
    model.eval()
    predictions = []
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

            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            predictions.extend(preds.cpu().tolist())
    return predictions

def get_finetuned_bert_embeddings(model_name,
                                  tokenizer_name,
                                  texts,
                                  device='cuda',
                                  ):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
        device)
    model.eval()

    embeddings = []
    
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


def get_sbert_embeddings(model_name, texts):
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
        max_length=128,
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
        max_length=128,
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



similarities = {
    'CKA Linear': CKASimilarity(kernel='linear'),
    'CKA RBF ($\\alpha=0.2$)': CKASimilarity(kernel='rbf', scale_by_alpha=0.2),
    'CKA RBF ($\\alpha=0.4$)': CKASimilarity(kernel='rbf', scale_by_alpha=0.4),
    'CKA RBF ($\\alpha=0.8$)': CKASimilarity(kernel='rbf', scale_by_alpha=0.8),
    'GULP': GULPSimilarity(),
    'Procrustes': ProcrustesSimilarity(),
    'GW': GWSimilarity(),
    'PWCCA': PWCCASimilarity(symmetric=True),
    #'RTD': RTDSimilarity(),
    'NNGS ($k=10$)': TASSimilarityTorch(k=10, normalize=True),
    'NNGS ($k=125$)': TASSimilarityTorch(k=125, normalize=True),
    'NNGS ($k=250$)': TASSimilarityTorch(k=250, normalize=True),
}

def calculate_all_similaritiees(X, Y, similarities):
    results = {}
    for name, sim in similarities.items():
        print("Calculating similarity:", name)
        results[name] = sim(X, Y)
    return results
