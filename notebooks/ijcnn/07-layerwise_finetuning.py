import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from scipy.stats import spearmanr
from tqdm import tqdm
import toml
from pathlib import Path


script_dir = Path(__file__).parent
config = toml.load(script_dir / "settings.toml")['figures']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def measure_false_similarity(model_name, dataset):
    print(f"Analyzing {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to('cuda')
    model.eval()
    
    # We want pairs that are SEMANTICALLY DIFFERENT (Score < 2)
    # But presumably share some domain features.
    dissimilar_pairs = [x for x in dataset if x['label'] < 2.0]
    # Limit to 1000 pairs
    dissimilar_pairs = dissimilar_pairs[:1000]
    
    layer_sims = [[] for _ in range(model.config.num_hidden_layers + 1)]
    
    with torch.no_grad():
        for item in dissimilar_pairs:
            inp1 = tokenizer(item['sentence1'], return_tensors='pt', padding=True, truncation=True).to('cuda')
            inp2 = tokenizer(item['sentence2'], return_tensors='pt', padding=True, truncation=True).to('cuda')
            
            out1 = model(**inp1, output_hidden_states=True)
            out2 = model(**inp2, output_hidden_states=True)
            
            for i in range(len(layer_sims)):
                v1 = out1.hidden_states[i][0, 0, :]
                v2 = out2.hidden_states[i][0, 0, :]
                sim = torch.nn.functional.cosine_similarity(v1, v2, dim=0).item()
                layer_sims[i].append(sim)
                
    return [np.mean(sims) for sims in layer_sims]


def get_layerwise_sts_scores(model_name, dataset):
    print(f"Evaluating {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to('cuda')
    model.eval()
    
    # Storage for correlations per layer
    num_layers = model.config.num_hidden_layers + 1
    layer_preds = [[] for _ in range(num_layers)]
    gold_scores = []
    
    print("Processing STS-B...")
    with torch.no_grad():
        for item in tqdm(dataset):
            sent1 = item['sentence1']
            sent2 = item['sentence2']
            score = item['label']
            if score < 1.0:
                continue
            gold_scores.append(score)
            
            # Tokenize pair
            # We process them separately to get cosine similarity of embeddings
            inp1 = tokenizer(sent1, return_tensors='pt', padding=True, truncation=True, max_length=128).to('cuda')
            inp2 = tokenizer(sent2, return_tensors='pt', padding=True, truncation=True, max_length=128).to('cuda')
            
            out1 = model(**inp1, output_hidden_states=True)
            out2 = model(**inp2, output_hidden_states=True)
            
            # Iterate layers
            for i in range(num_layers):
                # Get CLS token (Index 0)
                v1 = out1.hidden_states[i][0, 0, :]
                v2 = out2.hidden_states[i][0, 0, :]
                
                # Cosine Similarity
                cos = torch.nn.functional.cosine_similarity(v1, v2, dim=0).item()
                layer_preds[i].append(cos)
                
    # Compute Spearman Correlation per layer
    results = []
    for i in range(num_layers):
        corr, _ = spearmanr(layer_preds[i], gold_scores)
        results.append(corr)
        
    return results

if __name__ == "__main__":
    # 1. Load Data: STS Benchmark (Semantic Textual Similarity)
    # Part of GLUE
    dataset = load_dataset("glue", "stsb", split="validation")
    
    # 2. Evaluate Fine-Tuned Model (AG News)
    # This model is optimized for Topic, not Similarity.
    # NNGS predicts the "Topic" optimization destroys "Similarity" topology at the end.
    scores_ft = get_layerwise_sts_scores("textattack/bert-base-uncased-SST-2", dataset)
    
    # 3. Evaluate Baseline (Pre-Trained BERT)
    # Standard BERT is not optimized for STS either, but hasn't been "collapsed" by FT.
    scores_pt = get_layerwise_sts_scores("bert-base-uncased", dataset)

    # 4. Plot
    layers = range(len(scores_ft))
    plt.figure(figsize=(9, 6))
    
    plt.plot(layers, scores_pt, 'k--', label='Pre-Trained (Baseline)', alpha=0.5)
    plt.plot(layers, scores_ft, 'r-o', linewidth=3, label='Fine-Tuned (SST-2)')
    
    plt.xlabel('Layer Depth (0=Emb, 12=Output)')
    plt.ylabel('STS-B Correlation (Performance)')
    plt.title('The Consequence of Topological Collapse')
    plt.xticks(layers)
    plt.grid(True, alpha=0.3)
    
    # Highlight the "Sweet Spot" predicted by NNGS
    best_layer = np.argmax(scores_ft)
    plt.axvline(best_layer, color='green', linestyle=':', linewidth=2)
    plt.text(best_layer, min(scores_ft), f"  Best Layer: {best_layer}\n  (NNGS Sweet Spot)", verticalalignment='bottom')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(script_dir / config['output_dir'] / f"layerwise_similarity_accuracy.png", dpi=300, bbox_inches='tight')