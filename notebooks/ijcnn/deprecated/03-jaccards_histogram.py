from pointcloudsimilarity.similarities import (
    CKASimilarity,
    GULPSimilarity,
    ProcrustesSimilarity,
    GWSimilarity,
    TASSimilarityTorch,
    PWCCASimilarity,
)

from pointcloudsimilarity.tas_core.tas_torch import compute_jaccards_torch

import toml
from pathlib import Path
import numpy as np
from typing import DefaultDict
from tqdm import tqdm
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from itertools import product
import torch

script_dir = Path(__file__).parent
config = toml.load(script_dir / "settings.toml")['figures']


N = 10000
D = 5

c = 0.2
k = int(np.floor(N*c))
alpha = 0.9
#distribution, distribution_name = lambda : np.random.uniform(-1, 1, size=(N, D)), "uniform"
#distribution = lambda : np.random.exponential(scale=1.0, size=(N, D))
#distribution, distribution_name = lambda : np.random.normal(size=(N, D)), "normal"
distribution, distribution_name = lambda : np.random.standard_cauchy(size=(N, D)), "cauchy"
distribution, distribution_name = lambda : make_blobs(n_samples=N, n_features=D, centers=5000,)[0], "blobs"


X = distribution()
phi = distribution()
Y = alpha * X + (1-alpha) * phi

jaccards = compute_jaccards_torch(
    X=torch.tensor(X, device='cuda'),
    Y=torch.tensor(Y, device='cuda'),
    k=k,
    batch_size=50,
    device='cuda',
    only_intersection=True,
)

print(jaccards.shape)

import pandas as pd 
aux_bar = pd.Series(index=np.arange(k+1), data=0)
bar = pd.Series(jaccards).value_counts().sort_index()
aux_bar[bar.index] = bar.values
bar = aux_bar
bar.to_csv(script_dir / config['output_dir'] / f"jaccards_histogram_{k}_{alpha}_{distribution_name}.csv")
    
plt.figure(figsize=(config['width'], config['height']))
plt.bar(range(len(bar.values)), bar.values)
#plt.xticks(range(len(bar.values)), labels=np.round(bar.index,2 ), rotation=45)
plt.xlabel("$NNGS$")
plt.ylabel("Frequency")
    #plt.title("NNGS Similarity under Increasing Noise for Various k")
#plt.legend()
    
plt.tight_layout()
plt.savefig(script_dir / config['output_dir'] / f"jaccard_histogram_{k}_{alpha}_{distribution_name}.png", dpi=300)
