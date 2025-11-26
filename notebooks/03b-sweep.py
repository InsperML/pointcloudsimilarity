import os
from collections import defaultdict as DefaultDict
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from sklearn.datasets import (make_blobs, make_circles, make_classification,
                              make_moons)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from training import MLP

from pointcloudsimilarity.similarities import (CKASimilarity, GULPSimilarity,
                                               GWSimilarity, NNGSSimilarityFaiss, NNGSSimilarityTorch,
                                               ProcrustesSimilarity,
                                               PWCCASimilarity)

my_metrics = {
    #"pwcca": PWCCASimilarity(),
    "cka_linear": CKASimilarity(),
    #"cka_rbf_autosigma": CKASimilarity(kernel='rbf'),
    #"cka_rbf_sigma10": CKASimilarity(kernel='rbf', sigma=10.0),
    "cka_rbf_alpha1": CKASimilarity(kernel='rbf', scale_by_alpha=1.0),
    "cka_rbf_alpha03": CKASimilarity(kernel='rbf', scale_by_alpha=0.3),
    "cka_rbf_alpha01": CKASimilarity(kernel='rbf', scale_by_alpha=0.1),
    "cka_rbf_alpha003": CKASimilarity(kernel='rbf', scale_by_alpha=0.03),
    "cka_rbf_alpha001": CKASimilarity(kernel='rbf', scale_by_alpha=0.001),
    #"gulp": GULPSimilarity(),
    #"procrustes": ProcrustesSimilarity(),
    #"gw_sim": GWSimilarity(),
    "nngs_k100": NNGSSimilarityTorch(k=100, batch_size=5000, normalize=False),
    "nngs_k30": NNGSSimilarityTorch(k=30, batch_size=5000, normalize=False),
    #"nngs_k40": NNGSSimilarityFaiss(k=40, n_jobs=-1, normalize=True),
    "nngs_k10": NNGSSimilarityTorch(k=10, batch_size=5000, normalize=False),
    "nngs_k3": NNGSSimilarityTorch(k=3, batch_size=5000, normalize=False),
    "nngs_k1": NNGSSimilarityTorch(k=1, batch_size=5000, normalize=False),
}

input_size = 10
hidden_size = 30
output_size = 10
learning_rate = 1e-3
num_epochs = 500
batch_size = 200
n_layers = 5
n_samples = 1000
p_dropout = 0.0
n_models = 10


X, y = make_blobs(
    n_samples=n_samples,
    centers=output_size,
    n_features=input_size,
    cluster_std=0.2,
    random_state=43,
)


X_eval = torch.tensor(X, dtype=torch.float32)
y_eval = torch.tensor(y, dtype=torch.long)

#indices = torch.randperm(X_eval.size(0))[:n_samples]

eval_dataset = TensorDataset(X_eval, y_eval)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

loaded_models = []
loaded_optimizers = []
loaded_histories = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i in range(n_models):
    ckpt_path = os.path.join("saved_models", f"model_{i}.pth")
    ckpt = torch.load(ckpt_path, map_location=device)

    hp = ckpt["hyperparams"]
    model_args = {
        k: hp[k]
        for k in
        ["input_size", "hidden_size", "n_layers", "output_size", "p_dropout"]
    }
    loaded_model = MLP(**model_args).to(device)
    loaded_model.load_state_dict(ckpt["model_state_dict"])
    loaded_model.eval()

    loaded_models.append(loaded_model)

    loaded_histories.append({
        "train_losses": ckpt["train_losses"],
        "val_losses": ckpt["val_losses"],
        "val_accuracies": ckpt["val_accuracies"],
    })

print(f"Loaded {len(loaded_models)} models.")

# get each model embeddings
all_embeddings = DefaultDict(list)
for model in loaded_models:
    model.eval()
    with torch.no_grad():
        X_sampled = X_eval.cuda()
        X_transformed = model.get_final_embeddings(X_sampled).cpu().numpy()
        all_embeddings[id(model)].append(X_transformed)

print("Calculated embeddings")

def similarity_between_layers(emb : enumerate[list[torch.Tensor]], metric):
    layerwise_similarities = []
    n_models = len(emb)
    n_layers = len(emb[0])
    

    for k in range(n_layers):
        this_layer_similarities = []
        for j in range(n_models):
            for i in range(j + 1, n_models):
                m1 = emb[j][k].detach().cpu().numpy()
                m2 = emb[i][k].detach().cpu().numpy()
                this_layer_similarities.append(metric(m1, m2))#, label=f"Layer {k} Model {j} vs Model {i} - "))
        layerwise_similarities.append(np.median(this_layer_similarities))
    return layerwise_similarities


intermediate_embeddings =  [[model.get_intermediate_embeddings(X_eval.cuda())[6]] for model in loaded_models]
embeddings = intermediate_embeddings

# alphas = np.logspace(-5, 10, 200)
# similarities_cka = []
# for alpha in tqdm(alphas):
#     metric = CKASimilarity(kernel='rbf', scale_by_alpha=alpha)
#     #print(f"Calculating layerwise similarities using {metric.__class__.__name__}...")
#     t0 = time.perf_counter()
#     sim = similarity_between_layers(
#         embeddings,
#         metric,
#     )
#     t1 = time.perf_counter()
#     #print(f"Time taken: {t1 - t0:.2f} seconds.")
#     #print(f"Calculated layerwise similarities using {metric.__class__.__name__}.")
#     similarities_cka.append(sim)


ks = np.arange(1, 500, 1)
similarities_nngs = []
for k in tqdm(ks):
    metric = NNGSSimilarityTorch(k=k, batch_size=5000, normalize=True)
    #print(f"Calculating layerwise similarities using {metric.__class__.__name__}...")
    t0 = time.perf_counter()
    sim = similarity_between_layers(
        embeddings,
        metric,
    )
    t1 = time.perf_counter()
    #print(f"Time taken: {t1 - t0:.2f} seconds.")
    #print(f"Calculated layerwise similarities using {metric.__class__.__name__}.")
    similarities_nngs.append(sim)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.subplot(2,1,1)
# plt.plot(alphas, np.array(similarities_cka)[:,0], label='CKA RBF - Layer 6')
plt.xscale('log')
plt.title(f'Layer-wise Similarity between Models - CKA RBF')
plt.xlabel('Alpha (scaling factor for sigma)')
plt.ylabel('Average Similarity')
plt.grid()

plt.subplot(2,1,2)
plt.plot(ks, np.array(similarities_nngs)[:,0], label='NNGS - Layer 6')  
plt.title(f'Layer-wise Similarity between Models - NNGS')
plt.xlabel('Layer')
plt.ylabel('Average Similarity')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('model_similarity_sweep.png')
plt.close()

