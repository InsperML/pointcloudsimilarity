import os
from collections import defaultdict as DefaultDict

import numpy as np
import torch
import torch.optim as optim
from sklearn.datasets import (make_blobs, make_circles, make_classification,
                              make_moons)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from training import MLP

from pointcloudsimilarity.similarities import (CKASimilarity, GULPSimilarity,
                                               GWSimilarity, NNGSSimilarityFaiss,
                                               ProcrustesSimilarity,
                                               PWCCASimilarity)

my_metrics = {
    #"pwcca": PWCCASimilarity(),
    "cka_linear": CKASimilarity(),
    #"cka_rbf_autosigma": CKASimilarity(kernel='rbf'),
    #"cka_rbf_sigma10": CKASimilarity(kernel='rbf', sigma=10.0),
    "cka_rbf_sigma1": CKASimilarity(kernel='rbf', sigma=1.0),
    "cka_rbf_sigma01": CKASimilarity(kernel='rbf', sigma=0.1),
    "cka_rbf_sigma001": CKASimilarity(kernel='rbf', sigma=0.01),
    #"gulp": GULPSimilarity(),
    #"procrustes": ProcrustesSimilarity(),
    #"gw_sim": GWSimilarity(),
    "nngs_k90": NNGSSimilarityFaiss(k=90, batch_size=5000, normalize=False),
    "nngs_k40": NNGSSimilarityFaiss(k=40, batch_size=5000, normalize=False),
    #"nngs_k40": NNGSSimilarityFaiss(k=40, n_jobs=-1, normalize=True),
    "nngs_k10": NNGSSimilarityFaiss(k=10, batch_size=5000, normalize=False),
}

input_size = 20
hidden_size = 50
output_size = 20
learning_rate = 1e-4
num_epochs = 500
batch_size = 100
n_layers = 10
n_samples = 500
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

indices = torch.randperm(X_eval.size(0))[:n_samples]

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
        X_sampled = X_eval[indices].cuda()
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
                this_layer_similarities.append(metric(m1, m2))
        layerwise_similarities.append(np.mean(this_layer_similarities))
    return layerwise_similarities

intermediate_mebeddings = [model.get_intermediate_embeddings(X_eval[indices].cuda()) for model in loaded_models]
layerwise_similarities_dict = {}
for metric_name, metric in my_metrics.items():
    print(f"Calculating layerwise similarities using {metric.__class__.__name__}...")
    layerwise_similarities_dict[metric_name] = similarity_between_layers(
        intermediate_mebeddings,
        metric,
    )
    print(f"Calculated {len(layerwise_similarities_dict[metric_name])} layerwise similarities using {metric.__class__.__name__}.")


import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
for metric_name, metric in my_metrics.items():
    layerwise_similarities = layerwise_similarities_dict[metric_name]
    plt.plot(range(len(layerwise_similarities)), layerwise_similarities, marker='o', label=metric_name)
params_string = f"Models: {n_models}, Dim: {input_size}->{hidden_size}->{output_size}, Layers: {n_layers}"
plt.title(f'Layer-wise Similarity between Models\n{params_string}')
plt.xlabel('Layer')
plt.ylabel('Average Similarity')
plt.xticks(range(len(layerwise_similarities)))
plt.grid()
plt.legend()
plt.savefig('layerwise_model_similarity.png')
plt.close()

