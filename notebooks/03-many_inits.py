from typing import DefaultDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification, make_moons

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pointcloudsimilarity.similarities import (CKASimilarity, GULPSimilarity,
                                               GWSimilarity, NNGSSimilarity,
                                               ProcrustesSimilarity,
                                               PWCCASimilarity)

my_metrics = {
    "cka_linear": CKASimilarity(),
    "cka_rbf_autosigma": CKASimilarity(kernel='rbf'),
    "cka_rbf_sigma10": CKASimilarity(kernel='rbf', sigma=10.0),
    "cka_rbf_sigma1": CKASimilarity(kernel='rbf', sigma=1.0),
    "cka_rbf_sigma01": CKASimilarity(kernel='rbf', sigma=0.1),
    # "gulp": GULPSimilarity(),
    # "procrustes": ProcrustesSimilarity(),
    # "gw_sim": GWSimilarity(),
    "nngs_k80": NNGSSimilarity(k=80),
    "nngs_k40": NNGSSimilarity(k=40),
    "nngs_k10": NNGSSimilarity(k=10),
    # "pwcca": PWCCASimilarity(),
}


# Define a minimal MLP



# Hyperparameters
input_size = 2
hidden_size = 2
output_size = 2
learning_rate = 0.001
num_epochs = 1000
batch_size = 100
n_layers = 4

# Create dummy data
X, y = make_classification(
    n_samples=200,
    n_features=input_size,
    n_informative=input_size,
    n_redundant=0,
    n_classes=output_size,
    random_state=42,
    n_clusters_per_class=2,
)
X, y = make_moons(n_samples=200, noise=0.5, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
#Z = torch.randn(X.size(1), X.size(1))
#X = X @ Z
y = torch.tensor(y, dtype=torch.long)
train_dataset = TensorDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Initialize model, loss, and optimizer




def train_new_model(
    input_size,
    hidden_size,
    n_layers,
    output_size,
    learning_rate,
    train_loader,
    num_epochs,
):
    model = MLP(input_size, hidden_size, n_layers, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_loss = []
    val_loss = []
    avg_val_accuracy = []
    for epoch in tqdm(range(num_epochs)):
        avg_train_loss, avg_val_loss, avg_val_accuracy = train_one_epoch(
            train_loader, train_loader, model, optimizer, criterion
        )
        train_loss.append(avg_train_loss)
        val_loss.append(avg_val_loss)
        avg_val_accuracy.append(avg_val_accuracy)

    return train_loss, val_loss, avg_val_accuracy, model


n_models = 5
models = []
train_losses = []
embeddings = []
for i in range(n_models):
    train_loss, model = train_new_model(input_size, hidden_size, n_layers,
                                        output_size, learning_rate, train_loader,
                                        num_epochs)
    train_losses.append(train_loss)
    models.append(model)
    model.eval()
    with torch.no_grad():
        X_transformed = model(X).detach().numpy()
        embeddings.append(X_transformed)


plt.figure(figsize=(10, 6))
for train_loss in train_losses:
    plt.plot(range(num_epochs), train_loss, label='Training Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig('training_loss_many_trains.png')

# Pairwise similarities between all trained embeddings
similarity_results = {}
for name, metric in my_metrics.items():
    mat = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(i, n_models):
            s = metric(embeddings[i], embeddings[j])
            mat[i, j] = mat[j, i] = s
    similarity_results[name] = np.mean(mat)

print(similarity_results)