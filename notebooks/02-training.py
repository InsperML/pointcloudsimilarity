from typing import DefaultDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification, make_moons, make_circles

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pointcloudsimilarity.similarities import (CKASimilarity, GULPSimilarity,
                                               GWSimilarity, NNGSSimilarity,
                                               ProcrustesSimilarity,
                                               PWCCASimilarity)

from training import train_one_epoch, MLP

my_metrics = {
    # "cka_linear": CKASimilarity(),
    # "cka_rbf_autosigma": CKASimilarity(kernel='rbf'),
    # "cka_rbf_sigma10": CKASimilarity(kernel='rbf', sigma=10.0),
    # "cka_rbf_sigma1": CKASimilarity(kernel='rbf', sigma=1.0),
    # "cka_rbf_sigma01": CKASimilarity(kernel='rbf', sigma=0.1),
    # "gulp": GULPSimilarity(),
    # "procrustes": ProcrustesSimilarity(),
    # "gw_sim": GWSimilarity(),
    "nngs_k80": NNGSSimilarity(k=80),
    "nngs_k40": NNGSSimilarity(k=40),
    "nngs_k10": NNGSSimilarity(k=10),
    "nngs_k2": NNGSSimilarity(k=2),
    # "pwcca": PWCCASimilarity(),
}

    

# Hyperparameters
input_size = 2
hidden_size = 20
output_size = 2
learning_rate = 0.001
num_epochs = 1000
batch_size = 100
n_layers = 20

# Create dummy data
# X, y = make_classification(
#     n_samples=2000,
#     n_features=input_size,
#     n_informative=input_size,
#     n_redundant=0,
#     n_classes=output_size,
#     random_state=45,
#     n_clusters_per_class=2,
#     class_sep=2.0,
#     hypercube=True,
# )

#X, y = make_circles(n_samples=200, noise=0.3, random_state=42)
#X_eval, y_eval = make_circles(n_samples=2000, noise=0.1, random_state=24)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.title('Training Data Scatter Plot')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig('data_scatter.png')
plt.close()


X = torch.tensor(X, dtype=torch.float32)
Z = torch.randn(X.size(1), X.size(1))
#X = X @ Z
y = torch.tensor(y, dtype=torch.long)
indices = torch.randperm(X.size(0))[:100]


X_eval = torch.tensor(X_eval, dtype=torch.float32)
y_eval = torch.tensor(y_eval, dtype=torch.long)
#X_eval = X_eval @ Z

train_dataset = TensorDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

eval_dataset = TensorDataset(X_eval, y_eval)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss, and optimizer
model = MLP(input_size, hidden_size, n_layers,output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
train_loss = []
eval_loss = []
eval_accuracy = []
embeddings_over_time = []
for epoch in tqdm(range(num_epochs)):
    avg_train_loss, avg_val_loss, avg_val_accuracy = train_one_epoch(
        train_loader, eval_loader, model, optimizer, criterion
    )
    train_loss.append(avg_train_loss)
    eval_loss.append(avg_val_loss)
    eval_accuracy.append(avg_val_accuracy)
    model.eval()
    with torch.no_grad():
        X_sampled = X[indices]
        X_transformed = model(X_sampled).numpy()
        embeddings_over_time.append(X_transformed)
    
    

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(range(num_epochs), train_loss, label='Training Loss')
plt.plot(range(num_epochs), eval_loss, label='Evaluation Loss')
plt.legend()
plt.title('Training and Evaluation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.subplot(2,1,2)
plt.plot(range(num_epochs), eval_accuracy, label='Evaluation Accuracy', color='green')
plt.title('Evaluation Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig('training_loss.png')

print("Training complete! Calculating metrics...")
print("1: from current to next epoch")

similarities = DefaultDict(list)
for i in tqdm(range(len(embeddings_over_time)-1)):
    if i % 10 == 0:        
        for method in my_metrics.keys():
            sim = my_metrics[method](embeddings_over_time[i+1], embeddings_over_time[i])
            similarities[method].append(sim)
        
# Plotting the results

plt.figure(figsize=(10, 6))
for method, sims in similarities.items():
    plt.plot(range(0, num_epochs, 10), sims, label=method)
plt.title('Point Cloud Similarity between current and next epoch during MLP Training')
plt.xlabel('Epoch')
plt.ylabel('Similarity Score')
plt.legend()
plt.grid()
plt.savefig('training_similarity.png')

print("2: from trained to current epoch")

similarities2 = DefaultDict(list)
for i, X_emb in tqdm(enumerate(embeddings_over_time)):
    if i % 10 == 0:
        for method in my_metrics.keys():
            sim = my_metrics[method](embeddings_over_time[-1], X_emb)
            similarities2[method].append(sim)
        
# Plotting the results
plt.figure(figsize=(10, 6))
for method, sims in similarities2.items():
    plt.plot(range(0, num_epochs, 10), sims, label=method)
plt.title('Point Cloud Similarity (ref: last) during MLP Training')
plt.xlabel('Epoch')
plt.ylabel('Similarity Score')
plt.legend()
plt.grid()
plt.savefig('training_similarity2.png')