from typing import DefaultDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import (make_blobs, make_circles, make_classification,
                              make_moons)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from training import MLP, train_one_epoch
import os



# Hyperparameters
input_size = 10
hidden_size = 30
output_size = 10
learning_rate = 1e-3
num_epochs = 500
batch_size = 2000
n_layers = 5
n_samples = 2000
p_dropout = 0.1
n_models = 10

X, y = make_blobs(
    n_samples=n_samples,
    centers=output_size,
    n_features=input_size,
    cluster_std=0.2,
    random_state=42,
)

X_train, X_eval, y_train, y_eval = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42,
    stratify=y,
)

X_train = torch.tensor(X_train, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_train = y_train[torch.randperm(y_train.size(0))]

X_eval = torch.tensor(X_eval, dtype=torch.float32)
y_eval = torch.tensor(y_eval, dtype=torch.long)

indices = torch.randperm(X_eval.size(0))[:200]

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

eval_dataset = TensorDataset(X_eval, y_eval)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

# Initialize models and optimizers
models = [
    MLP(
        input_size=input_size,
        hidden_size=hidden_size,
        n_layers=n_layers,
        output_size=output_size,
        p_dropout=p_dropout,
    ) for _ in range(n_models)
]

for idx, model in enumerate(models):
    # model.apply(lambda m: nn.init.kaiming_uniform_(
    #     m.weight,
    #     generator=torch.Generator().manual_seed(idx),
    # ) if hasattr(m, 'weight') else None)
    model.cuda()

optimizers = [
    optim.Adam(model.parameters(), lr=learning_rate) for model in models
]
criterion = nn.CrossEntropyLoss()

# train each model
all_train_losses: DefaultDict[int, list] = DefaultDict(list)
all_val_losses: DefaultDict[int, list] = DefaultDict(list)
all_val_accuracies: DefaultDict[int, list] = DefaultDict(list)

for model, optimizer in zip(models, optimizers):
    train_losses = []
    val_losses = []
    val_accuracies = []
    for epoch in tqdm(range(num_epochs)):
        avg_train_loss, avg_val_loss, avg_val_accuracy = train_one_epoch(
            train_loader, eval_loader, model, optimizer, criterion)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)
    all_train_losses[id(model)] = train_losses
    all_val_losses[id(model)] = val_losses
    all_val_accuracies[id(model)] = val_accuracies

    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)

    for i, (model, optimizer) in enumerate(zip(models, optimizers)):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": all_train_losses[id(model)],
                "val_losses": all_val_losses[id(model)],
                "val_accuracies": all_val_accuracies[id(model)],
                "hyperparams": {
                    "input_size": input_size,
                    "hidden_size": hidden_size,
                    "output_size": output_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "n_layers": n_layers,
                    "n_samples": n_samples,
                    "p_dropout": p_dropout,
                },
            },
            os.path.join(save_dir, f"model_{i}.pth"),
        )
        
# Plot all training losses
plt.figure(figsize=(10, 6))
for model_id, losses in all_train_losses.items():
    plt.plot(losses, alpha=0.7, label=f'Model {model_id}', c='b')

for model_id, val_losses in all_val_losses.items():
    plt.plot(val_losses, alpha=0.7, label=f'Model {model_id} Val', c='r')
    
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training (blue) and validation (red) Losses for All Models')
plt.semilogy()
plt.grid(True)
plt.legend()
plt.savefig(os.path.join('training_losses.png'))
plt.close()

        
