import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from tqdm import tqdm
import torch
import pandas as pd

# import pointcloudsimilarity.similarities as pcsim
from pointcloudsimilarity.similarities import (CKASimilarity, GULPSimilarity,
                                               GWSimilarity,
                                               TASSimilarityTorch,
                                               ProcrustesSimilarity,
                                               PWCCASimilarity, RTDSimilarity)
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import toml

script_dir = Path(__file__).parent
config = toml.load(script_dir / "settings.toml")['figures']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sweep_model_similarity(X, Y):
    ks = np.arange(1, X.shape[0] - 1, 20)
    #ks = np.arange(1, 500, 1)
    similarities_nngs = []
    for k in tqdm(ks):
        metric = TASSimilarityTorch(k=k, batch_size=100, normalize=True)
        sim = metric(torch.Tensor(X).cuda(), torch.Tensor(Y).cuda())
        similarities_nngs.append(sim)
    return ks, similarities_nngs


class MLPNetwork(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_one_batch(X, y, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def experiment(
    n_samples,
    n_features,
    n_classes,
    hidden_dim,
    epochs,
    cluster_std,
):
    X_, y_ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_classes,
        cluster_std=cluster_std,
        random_state=42,
    )
    X_tensor = torch.Tensor(X_).to(device)
    y_tensor = torch.LongTensor(y_).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model1 = MLPNetwork(
        input_dim=n_features,
        hidden_dim=hidden_dim,
        output_dim=n_classes,
    ).to(device)
    model2 = MLPNetwork(
        input_dim=n_features,
        hidden_dim=hidden_dim,
        output_dim=n_classes,
    ).to(device)

    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)

    # Training loop

    similarities = []
    losses1 = []
    losses2 = []
    for epoch in range(epochs):
        loss1_local = 0.0
        loss2_local = 0.0
        for X, y in dataloader:
            loss1 = train_one_batch(X.to(device), y.to(device), model1,
                                    optimizer1, criterion)
            loss2 = train_one_batch(X.to(device), y.to(device), model2,
                                    optimizer2, criterion)
            loss1_local += loss1
            loss2_local += loss2

        losses1.append(loss1_local / len(dataloader))
        losses2.append(loss2_local / len(dataloader))

        with torch.no_grad():
            z1 = model1(X_tensor)
            z2 = model2(X_tensor)
            ks, similarities_nngs = sweep_model_similarity(
                z1.cpu().numpy(),
                z2.cpu().numpy())
            similarities.append(similarities_nngs)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss1: {loss1:.4f}, Loss2: {loss2:.4f}"
            )

    with torch.no_grad():
        z1 = model1(X_tensor)
        z2 = model2(X_tensor)
        ks, similarities_nngs = sweep_model_similarity(z1.cpu().numpy(),
                                                       z2.cpu().numpy())
        similarities.append(similarities_nngs)

    print(z1[:2, :])
    print(F.softmax(z1[:2, :], dim=1))
    print(z2[:2, :])
    print(F.softmax(z2[:2, :], dim=1))

    plt.figure(figsize=(config['width'], config['height']))
    similarities = np.array(similarities).T
    plt.imshow(similarities, aspect='auto', cmap='viridis')
    plt.colorbar(label='Similarity')

    plt.ylabel("$\\alpha$")
    plt.xlabel("Epoch")
    plt.title("Model Similarity Over Time")
    plt.savefig(
        script_dir / config['output_dir'] / f'training_similarity_{n_samples}_{n_features}_{n_classes}_{hidden_dim}_{epochs}_{cluster_std}.pdf',
        dpi=300,
        bbox_inches='tight',
    )

    plt.figure(figsize=(config['width'], config['height']))
    plt.plot(losses1, label='Model 1 Loss')
    plt.plot(losses2, label='Model 2 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.title('Training Loss Over Time')
    plt.legend()
    plt.savefig(
        script_dir / config['output_dir'] / f'training_loss_{n_samples}_{n_features}_{n_classes}_{hidden_dim}_{epochs}_{cluster_std}.pdf',
        dpi=300,
        bbox_inches='tight',
    )


def main():
    from itertools import product
    for n_samples, n_features, n_classes, hidden_dim, epochs, cluster_std in product(
        [1000],
        [5, 10],
        [5],
        [2, 5, 50],
        [40],
        [0.1, 5, 10],
    ):
        print(
            f"Running experiment with n_samples={n_samples}, n_features={n_features}, n_classes={n_classes}, hidden_dim={hidden_dim}, epochs={epochs}, cluster_std={cluster_std}"
        )
        experiment(
            n_samples,
            n_features,
            n_classes,
            hidden_dim,
            epochs,
            cluster_std,
        )


if __name__ == "__main__":
    main()
