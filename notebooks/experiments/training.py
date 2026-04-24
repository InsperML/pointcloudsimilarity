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
    ks = np.arange(1, X.shape[0] - 1, 1)
    #ks = np.arange(1, 500, 1)
    similarities_nngs = []
    for k in tqdm(ks, disable=True):
        metric = TASSimilarityTorch(k=k, batch_size=100, normalize=True)
        sim = metric(torch.Tensor(X).cuda(), torch.Tensor(Y).cuda())
        similarities_nngs.append(sim)
    return ks, similarities_nngs


class ResidualBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim)
        self.dropout = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(dim, dim)

    def forward(self, x):
        identity = x
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out += identity
        return F.relu(out)

class MLPNetwork(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.resblock = ResidualBlock(hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x_ = self.resblock(x)
        x = self.fc2(x_)
        return x, x_


def train_one_batch(X, y, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    outputs, _ = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def make_blobs_dataset(n_samples, n_features, n_classes, cluster_std):
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_classes,
        cluster_std=cluster_std,
        random_state=42,
    )
    return X, y

def make_geometric_dataset(n_samples, n_features, n_classes, cluster_std):
    # Make a set of n_classes points in n_features dimensions.
    corners = np.random.randn(n_classes, n_features)

    # Make a Delauney triangulation of the corners to get simplices.
    from scipy.spatial import Delaunay
    tri = Delaunay(corners)

    # Sample points uniformly from the simplices.
    points = []
    for simplex in tri.simplices:
        simplex_corners = corners[simplex]
        for _ in range(n_samples // len(tri.simplices)):
            weights = np.random.dirichlet(np.ones(len(simplex)))
            point = np.dot(weights, simplex_corners)
            points.append(point)

    return np.array(points), np.repeat(np.arange(n_classes), n_samples // len(tri.simplices))


make_dataset = make_blobs_dataset


def experiment(
    n_samples,
    n_features,
    n_classes,
    hidden_dim,
    epochs,
    cluster_std,
):
    X_, y_ = make_dataset(n_samples, n_features, n_classes, cluster_std)

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
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

    # Training loop

    similarities = []
    losses1 = []
    losses2 = []
    best_loss = float('inf')
    no_improve_count = 0
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

        avg_loss1 = loss1_local / len(dataloader)
        avg_loss2 = loss2_local / len(dataloader)
        losses1.append(avg_loss1)
        losses2.append(avg_loss2)

        combined_loss = avg_loss1 + avg_loss2
        if combined_loss < best_loss:
            best_loss = combined_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        with torch.no_grad():
            _, z1 = model1(X_tensor)
            _, z2 = model2(X_tensor)
            ks, similarities_nngs = sweep_model_similarity(
                z1.cpu().numpy(),
                z2.cpu().numpy())
            similarities.append(similarities_nngs)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss1: {loss1:.4f}, Loss2: {loss2:.4f}"
            )

        if no_improve_count >= 10:
            print(f"Early stopping at epoch {epoch + 1}: no improvement for 10 epochs.")
            break

    with torch.no_grad():
        _, z1 = model1(X_tensor)
        _, z2 = model2(X_tensor)
        ks, similarities_nngs = sweep_model_similarity(z1.cpu().numpy(),
                                                       z2.cpu().numpy())
        similarities.append(similarities_nngs)

    print(z1[:2, :])
    print(F.softmax(z1[:2, :], dim=1))
    print(z2[:2, :])
    print(F.softmax(z2[:2, :], dim=1))

    plt.figure(figsize=(config['width'], config['height']))
    similarities = np.array(similarities).T
    im = plt.imshow(similarities, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    cs = plt.contour(similarities, levels=np.arange(0, 1.1, 0.1), colors='white', linewidths=0.5, alpha=0.6)
    plt.clabel(cs, fmt='%.1f', fontsize=7)
    plt.colorbar(im, label='Similarity')

    plt.ylabel("$n$")
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
        [200],
        [10],
        [2, 4, 20],
        [2, 6, 10, 20, 100],
        [250],
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
