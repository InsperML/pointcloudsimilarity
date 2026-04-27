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
    # Params: 2*(dim² + dim)
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


class SimpleMLP(torch.nn.Module):
    # Same params as ResidualBlock (2*(dim² + dim)), no skip connection.
    def __init__(self, dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim)
        self.dropout = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(dim, dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return F.relu(out)


class SelfAttentionBlock(torch.nn.Module):
    # Params ≈ 2*(dim² + dim): attention over two token_dim-sized feature halves
    # (dim²+2*dim) plus a linear layer (dim²+dim) ≈ 2*dim²+3*dim.
    # Requires even dim.
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, f"SelfAttentionBlock requires even dim, got {dim}"
        self.token_dim = dim // 2
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=1,
            dropout=0.0,
            batch_first=True,
        )

    def forward(self, x):
        B, D = x.shape
        x_tokens = x.view(B, 2,
                          self.token_dim)  # treat two feature halves as tokens
        attn_out, _ = self.attn(x_tokens, x_tokens, x_tokens)
        out = attn_out.reshape(B, D)
        return F.relu(out)


class SelfAttentionResidualBlock(torch.nn.Module):
    # Self-attention with residual + LayerNorm; no feedforward.
    # Params ≈ dim² + 4*dim (roughly half of ResidualBlock). Requires even dim.
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, f"SelfAttentionResidualBlock requires even dim, got {dim}"
        self.token_dim = dim // 2
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=1,
            dropout=0.2,
            batch_first=True,
        )
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        B, D = x.shape
        x_tokens = x.view(B, 2, self.token_dim)
        attn_out, _ = self.attn(x_tokens, x_tokens, x_tokens)
        return F.relu(self.norm(x + attn_out.reshape(B, D)))


class TransformerBlock(torch.nn.Module):
    # Standard transformer encoder block: attention + feedforward, both with residual + LayerNorm.
    # Params ≈ 2*dim² + 7*dim (attention: dim²+2*dim, FFN with dim//2 width: dim²+1.5*dim, norms: 4*dim).
    # Requires even dim.
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, f"TransformerBlock requires even dim, got {dim}"
        self.token_dim = dim // 2
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=1,
            dropout=0.2,
            batch_first=True,
        )
        self.ff1 = torch.nn.Linear(dim, dim // 2)
        self.ff2 = torch.nn.Linear(dim // 2, dim)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):
        B, D = x.shape
        x_tokens = x.view(B, 2, self.token_dim)
        attn_out, _ = self.attn(x_tokens, x_tokens, x_tokens)
        x = self.norm1(x + attn_out.reshape(B, D))
        x = self.norm2(x + self.ff2(self.dropout(F.relu(self.ff1(x)))))
        return x


BLOCK_CLASS_MAP = {
    'ResidualBlock': ResidualBlock,
    'SimpleMLP': SimpleMLP,
    'SelfAttentionBlock': SelfAttentionBlock,
    'SelfAttentionResidualBlock': SelfAttentionResidualBlock,
    'TransformerBlock': TransformerBlock,
}


class MLPNetwork(torch.nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 block_class=ResidualBlock,
                 n_blocks=1):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.blocks = torch.nn.Sequential(
            *[block_class(hidden_dim) for _ in range(n_blocks)]
        )
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x_ = self.blocks(x)
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


def experiment(
    n_samples,
    n_features,
    n_classes,
    hidden_dim,
    epochs,
    cluster_std,
    block_class=ResidualBlock,
    n_blocks=1,
    patience=200,
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
    dataloader = DataLoader(dataset, batch_size=X_tensor.shape[0], shuffle=True)

    model1 = MLPNetwork(
        input_dim=n_features,
        hidden_dim=hidden_dim,
        output_dim=n_classes,
        block_class=block_class,
        n_blocks=n_blocks,
    ).to(device)
    model2 = MLPNetwork(
        input_dim=n_features,
        hidden_dim=hidden_dim,
        output_dim=n_classes,
        block_class=block_class,
        n_blocks=n_blocks,
    ).to(device)

    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    lr = 5e-4
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr)

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

        if no_improve_count >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}: no improvement for 10 epochs."
            )
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

    block_name = block_class.__name__
    fname_base = f'{n_samples}_{n_features}_{n_classes}_{hidden_dim}_{epochs}_{cluster_std}_{block_name}_{n_blocks}blocks'

    out_dir = script_dir / config['output_dir']
    torch.save(model1.state_dict(), out_dir / f'model1_{fname_base}.pt')
    torch.save(model2.state_dict(), out_dir / f'model2_{fname_base}.pt')

    plt.figure(figsize=(config['width'], config['height']))
    similarities = np.array(similarities).T
    im = plt.imshow(similarities,
                    aspect='auto',
                    cmap='viridis',
                    vmin=0,
                    vmax=1)
    cs = plt.contour(similarities,
                     levels=np.arange(0, 1.1, 0.1),
                     colors='white',
                     linewidths=0.5,
                     alpha=0.6)
    plt.clabel(cs, fmt='%.1f', fontsize=7)
    plt.colorbar(im, label='Similarity')

    plt.ylabel("$n$")
    plt.xlabel("Epoch")
    plt.title(f"Model Similarity Over Time ({block_name}, {n_blocks} blocks)")
    plt.savefig(
        out_dir /
        f'training_similarity_{fname_base}.pdf',
        dpi=300,
        bbox_inches='tight',
    )

    plt.figure(figsize=(config['width'], config['height']))
    plt.plot(losses1, label='Model 1 Loss')
    plt.plot(losses2, label='Model 2 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 1.0)

    plt.title(f'Training Loss Over Time ({block_name}, {n_blocks} blocks)')
    plt.legend()
    plt.savefig(
        out_dir / f'training_loss_{fname_base}.pdf',
        dpi=300,
        bbox_inches='tight',
    )


def main():
    from itertools import product
    sweep = toml.load(script_dir / "experiments.toml")['sweep']
    block_classes = [BLOCK_CLASS_MAP[name] for name in sweep['block_classes']]

    for (
            block_class,
            n_blocks,
            n_samples,
            n_features,
            n_classes,
            hidden_dim,
            epochs,
            cluster_std,
    ) in product(
            block_classes,
            sweep['n_blocks'],
            sweep['n_samples'],
            sweep['n_features'],
            sweep['n_classes'],
            sweep['hidden_dim'],
            sweep['epochs'],
            sweep['cluster_std'],
    ):
        print(
            f"Running experiment: block={block_class.__name__}, n_blocks={n_blocks}, "
            f"n_samples={n_samples}, n_features={n_features}, n_classes={n_classes}, "
            f"hidden_dim={hidden_dim}, epochs={epochs}, cluster_std={cluster_std}")
        experiment(
            n_samples,
            n_features,
            n_classes,
            hidden_dim,
            epochs,
            cluster_std,
            block_class=block_class,
            n_blocks=n_blocks,
        )


if __name__ == "__main__":
    main()
