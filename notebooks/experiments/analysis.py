import torch
import toml
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from training import BLOCK_CLASS_MAP, MLPNetwork, make_equidistant_centers

script_dir = Path(__file__).parent
config = toml.load(script_dir / "settings.toml")['figures']
sweep = toml.load(script_dir / "experiments.toml")['sweep']

out_dir = script_dir / config['output_dir']


def iter_experiments():
    for (
            block_name,
            n_blocks,
            n_samples,
            n_features,
            n_classes,
            hidden_dim,
            epochs,
            cluster_std,
            center_scale,
    ) in product(
            sweep['block_classes'],
            sweep['n_blocks'],
            sweep['n_samples'],
            sweep['n_features'],
            sweep['n_classes'],
            sweep['hidden_dim'],
            sweep['epochs'],
            sweep['cluster_std'],
            sweep['center_scale'],
    ):
        block_class = BLOCK_CLASS_MAP[block_name]
        fname_base = (
            f'{n_samples}_{n_features}_{n_classes}_{hidden_dim}'
            f'_{epochs}_{cluster_std}_{center_scale}_{block_name}_{n_blocks}blocks'
        )

        path1 = out_dir / f'model1_{fname_base}.pt'
        path2 = out_dir / f'model2_{fname_base}.pt'
        if not path1.exists() or not path2.exists():
            print(f"Skipping {fname_base}: weights not found")
            continue

        model1 = MLPNetwork(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            output_dim=n_classes,
            block_class=block_class,
            n_blocks=n_blocks,
        )
        model2 = MLPNetwork(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            output_dim=n_classes,
            block_class=block_class,
            n_blocks=n_blocks,
        )
        model1.load_state_dict(torch.load(path1, map_location='cpu'))
        model2.load_state_dict(torch.load(path2, map_location='cpu'))
        model1.eval()
        model2.eval()

        centers = make_equidistant_centers(n_classes, n_features, scale=center_scale)
        X_, y_ = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=centers,
            cluster_std=cluster_std,
            random_state=42,
        )
        X_tensor = torch.tensor(X_, dtype=torch.float32)

        params = dict(
            block_name=block_name,
            block_class=block_class,
            n_blocks=n_blocks,
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            epochs=epochs,
            cluster_std=cluster_std,
            center_scale=center_scale,
            fname_base=fname_base,
        )
        yield model1, model2, X_tensor, y_, params


def _compute_stats(z, y_, n_classes):
    class_means = np.stack([z[y_ == c].mean(axis=0) for c in range(n_classes)])
    class_stds  = np.stack([z[y_ == c].std(axis=0)  for c in range(n_classes)])
    corr_matrix = np.corrcoef(class_stds)
    off_diag = corr_matrix[~np.eye(n_classes, dtype=bool)]
    return class_means, class_stds, corr_matrix, off_diag.mean()


def _plot_stats(class_means, corr_matrix, mean_corr, layer_label, params, fname_prefix):
    n_classes = params['n_classes']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(config['width'], config['height']))

    im1 = ax1.imshow(class_means, aspect='auto', cmap='RdBu_r')
    plt.colorbar(im1, ax=ax1, label='Mean activation')
    ax1.set_xlabel('Feature index')
    ax1.set_ylabel('Class')
    ax1.set_yticks(range(n_classes))
    ax1.set_title(
        f"{layer_label} class means — {params['block_name']}, {params['n_blocks']} blocks"
    )

    im2 = ax2.imshow(corr_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im2, ax=ax2, label='Correlation of stds')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Class')
    ax2.set_xticks(range(n_classes))
    ax2.set_yticks(range(n_classes))
    ax2.set_title(f'Inter-class correlation of stds\nmean: {mean_corr:.4f}')

    fig.tight_layout()
    fig.savefig(
        out_dir / f"{fname_prefix}_{params['fname_base']}.pdf",
        dpi=300,
        bbox_inches='tight',
    )
    plt.close(fig)


def plot_fc1_class_means(model, X_tensor, y_, params):
    with torch.no_grad():
        z = model.fc1(X_tensor).numpy()
    class_means, class_stds, corr_matrix, mean_corr = _compute_stats(z, y_, params['n_classes'])
    print(f"  [fc1]    mean inter-class std correlation: {mean_corr:.4f}")
    _plot_stats(class_means, corr_matrix, mean_corr, 'fc1', params, 'fc1_class_means')


def plot_blocks_class_means(model, X_tensor, y_, params):
    with torch.no_grad():
        _, z = model(X_tensor)
        z = z.numpy()
    class_means, class_stds, corr_matrix, mean_corr = _compute_stats(z, y_, params['n_classes'])
    print(f"  [blocks] mean inter-class std correlation: {mean_corr:.4f}")
    _plot_stats(class_means, corr_matrix, mean_corr, 'blocks out', params, 'blocks_class_means')


def plot_class_selectivity(model, X_tensor, y_, params):
    with torch.no_grad():
        _, z = model(X_tensor)
        z = z.numpy()

    n_classes = params['n_classes']
    class_means = np.stack([z[y_ == c].mean(axis=0) for c in range(n_classes)])

    # Z-score each feature across classes: reveals which features are class-specific
    z_scored = (class_means - class_means.mean(axis=0)) / (class_means.std(axis=0) + 1e-8)

    # For each feature, which class has the highest z-score?
    preferred_class = z_scored.argmax(axis=0)  # (hidden_dim,)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(config['width'], config['height']),
                                   gridspec_kw={'height_ratios': [4, 1]})

    vmax = np.abs(z_scored).max()
    im = ax1.imshow(z_scored, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax1, label='Z-score across classes')
    ax1.set_ylabel('Class')
    ax1.set_yticks(range(n_classes))
    ax1.set_title(
        f"Class selectivity — {params['block_name']}, {params['n_blocks']} blocks"
    )
    ax1.set_xticks([])

    ax2.bar(range(z_scored.shape[1]), preferred_class, width=1.0, color=[
        plt.cm.tab10(c / n_classes) for c in preferred_class
    ])
    ax2.set_xlabel('Feature index')
    ax2.set_ylabel('Preferred\nclass')
    ax2.set_yticks(range(n_classes))
    ax2.set_xlim(0, z_scored.shape[1])

    fig.tight_layout()
    fig.savefig(
        out_dir / f"selectivity_{params['fname_base']}.pdf",
        dpi=300,
        bbox_inches='tight',
    )
    plt.close(fig)


def plot_tsne(model, X_tensor, y_, params):
    with torch.no_grad():
        _, z = model(X_tensor)
        z = z.numpy()

    embedding = TSNE(n_components=2, random_state=42).fit_transform(z)

    n_classes = params['n_classes']
    fig, ax = plt.subplots(figsize=(config['height'], config['height']))
    for c in range(n_classes):
        mask = y_ == c
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   label=f'Class {c}', s=15, alpha=0.7, color=plt.cm.tab10(c / n_classes))
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(markerscale=2, fontsize=7)
    ax.set_title(f"t-SNE — {params['block_name']}, {params['n_blocks']} blocks")
    fig.tight_layout()
    fig.savefig(
        out_dir / f"tsne_{params['fname_base']}.pdf",
        dpi=300,
        bbox_inches='tight',
    )
    plt.close(fig)


def plot_confusion_matrix(model, X_tensor, y_, params):
    with torch.no_grad():
        logits, _ = model(X_tensor)
        y_pred = logits.argmax(dim=1).numpy()

    n_classes = params['n_classes']
    cm = confusion_matrix(y_, y_pred, labels=range(n_classes))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    accuracy = np.diag(cm).sum() / cm.sum()

    fig, ax = plt.subplots(figsize=(config['height'], config['height']))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Fraction of true class')
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_title(
        f"Confusion matrix — {params['block_name']}, {params['n_blocks']} blocks"
        f"\naccuracy: {accuracy:.3f}"
    )
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, f'{cm_norm[i, j]:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if cm_norm[i, j] > 0.5 else 'black')
    fig.tight_layout()
    fig.savefig(
        out_dir / f"confusion_{params['fname_base']}.pdf",
        dpi=300,
        bbox_inches='tight',
    )
    plt.close(fig)


if __name__ == "__main__":
    for model1, model2, X_tensor, y_, params in iter_experiments():
        print(f"Loaded: {params['fname_base']}")
        plot_blocks_class_means(model1, X_tensor, y_, params)
        plot_class_selectivity(model1, X_tensor, y_, params)
        plot_tsne(model1, X_tensor, y_, params)
        plot_confusion_matrix(model1, X_tensor, y_, params)
