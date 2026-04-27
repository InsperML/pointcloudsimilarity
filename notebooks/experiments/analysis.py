import torch
import toml
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path
from sklearn.datasets import make_blobs

from training import BLOCK_CLASS_MAP, MLPNetwork

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
    ) in product(
            sweep['block_classes'],
            sweep['n_blocks'],
            sweep['n_samples'],
            sweep['n_features'],
            sweep['n_classes'],
            sweep['hidden_dim'],
            sweep['epochs'],
            sweep['cluster_std'],
    ):
        block_class = BLOCK_CLASS_MAP[block_name]
        fname_base = (
            f'{n_samples}_{n_features}_{n_classes}_{hidden_dim}'
            f'_{epochs}_{cluster_std}_{block_name}_{n_blocks}blocks'
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

        X_, y_ = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_classes,
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
            fname_base=fname_base,
        )
        yield model1, model2, X_tensor, y_, params


def _class_means_and_corr(z, y_, n_classes):
    class_means = np.stack([z[y_ == c].mean(axis=0) for c in range(n_classes)])
    corr_matrix = np.corrcoef(class_means)
    off_diag = corr_matrix[~np.eye(n_classes, dtype=bool)]
    return class_means, corr_matrix, off_diag.mean()


def _plot_means_and_corr(class_means, corr_matrix, mean_corr, layer_label, params, fname_prefix):
    n_classes = params['n_classes']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(config['width'], config['height']))

    im1 = ax1.imshow(class_means, aspect='auto', cmap='RdBu_r')
    plt.colorbar(im1, ax=ax1, label='Mean activation')
    ax1.set_xlabel('Feature index')
    ax1.set_ylabel('Class')
    ax1.set_yticks(range(n_classes))
    ax1.set_title(
        f"{layer_label} class means — {params['block_name']}, {params['n_blocks']} blocks"
        f"\nmean inter-class correlation: {mean_corr:.4f}"
    )

    im2 = ax2.imshow(corr_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im2, ax=ax2, label='Correlation')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Class')
    ax2.set_xticks(range(n_classes))
    ax2.set_yticks(range(n_classes))
    ax2.set_title('Inter-class correlation matrix')

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
    class_means, corr_matrix, mean_corr = _class_means_and_corr(z, y_, params['n_classes'])
    print(f"  [fc1]    mean inter-class correlation: {mean_corr:.4f}")
    _plot_means_and_corr(class_means, corr_matrix, mean_corr, 'fc1', params, 'fc1_class_means')


def plot_blocks_class_means(model, X_tensor, y_, params):
    with torch.no_grad():
        _, z = model(X_tensor)  # x_ is the post-block representation
        z = z.numpy()
    class_means, corr_matrix, mean_corr = _class_means_and_corr(z, y_, params['n_classes'])
    print(f"  [blocks] mean inter-class correlation: {mean_corr:.4f}")
    _plot_means_and_corr(class_means, corr_matrix, mean_corr, 'blocks out', params, 'blocks_class_means')


if __name__ == "__main__":
    for model1, model2, X_tensor, y_, params in iter_experiments():
        print(f"Loaded: {params['fname_base']}")
        plot_blocks_class_means(model1, X_tensor, y_, params)
