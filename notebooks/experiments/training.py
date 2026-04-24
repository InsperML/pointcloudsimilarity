from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import toml
import torch
from dataset_ import make_dataloader
from model_ import BLOCK_CLASS_MAP, ResidualBlock, make_criterion, make_model_and_optimizer
from similarity_ import SimilarityTracker

script_dir = Path(__file__).parent


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrainingSession:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.losses = []

    def init_epoch(self) -> None:
        self.epoch_loss = 0.0
        self.num_batches = 0

    def train_batch(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.model.train()

        self.optimizer.zero_grad()

        outputs, _ = self.model(X)
        batch_loss = self.criterion(outputs, y)

        batch_loss.backward()
        self.optimizer.step()

        self.epoch_loss += batch_loss.item()
        self.num_batches += 1

    def end_epoch(self) -> float:
        avg_loss = self.epoch_loss / self.num_batches if self.num_batches > 0 else 0.0
        self.losses.append(avg_loss)
        return avg_loss


class EarlyStopping:
    def __init__(self, patience: int = 10) -> None:
        self.patience = patience
        self.best_loss = float('inf')
        self.no_improve_count = 0

    def step(self, current_loss: float) -> bool:
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.no_improve_count = 0
            return False  # Not stopping
        else:
            self.no_improve_count += 1
            return (
                self.no_improve_count >= self.patience
            )  # Stop if no improvement for 'patience' epochs


def plot_results(
    similarities: np.ndarray,
    losses_1: np.ndarray,
    losses_2: np.ndarray,
    fname_base: str,
    title_suffix: str,
) -> None:
    config = toml.load(script_dir / 'settings.toml')['figures']
    out_dir = script_dir / config['output_dir']

    def make_figure_filename(metric: str) -> Path:
        return out_dir / f'training_{metric}_{fname_base}.pdf'

    figsize = (config['width'], config['height'])

    plt.figure(figsize=figsize)
    im = plt.imshow(similarities, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    cs = plt.contour(
        similarities,
        levels=np.arange(0, 1.1, 0.1),
        colors='white',
        linewidths=0.5,
        alpha=0.6,
    )
    plt.clabel(cs, fmt='%.1f', fontsize=7)
    plt.colorbar(im, label='Similarity')
    plt.ylabel('$n$')
    plt.xlabel('Epoch')
    plt.title(f'Model Similarity Over Time ({title_suffix})')
    plt.savefig(make_figure_filename('similarity'), dpi=300, bbox_inches='tight')

    plt.figure(figsize=figsize)
    plt.plot(losses_1, label='Model 1 Loss')
    plt.plot(losses_2, label='Model 2 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 1.0)
    plt.title(f'Training Loss Over Time ({title_suffix})')
    plt.legend()
    plt.savefig(make_figure_filename('loss'), dpi=300, bbox_inches='tight')


def save_models(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    fname_base: str,
) -> None:
    config = toml.load(script_dir / 'settings.toml')['figures']
    out_dir = script_dir / config['output_dir']
    torch.save(model1.state_dict(), out_dir / f'model1_{fname_base}.pt')
    torch.save(model2.state_dict(), out_dir / f'model2_{fname_base}.pt')


def experiment(
    n_samples: int,
    n_features: int,
    n_classes: int,
    hidden_dim: int,
    epochs: int,
    cluster_std: float,
    block_class: type[torch.nn.Module] = ResidualBlock,
    n_blocks: int = 1,
    patience: int = 200,
    center_scale: float = 10.0,
):
    print(
        ', '.join(
            [
                f'Running experiment: block={block_class.__name__}',
                f'n_blocks={n_blocks}',
                f'n_samples={n_samples}',
                f'n_features={n_features}',
                f'n_classes={n_classes}',
                f'hidden_dim={hidden_dim}',
                f'epochs={epochs}',
                f'cluster_std={cluster_std}',
                f'center_scale={center_scale}',
            ]
        )
    )

    device = get_device()
    X_tensor, y_tensor, dataloader = make_dataloader(
        n_samples,
        n_features,
        n_classes,
        cluster_std,
        center_scale,
        device=device,
    )

    model1, optimizer1 = make_model_and_optimizer(
        n_features,
        hidden_dim,
        n_classes,
        device,
        lr=1e-3,
        block_class=block_class,
        n_blocks=n_blocks,
    )

    model2, optimizer2 = make_model_and_optimizer(
        n_features,
        hidden_dim,
        n_classes,
        device,
        lr=1e-3,
        block_class=block_class,
        n_blocks=n_blocks,
    )

    # Training setup
    criterion = make_criterion()

    # Training loop
    trainer_1 = TrainingSession(model1, optimizer1, criterion)
    trainer_2 = TrainingSession(model2, optimizer2, criterion)
    early_stopping = EarlyStopping(patience=patience)
    similarity_tracker = SimilarityTracker(model1, model2, X_tensor)

    for epoch in range(1, epochs + 1):
        trainer_1.init_epoch()
        trainer_2.init_epoch()

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            trainer_1.train_batch(X, y)
            trainer_2.train_batch(X, y)

        avg_loss1 = trainer_1.end_epoch()
        avg_loss2 = trainer_2.end_epoch()
        if epoch % 10 == 0:
            print(
                f'Epoch {epoch}/{epochs}, Loss1: {avg_loss1:.4f}, Loss2: {avg_loss2:.4f}'
            )

        similarity_tracker.add_similarity()

        combined_loss = avg_loss1 + avg_loss2
        if early_stopping.step(combined_loss):
            print(
                f'Early stopping at epoch {epoch}: '
                f'no improvement for {early_stopping.patience} epochs.'
            )
            break

    similarity_tracker.add_similarity()

    block_name = block_class.__name__
    fname_base = (
        f'{n_samples}_{n_features}_{n_classes}_{hidden_dim}_{epochs}_{cluster_std}'
        f'_{center_scale}_{block_name}_{n_blocks}blocks'
    )
    save_models(model1, model2, fname_base)

    similarities = np.array(similarity_tracker.similarities).T
    losses_1 = np.array(trainer_1.losses)
    losses_2 = np.array(trainer_2.losses)
    plot_results(
        similarities,
        losses_1,
        losses_2,
        fname_base,
        title_suffix=f'{block_name}, {n_blocks} blocks',
    )


def main():
    sweep = toml.load(script_dir / 'experiments.toml')['sweep']
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
        center_scale,
    ) in product(
        block_classes,
        sweep['n_blocks'],
        sweep['n_samples'],
        sweep['n_features'],
        sweep['n_classes'],
        sweep['hidden_dim'],
        sweep['epochs'],
        sweep['cluster_std'],
        sweep['center_scale'],
    ):
        experiment(
            n_samples,
            n_features,
            n_classes,
            hidden_dim,
            epochs,
            cluster_std,
            block_class=block_class,
            n_blocks=n_blocks,
            center_scale=center_scale,
        )


if __name__ == '__main__':
    main()
