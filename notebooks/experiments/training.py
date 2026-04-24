from itertools import product
from pathlib import Path
from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
import toml
import torch
from dataset_ import make_dataloader
from model_ import make_criterion, make_model_and_optimizer
from sentence_transformers.util import get_device_name
from similarity_ import SimilarityTracker


def get_device() -> torch.device:
    return torch.device(get_device_name())


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
    n_samples: int,
    n_features: int,
    n_classes: int,
    hidden_dim: int,
    epochs: int,
    cluster_std: float,
) -> None:
    script_dir = Path(__file__).parent
    config = toml.load(script_dir / 'settings.toml')['figures']

    def make_figure_filename(metric: str) -> Path:
        return (
            script_dir
            / config['output_dir']
            / f'training_{metric}_{n_samples}_{n_features}_{n_classes}_{hidden_dim}_{epochs}_{cluster_std}.pdf'
        )

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
    plt.title('Model Similarity Over Time')
    plt.savefig(make_figure_filename('similarity'), dpi=300, bbox_inches='tight')

    plt.figure(figsize=figsize)
    plt.plot(losses_1, label='Model 1 Loss')
    plt.plot(losses_2, label='Model 2 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.savefig(make_figure_filename('loss'), dpi=300, bbox_inches='tight')


def experiment(
    n_samples: int,
    n_features: int,
    n_classes: int,
    hidden_dim: int,
    epochs: int,
    cluster_std: float,
):
    print(
        ', '.join(
            [
                f'Running experiment with n_samples={n_samples}',
                f'n_features={n_features}',
                f'n_classes={n_classes}',
                f'hidden_dim={hidden_dim}',
                f'epochs={epochs}',
                f'cluster_std={cluster_std}',
            ]
        )
    )

    device = get_device()
    X_tensor, y_tensor, dataloader = make_dataloader(
        n_samples,
        n_features,
        n_classes,
        cluster_std,
        batch_size=32,
        device=device,
    )

    output_dim = y_tensor.shape[1] if len(y_tensor.shape) > 1 else n_classes

    model1, optimizer1 = make_model_and_optimizer(
        n_features,
        hidden_dim,
        output_dim,
        device,
        lr=1e-3,
    )

    model2, optimizer2 = make_model_and_optimizer(
        n_features,
        hidden_dim,
        output_dim,
        device,
        lr=1e-3,
    )

    # Training setup
    criterion = make_criterion()

    # Training loop
    trainer_1 = TrainingSession(model1, optimizer1, criterion)
    trainer_2 = TrainingSession(model2, optimizer2, criterion)
    early_stopping = EarlyStopping(patience=10)
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

    similarities = np.array(similarity_tracker.similarities).T
    losses_1 = np.array(trainer_1.losses)
    losses_2 = np.array(trainer_2.losses)
    plot_results(
        similarities,
        losses_1,
        losses_2,
        n_samples,
        n_features,
        n_classes,
        hidden_dim,
        epochs,
        cluster_std,
    )


def main():
    def experiment_options() -> Generator[tuple[int, int, int, int, int, float]]:
        options = {
            'n_samples': [200],
            'n_features': [10],
            'n_classes': [2, 4, 20],
            'hidden_dim': [2, 6, 10, 20, 100],
            'epochs': [250],
            'cluster_std': [0.1, 5, 10],
        }
        yield from product(
            options['n_samples'],
            options['n_features'],
            options['n_classes'],
            options['hidden_dim'],
            options['epochs'],
            options['cluster_std'],
        )

    for options in experiment_options():
        experiment(*options)


if __name__ == '__main__':
    main()
