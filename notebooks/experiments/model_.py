import torch
import torch.nn.functional as F


class ResidualBlock(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim)
        self.dropout = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out += identity
        return F.relu(out)


class MLPNetwork(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.resblock = ResidualBlock(hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.fc1(x)
        x_ = self.resblock(x)
        x = self.fc2(x_)
        return x, x_


def make_model_and_optimizer(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    device: torch.device,
    lr: float = 1e-3,
) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = MLPNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def make_criterion() -> torch.nn.Module:
    return torch.nn.BCEWithLogitsLoss()
