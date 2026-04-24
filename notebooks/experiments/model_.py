import torch
import torch.nn.functional as F


class ResidualBlock(torch.nn.Module):
    # Params: 2*(dim² + dim)
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


class SimpleMLP(torch.nn.Module):
    # Same params as ResidualBlock (2*(dim² + dim)), no skip connection.
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim)
        self.dropout = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return F.relu(out)


class SelfAttentionBlock(torch.nn.Module):
    # Params ≈ 2*(dim² + dim): attention over two token_dim-sized feature halves
    # (dim²+2*dim) plus a linear layer (dim²+dim) ≈ 2*dim²+3*dim.
    # Requires even dim.
    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0, f'SelfAttentionBlock requires even dim, got {dim}'
        self.token_dim = dim // 2
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=1,
            dropout=0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        x_tokens = x.view(B, 2, self.token_dim)  # treat two feature halves as tokens
        attn_out, _ = self.attn(x_tokens, x_tokens, x_tokens)
        out = attn_out.reshape(B, D)
        return F.relu(out)


class SelfAttentionResidualBlock(torch.nn.Module):
    # Self-attention with residual + LayerNorm; no feedforward.
    # Params ≈ dim² + 4*dim (roughly half of ResidualBlock). Requires even dim.
    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0, (
            f'SelfAttentionResidualBlock requires even dim, got {dim}'
        )
        self.token_dim = dim // 2
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=1,
            dropout=0.2,
            batch_first=True,
        )
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        x_tokens = x.view(B, 2, self.token_dim)
        attn_out, _ = self.attn(x_tokens, x_tokens, x_tokens)
        return F.relu(self.norm(x + attn_out.reshape(B, D)))


class TransformerBlock(torch.nn.Module):
    # Standard transformer encoder block: attention + feedforward, both with residual + LayerNorm.
    # Params ≈ 2*dim² + 7*dim (attention: dim²+2*dim, FFN with dim//2 width: dim²+1.5*dim, norms: 4*dim).
    # Requires even dim.
    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0, f'TransformerBlock requires even dim, got {dim}'
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        x_tokens = x.view(B, 2, self.token_dim)
        attn_out, _ = self.attn(x_tokens, x_tokens, x_tokens)
        x = self.norm1(x + attn_out.reshape(B, D))
        x = self.norm2(x + self.ff2(self.dropout(F.relu(self.ff1(x)))))
        return x


BLOCK_CLASS_MAP: dict[str, type[torch.nn.Module]] = {
    'ResidualBlock': ResidualBlock,
    'SimpleMLP': SimpleMLP,
    'SelfAttentionBlock': SelfAttentionBlock,
    'SelfAttentionResidualBlock': SelfAttentionResidualBlock,
    'TransformerBlock': TransformerBlock,
}


class MLPNetwork(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        block_class: type[torch.nn.Module] = ResidualBlock,
        n_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.blocks = torch.nn.Sequential(
            *[block_class(hidden_dim) for _ in range(n_blocks)]
        )
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.fc1(x)
        x_ = self.blocks(x)
        x = self.fc2(x_)
        return x, x_


def make_model_and_optimizer(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    device: torch.device,
    lr: float = 1e-3,
    block_class: type[torch.nn.Module] = ResidualBlock,
    n_blocks: int = 1,
) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = MLPNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        block_class=block_class,
        n_blocks=n_blocks,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def make_criterion() -> torch.nn.Module:
    return torch.nn.CrossEntropyLoss()
