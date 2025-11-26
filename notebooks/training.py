import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        n_layers,
        output_size,
        p_dropout=0.0,
    ):
        super().__init__()
        self.input_adapter = nn.Linear(input_size, hidden_size)
        self.layers = nn.Sequential(*[
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p_dropout),
        ] * n_layers)
        self.output_adapter = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_adapter(x)
        x = self.layers(x)
        x = self.output_adapter(x)
        return x
    
    def get_final_embeddings(self, x):
        x = self.input_adapter(x)
        x = self.layers(x)
        return x
    
    def get_intermediate_embeddings(self, x):
        y = [x]
        x = self.input_adapter(x)
        y.append(x)
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                y.append(x)
        x = self.output_adapter(x)
        y.append(x)
        return y

def train_one_batch(batch_x, batch_y, model, optimizer, criterion):
    model.train()
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_one_batch(batch_x, batch_y, model, criterion, output_accuracy=True):
    model.eval()
    with torch.no_grad():
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        predicted = torch.argmax(outputs, dim=1)
        accuracy = (predicted == batch_y).float().mean().item() if output_accuracy else None
    return loss.item(), accuracy


def train_one_epoch(train_loader, eval_loader, model, optimizer, criterion):
    total_loss = 0
    total_val_loss = 0
    total_val_accuracy = 0

    device = next(model.parameters()).device

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        total_loss += train_one_batch(batch_x, batch_y, model, optimizer,
                                      criterion)

    for batch_x, batch_y in eval_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        val_loss_batch, eval_acc = eval_one_batch(batch_x, batch_y, model,
                                                  criterion)
        total_val_loss += val_loss_batch
        total_val_accuracy += eval_acc
    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(eval_loader)
    avg_val_accuracy = total_val_accuracy / len(eval_loader)
    return avg_train_loss, avg_val_loss, avg_val_accuracy
