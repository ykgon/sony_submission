import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config, fix_seed
from data_loader import get_train_val_loaders
from model import ImplicitGNN
from utils import save_best_model

DEVICE = Config.DEVICE


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def validate_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def main():
    fix_seed(Config.SEED)

    train_loader, val_loader = get_train_val_loaders()
    model = ImplicitGNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10, threshold=1e-4
    )
    criterion = torch.nn.MSELoss()

    best_val = float("inf")
    patience = 0
    history = []

    for epoch in range(Config.MAX_EPOCH):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate_epoch(model, val_loader, criterion)

        # adjust lr
        scheduler.step(val_loss)

        # save best
        best_val, patience = save_best_model(model, best_val, val_loss, patience)

        history.append((epoch, train_loss, val_loss))

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{Config.MAX_EPOCH} — "
                  f"Train: {train_loss:.5f}, Val: {val_loss:.5f}")

        if patience >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping.")
            break

    # Plot losses over epochs
    hist = np.array(history)
    plt.plot(hist[:, 0], hist[:, 1], label="Train")
    plt.plot(hist[:, 0], hist[:, 2], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save final checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": val_loss
    }, "Implicit_GNN_checkpoint.pth")


if __name__ == "__main__":
    main()
