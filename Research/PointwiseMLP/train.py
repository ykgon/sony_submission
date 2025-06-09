import torch
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from data_loader import get_train_val_test_loaders
from model import PointwiseMLP
from utils import save_model

def train():
    torch.manual_seed(Config.SEED)
    train_loader, val_loader, _ = get_train_val_test_loaders()

    model = PointwiseMLP(Config.HIDDEN_DIM).to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.HuberLoss()

    history = []  # [(epoch, train_loss, val_loss), ...]

    for epoch in range(1, Config.MAX_EPOCH+1):
        # --- training ---
        model.train()
        total_train = 0.0
        for x, y in train_loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train += loss.item() * x.size(0)
        avg_train = total_train / len(train_loader.dataset)

        # --- validation ---
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                pred = model(x)
                loss = criterion(pred, y)
                total_val += loss.item() * x.size(0)
        avg_val = total_val / len(val_loader.dataset)

        history.append((epoch, avg_train, avg_val))
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{Config.MAX_EPOCH} — "
                  f"Train: {avg_train:.5f}, Val: {avg_val:.5f}")

    # --- モデル保存 ---
    save_model(model)

    # --- 損失推移プロット ---
    hist = np.array(history)
    plt.plot(hist[:,0], hist[:,1], label="Train")
    plt.plot(hist[:,0], hist[:,2], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Huber Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train()
