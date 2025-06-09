import torch
import numpy as np
import csv
from config import Config


def save_best_model(model: torch.nn.Module,
                    best_val_loss: float,
                    current_val_loss: float,
                    patience_counter: int):
    """
    Save model.state_dict() if improved.
    Returns (new_best_val_loss, new_patience_counter).
    """
    if current_val_loss < best_val_loss:
        torch.save(model.state_dict(), "best_model.pth")
        return current_val_loss, 0
    else:
        return best_val_loss, patience_counter + 1


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the L2‐type accuracy:
      sqrt( sum((y_true - y_pred)^2) / sum(y_true^2) ).
    """
    diff_sq = np.abs(y_true - y_pred) ** 2
    ans_sq = np.abs(y_true) ** 2
    return np.sqrt(diff_sq.sum() / ans_sq.sum())


def save_predictions(predictions: list, file_path: str):
    """Write one prediction per row to CSV."""
    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for p in predictions:
            writer.writerow([p])
