import csv
import numpy as np
import torch
from config import Config

def save_model(model: torch.nn.Module, path: str = Config.MODEL_SAVE_PATH):
    """モデルの state_dict を保存"""
    torch.save({'model_state_dict': model.state_dict()}, path)

def load_model(model: torch.nn.Module, path: str = Config.MODEL_SAVE_PATH):
    """保存済みモデルを読み込む"""
    checkpoint = torch.load(path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(Config.DEVICE)
    model.eval()
    return model

def save_predictions(preds: list, path: str = Config.PREDICTION_CSV):
    """予測値リストを CSV に書き出し"""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for p in preds:
            writer.writerow([p])

def relative_error(true: np.ndarray, pred: np.ndarray) -> float:
    """
    sqrt( sum(|t - p|) / sum(|t|) )
    """
    diff = np.abs(true - pred)
    denom = np.abs(true)
    return np.sqrt(diff.sum() / denom.sum())
