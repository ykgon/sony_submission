import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from config import Config


def load_csv(path: str, column_names: list) -> pd.DataFrame:
    """Load a CSV file with given column names."""
    return pd.read_csv(path, names=column_names, encoding="UTF8")


def prepare_tensor_dataset(df1: pd.DataFrame, df2: pd.DataFrame) -> TensorDataset:
    """
    Given two DataFrames (features and targets), return a TensorDataset.
    df1 must contain all Config.COLUMN_NAMES; df2 must contain 'u'.
    """
    # features: all 10 columns
    teach = df1[Config.COLUMN_NAMES].astype(float).values
    # target: only 'u' from the second file
    answer = df2['u'].astype(float).values

    teach_tensor = torch.tensor(teach, dtype=torch.float32)
    answer_tensor = torch.tensor(answer, dtype=torch.float32).view(-1, 1)
    return TensorDataset(teach_tensor, answer_tensor)


def get_train_val_loaders():
    """Return train and validation DataLoaders (80/20 split)."""
    df1 = load_csv(Config.TRAIN_CSV1_PATH, Config.COLUMN_NAMES)
    df2 = load_csv(Config.TRAIN_CSV2_PATH, Config.COLUMN_NAMES)
    full_ds = prepare_tensor_dataset(df1, df2)

    train_size = int(len(full_ds) * 0.8)
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    return train_loader, val_loader
