import pandas as pd
import torch
from torch.utils.data import TensorDataset, Subset, DataLoader
from config import Config

def load_filtered_df(csv_t0_path: str, csv_t1_path: str) -> pd.DataFrame:
    """
    pointdata_t0.csv と pointdata_t1.csv を読み込み、
    列名 ['t0','t1'] で結合する。
    """
    df_t0 = pd.read_csv(csv_t0_path, names=['t0'], encoding="UTF8")
    df_t1 = pd.read_csv(csv_t1_path, names=['t1'], encoding="UTF8")
    df = pd.concat([df_t0, df_t1], axis=1)
    return df

def prepare_dataset(df: pd.DataFrame) -> TensorDataset:
    """
    DataFrame から (teach, answer) の TensorDataset を生成。
    teach: [n,1]、answer: [n,1]
    """
    teach = df['t0'].astype(float).values
    answer = df['t1'].astype(float).values
    teach_tensor = torch.tensor(teach, dtype=torch.float32).view(-1,1)
    answer_tensor = torch.tensor(answer, dtype=torch.float32).view(-1,1)
    return TensorDataset(teach_tensor, answer_tensor)

def get_train_val_test_loaders():
    """
    データを 80%:10%:10% で分割し、DataLoader を返す。
    学習: shuffle=True, drop_last=True
    検証/テスト: shuffle=False
    """
    df = load_filtered_df(Config.TRAIN_CSV_T0_PATH, Config.TRAIN_CSV_T1_PATH)
    dataset = prepare_dataset(df)

    n = len(dataset)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    n_test  = n - n_train - n_val
    idx = list(range(n))

    train_ds = Subset(dataset, idx[:n_train])
    val_ds   = Subset(dataset, idx[n_train:n_train+n_val])
    test_ds  = Subset(dataset, idx[n_train+n_val:])

    train_loader = DataLoader(train_ds,
                              batch_size=Config.BATCH_SIZE,
                              shuffle=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,
                              batch_size=Config.BATCH_SIZE,
                              shuffle=False)
    test_loader  = DataLoader(test_ds,
                              batch_size=Config.BATCH_SIZE,
                              shuffle=False)

    return train_loader, val_loader, test_loader
