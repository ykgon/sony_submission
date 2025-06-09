import numpy as np
import pandas as pd
import torch

from config import Config
from data_loader import load_filtered_df, prepare_dataset
from model import PointwiseMLP
from utils import load_model, save_predictions, relative_error

def test():
    # --- モデル準備 ---
    model = PointwiseMLP(Config.HIDDEN_DIM)
    load_model(model)

    # --- テストデータ読み込み・評価 ---
    df = load_filtered_df(Config.TEST_CSV_T0_PATH, Config.TEST_CSV_T1_PATH)
    dataset = prepare_dataset(df)

    preds = []
    trues = []
    with torch.no_grad():
        for x, y in dataset:
            p = model(x.view(1,1).to(Config.DEVICE)).cpu().item()
            preds.append(p)
            trues.append(y.item())

    # --- CSV出力 & フィルタ済み保存 ---
    save_predictions(preds, Config.PREDICTION_CSV)
    df_pred = pd.read_csv(Config.PREDICTION_CSV, names=['u'], encoding="UTF8")
    df_pred.to_csv(Config.FILTERED_PREDICTION_CSV, index=False, header=False)

    # --- 相対誤差計算 ---
    true_arr = np.array(trues)
    pred_arr = df_pred['u'].values.astype(float)
    err = relative_error(true_arr, pred_arr)
    print(f"Relative error: {err:.5f}")

if __name__ == "__main__":
    test()
