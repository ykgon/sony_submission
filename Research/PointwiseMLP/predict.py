import torch
import numpy as np
import pandas as pd
from config import Config, fix_seed
from data_loader import load_csv, prepare_tensor_dataset
from model import ImplicitGNN
from utils import calculate_accuracy, save_predictions

DEVICE = Config.DEVICE


def predict_all():
    fix_seed(Config.SEED)
    model = ImplicitGNN().to(DEVICE)
    checkpoint = torch.load("Implicit_GNN_checkpoint.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    accuracies = []
    for i in range(200):
        # load test pairs
        path1 = Config.TEST_CSV1_PATTERN.format(i=i)
        path2 = Config.TEST_CSV2_PATTERN.format(i=i)
        df1 = load_csv(path1, Config.COLUMN_NAMES)
        df2 = load_csv(path2, Config.COLUMN_NAMES)

        ds = prepare_tensor_dataset(df1, df2)
        preds, trues = [], []

        with torch.no_grad():
            for x, y in ds:
                out = model(x.to(DEVICE).unsqueeze(0))
                preds.append(out.cpu().item())
                trues.append(y.item())

        # save raw predictions
        out_path = Config.PREDICTION_OUTPUT_PATTERN.format(i=i)
        save_predictions(preds, out_path)

        # reload and save filtered
        df_pred = pd.read_csv(out_path, names=["u"], encoding="UTF8")
        filt_path = Config.FILTERED_OUTPUT_PATTERN.format(i=i)
        df_pred.to_csv(filt_path, header=False, index=False)

        acc = calculate_accuracy(np.array(trues), df_pred["u"].values)
        print(f"Instance {i:03d} — accuracy: {acc:.5f}")
        accuracies.append(acc)

    avg_acc = sum(accuracies) / len(accuracies)
    print(f"\nAverage accuracy over 200 runs: {avg_acc:.5f}")


if __name__ == "__main__":
    predict_all()
