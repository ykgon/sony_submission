import torch

class Config:
    # --- データパス ---
    TRAIN_CSV_T0_PATH = "pointdata_t0.csv"
    TRAIN_CSV_T1_PATH = "pointdata_t1.csv"
    TEST_CSV_T0_PATH = "../input_data/pointdata_test_t0=0.1_ver5.csv"
    TEST_CSV_T1_PATH = "../answer_data/pointdata_test_t1=0.2_ver5.csv"

    # --- 出力パス ---
    MODEL_SAVE_PATH = "model.pth"
    PREDICTION_CSV = "predictions_test.csv"
    FILTERED_PREDICTION_CSV = "filtered_predictions.csv"

    # --- ハイパーパラメータ ---
    BATCH_SIZE = 64
    HIDDEN_DIM = 64
    MAX_EPOCH = 1000
    SEED = 3407

    # --- デバイス ---
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
