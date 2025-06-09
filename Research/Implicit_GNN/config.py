import random
import torch
import numpy as np

class Config:
    # Paths to training data
    TRAIN_CSV1_PATH = "../train_data/N=41_update_message_ver2/pointdata_t1=0.100000_N=41.csv"
    TRAIN_CSV2_PATH = "../train_data/N=41_update_message_ver2/pointdata_t2=0.200000_N=41.csv"

    # Patterns for testing data
    TEST_CSV1_PATTERN = "../test_data/N=41_update_message/t=0.1/pointdata_test_N=41_t1=0.100000_{i}.csv"
    TEST_CSV2_PATTERN = "../test_data/N=41_update_message/t=0.2/pointdata_test_N=41_t2=0.200000_{i}.csv"

    # Where to write predictions
    PREDICTION_OUTPUT_PATTERN = (
        "../prediction_data/"
        "N=41_update_message_dim_64_batch_512_ver2/before_filtered/"
        "predictions_test_N=41_t=0.2_{i}.csv"
    )
    FILTERED_OUTPUT_PATTERN = (
        "../prediction_data/"
        "N=41_update_message_dim_64_batch_512_ver2/filtered/"
        "filterd_data_N=41_{i}.csv"
    )

    # CSV column names
    COLUMN_NAMES = [
        'top', 'top_phi',
        'bottom', 'bottom_phi',
        'right', 'right_phi',
        'left', 'left_phi',
        'u', 'u_phi'
    ]

    # Model architecture
    MESSAGE_INPUT_DIM = 2
    MESSAGE_HIDDEN_SIZES = (36, 8, 3)
    UPDATE_INPUT_DIM = 4
    UPDATE_HIDDEN_SIZES = (36, 8, 3)

    # Training hyperparameters
    BATCH_SIZE = 34040
    LEARNING_RATE = 0.001
    MAX_EPOCH = 4052
    EARLY_STOPPING_PATIENCE = 200
    SEED = 3407

    # Device
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fix_seed(seed: int):
    """Fix the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
