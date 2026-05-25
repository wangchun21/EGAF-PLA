from pathlib import Path
from datetime import datetime

BASE_DIR = Path("./").resolve()
DATA_DIR = BASE_DIR / "pre_data"
LABEL_FILE = BASE_DIR / "data" / "INDEX_general_PL_data.csv"

SEED = 123
BATCH_SIZE = 32
NUM_EPOCHS = 120
LR = 3e-4
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS = 30
FIXED_ENTROPY_WEIGHT = 0.10

DEVICE = "cuda:0"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")