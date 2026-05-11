"""Hyperparameters and paths for CAIL2018 charge prediction."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Practice split filenames (CAIL2018)
TRAIN_FILE = DATA_DIR / "data_train.json"
VALID_FILE = DATA_DIR / "data_valid.json"
TEST_FILE = DATA_DIR / "data_test.json"

# Hugging Face backbone (default: legal-domain simplified Chinese ELECTRA)
PRETRAINED_MODEL_NAME = "hfl/chinese-legal-electra-large-discriminator"
# If None, use PRETRAINED_MODEL_NAME for tokenizer as well.
TOKENIZER_NAME = None

MAX_LEN = 512
BATCH_SIZE = 4
EPOCHS = 10
ENCODER_LR = 2e-5
BERT_LR = ENCODER_LR  # alias for original spec
HEAD_LR = 1e-3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
DROPOUT = 0.3
FC_HIDDEN = 512

MIN_LABEL_FREQ = 50
SEED = 42

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
LABEL_MAP_PATH = CHECKPOINT_DIR / "label_map.pt"

USE_GRADIENT_CHECKPOINTING = False
# Optional: convert simplified to traditional before tokenization (requires opencc).
USE_OPENCC_S2T = False

# Inference
PRED_THRESHOLD = 0.5
