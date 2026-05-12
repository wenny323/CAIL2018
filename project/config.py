"""
Hyperparameters and paths for CAIL2018 charge prediction.

Defaults target laptops (e.g. MacBook Air, Apple M): small legal ELECTRA + gradient
checkpointing. For a desktop GPU you can switch to the large model and disable
checkpointing in this file.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def _resolve_data_file(stem: str) -> Path:
    """Prefer uncompressed .json; fall back to .json.gz (for GitHub-friendly repos)."""
    for suffix in (".json", ".json.gz"):
        p = DATA_DIR / f"{stem}{suffix}"
        if p.is_file():
            return p
    return DATA_DIR / f"{stem}.json"


# Practice split filenames (CAIL2018)
TRAIN_FILE = _resolve_data_file("data_train")
VALID_FILE = _resolve_data_file("data_valid")
TEST_FILE = _resolve_data_file("data_test")

# Hugging Face backbone (legal-domain ELECTRA). Default: small + checkpointing for memory.
PRETRAINED_MODEL_NAME = "hfl/chinese-legal-electra-small-discriminator"
# PRETRAINED_MODEL_NAME = "hfl/chinese-legal-electra-large-discriminator"
# If None, use PRETRAINED_MODEL_NAME for tokenizer as well.
TOKENIZER_NAME = None

MAX_LEN = 512
BATCH_SIZE = 4
EPOCHS = 1  # set back to 10 for full training; use 1 to smoke-test / time one full pass
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

USE_GRADIENT_CHECKPOINTING = True
# Optional: convert simplified to traditional before tokenization (requires opencc).
USE_OPENCC_S2T = False

# Training progress: tqdm bar + running loss / LR in postfix every N steps (0 = postfix only at last batch).
LOG_EVERY_N_STEPS = 50

# Inference
PRED_THRESHOLD = 0.5

# torch device: None = auto (CUDA > MPS > CPU). Set "cpu" / "mps" / "cuda" to override.
FORCE_DEVICE = None
