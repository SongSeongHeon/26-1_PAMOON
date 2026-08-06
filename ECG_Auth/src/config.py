from pathlib import Path


# =========================================================
# Reproducibility
# =========================================================
SEED = 42


# =========================================================
# Project paths
# =========================================================

# config.py 위치:
# ECG_Auth/src/config.py
#
# parents[1]:
# ECG_Auth
PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"


# 실제 ECG-ID 데이터 위치에 맞게 둘 중 하나를 사용
DATASET_DIR = PROJECT_ROOT / "data" / "ecg-id"

# 데이터 폴더명이 아래와 같다면 위 줄 대신 사용
# DATASET_DIR = PROJECT_ROOT / "data" / "ecg-id-database-1.0.0"


# =========================================================
# ECG preprocessing
# =========================================================
SAMPLING_RATE = 500
CHANNEL_INDEX = 0

SEGMENT_MODE = "rr_segment"

RPEAK_LEFT = 200
RPEAK_RIGHT = 400

BEAT_LEN = 400

USE_RR_RIGID_THRESHOLD = True
RR_STD_FACTOR = 1.5
MIN_RR_COUNT_FOR_THRESHOLD = 8

BANDPASS_LOW = 0.5
BANDPASS_HIGH = 40.0
FILTER_ORDER = 3


# =========================================================
# Dataset split
# =========================================================
TEST_RATIO = 0.2
VAL_RATIO = 0.2


# =========================================================
# Deep-learning configuration
# =========================================================
BATCH_SIZE = 64
EPOCHS = 80

EMBED_DIM = 256
LEARNING_RATE = 1e-3


# =========================================================
# Model checkpoint
# =========================================================
BEST_MODEL_PATH = MODEL_DIR / "best_record_model_rr.keras"


# 다른 코드에서 문자열 경로를 요구할 경우를 대비
DATASET_DIR = str(DATASET_DIR)
OUTPUT_DIR = str(OUTPUT_DIR)
MODEL_DIR = str(MODEL_DIR)
BEST_MODEL_PATH = str(BEST_MODEL_PATH)