import os

SEED = 42

# --- 여기서부터 수정 ---
# 1. 윈도우 환경의 실제 프로젝트 폴더 경로로 변경 (경로 앞 'r' 필수!)
PROJECT_ROOT = r"C:\Users\User\Downloads\26-1_Capstone-main\26-1_Capstone-main"

DATA_ZIP_PATH = os.path.join(PROJECT_ROOT, "data", "ecg-id-database-1.0.0.zip")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

# 2. 데이터가 풀려있을 경로를 윈도우 경로로 변경
EXTRACT_DIR = os.path.join(PROJECT_ROOT, "data")
# 데이터셋 폴더명이 다르다면 아래 이름을 변경하세요 (예: "ECG_ID")
DATASET_DIR = os.path.join(EXTRACT_DIR, "ecg-id-database-1.0.0")

SAMPLING_RATE = 500
CHANNEL_INDEX = 0

SEGMENT_MODE = "rr_segment"
RPEAK_LEFT = 200
RPEAK_RIGHT = 400
BEAT_LEN = 400

USE_RR_RIGID_THRESHOLD = True
RR_STD_FACTOR = 1.5
MIN_RR_COUNT_FOR_THRESHOLD = 8

TEST_RATIO = 0.2
VAL_RATIO = 0.2

BATCH_SIZE = 64
EPOCHS = 80
EMBED_DIM = 256
LEARNING_RATE = 1e-3

BANDPASS_LOW = 0.5
BANDPASS_HIGH = 40.0
FILTER_ORDER = 3

FIG_SIZE = (10, 5)
HIST_BINS = 50

TSNE_MAX_CLASSES = 15
TSNE_MAX_SAMPLES_PER_CLASS = 30
TSNE_PERPLEXITY = 30

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_record_model_rr.keras")
MASTER_SUMMARY_PATH = os.path.join(TABLE_DIR, "master_summary.csv")
IDENTIFICATION_SUMMARY_PATH = os.path.join(TABLE_DIR, "identification_summary.csv")
VERIFICATION_SUMMARY_PATH = os.path.join(TABLE_DIR, "verification_summary.csv")
CLASSIFICATION_SUMMARY_PATH = os.path.join(TABLE_DIR, "classification_summary.csv")
RESULT_DF_PATH = os.path.join(TABLE_DIR, "beat_level_result_detail.csv")
