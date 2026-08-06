import os

# TensorFlow를 import하기 전에 설정해야 적용됨
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

from src.config import (
    SEED,
    DATASET_DIR,
    SEGMENT_MODE,
    BEAT_LEN,
    USE_RR_RIGID_THRESHOLD,
    RR_STD_FACTOR,
    MIN_RR_COUNT_FOR_THRESHOLD,
    BATCH_SIZE,
    EPOCHS,
    EMBED_DIM,
    LEARNING_RATE,
    BEST_MODEL_PATH,
)

from src.dataset import build_dataset_from_ecg_id
from src.split import make_split_data
from src.model import build_model


MODEL_NAME = "resnet1d"

# 사용할 수 있는 모델:
# "resnet1d"
# "plain_cnn1d"
# "bilstm"
# "cnn_bilstm"


def set_global_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_model_output_dir(model_name=MODEL_NAME):
    best_model_path = Path(BEST_MODEL_PATH)
    base_dir = best_model_path.parent

    output_dir = base_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def get_model_checkpoint_path(model_name=MODEL_NAME):
    output_dir = get_model_output_dir(model_name)

    return str(output_dir / "best_model.keras")


def train_model(split_data, model_name=MODEL_NAME):
    model = build_model(
        model_name=model_name,
        input_shape=(BEAT_LEN, 1),
        num_classes=split_data["num_classes"],
        embed_dim=EMBED_DIM,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=3,
                name="top3_acc",
            ),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=5,
                name="top5_acc",
            ),
        ],
    )

    checkpoint_path = get_model_checkpoint_path(model_name)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("=" * 60)
    print(f"[INFO] Training model: {model_name}")
    print(f"[INFO] Number of classes: {split_data['num_classes']}")
    print(f"[INFO] Model checkpoint: {checkpoint_path}")
    print("=" * 60)

    history = model.fit(
        split_data["X_train"],
        split_data["y_train"],
        validation_data=(
            split_data["X_val"],
            split_data["y_val"],
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=split_data["class_weight_dict"],
        callbacks=callbacks,
        verbose=1,
    )

    return model, history


def run_training_only(model_name=MODEL_NAME):
    set_global_seed()

    print("=" * 60)
    print(f"[INFO] MODEL_NAME: {model_name}")
    print(f"[INFO] DATASET_DIR: {DATASET_DIR}")
    print("=" * 60)

    # 1. ECG 데이터 구성
    X, y, meta_df = build_dataset_from_ecg_id(
        dataset_dir=DATASET_DIR,
        segment_mode=SEGMENT_MODE,
        target_len=BEAT_LEN,
        use_rr_threshold=USE_RR_RIGID_THRESHOLD,
        rr_std_factor=RR_STD_FACTOR,
        min_rr_count=MIN_RR_COUNT_FOR_THRESHOLD,
    )

    print(f"[INFO] X shape: {X.shape}")
    print(f"[INFO] y shape: {y.shape}")
    print(f"[INFO] Subjects: {len(np.unique(y))}")

    # 2. Train / Validation / Test 분할
    split_data = make_split_data(
        X,
        y,
        meta_df,
    )

    print(f"[INFO] Train samples: {len(split_data['X_train'])}")
    print(f"[INFO] Validation samples: {len(split_data['X_val'])}")
    print(f"[INFO] Test samples: {len(split_data['X_test'])}")

    # 3. 모델 학습
    model, history = train_model(
        split_data=split_data,
        model_name=model_name,
    )

    checkpoint_path = get_model_checkpoint_path(model_name)

    print("=" * 60)
    print("[INFO] Training completed")
    print(f"[INFO] Best model saved: {checkpoint_path}")
    print("=" * 60)

    return {
        "model_name": model_name,
        "model": model,
        "history": history,
        "split_data": split_data,
        "checkpoint_path": checkpoint_path,
    }


if __name__ == "__main__":
    run_training_only(
        model_name=MODEL_NAME,
    )