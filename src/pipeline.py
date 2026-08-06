
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

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
from src.dataset import (
    build_dataset_from_ecg_id,
    build_dataset_summary_df,
    build_subject_count_df,
)
from src.split import make_split_data, build_split_summary_df
from src.model import build_model
from src.evaluation import (
    evaluate_classification,
    evaluate_identification_modes,
    evaluate_verification_modes,
)
from src.visualization import (
    visualize_single_record,
    plot_subject_count_bar,
    plot_training_curves,
    plot_all_tsne,
    plot_all_verification_curves,
)
from src.reporting import (
    build_master_summary_table,
    build_pretty_tables,
    save_tables,
)

MODEL_NAME = "resnet1d"
# available:
# "resnet1d"
# "plain_cnn1d"
# "bilstm"
# "cnn_bilstm"


def set_global_seed(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc"),
        ],
    )

    checkpoint_path = get_model_checkpoint_path(model_name)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1),
    ]

    history = model.fit(
        split_data["X_train"],
        split_data["y_train"],
        validation_data=(split_data["X_val"], split_data["y_val"]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=split_data["class_weight_dict"],
        callbacks=callbacks,
        verbose=1,
    )

    return model, history


def run_full_experiment(example_record_path=None, do_visualize_example=True, model_name=MODEL_NAME):
    set_global_seed()

    output_dir = get_model_output_dir(model_name)

    print("=" * 60)
    print(f"[INFO] MODEL_NAME : {model_name}")
    print(f"[INFO] OUTPUT_DIR : {output_dir}")
    print("=" * 60)

    if do_visualize_example and example_record_path is not None:
        visualize_single_record(
            example_record_path,
            save_prefix="example_record",
            output_dir=str(output_dir),
        )

    X, y, meta_df = build_dataset_from_ecg_id(
        dataset_dir=DATASET_DIR,
        segment_mode=SEGMENT_MODE,
        target_len=BEAT_LEN,
        use_rr_threshold=USE_RR_RIGID_THRESHOLD,
        rr_std_factor=RR_STD_FACTOR,
        min_rr_count=MIN_RR_COUNT_FOR_THRESHOLD,
    )

    dataset_summary_df = build_dataset_summary_df(
        X,
        y,
        meta_df,
        segment_mode=SEGMENT_MODE,
        use_rr_threshold=USE_RR_RIGID_THRESHOLD,
        rr_std_factor=RR_STD_FACTOR,
    )

    subject_count_df = build_subject_count_df(y)
    plot_subject_count_bar(
        subject_count_df,
        top_n=20,
        save_name="subject_count_top20.png",
        output_dir=str(output_dir),
    )

    split_data = make_split_data(X, y, meta_df)
    split_summary_df = build_split_summary_df(split_data)

    model, history = train_model(split_data, model_name=model_name)

    plot_training_curves(
        history,
        save_prefix="training_curve",
        output_dir=str(output_dir),
    )

    classification_summary_df, result_dict, y_prob, y_pred = evaluate_classification(
        model,
        split_data["X_test"],
        split_data["y_test"],
        split_data["num_classes"],
    )

    identification_summary_df, result_df, emb_dict = evaluate_identification_modes(
        model=model,
        X_train=split_data["X_train"],
        y_train=split_data["y_train"],
        meta_train=split_data["meta_train"],
        X_test=split_data["X_test"],
        y_test=split_data["y_test"],
        meta_test=split_data["meta_test"],
        y_pred_cls=y_pred,
    )

    verification_summary_df, verification_detail_dict = evaluate_verification_modes(emb_dict)

    plot_all_tsne(emb_dict, output_dir=str(output_dir))
    plot_all_verification_curves(verification_detail_dict, output_dir=str(output_dir))

    master_summary_df = build_master_summary_table(
        dataset_summary_df,
        split_summary_df,
        classification_summary_df,
    )

    pretty_tables = build_pretty_tables(
        master_summary_df=master_summary_df,
        classification_summary_df=classification_summary_df,
        identification_summary_df=identification_summary_df,
        verification_summary_df=verification_summary_df,
    )

    save_tables(
        master_summary_df=master_summary_df,
        identification_summary_df=identification_summary_df,
        verification_summary_df=verification_summary_df,
        classification_summary_df=classification_summary_df,
        result_df=result_df,
        output_dir=str(output_dir),
    )

    return {
        "model_name": model_name,
        "output_dir": str(output_dir),
        "X": X,
        "y": y,
        "meta_df": meta_df,
        "split_data": split_data,
        "model": model,
        "history": history,
        "master_summary_df": master_summary_df,
        "classification_summary_df": classification_summary_df,
        "identification_summary_df": identification_summary_df,
        "verification_summary_df": verification_summary_df,
        "result_df": result_df,
        "emb_dict": emb_dict,
        "verification_detail_dict": verification_detail_dict,
        "pretty_tables": pretty_tables,
    }
