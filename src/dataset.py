import os
import glob
import numpy as np
import pandas as pd

from collections import Counter

from src.config import (
    DATASET_DIR,
    SEGMENT_MODE,
    BEAT_LEN,
    USE_RR_RIGID_THRESHOLD,
    RR_STD_FACTOR,
    MIN_RR_COUNT_FOR_THRESHOLD,
)
from src.segmentation import extract_beats_from_record


def build_dataset_from_ecg_id(
    dataset_dir=DATASET_DIR,
    segment_mode=SEGMENT_MODE,
    target_len=BEAT_LEN,
    use_rr_threshold=USE_RR_RIGID_THRESHOLD,
    rr_std_factor=RR_STD_FACTOR,
    min_rr_count=MIN_RR_COUNT_FOR_THRESHOLD,
):
    X_list = []
    y_list = []
    meta_rows = []

    person_dirs = sorted(glob.glob(os.path.join(dataset_dir, "Person_*")))

    for label, person_dir in enumerate(person_dirs):
        person_name = os.path.basename(person_dir)
        record_headers = sorted(glob.glob(os.path.join(person_dir, "rec_*.hea")))

        for header_path in record_headers:
            record_path = header_path.replace(".hea", "")
            record_name = os.path.basename(record_path)

            try:
                beats, raw_signal, proc_signal, rpeaks, beat_meta = extract_beats_from_record(
                    record_path,
                    segment_mode=segment_mode,
                    target_len=target_len,
                    use_rr_threshold=use_rr_threshold,
                    rr_std_factor=rr_std_factor,
                    min_rr_count=min_rr_count,
                )

                for beat, meta in zip(beats, beat_meta):
                    X_list.append(beat)
                    y_list.append(label)

                    meta_rows.append({
                        "label": label,
                        "person_name": person_name,
                        "record_name": record_name,
                        "record_key": f"{label}_{record_name}",
                        "beat_index": int(meta["beat_index"]),
                        "record_path": record_path,
                        "segment_type": meta["segment_type"],
                        "start_idx": int(meta["start_idx"]),
                        "end_idx": int(meta["end_idx"]),
                        "rpeak_idx": int(meta["rpeak_idx"]),
                        "rr_len_raw": int(meta["rr_len_raw"]),
                        "kept_by_threshold": bool(meta["kept_by_threshold"]),
                    })

            except Exception as e:
                print(f"[SKIP] {record_path}: {e}")

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    meta_df = pd.DataFrame(meta_rows)

    return X, y, meta_df


def build_dataset_summary_df(X, y, meta_df, segment_mode, use_rr_threshold, rr_std_factor):
    rows = [
        {"section": "dataset", "metric": "SEGMENT_MODE", "value": segment_mode},
        {"section": "dataset", "metric": "USE_RR_RIGID_THRESHOLD", "value": bool(use_rr_threshold)},
        {"section": "dataset", "metric": "RR_STD_FACTOR", "value": float(rr_std_factor)},
        {"section": "dataset", "metric": "X_shape", "value": str(X.shape)},
        {"section": "dataset", "metric": "y_shape", "value": str(y.shape)},
        {"section": "dataset", "metric": "num_people", "value": int(len(np.unique(y)))},
        {"section": "dataset", "metric": "num_records", "value": int(meta_df[["label", "record_name"]].drop_duplicates().shape[0])},
    ]

    if "rr_len_raw" in meta_df.columns and len(meta_df) > 0:
        rr_desc = meta_df["rr_len_raw"].describe()
        for stat_name in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
            rows.append({
                "section": "dataset_rr",
                "metric": f"rr_len_{stat_name}",
                "value": float(rr_desc[stat_name]),
            })

    return pd.DataFrame(rows)


def build_subject_count_df(y):
    counter = Counter(y)
    count_df = pd.DataFrame(counter.items(), columns=["label", "num_beats"])
    count_df = count_df.sort_values("num_beats", ascending=False).reset_index(drop=True)
    return count_df
