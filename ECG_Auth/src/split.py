import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight

from src.config import TEST_RATIO, VAL_RATIO, SEED


def split_records_train_val_test(meta_df, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, seed=SEED):
    rng = np.random.RandomState(seed)

    grouped = meta_df[["label", "person_name", "record_name"]].drop_duplicates()

    train_keys, val_keys, test_keys = [], [], []

    for label in sorted(grouped["label"].unique()):
        person_records = sorted(grouped[grouped["label"] == label]["record_name"].tolist())
        n = len(person_records)

        if n == 1:
            train_keys.append((label, person_records[0]))
            continue

        shuffled = person_records.copy()
        rng.shuffle(shuffled)

        n_test = max(1, int(round(n * test_ratio)))
        n_test = min(n_test, n - 1)

        remain_after_test = n - n_test
        n_val = max(1, int(round(remain_after_test * val_ratio))) if remain_after_test >= 2 else 0
        n_val = min(n_val, remain_after_test - 1) if remain_after_test >= 2 else 0

        test_records = shuffled[:n_test]
        val_records = shuffled[n_test:n_test + n_val]
        train_records = shuffled[n_test + n_val:]

        if len(train_records) == 0 and len(val_records) > 0:
            train_records.append(val_records.pop())

        train_keys.extend([(label, r) for r in train_records])
        val_keys.extend([(label, r) for r in val_records])
        test_keys.extend([(label, r) for r in test_records])

    return set(train_keys), set(val_keys), set(test_keys)


def compute_class_weights(y_train):
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    return {int(c): float(w) for c, w in zip(classes, class_weights)}


def make_split_data(X, y, meta_df):
    X_cnn = X[..., np.newaxis]

    train_record_keys, val_record_keys, test_record_keys = split_records_train_val_test(meta_df)

    train_mask = meta_df.apply(lambda row: (row["label"], row["record_name"]) in train_record_keys, axis=1).values
    val_mask   = meta_df.apply(lambda row: (row["label"], row["record_name"]) in val_record_keys, axis=1).values
    test_mask  = meta_df.apply(lambda row: (row["label"], row["record_name"]) in test_record_keys, axis=1).values

    X_train, X_val, X_test = X_cnn[train_mask], X_cnn[val_mask], X_cnn[test_mask]
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]

    meta_train = meta_df[train_mask].reset_index(drop=True)
    meta_val   = meta_df[val_mask].reset_index(drop=True)
    meta_test  = meta_df[test_mask].reset_index(drop=True)

    class_weight_dict = compute_class_weights(y_train)
    num_classes = len(np.unique(y))

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "meta_train": meta_train,
        "meta_val": meta_val,
        "meta_test": meta_test,
        "class_weight_dict": class_weight_dict,
        "num_classes": num_classes,
        "train_record_keys": train_record_keys,
        "val_record_keys": val_record_keys,
        "test_record_keys": test_record_keys,
    }


def build_split_summary_df(split_data):
    X_train = split_data["X_train"]
    X_val = split_data["X_val"]
    X_test = split_data["X_test"]
    y_train = split_data["y_train"]
    y_val = split_data["y_val"]
    y_test = split_data["y_test"]
    meta_train = split_data["meta_train"]
    meta_val = split_data["meta_val"]
    meta_test = split_data["meta_test"]

    rows = [
        {"section": "split", "metric": "X_train_shape", "value": str(X_train.shape)},
        {"section": "split", "metric": "X_val_shape", "value": str(X_val.shape)},
        {"section": "split", "metric": "X_test_shape", "value": str(X_test.shape)},
        {"section": "split", "metric": "y_train_shape", "value": str(y_train.shape)},
        {"section": "split", "metric": "y_val_shape", "value": str(y_val.shape)},
        {"section": "split", "metric": "y_test_shape", "value": str(y_test.shape)},
        {"section": "split", "metric": "train_classes", "value": int(len(np.unique(y_train)))},
        {"section": "split", "metric": "val_classes", "value": int(len(np.unique(y_val)))},
        {"section": "split", "metric": "test_classes", "value": int(len(np.unique(y_test)))},
        {"section": "split", "metric": "train_records", "value": int(meta_train[["label", "record_name"]].drop_duplicates().shape[0])},
        {"section": "split", "metric": "val_records", "value": int(meta_val[["label", "record_name"]].drop_duplicates().shape[0])},
        {"section": "split", "metric": "test_records", "value": int(meta_test[["label", "record_name"]].drop_duplicates().shape[0])},
    ]

    def add_group_stats(name, meta_part):
        if len(meta_part) == 0:
            rows.extend([
                {"section": f"split_{name}", "metric": "beats_per_subject_min", "value": 0},
                {"section": f"split_{name}", "metric": "beats_per_subject_mean", "value": 0},
                {"section": f"split_{name}", "metric": "beats_per_subject_max", "value": 0},
            ])
            return

        grp = meta_part.groupby("label").size()
        rows.extend([
            {"section": f"split_{name}", "metric": "beats_per_subject_min", "value": int(grp.min())},
            {"section": f"split_{name}", "metric": "beats_per_subject_mean", "value": float(grp.mean())},
            {"section": f"split_{name}", "metric": "beats_per_subject_max", "value": int(grp.max())},
        ])

        if "rr_len_raw" in meta_part.columns:
            rows.append({
                "section": f"split_{name}",
                "metric": "rr_len_raw_mean",
                "value": float(meta_part["rr_len_raw"].mean()),
            })

    add_group_stats("train", meta_train)
    add_group_stats("val", meta_val)
    add_group_stats("test", meta_test)

    return pd.DataFrame(rows)
