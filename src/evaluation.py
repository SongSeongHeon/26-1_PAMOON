import numpy as np
import pandas as pd

from sklearn.metrics import top_k_accuracy_score, roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity

from src.config import SEED
from src.embedding import (
    extract_embeddings,
    beat_level_identification,
    gallery_probe_identification_record_centroid,
    make_multi_beat_queries,
)


def evaluate_classification(model, X_test, y_test, num_classes):
    eval_results = model.evaluate(X_test, y_test, verbose=0)
    result_dict = {name: float(value) for name, value in zip(model.metrics_names, eval_results)}

    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    top1 = float(np.mean(y_pred == y_test))
    top3 = float(top_k_accuracy_score(y_test, y_prob, k=3, labels=np.arange(num_classes)))
    top5 = float(top_k_accuracy_score(y_test, y_prob, k=5, labels=np.arange(num_classes)))

    summary_df = pd.DataFrame([
        {"section": "classification", "metric": "loss", "value": float(result_dict["loss"])},
        {"section": "classification", "metric": "top1_acc", "value": top1},
        {"section": "classification", "metric": "top3_acc", "value": top3},
        {"section": "classification", "metric": "top5_acc", "value": top5},
    ])

    return summary_df, result_dict, y_prob, y_pred


def evaluate_identification_modes(model, X_train, y_train, meta_train, X_test, y_test, meta_test, y_pred_cls):
    embedding_model, train_embeddings, test_embeddings = extract_embeddings(model, X_train, X_test)

    beat_acc, beat_pred_labels, beat_scores = beat_level_identification(
        train_embeddings, y_train, test_embeddings, y_test
    )

    gallery_acc, gallery_pred_labels, gallery_scores = gallery_probe_identification_record_centroid(
        train_embeddings, meta_train, test_embeddings, y_test
    )

    result_df = meta_test.copy()
    result_df["true_label"] = y_test
    result_df["cls_pred"] = y_pred_cls
    result_df["beat_pred"] = beat_pred_labels
    result_df["beat_sim_score"] = beat_scores
    result_df["gallery_pred"] = gallery_pred_labels
    result_df["gallery_sim_score"] = gallery_scores
    result_df["cls_correct"] = (result_df["true_label"] == result_df["cls_pred"]).astype(int)
    result_df["beat_correct"] = (result_df["true_label"] == result_df["beat_pred"]).astype(int)
    result_df["gallery_correct"] = (result_df["true_label"] == result_df["gallery_pred"]).astype(int)

    test_embeddings_2b, y_test_2b, meta_test_2b = make_multi_beat_queries(
        test_embeddings, meta_test, y_test, n_beats=2, step=1
    )

    beat_acc_2b, _, _ = beat_level_identification(
        train_embeddings, y_train, test_embeddings_2b, y_test_2b
    )

    gallery_acc_2b, _, _ = gallery_probe_identification_record_centroid(
        train_embeddings, meta_train, test_embeddings_2b, y_test_2b
    )

    test_embeddings_3b, y_test_3b, meta_test_3b = make_multi_beat_queries(
        test_embeddings, meta_test, y_test, n_beats=3, step=1
    )

    beat_acc_3b, _, _ = beat_level_identification(
        train_embeddings, y_train, test_embeddings_3b, y_test_3b
    )

    gallery_acc_3b, _, _ = gallery_probe_identification_record_centroid(
        train_embeddings, meta_train, test_embeddings_3b, y_test_3b
    )

    summary_df = pd.DataFrame([
        {
            "section": "identification",
            "mode": "single_beat",
            "num_queries": len(test_embeddings),
            "similarity_acc": beat_acc,
            "gallery_acc": gallery_acc,
        },
        {
            "section": "identification",
            "mode": "2beat",
            "num_queries": len(test_embeddings_2b),
            "similarity_acc": beat_acc_2b,
            "gallery_acc": gallery_acc_2b,
        },
        {
            "section": "identification",
            "mode": "3beat",
            "num_queries": len(test_embeddings_3b),
            "similarity_acc": beat_acc_3b,
            "gallery_acc": gallery_acc_3b,
        },
    ])

    emb_dict = {
        "embedding_model": embedding_model,
        "train_embeddings": train_embeddings,
        "test_embeddings": test_embeddings,
        "test_embeddings_2b": test_embeddings_2b,
        "test_embeddings_3b": test_embeddings_3b,
        "y_test": y_test,
        "y_test_2b": y_test_2b,
        "y_test_3b": y_test_3b,
        "meta_test_2b": meta_test_2b,
        "meta_test_3b": meta_test_3b,
    }

    return summary_df, result_df, emb_dict


def sample_verification_pairs(embeddings, labels, max_positive_pairs=3000, max_negative_pairs=3000, seed=SEED):
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)

    label_to_idx = {lbl: np.where(labels == lbl)[0] for lbl in np.unique(labels)}

    pos_scores, neg_scores = [], []

    pos_count = 0
    label_order = list(label_to_idx.keys())
    rng.shuffle(label_order)

    for lbl in label_order:
        idxs = label_to_idx[lbl]
        if len(idxs) < 2:
            continue
        for _ in range(min(len(idxs), 50)):
            a, b = rng.choice(idxs, size=2, replace=False)
            score = cosine_similarity(embeddings[a].reshape(1, -1), embeddings[b].reshape(1, -1))[0, 0]
            pos_scores.append(score)
            pos_count += 1
            if pos_count >= max_positive_pairs:
                break
        if pos_count >= max_positive_pairs:
            break

    neg_count = 0
    all_labels = np.unique(labels)
    while neg_count < max_negative_pairs:
        lbl1, lbl2 = rng.choice(all_labels, size=2, replace=False)
        idx1 = rng.choice(label_to_idx[lbl1])
        idx2 = rng.choice(label_to_idx[lbl2])
        score = cosine_similarity(embeddings[idx1].reshape(1, -1), embeddings[idx2].reshape(1, -1))[0, 0]
        neg_scores.append(score)
        neg_count += 1

    pos_scores = np.asarray(pos_scores)
    neg_scores = np.asarray(neg_scores)

    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_score = np.concatenate([pos_scores, neg_scores])

    return pos_scores, neg_scores, y_true, y_score


def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, eer, eer_threshold, roc_auc


def run_verification(embeddings, labels, title_prefix="Single-beat", seed=SEED):
    pos_scores, neg_scores, y_true_ver, y_score_ver = sample_verification_pairs(
        embeddings, labels,
        max_positive_pairs=3000,
        max_negative_pairs=3000,
        seed=seed
    )

    fpr, tpr, thresholds, eer, eer_threshold, roc_auc = compute_eer(y_true_ver, y_score_ver)

    return {
        "section": "verification",
        "mode": title_prefix,
        "auc": float(roc_auc),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "fpr": fpr,
        "tpr": tpr,
        "pos_scores": pos_scores,
        "neg_scores": neg_scores,
    }


def evaluate_verification_modes(emb_dict):
    ver_single = run_verification(
        emb_dict["test_embeddings"],
        emb_dict["y_test"],
        title_prefix="single_beat",
        seed=SEED,
    )

    ver_2b = run_verification(
        emb_dict["test_embeddings_2b"],
        emb_dict["y_test_2b"],
        title_prefix="2beat",
        seed=SEED,
    )

    ver_3b = run_verification(
        emb_dict["test_embeddings_3b"],
        emb_dict["y_test_3b"],
        title_prefix="3beat",
        seed=SEED,
    )

    summary_df = pd.DataFrame([
        {
            "section": "verification",
            "mode": ver_single["mode"],
            "auc": ver_single["auc"],
            "eer": ver_single["eer"],
            "eer_threshold": ver_single["eer_threshold"],
        },
        {
            "section": "verification",
            "mode": ver_2b["mode"],
            "auc": ver_2b["auc"],
            "eer": ver_2b["eer"],
            "eer_threshold": ver_2b["eer_threshold"],
        },
        {
            "section": "verification",
            "mode": ver_3b["mode"],
            "auc": ver_3b["auc"],
            "eer": ver_3b["eer"],
            "eer_threshold": ver_3b["eer_threshold"],
        },
    ])

    return summary_df, {
        "single": ver_single,
        "2beat": ver_2b,
        "3beat": ver_3b,
    }
