import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics.pairwise import cosine_similarity


def extract_embeddings(model, X_ref, X_query):
    embedding_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("embedding_l2norm").output,
        name="ECG_Fingerprint_Extractor"
    )
    ref_embeddings = embedding_model.predict(X_ref, verbose=0)
    query_embeddings = embedding_model.predict(X_query, verbose=0)
    return embedding_model, ref_embeddings, query_embeddings


def beat_level_identification(ref_embeddings, ref_labels, query_embeddings, query_labels):
    pred_labels, pred_scores = [], []
    correct = 0

    for i in range(len(query_embeddings)):
        sims = cosine_similarity(query_embeddings[i].reshape(1, -1), ref_embeddings)[0]
        best_idx = np.argmax(sims)
        pred_label = int(ref_labels[best_idx])
        pred_score = float(sims[best_idx])

        pred_labels.append(pred_label)
        pred_scores.append(pred_score)

        if pred_label == query_labels[i]:
            correct += 1

    acc = correct / len(query_embeddings) if len(query_embeddings) > 0 else 0.0
    return acc, pred_labels, pred_scores


def gallery_probe_identification_record_centroid(ref_embeddings, ref_meta, query_embeddings, query_labels):
    ref_df = ref_meta.copy().reset_index(drop=True)
    ref_df["emb_idx"] = np.arange(len(ref_df))

    gallery_vectors = []
    gallery_labels = []

    for (label, record_name), group in ref_df.groupby(["label", "record_name"]):
        idxs = group["emb_idx"].values
        centroid = np.mean(ref_embeddings[idxs], axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        gallery_vectors.append(centroid)
        gallery_labels.append(label)

    gallery_vectors = np.asarray(gallery_vectors)
    gallery_labels = np.asarray(gallery_labels)

    pred_labels, pred_scores = [], []
    correct = 0

    for i in range(len(query_embeddings)):
        sims = cosine_similarity(query_embeddings[i].reshape(1, -1), gallery_vectors)[0]
        best_idx = np.argmax(sims)
        pred_label = int(gallery_labels[best_idx])
        pred_score = float(sims[best_idx])

        pred_labels.append(pred_label)
        pred_scores.append(pred_score)

        if pred_label == query_labels[i]:
            correct += 1

    acc = correct / len(query_embeddings) if len(query_embeddings) > 0 else 0.0
    return acc, pred_labels, pred_scores


def make_multi_beat_queries(query_embeddings, query_meta, query_labels, n_beats=2, step=None):
    if step is None:
        step = n_beats

    query_meta = query_meta.copy().reset_index(drop=True)
    query_meta["emb_idx"] = np.arange(len(query_meta))

    agg_embeddings = []
    agg_labels = []
    agg_meta_rows = []

    grouped = query_meta.groupby(["label", "record_name"])

    for (label, record_name), group in grouped:
        group = group.sort_values("beat_index").reset_index(drop=True)
        emb_indices = group["emb_idx"].values

        if len(emb_indices) < n_beats:
            continue

        for start in range(0, len(emb_indices) - n_beats + 1, step):
            chunk = emb_indices[start:start + n_beats]

            emb_mean = np.mean(query_embeddings[chunk], axis=0)
            emb_mean = emb_mean / (np.linalg.norm(emb_mean) + 1e-8)

            agg_embeddings.append(emb_mean)
            agg_labels.append(int(label))

            agg_meta_rows.append({
                "label": int(label),
                "record_name": record_name,
                "start_beat_index": int(group.loc[start, "beat_index"]),
                "end_beat_index": int(group.loc[start + n_beats - 1, "beat_index"]),
                "num_beats": int(n_beats),
            })

    if len(agg_embeddings) == 0:
        return (
            np.empty((0, query_embeddings.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            pd.DataFrame(columns=["label", "record_name", "start_beat_index", "end_beat_index", "num_beats"]),
        )

    return (
        np.asarray(agg_embeddings, dtype=np.float32),
        np.asarray(agg_labels, dtype=np.int64),
        pd.DataFrame(agg_meta_rows),
    )
