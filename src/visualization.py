
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from src.config import (
    FIG_SIZE,
    HIST_BINS,
    FIG_DIR,
    SEGMENT_MODE,
    TSNE_MAX_CLASSES,
    TSNE_MAX_SAMPLES_PER_CLASS,
    TSNE_PERPLEXITY,
    SEED,
)
from src.segmentation import extract_beats_from_record


def maybe_savefig(filename=None, output_dir=None):
    if filename is None:
        return

    save_dir = output_dir if output_dir is not None else FIG_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    print("Saved figure:", save_path)


def visualize_single_record(record_path, plot_len=2000, save_prefix=None, output_dir=None):
    beats, raw_signal, proc_signal, rpeaks, beat_meta = extract_beats_from_record(record_path)

    plt.figure(figsize=FIG_SIZE)
    plt.plot(raw_signal[:plot_len], label="raw")
    plt.title("Raw ECG")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    maybe_savefig(None if save_prefix is None else f"{save_prefix}_raw.png", output_dir=output_dir)
    plt.show()

    plt.figure(figsize=FIG_SIZE)
    plt.plot(proc_signal[:plot_len], label="preprocessed")
    rp = rpeaks[rpeaks < plot_len]
    plt.scatter(rp, proc_signal[rp], s=20, label="R-peaks")
    plt.title(f"Preprocessed ECG with R-peaks ({SEGMENT_MODE})")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    maybe_savefig(None if save_prefix is None else f"{save_prefix}_preprocessed.png", output_dir=output_dir)
    plt.show()

    if len(beats) > 0:
        plt.figure(figsize=FIG_SIZE)
        plt.plot(beats[0])
        plt.title(f"Single heartbeat ({SEGMENT_MODE})")
        plt.xlabel("Sample")
        plt.ylabel("Normalized amplitude")
        maybe_savefig(None if save_prefix is None else f"{save_prefix}_single_beat.png", output_dir=output_dir)
        plt.show()

        plt.figure(figsize=FIG_SIZE)
        for i in range(min(5, len(beats))):
            plt.plot(beats[i], alpha=0.8)
        plt.title(f"Several heartbeats ({SEGMENT_MODE})")
        plt.xlabel("Sample")
        plt.ylabel("Normalized amplitude")
        maybe_savefig(None if save_prefix is None else f"{save_prefix}_multi_beats.png", output_dir=output_dir)
        plt.show()

        if SEGMENT_MODE == "rr_segment":
            rr_lengths = [m["rr_len_raw"] for m in beat_meta]
            if len(rr_lengths) > 0:
                plt.figure(figsize=FIG_SIZE)
                plt.hist(rr_lengths, bins=20)
                plt.title("Raw RR length distribution")
                plt.xlabel("RR length (samples)")
                plt.ylabel("Count")
                maybe_savefig(None if save_prefix is None else f"{save_prefix}_rr_hist.png", output_dir=output_dir)
                plt.show()


def plot_subject_count_bar(subject_count_df, top_n=20, save_name=None, output_dir=None):
    plot_df = subject_count_df.head(top_n)

    plt.figure(figsize=FIG_SIZE)
    plt.bar(plot_df["label"].astype(str), plot_df["num_beats"])
    plt.title(f"Top {top_n} subjects by number of kept beats")
    plt.xlabel("Label")
    plt.ylabel("Number of beats")
    plt.xticks(rotation=45)
    maybe_savefig(save_name, output_dir=output_dir)
    plt.show()


def plot_training_curves(history, save_prefix=None, output_dir=None, smooth_window=5):
    train_acc = pd.Series(history.history["accuracy"])
    val_acc = pd.Series(history.history["val_accuracy"])
    train_loss = pd.Series(history.history["loss"])
    val_loss = pd.Series(history.history["val_loss"])

    # moving average smoothing
    train_acc_smooth = train_acc.rolling(window=smooth_window, min_periods=1).mean()
    val_acc_smooth = val_acc.rolling(window=smooth_window, min_periods=1).mean()
    train_loss_smooth = train_loss.rolling(window=smooth_window, min_periods=1).mean()
    val_loss_smooth = val_loss.rolling(window=smooth_window, min_periods=1).mean()

    plt.figure(figsize=FIG_SIZE)
    plt.plot(train_acc_smooth, label="train_acc")
    plt.plot(val_acc_smooth, label="val_acc")
    plt.title("Record-level Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    maybe_savefig(None if save_prefix is None else f"{save_prefix}_acc.png", output_dir=output_dir)
    plt.show()

    plt.figure(figsize=FIG_SIZE)
    plt.plot(train_loss_smooth, label="train_loss")
    plt.plot(val_loss_smooth, label="val_loss")
    plt.title("Record-level Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    maybe_savefig(None if save_prefix is None else f"{save_prefix}_loss.png", output_dir=output_dir)
    plt.show()


def plot_tsne_embeddings(
    embeddings,
    labels,
    title="t-SNE",
    max_classes=TSNE_MAX_CLASSES,
    max_samples_per_class=TSNE_MAX_SAMPLES_PER_CLASS,
    seed=SEED,
    save_name=None,
    output_dir=None,
):
    if embeddings is None or len(embeddings) == 0:
        print(f"[SKIP] {title}: no embeddings")
        return

    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)

    selected_indices = []
    unique_labels = sorted(np.unique(labels))[:max_classes]

    for lbl in unique_labels:
        idxs = np.where(labels == lbl)[0]
        if len(idxs) > max_samples_per_class:
            idxs = rng.choice(idxs, size=max_samples_per_class, replace=False)
        selected_indices.extend(idxs.tolist())

    if len(selected_indices) < 2:
        print(f"[SKIP] {title}: not enough samples")
        return

    selected_indices = np.array(selected_indices)
    emb_subset = embeddings[selected_indices]
    y_subset = labels[selected_indices]

    perplexity = min(TSNE_PERPLEXITY, max(5, len(emb_subset) - 1))

    tsne = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    emb_2d = tsne.fit_transform(emb_subset)

    plt.figure(figsize=FIG_SIZE)
    for lbl in sorted(np.unique(y_subset)):
        mask = y_subset == lbl
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=25, alpha=0.8, label=f"ID {lbl}")
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.6)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    maybe_savefig(save_name, output_dir=output_dir)
    plt.show()


def plot_similarity_distribution(pos_scores, neg_scores, eer_threshold, title, save_name=None, output_dir=None):
    plt.figure(figsize=FIG_SIZE)
    plt.hist(pos_scores, bins=HIST_BINS, alpha=0.6, label="Same person")
    plt.hist(neg_scores, bins=HIST_BINS, alpha=0.6, label="Different person")
    plt.axvline(eer_threshold, linestyle="--", label="EER threshold")
    plt.title(title)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.legend()
    maybe_savefig(save_name, output_dir=output_dir)
    plt.show()


def plot_roc_curve(fpr, tpr, roc_auc, title, save_name=None, output_dir=None):
    plt.figure(figsize=FIG_SIZE)
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    maybe_savefig(save_name, output_dir=output_dir)
    plt.show()


def plot_all_tsne(emb_dict, output_dir=None):
    if "test_embeddings" in emb_dict and "y_test" in emb_dict:
        plot_tsne_embeddings(
            emb_dict["test_embeddings"],
            emb_dict["y_test"],
            title="Record-level ECG Fingerprint t-SNE (Single-beat)",
            save_name="tsne_single_beat.png",
            output_dir=output_dir,
        )

    if "test_embeddings_2b" in emb_dict and "y_test_2b" in emb_dict:
        plot_tsne_embeddings(
            emb_dict["test_embeddings_2b"],
            emb_dict["y_test_2b"],
            title="Record-level ECG Fingerprint t-SNE (2-Beat Aggregation)",
            save_name="tsne_2beat.png",
            output_dir=output_dir,
        )

    if "test_embeddings_3b" in emb_dict and "y_test_3b" in emb_dict:
        plot_tsne_embeddings(
            emb_dict["test_embeddings_3b"],
            emb_dict["y_test_3b"],
            title="Record-level ECG Fingerprint t-SNE (3-Beat Aggregation)",
            save_name="tsne_3beat.png",
            output_dir=output_dir,
        )


def plot_all_verification_curves(verification_detail_dict, output_dir=None):
    for _, item in verification_detail_dict.items():
        mode = item["mode"]

        plot_similarity_distribution(
            item["pos_scores"],
            item["neg_scores"],
            item["eer_threshold"],
            title=f"{mode} Similarity Distribution",
            save_name=f"hist_{mode}.png",
            output_dir=output_dir,
        )

        plot_roc_curve(
            item["fpr"],
            item["tpr"],
            item["auc"],
            title=f"{mode} ROC Curve",
            save_name=f"roc_{mode}.png",
            output_dir=output_dir,
        )
