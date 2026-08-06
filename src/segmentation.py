import numpy as np
import wfdb

from src.config import (
    SAMPLING_RATE,
    CHANNEL_INDEX,
    SEGMENT_MODE,
    RPEAK_LEFT,
    RPEAK_RIGHT,
    BEAT_LEN,
    USE_RR_RIGID_THRESHOLD,
    RR_STD_FACTOR,
    MIN_RR_COUNT_FOR_THRESHOLD,
)
from src.preprocessing import (
    preprocess_ecg,
    detect_rpeaks,
    resample_to_fixed_length,
    zscore_normalize,
)


def segment_beats_r_centered(proc_signal, rpeaks, left=RPEAK_LEFT, right=RPEAK_RIGHT):
    beats = []
    beat_meta = []

    for i, r in enumerate(rpeaks):
        start = r - left
        end = r + right

        if start < 0 or end > len(proc_signal):
            continue

        beat = proc_signal[start:end]
        beat = zscore_normalize(beat)

        beats.append(beat)
        beat_meta.append({
            "beat_index": i,
            "segment_type": "r_centered",
            "start_idx": int(start),
            "end_idx": int(end),
            "rpeak_idx": int(r),
            "rr_len_raw": int(len(beat)),
            "kept_by_threshold": True,
        })

    return beats, beat_meta


def compute_rr_lengths(rpeaks):
    if len(rpeaks) < 2:
        return np.asarray([], dtype=np.int64)
    return np.diff(rpeaks)


def rigid_rr_threshold(rr_lengths, std_factor=RR_STD_FACTOR, min_rr_count=MIN_RR_COUNT_FOR_THRESHOLD):
    rr_lengths = np.asarray(rr_lengths, dtype=np.float32)

    if len(rr_lengths) < min_rr_count:
        return np.ones(len(rr_lengths), dtype=bool), None, None, None, None

    mu = float(np.mean(rr_lengths))
    sigma = float(np.std(rr_lengths))
    lower = mu - std_factor * sigma
    upper = mu + std_factor * sigma

    keep_mask = (rr_lengths >= lower) & (rr_lengths <= upper)
    return keep_mask, mu, sigma, lower, upper


def segment_beats_rr(
    proc_signal,
    rpeaks,
    target_len=BEAT_LEN,
    use_threshold=USE_RR_RIGID_THRESHOLD,
    std_factor=RR_STD_FACTOR,
    min_rr_count=MIN_RR_COUNT_FOR_THRESHOLD,
):
    beats = []
    beat_meta = []

    if len(rpeaks) < 2:
        return beats, beat_meta

    rr_lengths = compute_rr_lengths(rpeaks)

    if use_threshold:
        keep_mask, mu, sigma, lower, upper = rigid_rr_threshold(
            rr_lengths,
            std_factor=std_factor,
            min_rr_count=min_rr_count,
        )
    else:
        keep_mask = np.ones(len(rr_lengths), dtype=bool)
        mu = sigma = lower = upper = None

    for i in range(len(rpeaks) - 1):
        start = int(rpeaks[i])
        end = int(rpeaks[i + 1])
        rr_len = end - start
        keep = bool(keep_mask[i])

        if start < 0 or end > len(proc_signal) or rr_len < 2:
            continue

        if not keep:
            continue

        beat_raw = proc_signal[start:end]
        beat = resample_to_fixed_length(beat_raw, target_len=target_len)

        if beat is None:
            continue

        beat = zscore_normalize(beat)
        beats.append(beat)

        beat_meta.append({
            "beat_index": i,
            "segment_type": "rr_segment",
            "start_idx": start,
            "end_idx": end,
            "rpeak_idx": start,
            "rr_len_raw": int(rr_len),
            "kept_by_threshold": keep,
            "rr_mu": None if mu is None else float(mu),
            "rr_sigma": None if sigma is None else float(sigma),
            "rr_lower": None if lower is None else float(lower),
            "rr_upper": None if upper is None else float(upper),
        })

    return beats, beat_meta


def extract_beats_from_record(
    record_path,
    sampling_rate=SAMPLING_RATE,
    channel_index=CHANNEL_INDEX,
    segment_mode=SEGMENT_MODE,
    left=RPEAK_LEFT,
    right=RPEAK_RIGHT,
    target_len=BEAT_LEN,
    use_rr_threshold=USE_RR_RIGID_THRESHOLD,
    rr_std_factor=RR_STD_FACTOR,
    min_rr_count=MIN_RR_COUNT_FOR_THRESHOLD,
):
    record = wfdb.rdrecord(record_path)
    raw_signal = np.asarray(record.p_signal[:, channel_index], dtype=np.float32)
    proc_signal = preprocess_ecg(raw_signal)
    rpeaks = detect_rpeaks(proc_signal, sampling_rate=sampling_rate)

    if segment_mode == "r_centered":
        beats, beat_meta = segment_beats_r_centered(proc_signal, rpeaks, left=left, right=right)
    elif segment_mode == "rr_segment":
        beats, beat_meta = segment_beats_rr(
            proc_signal,
            rpeaks,
            target_len=target_len,
            use_threshold=use_rr_threshold,
            std_factor=rr_std_factor,
            min_rr_count=min_rr_count,
        )
    else:
        raise ValueError(f"Unknown segment_mode: {segment_mode}")

    if len(beats) == 0:
        return np.empty((0, target_len), dtype=np.float32), raw_signal, proc_signal, rpeaks, []

    return np.asarray(beats, dtype=np.float32), raw_signal, proc_signal, rpeaks, beat_meta
