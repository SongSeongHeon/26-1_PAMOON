import numpy as np
import neurokit2 as nk

from scipy.signal import butter, filtfilt, detrend

from src.config import (
    SAMPLING_RATE,
    BANDPASS_LOW,
    BANDPASS_HIGH,
    FILTER_ORDER,
)


def bandpass_filter(signal_1d, fs=SAMPLING_RATE, low=BANDPASS_LOW, high=BANDPASS_HIGH, order=FILTER_ORDER):
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = butter(order, [low_norm, high_norm], btype="band")
    return filtfilt(b, a, signal_1d)


def zscore_normalize(signal_1d):
    signal_1d = np.asarray(signal_1d, dtype=np.float32)
    return (signal_1d - np.mean(signal_1d)) / (np.std(signal_1d) + 1e-8)


def preprocess_ecg(signal_1d):
    signal_1d = np.asarray(signal_1d, dtype=np.float32)
    signal_1d = detrend(signal_1d)
    signal_1d = bandpass_filter(signal_1d)
    return signal_1d.astype(np.float32)


def resample_to_fixed_length(signal_1d, target_len):
    signal_1d = np.asarray(signal_1d, dtype=np.float32)
    if len(signal_1d) < 2:
        return None
    old_x = np.linspace(0, 1, len(signal_1d))
    new_x = np.linspace(0, 1, target_len)
    resampled = np.interp(new_x, old_x, signal_1d)
    return resampled.astype(np.float32)


def detect_rpeaks(proc_signal, sampling_rate=SAMPLING_RATE):
    _, info = nk.ecg_process(proc_signal, sampling_rate=sampling_rate)
    rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
    rpeaks = np.unique(rpeaks)
    return rpeaks
