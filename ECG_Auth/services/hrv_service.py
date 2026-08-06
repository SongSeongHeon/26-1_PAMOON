# services/hrv_service.py

import numpy as np
from scipy.signal import find_peaks


def calculate_hrv_metrics(waveform_values, sampling_rate=500, pdf_heart_rate=None):
    signal = np.asarray(waveform_values, dtype=np.float32).reshape(-1)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    if signal.size < int(sampling_rate * 5):
        return build_empty_hrv_result(
            pdf_heart_rate=pdf_heart_rate,
            message="ECG 데이터 길이가 짧아 신호 품질을 계산할 수 없습니다.",
        )

    peaks = detect_r_peaks_simple(signal, sampling_rate=sampling_rate)

    if len(peaks) < 3:
        return build_empty_hrv_result(
            pdf_heart_rate=pdf_heart_rate,
            message="R-peak 검출 수가 부족해 신호 품질을 계산할 수 없습니다.",
        )

    rr_sec = np.diff(peaks) / float(sampling_rate)
    valid_mask = (rr_sec >= 0.4) & (rr_sec <= 1.5)
    valid_rr = rr_sec[valid_mask]

    if len(valid_rr) < 2:
        return build_empty_hrv_result(
            pdf_heart_rate=pdf_heart_rate,
            message="유효 RR interval이 부족해 신호 품질을 계산할 수 없습니다.",
        )

    rr_mean_sec = float(np.mean(valid_rr))
    ecg_heart_rate = float(60.0 / max(rr_mean_sec, 1e-8))

    rr_std_sec = float(np.std(valid_rr))
    sdnn_ms = float(np.std(valid_rr, ddof=1) * 1000.0) if len(valid_rr) > 1 else 0.0

    rr_diff = np.diff(valid_rr)

    if len(rr_diff) > 0:
        rmssd_ms = float(np.sqrt(np.mean(np.square(rr_diff))) * 1000.0)
        pnn50 = float(np.mean(np.abs(rr_diff) > 0.05))
    else:
        rmssd_ms = 0.0
        pnn50 = 0.0

    valid_rr_ratio = float(len(valid_rr) / max(len(rr_sec), 1))

    rr_cv = rr_std_sec / max(rr_mean_sec, 1e-8)
    rr_stability_score = float(1.0 / (1.0 + rr_cv * 8.0))
    rr_stability_score = float(np.clip(rr_stability_score, 0.0, 1.0))

    heart_rate_bpm = resolve_display_heart_rate(
        pdf_heart_rate=pdf_heart_rate,
        ecg_heart_rate=ecg_heart_rate,
    )

    quality = classify_ecg_reliability(
        heart_rate_bpm=heart_rate_bpm,
        ecg_heart_rate=ecg_heart_rate,
        pdf_heart_rate=pdf_heart_rate,
        valid_rr_ratio=valid_rr_ratio,
        rr_stability_score=rr_stability_score,
        valid_rr_count=len(valid_rr),
    )

    return {
        "available": True,
        "heart_rate_bpm": round_or_none(heart_rate_bpm, 1),
        "pdf_heart_rate_bpm": (
            int(pdf_heart_rate) if pdf_heart_rate is not None else None
        ),
        "ecg_estimated_heart_rate_bpm": round_or_none(ecg_heart_rate, 1),
        "rr_mean_sec": round_or_none(rr_mean_sec, 3),
        "sdnn_ms": round_or_none(sdnn_ms, 1),
        "rmssd_ms": round_or_none(rmssd_ms, 1),
        "pnn50": round_or_none(pnn50, 3),
        "r_peak_count": int(len(peaks)),
        "valid_rr_count": int(len(valid_rr)),
        "valid_rr_ratio": round_or_none(valid_rr_ratio, 3),
        "rr_stability_score": round_or_none(rr_stability_score, 3),
        "quality_status": quality["status"],
        "quality_label": quality["label"],
        "trust_level": quality["trust_level"],
        "quality_message": quality["message"],
    }


def detect_r_peaks_simple(signal, sampling_rate=500):
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    centered = signal - np.median(signal)

    if np.std(centered) < 1e-8:
        return np.array([], dtype=np.int64)

    min_distance = max(1, int(0.35 * sampling_rate))
    prominence = max(float(np.std(centered) * 0.35), 1e-6)

    positive_peaks, _ = find_peaks(
        centered,
        distance=min_distance,
        prominence=prominence,
    )

    negative_peaks, _ = find_peaks(
        -centered,
        distance=min_distance,
        prominence=prominence,
    )

    if len(negative_peaks) > len(positive_peaks):
        peaks = negative_peaks
    else:
        peaks = positive_peaks

    return np.asarray(peaks, dtype=np.int64)


def resolve_display_heart_rate(pdf_heart_rate=None, ecg_heart_rate=None):
    if pdf_heart_rate is not None:
        try:
            reference_hr = float(pdf_heart_rate)

            if 30 <= reference_hr <= 220:
                return reference_hr
        except Exception:
            pass

    if ecg_heart_rate is not None and np.isfinite(ecg_heart_rate):
        return float(ecg_heart_rate)

    return None


def classify_ecg_reliability(
    heart_rate_bpm,
    ecg_heart_rate,
    pdf_heart_rate,
    valid_rr_ratio,
    rr_stability_score,
    valid_rr_count,
):
    if valid_rr_count < 3:
        return {
            "status": "unavailable",
            "label": "확인 불가",
            "trust_level": "확인 필요",
            "message": "유효한 ECG beat 수가 부족해 신호 품질을 판단하기 어렵습니다.",
        }

    if heart_rate_bpm is None:
        return {
            "status": "unavailable",
            "label": "확인 불가",
            "trust_level": "확인 필요",
            "message": "심박수 정보를 확인할 수 없어 신호 품질 판단이 제한됩니다.",
        }

    if heart_rate_bpm < 35 or heart_rate_bpm > 170:
        return {
            "status": "warning",
            "label": "확인 필요",
            "trust_level": "낮음",
            "message": "심박수가 일반적인 웨어러블 ECG 측정 범위를 벗어나 재측정이 권장됩니다.",
        }

    hr_diff = None

    if pdf_heart_rate is not None and ecg_heart_rate is not None:
        try:
            hr_diff = abs(float(pdf_heart_rate) - float(ecg_heart_rate))
        except Exception:
            hr_diff = None

    if (
        valid_rr_count >= 22
        and valid_rr_ratio >= 0.55
        and rr_stability_score >= 0.40
        and (hr_diff is None or hr_diff <= 35)
    ):
        return {
            "status": "stable",
            "label": "안정적",
            "trust_level": "높음",
            "message": "현재 ECG는 인증 비교에 사용할 수 있는 안정적인 신호로 판단됩니다.",
        }

    if (
        valid_rr_count >= 15
        and valid_rr_ratio >= 0.35
        and rr_stability_score >= 0.20
        and (hr_diff is None or hr_diff <= 55)
    ):
        return {
            "status": "caution",
            "label": "주의",
            "trust_level": "중간",
            "message": "일부 RR 간격 변동이 있으나 인증 비교에 참고 가능한 ECG 신호입니다.",
        }

    return {
        "status": "warning",
        "label": "확인 필요",
        "trust_level": "낮음",
        "message": "RR 간격이 불안정하게 검출되어 ECG 재측정이 권장됩니다.",
    }


def build_empty_hrv_result(
    pdf_heart_rate=None, message="ECG 신호 품질을 계산할 수 없습니다."
):
    return {
        "available": False,
        "heart_rate_bpm": int(pdf_heart_rate) if pdf_heart_rate is not None else None,
        "pdf_heart_rate_bpm": (
            int(pdf_heart_rate) if pdf_heart_rate is not None else None
        ),
        "ecg_estimated_heart_rate_bpm": None,
        "rr_mean_sec": None,
        "sdnn_ms": None,
        "rmssd_ms": None,
        "pnn50": None,
        "r_peak_count": 0,
        "valid_rr_count": 0,
        "valid_rr_ratio": 0.0,
        "rr_stability_score": 0.0,
        "quality_status": "unavailable",
        "quality_label": "확인 불가",
        "trust_level": "확인 필요",
        "quality_message": message,
    }


def round_or_none(value, digits=1):
    if value is None:
        return None

    try:
        value = float(value)
    except Exception:
        return None

    if not np.isfinite(value):
        return None

    return round(value, digits)
