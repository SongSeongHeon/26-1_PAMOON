# ECG JSON 처리 서비스

from pathlib import Path
import json
import re

import numpy as np

MIN_ECG_POINTS = 100
DEFAULT_DISPLAY_POINTS = 1200
EPS = 1e-8

REGISTER_MIN_SAMPLE_COUNT = 14000
REGISTER_MIN_DURATION_SECONDS = 28.0
REGISTER_MIN_COMPLETENESS = 0.933
REGISTER_MAX_LEAD_OFF_COUNT = 20
REGISTER_MAX_INVALID_SAMPLE_COUNT = 100
REGISTER_MAX_MISSING_SAMPLE_ESTIMATE = 1000

TAIL_CHECK_SECONDS = 5
TAIL_LOW_VARIABILITY_RATIO = 0.15
TAIL_HARD_FLAT_RATIO = 0.05
TAIL_HARD_FLAT_STD = 0.005
DISPLAY_CLIP_LIMIT = 1.35
DISPLAY_TARGET_PERCENTILE = 98.0


# ECG JSON 로드
def load_ecg_json(json_path):
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"ECG JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("ECG JSON must be an object.")

    ecg_values = extract_ecg_values(data)
    sampling_rate = extract_sampling_rate(data)

    return {
        "raw": data,
        "ecg": ecg_values,
        "sampling_rate": sampling_rate,
    }


# ECG 배열 추출
def extract_ecg_values(data):
    candidate_keys = [
        "ecg_mv",
        "ecg",
        "signal",
        "waveform",
        "amplitude",
        "values",
    ]

    values = None

    for key in candidate_keys:
        if key in data:
            values = data[key]
            break

    if values is None:
        raise ValueError(
            "ECG JSON must contain one of: "
            "ecg_mv, ecg, signal, waveform, amplitude, values."
        )

    array = np.asarray(values, dtype=np.float32).reshape(-1)
    array = interpolate_nonfinite(array)
    array = clean_samsung_sdk_ecg(array, data)

    if len(array) < MIN_ECG_POINTS:
        raise ValueError(f"ECG signal is too short: {len(array)} points.")

    if np.std(array) < EPS:
        raise ValueError("ECG signal variance is too small.")

    return array.astype(np.float32)


# sampling rate 추출
def extract_sampling_rate(data):
    for key in ["sampling_rate", "fs", "sample_rate", "frequency"]:
        if key in data:
            try:
                sampling_rate = int(float(data[key]))

                if sampling_rate > 0:
                    return sampling_rate
            except Exception:
                pass

    return 500


def extract_sanity_check(raw):
    sanity = raw.get("sanity_check")

    if not isinstance(sanity, dict):
        return {
            "status": "unavailable",
            "warnings": [],
            "corrected_fields": [],
        }

    warnings = sanity.get("warnings")
    corrected_fields = sanity.get("corrected_fields")

    return {
        "status": str(sanity.get("status", "ok")),
        "warnings": warnings if isinstance(warnings, list) else [],
        "corrected_fields": (
            corrected_fields if isinstance(corrected_fields, list) else []
        ),
        "actual_sample_count": safe_int(sanity.get("actual_sample_count"), None),
        "actual_duration_seconds": safe_float(
            sanity.get("actual_duration_seconds"), None
        ),
    }


# NaN / Inf 보간
def interpolate_nonfinite(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = np.isfinite(values)

    if np.all(finite):
        return values

    if np.sum(finite) < 2:
        raise ValueError("ECG signal has too few finite values.")

    x = np.arange(len(values), dtype=np.float32)

    return np.interp(x, x[finite], values[finite]).astype(np.float32)


# Samsung Health Sensor SDK ECG 보정
def clean_samsung_sdk_ecg(values, raw):
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    if len(values) < MIN_ECG_POINTS:
        return values

    source = str(raw.get("source", "")).lower()
    tracker_type = str(raw.get("tracker_type", "")).lower()
    is_samsung_sdk = (
        "samsung health sensor sdk" in source
        or "ecg_on_demand" in tracker_type
        or "ecg_mv" in raw
    )

    if not is_samsung_sdk:
        return values.astype(np.float32)

    cleaned = values.copy()

    # ECGMonitor에서 이미 warmup 제거를 하지 않도록 변경했으므로,
    # 서버에서는 극단적인 초반 이상치가 있을 때만 제한적으로 제거
    sampling_rate = extract_sampling_rate(raw)
    warmup_samples = min(len(cleaned) // 10, int(sampling_rate * 1.0))

    if warmup_samples > 0:
        warmup = cleaned[:warmup_samples]
        body = cleaned[warmup_samples:]

        if len(body) >= MIN_ECG_POINTS:
            body_median = np.median(body)
            body_mad = np.median(np.abs(body - body_median)) + EPS
            warmup_deviation = np.abs(warmup - body_median) / (1.4826 * body_mad)

            if np.max(warmup_deviation) > 20:
                cleaned = body

    median = np.median(cleaned)
    mad = np.median(np.abs(cleaned - median))

    if mad < EPS:
        mean = np.mean(cleaned)
        std = np.std(cleaned)

        if std < EPS:
            return cleaned.astype(np.float32)

        z = np.abs((cleaned - mean) / (std + EPS))
        valid_mask = z < 12
    else:
        robust_z = np.abs((cleaned - median) / (1.4826 * mad + EPS))
        valid_mask = robust_z < 12

    if np.sum(valid_mask) >= MIN_ECG_POINTS:
        x = np.arange(len(cleaned), dtype=np.float32)
        cleaned = np.interp(
            x,
            x[valid_mask],
            cleaned[valid_mask],
        ).astype(np.float32)

    cleaned = cleaned - np.median(cleaned)

    q_low, q_high = np.percentile(cleaned, [1, 99])
    robust_range = float(q_high - q_low)

    if robust_range > EPS:
        cleaned = cleaned / robust_range

    cleaned = np.clip(cleaned, -5.0, 5.0)

    return cleaned.astype(np.float32)


# 1D 리샘플링
def resample_1d(values, target_len):
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    if len(values) == target_len:
        return values.astype(np.float32)

    if len(values) < 2:
        raise ValueError("Signal must contain at least 2 points for resampling.")

    old_x = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)

    return np.interp(new_x, old_x, values).astype(np.float32)


# 웹 표시용 normalization
def normalize_for_display(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    if values.size == 0:
        return values.astype(np.float32)

    centered = values - np.median(values)
    abs_scale = np.percentile(np.abs(centered), DISPLAY_TARGET_PERCENTILE)

    if abs_scale < EPS:
        std = np.std(centered)
        abs_scale = std if std > EPS else 1.0

    normalized = centered / (abs_scale + EPS)
    normalized = np.clip(normalized, -DISPLAY_CLIP_LIMIT, DISPLAY_CLIP_LIMIT)

    return normalized.astype(np.float32)


# time axis 생성
def make_time_axis(sample_count, sampling_rate):
    if sample_count <= 0:
        return []

    time = np.arange(sample_count, dtype=np.float32) / float(sampling_rate)

    return time.astype(float).tolist()


# JSON 저장
def save_json(payload, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return output_path


# subject_id 정규화
def normalize_subject_id(value):
    text = str(value or "").strip().upper().replace(" ", "")

    if re.fullmatch(r"S\d{3}", text):
        return text

    return ""


# session_id 정규화
def normalize_session_id(value):
    text = str(value or "").strip().upper().replace(" ", "")

    if re.fullmatch(r"T\d{2}", text):
        return text

    if re.fullmatch(r"\d+", text):
        return f"T{text.zfill(2)}"

    return ""


# 생년월일 텍스트 확인
def is_birth_date_text(value):
    text = str(value or "").strip()

    if len(text) != 8:
        return False

    if not text.isdigit():
        return False

    year = int(text[:4])
    month = int(text[4:6])
    day = int(text[6:8])

    return 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31


# 안전한 숫자 변환
def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default

        return int(float(value))
    except Exception:
        return default


def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default

        return float(value)
    except Exception:
        return default


def safe_bool(value, default=False):
    if isinstance(value, bool):
        return value

    if value is None:
        return default

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y"}:
        return True

    if text in {"false", "0", "no", "n"}:
        return False

    return default


# 파일명에서 subject/session 정보 추출
def parse_subject_info_from_filename(json_path):
    path = Path(json_path)
    stem = path.stem
    parts = [part.strip() for part in stem.split("_") if part.strip()]

    subject_id = ""
    session_id = ""
    birth_date = ""

    # 저장된 파일명 예:
    # manual_S001_20260530_xxxxx_S001_T01_20260527_163142_watch6_leadI_500Hz
    # watch_S001_20260530_xxxxx_S001_T01_20260530_153000_watch6_leadI_500Hz
    for index, part in enumerate(parts):
        normalized_subject = normalize_subject_id(part)

        if normalized_subject:
            subject_id = normalized_subject

            if index + 1 < len(parts):
                session_id = normalize_session_id(parts[index + 1])

            break

    for part in parts:
        if is_birth_date_text(part):
            birth_date = part
            break

    return {
        "subject_id": subject_id or "USER-UNKNOWN",
        "session_id": session_id,
        "birth_date": birth_date or "-",
        "filename": path.name,
    }


# subject 정보 결정
def resolve_subject_info(raw, json_path):
    filename_info = parse_subject_info_from_filename(json_path)

    subject_id = (
        normalize_subject_id(raw.get("subject_id"))
        or normalize_subject_id(raw.get("user_id"))
        or normalize_subject_id(raw.get("patient_id"))
        or filename_info["subject_id"]
    )

    session_id = (
        normalize_session_id(raw.get("session_id"))
        or normalize_session_id(raw.get("trial_id"))
        or normalize_session_id(raw.get("record_id"))
        or filename_info["session_id"]
    )

    birth_date = raw.get("birth_date") or filename_info["birth_date"]

    raw_name = str(
        raw.get("subject_name")
        or raw.get("display_name")
        or raw.get("real_name")
        or raw.get("name")
        or ""
    ).strip()

    blocked_names = {
        "",
        "-",
        "USER-01",
        "USER-UNKNOWN",
        "UNASSIGNED",
        "ECG 데이터",
        "ECG DATA",
        "ECG JSON Sample",
        "ecg data",
    }

    if raw_name in blocked_names or raw_name.upper() in blocked_names:
        name = subject_id
    else:
        name = raw_name

    if not subject_id or subject_id == "USER-UNKNOWN":
        subject_id = name if normalize_subject_id(name) else "USER-UNKNOWN"

    if not name or name == "USER-UNKNOWN":
        name = subject_id

    return {
        "subject_id": subject_id,
        "session_id": session_id,
        "birth_date": birth_date or "-",
        "name": name,
        "filename": raw.get("filename") or filename_info["filename"],
    }


# ECGMonitor 품질 메타데이터 추출
def extract_collection_quality(raw, ecg, sampling_rate):
    sample_count = int(len(ecg))
    sanity_check = extract_sanity_check(raw)
    expected_sample_count = safe_int(
        raw.get("expected_sample_count"),
        int(round(sampling_rate * 30)),
    )

    if expected_sample_count <= 0:
        expected_sample_count = int(round(sampling_rate * 30))

    duration_seconds = sample_count / float(sampling_rate)

    collection_completeness = safe_float(
        raw.get("collection_completeness"),
        sample_count / float(expected_sample_count),
    )

    missing_sample_estimate = safe_int(
        raw.get("missing_sample_estimate"),
        max(0, expected_sample_count - sample_count),
    )

    lead_off_count = safe_int(raw.get("lead_off_count"), 0)
    invalid_sample_count = safe_int(raw.get("invalid_sample_count"), 0)
    valid_sample_count = safe_int(raw.get("valid_sample_count"), sample_count)
    total_sample_count = safe_int(raw.get("total_sample_count"), sample_count)

    if valid_sample_count != sample_count:
        valid_sample_count = sample_count

    if total_sample_count < sample_count:
        total_sample_count = sample_count
    warmup_skip_count = safe_int(raw.get("warmup_skip_count"), 0)
    received_batch_count = safe_int(raw.get("received_batch_count"), 0)

    overall_std = safe_float(raw.get("overall_std"), float(np.std(ecg)))
    tail_5s_std = safe_float(
        raw.get("tail_5s_std"),
        calculate_tail_std(ecg, sampling_rate, seconds=TAIL_CHECK_SECONDS),
    )

    tail_std_ratio = safe_float(
        raw.get("tail_std_ratio"),
        tail_5s_std / (overall_std + EPS),
    )

    # ECGMonitor가 보낸 기존 flat_tail_detected는 기록용으로만 보관
    # 서버 등록 차단에는 더 엄격한 hard flatline 기준만 사용
    raw_flat_tail_detected = safe_bool(raw.get("flat_tail_detected"), False)

    tail_low_variability = (
        sample_count >= REGISTER_MIN_SAMPLE_COUNT
        and tail_std_ratio < TAIL_LOW_VARIABILITY_RATIO
    )

    hard_flat_tail_detected = (
        sample_count >= REGISTER_MIN_SAMPLE_COUNT
        and tail_std_ratio < TAIL_HARD_FLAT_RATIO
        and tail_5s_std < TAIL_HARD_FLAT_STD
    )

    flat_tail_detected = bool(hard_flat_tail_detected)

    quality_status = str(raw.get("quality_status") or "").strip().lower()
    quality_reason = str(raw.get("quality_reason") or "").strip()

    # 앱에서 flat_tail_detected로 보낸 사유가 있어도,
    # 서버 hard flatline 기준에 걸리지 않으면 등록 차단 사유로 쓰지 않음
    if quality_reason == "flat_tail_detected" and not flat_tail_detected:
        if tail_low_variability:
            quality_reason = "tail_low_variability"
        else:
            quality_reason = "ok"

    if not quality_status:
        quality_status = estimate_quality_status(
            sample_count=sample_count,
            duration_seconds=duration_seconds,
            collection_completeness=collection_completeness,
            missing_sample_estimate=missing_sample_estimate,
            lead_off_count=lead_off_count,
            invalid_sample_count=invalid_sample_count,
            flat_tail_detected=flat_tail_detected,
            overall_std=overall_std,
        )

    if not quality_reason:
        quality_reason = estimate_quality_reason(
            sample_count=sample_count,
            duration_seconds=duration_seconds,
            collection_completeness=collection_completeness,
            missing_sample_estimate=missing_sample_estimate,
            lead_off_count=lead_off_count,
            invalid_sample_count=invalid_sample_count,
            flat_tail_detected=flat_tail_detected,
            overall_std=overall_std,
        )

        if quality_reason == "ok" and tail_low_variability:
            quality_reason = "tail_low_variability"

    sample_index = raw.get("sample_index")
    has_sample_index = (
        isinstance(sample_index, list) and len(sample_index) == sample_count
    )

    if sanity_check.get("actual_sample_count") not in (None, sample_count):
        sanity_check["warnings"] = sorted(
            set(
                list(sanity_check.get("warnings", []))
                + ["actual_sample_count_mismatch"]
            )
        )
        sanity_check["status"] = "corrected"

    return {
        "expected_sample_count": int(expected_sample_count),
        "sample_count": int(sample_count),
        "duration_seconds": round(float(duration_seconds), 4),
        "collection_completeness": round(float(collection_completeness), 6),
        "missing_sample_estimate": int(missing_sample_estimate),
        "lead_off_count": int(lead_off_count),
        "invalid_sample_count": int(invalid_sample_count),
        "valid_sample_count": int(valid_sample_count),
        "total_sample_count": int(total_sample_count),
        "warmup_skip_count": int(warmup_skip_count),
        "received_batch_count": int(received_batch_count),
        "overall_std": round(float(overall_std), 6),
        "tail_5s_std": round(float(tail_5s_std), 6),
        "tail_std_ratio": round(float(tail_std_ratio), 6),
        "tail_low_variability": bool(tail_low_variability),
        "raw_flat_tail_detected": bool(raw_flat_tail_detected),
        "flat_tail_detected": bool(flat_tail_detected),
        "quality_status": quality_status,
        "quality_reason": quality_reason,
        "has_sample_index": bool(has_sample_index),
        "sanity_check": sanity_check,
    }


# tail std 계산
def calculate_tail_std(ecg, sampling_rate, seconds=5):
    values = np.asarray(ecg, dtype=np.float32).reshape(-1)

    if values.size == 0:
        return 0.0

    tail_count = min(values.size, max(1, int(float(sampling_rate) * seconds)))
    tail = values[-tail_count:]

    return float(np.std(tail))


# 품질 상태 추정
def estimate_quality_status(
    sample_count,
    duration_seconds,
    collection_completeness,
    missing_sample_estimate,
    lead_off_count,
    invalid_sample_count,
    flat_tail_detected,
    overall_std,
):
    if (
        sample_count >= 14500
        and duration_seconds >= 29.0
        and collection_completeness >= 0.965
        and missing_sample_estimate <= 500
        and lead_off_count <= 5
        and invalid_sample_count <= 30
        and not flat_tail_detected
        and overall_std > EPS
    ):
        return "stable"

    if (
        sample_count >= REGISTER_MIN_SAMPLE_COUNT
        and duration_seconds >= REGISTER_MIN_DURATION_SECONDS
        and collection_completeness >= REGISTER_MIN_COMPLETENESS
        and missing_sample_estimate <= REGISTER_MAX_MISSING_SAMPLE_ESTIMATE
        and lead_off_count <= REGISTER_MAX_LEAD_OFF_COUNT
        and invalid_sample_count <= REGISTER_MAX_INVALID_SAMPLE_COUNT
        and not flat_tail_detected
        and overall_std > EPS
    ):
        return "caution"

    return "warning"


# 품질 사유 추정
def estimate_quality_reason(
    sample_count,
    duration_seconds,
    collection_completeness,
    missing_sample_estimate,
    lead_off_count,
    invalid_sample_count,
    flat_tail_detected,
    overall_std,
):
    if sample_count < REGISTER_MIN_SAMPLE_COUNT:
        return "low_sample_count"

    if duration_seconds < REGISTER_MIN_DURATION_SECONDS:
        return "short_duration"

    if collection_completeness < REGISTER_MIN_COMPLETENESS:
        return "low_collection_completeness"

    if missing_sample_estimate > REGISTER_MAX_MISSING_SAMPLE_ESTIMATE:
        return "high_missing_sample_estimate"

    if lead_off_count > REGISTER_MAX_LEAD_OFF_COUNT:
        return "high_lead_off_count"

    if invalid_sample_count > REGISTER_MAX_INVALID_SAMPLE_COUNT:
        return "high_invalid_sample_count"

    if flat_tail_detected:
        return "flat_tail_detected"

    if overall_std <= EPS:
        return "low_signal_variance"

    return "ok"


# 등록 템플릿으로 쓸 수 있는지 판단
def is_registration_eligible(quality):
    return (
        quality["sample_count"] >= REGISTER_MIN_SAMPLE_COUNT
        and quality["duration_seconds"] >= REGISTER_MIN_DURATION_SECONDS
        and quality["collection_completeness"] >= REGISTER_MIN_COMPLETENESS
        and quality["missing_sample_estimate"] <= REGISTER_MAX_MISSING_SAMPLE_ESTIMATE
        and quality["lead_off_count"] <= REGISTER_MAX_LEAD_OFF_COUNT
        and quality["invalid_sample_count"] <= REGISTER_MAX_INVALID_SAMPLE_COUNT
        and not quality["flat_tail_detected"]
        and quality["overall_std"] > EPS
        and quality["quality_status"] in {"stable", "caution"}
    )


# 인증 비교에 쓸 수 있는지 판단
def is_verification_eligible(quality):
    return (
        quality["sample_count"] >= 12500
        and quality["duration_seconds"] >= 25.0
        and quality["collection_completeness"] >= 0.80
        and not quality["flat_tail_detected"]
        and quality["overall_std"] > EPS
        and quality["quality_status"] in {"stable", "caution"}
    )


# 모델 입력용 waveform 생성
def build_model_waveform(raw, ecg, sampling_rate, json_path):
    duration_seconds = len(ecg) / float(sampling_rate)
    subject_info = resolve_subject_info(raw, json_path)
    quality = extract_collection_quality(raw, ecg, sampling_rate)

    waveform = {
        "source": raw.get("source", "Samsung Health Sensor SDK"),
        "tracker_type": raw.get("tracker_type", "-"),
        "subject_id": subject_info["subject_id"],
        "session_id": subject_info["session_id"],
        "name": subject_info["name"],
        "birth_date": subject_info["birth_date"],
        "device": raw.get("device", "Samsung Galaxy Watch"),
        "record": raw.get("record", "-"),
        "unit": raw.get("unit", "mV"),
        "signal_name": raw.get("signal_name", "ECG"),
        "sampling_rate": int(sampling_rate),
        "duration_seconds": round(float(duration_seconds), 4),
        "sample_count": int(len(ecg)),
        "expected_sample_count": quality["expected_sample_count"],
        "collection_completeness": quality["collection_completeness"],
        "missing_sample_estimate": quality["missing_sample_estimate"],
        "lead_off_count": quality["lead_off_count"],
        "invalid_sample_count": quality["invalid_sample_count"],
        "flat_tail_detected": quality["flat_tail_detected"],
        "tail_low_variability": quality["tail_low_variability"],
        "raw_flat_tail_detected": quality["raw_flat_tail_detected"],
        "quality_status": quality["quality_status"],
        "quality_reason": quality["quality_reason"],
        "registration_eligible": is_registration_eligible(quality),
        "verification_eligible": is_verification_eligible(quality),
        "time": make_time_axis(len(ecg), sampling_rate),
        "amplitude": ecg.astype(float).tolist(),
        "source_json": str(json_path),
        "sanity_check": quality["sanity_check"],
    }

    sample_index = raw.get("sample_index")
    if isinstance(sample_index, list) and len(sample_index) == len(ecg):
        waveform["sample_index"] = sample_index

    return waveform


# 웹 표시용 waveform 생성
def build_display_waveform(raw, ecg, sampling_rate, json_path, display_points):
    duration_seconds = len(ecg) / float(sampling_rate)
    subject_info = resolve_subject_info(raw, json_path)
    quality = extract_collection_quality(raw, ecg, sampling_rate)

    display_signal = resample_1d(ecg, display_points)
    display_signal = normalize_for_display(display_signal)
    display_time = np.linspace(0, duration_seconds, display_points).astype(float)

    waveform = {
        "subject_id": subject_info["subject_id"],
        "session_id": subject_info["session_id"],
        "time": display_time.tolist(),
        "amplitude": display_signal.astype(float).tolist(),
        "sample_count": int(len(display_signal)),
        "source_json": str(json_path),
        "raw_sample_count": int(len(ecg)),
        "raw_duration_seconds": round(float(duration_seconds), 4),
        "sampling_rate": int(sampling_rate),
        "expected_sample_count": quality["expected_sample_count"],
        "collection_completeness": quality["collection_completeness"],
        "missing_sample_estimate": quality["missing_sample_estimate"],
        "flat_tail_detected": quality["flat_tail_detected"],
        "tail_low_variability": quality["tail_low_variability"],
        "raw_flat_tail_detected": quality["raw_flat_tail_detected"],
        "quality_status": quality["quality_status"],
        "quality_reason": quality["quality_reason"],
        "display_normalization": "robust_percentile",
        "display_clip_limit": DISPLAY_CLIP_LIMIT,
        "sanity_check": quality["sanity_check"],
    }

    return waveform


# report 생성
def build_report(raw, sampling_rate, json_path):
    subject_info = resolve_subject_info(raw, json_path)
    package = load_ecg_json_for_quality_only(json_path)
    ecg = package["ecg"]
    quality = extract_collection_quality(raw, ecg, sampling_rate)

    return {
        "subject_id": subject_info["subject_id"],
        "session_id": subject_info["session_id"],
        "name": subject_info["name"],
        "birth_date": subject_info["birth_date"],
        "rhythm": raw.get("rhythm", "-"),
        "measured_at": raw.get("measured_at", "-"),
        "heart_rate": raw.get("heart_rate", None),
        "device": raw.get("device", "Samsung Galaxy Watch"),
        "sampling_rate": f"{sampling_rate} Hz",
        "lead": raw.get("lead", "Lead I ECG"),
        "filename": subject_info["filename"] or Path(json_path).name,
        "raw_text": "",
        "quality_status": quality["quality_status"],
        "quality_reason": quality["quality_reason"],
        "collection_completeness": quality["collection_completeness"],
        "missing_sample_estimate": quality["missing_sample_estimate"],
        "lead_off_count": quality["lead_off_count"],
        "invalid_sample_count": quality["invalid_sample_count"],
        "flat_tail_detected": quality["flat_tail_detected"],
        "tail_low_variability": quality["tail_low_variability"],
        "raw_flat_tail_detected": quality["raw_flat_tail_detected"],
        "tail_5s_std": quality["tail_5s_std"],
        "tail_std_ratio": quality["tail_std_ratio"],
        "registration_eligible": is_registration_eligible(quality),
        "verification_eligible": is_verification_eligible(quality),
        "sanity_check": quality["sanity_check"],
    }


# build_report에서 재귀 방지용 ECG 로드
def load_ecg_json_for_quality_only(json_path):
    json_path = Path(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ecg_values = extract_ecg_values(data)

    return {
        "raw": data,
        "ecg": ecg_values,
    }


# ECG summary 생성
def build_raw_summary(raw, ecg, sampling_rate, json_path=None):
    duration_seconds = len(ecg) / float(sampling_rate)
    quality = extract_collection_quality(raw, ecg, sampling_rate)

    if json_path is not None:
        subject_info = resolve_subject_info(raw, json_path)
    else:
        subject_info = {
            "subject_id": raw.get("subject_id", "USER-UNKNOWN"),
            "session_id": raw.get("session_id", ""),
            "birth_date": raw.get("birth_date", "-"),
            "name": raw.get("name", raw.get("subject_id", "ECG 데이터")),
            "filename": raw.get("filename", ""),
        }

    return {
        "source": raw.get("source", "-"),
        "tracker_type": raw.get("tracker_type", "-"),
        "subject_id": subject_info["subject_id"],
        "session_id": subject_info["session_id"],
        "name": subject_info["name"],
        "birth_date": subject_info["birth_date"],
        "filename": subject_info["filename"],
        "device": raw.get("device", "Samsung Galaxy Watch"),
        "record": raw.get("record", "-"),
        "unit": raw.get("unit", "mV"),
        "signal_name": raw.get("signal_name", "ECG"),
        "sampling_rate": int(sampling_rate),
        "expected_sample_count": quality["expected_sample_count"],
        "raw_sample_count": int(len(ecg)),
        "sample_count": int(len(ecg)),
        "duration_seconds": round(float(duration_seconds), 4),
        "collection_completeness": quality["collection_completeness"],
        "missing_sample_estimate": quality["missing_sample_estimate"],
        "total_sample_count": quality["total_sample_count"],
        "valid_sample_count": quality["valid_sample_count"],
        "invalid_sample_count": quality["invalid_sample_count"],
        "warmup_skip_count": quality["warmup_skip_count"],
        "lead_off_count": quality["lead_off_count"],
        "received_batch_count": quality["received_batch_count"],
        "overall_std": quality["overall_std"],
        "tail_5s_std": quality["tail_5s_std"],
        "tail_std_ratio": quality["tail_std_ratio"],
        "flat_tail_detected": quality["flat_tail_detected"],
        "tail_low_variability": quality["tail_low_variability"],
        "raw_flat_tail_detected": quality["raw_flat_tail_detected"],
        "quality_status": quality["quality_status"],
        "quality_reason": quality["quality_reason"],
        "registration_eligible": is_registration_eligible(quality),
        "verification_eligible": is_verification_eligible(quality),
        "has_sample_index": quality["has_sample_index"],
        "sanity_check": quality["sanity_check"],
        "min": round(float(np.min(ecg)), 6),
        "max": round(float(np.max(ecg)), 6),
        "mean": round(float(np.mean(ecg)), 6),
        "std": round(float(np.std(ecg)), 6),
    }


# 등록용 품질 게이트
def validate_registration_quality(raw_summary):
    if raw_summary.get("registration_eligible"):
        return

    reason = raw_summary.get("quality_reason", "unknown")
    status = raw_summary.get("quality_status", "unknown")

    sanity = raw_summary.get("sanity_check") or {}
    sanity_status = sanity.get("status", "unavailable")
    sanity_warnings = (
        ",".join(sanity.get("warnings", []))
        if isinstance(sanity.get("warnings"), list)
        else ""
    )

    message = (
        "등록 템플릿으로 사용할 수 없는 ECG입니다. "
        f"quality_status={status}, reason={reason}, "
        f"sample_count={raw_summary.get('sample_count')}, "
        f"duration={raw_summary.get('duration_seconds')}, "
        f"collection_completeness={raw_summary.get('collection_completeness')}, "
        f"lead_off_count={raw_summary.get('lead_off_count')}, "
        f"invalid_sample_count={raw_summary.get('invalid_sample_count')}, "
        f"flat_tail_detected={raw_summary.get('flat_tail_detected')}, "
        f"sanity_status={sanity_status}, sanity_warnings={sanity_warnings}"
    )

    raise ValueError(message)


# ECG JSON을 웹/모델 응답용 waveform으로 변환
def build_raw_waveform_response(
    json_path,
    display_points=DEFAULT_DISPLAY_POINTS,
    display_waveform_json_path=None,
    model_waveform_json_path=None,
):
    package = load_ecg_json(json_path)

    raw = package["raw"]
    ecg = package["ecg"]
    sampling_rate = package["sampling_rate"]

    report = build_report(raw, sampling_rate, json_path)
    display_waveform = build_display_waveform(
        raw=raw,
        ecg=ecg,
        sampling_rate=sampling_rate,
        json_path=json_path,
        display_points=display_points,
    )
    model_waveform = build_model_waveform(
        raw=raw,
        ecg=ecg,
        sampling_rate=sampling_rate,
        json_path=json_path,
    )
    raw_summary = build_raw_summary(raw, ecg, sampling_rate, json_path)

    if display_waveform_json_path is not None:
        save_json(display_waveform, display_waveform_json_path)

    if model_waveform_json_path is not None:
        save_json(model_waveform, model_waveform_json_path)

    return {
        "report": report,
        "waveform": display_waveform,
        "model_waveform": model_waveform,
        "raw_summary": raw_summary,
        "display_waveform_json_path": (
            str(display_waveform_json_path) if display_waveform_json_path else None
        ),
        "model_waveform_json_path": (
            str(model_waveform_json_path) if model_waveform_json_path else None
        ),
    }
