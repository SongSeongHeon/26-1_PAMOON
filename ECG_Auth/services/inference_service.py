# 기본 설정
from pathlib import Path
import json
import re
import tempfile

import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
TEMPLATE_DIR = BASE_DIR / "data" / "templates"

CONFIG_PATH = MODEL_DIR / "plain_cnn1d_config.json"
EMBEDDING_MODEL_PATH = MODEL_DIR / "plain_cnn1d_embedding.keras"

TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

_embedding_model = None
_config = None

DEFAULT_INPUT_LENGTH = 400
DEFAULT_MODEL_NAME = "Plain CNN1D"
DEFAULT_NORMALIZATION = "z_score"
DEFAULT_THRESHOLD_FLOOR = 0.8500

MIN_WAVEFORM_POINTS = 80
MIN_SIGNAL_STD = 1e-6
MIN_PEAK_TO_PEAK = 0.05
EPS = 1e-8

DEFAULT_SAMPLING_RATE = 500
MIN_R_PEAK_COUNT = 8
MIN_SEGMENT_COUNT = 6
MAX_SEGMENT_COUNT = 40
R_PEAK_MIN_DISTANCE_SECONDS = 0.35
SEGMENT_WINDOW_SECONDS = 0.8
TOP_MATCH_RATIO = 0.30
ROW_BEST_RATIO = 0.50
COL_BEST_RATIO = 0.50


# 모델 설정 로드
def load_config():
    global _config

    if _config is not None:
        return _config

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Model config not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _config = json.load(f)

    if not isinstance(_config, dict):
        raise ValueError("Model config must be a JSON object.")

    return _config


# embedding 모델 로드
def load_embedding_model():
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    if not EMBEDDING_MODEL_PATH.exists():
        raise FileNotFoundError(f"Embedding model not found: {EMBEDDING_MODEL_PATH}")

    _embedding_model = tf.keras.models.load_model(
        EMBEDDING_MODEL_PATH,
        compile=False,
        safe_mode=False,
    )

    return _embedding_model


# 설정값 조회
def get_config_value(key, default=None):
    config = load_config()
    return config.get(key, default)


# 모델 입력 길이
def get_input_length():
    value = get_config_value("input_length", DEFAULT_INPUT_LENGTH)

    try:
        input_length = int(value)
    except Exception as error:
        raise ValueError(f"Invalid input_length in config: {value}") from error

    if input_length < 64:
        raise ValueError(f"input_length is too small: {input_length}")

    return input_length


# 모델명
def get_model_name():
    return str(get_config_value("model_name", DEFAULT_MODEL_NAME))


# 정규화 방식
def get_normalization_name():
    return str(get_config_value("normalization", DEFAULT_NORMALIZATION)).lower()


# 인증 임계값
def get_effective_threshold():
    value = get_config_value("threshold", DEFAULT_THRESHOLD_FLOOR)

    try:
        config_threshold = float(value)
    except Exception as error:
        raise ValueError(f"Invalid threshold in config: {value}") from error

    allow_low_threshold = bool(get_config_value("allow_low_threshold", False))

    if allow_low_threshold:
        return config_threshold

    return max(config_threshold, DEFAULT_THRESHOLD_FLOOR)


# subject_id 정규화
def normalize_subject_id(value):
    text = str(value or "").strip().upper().replace(" ", "")

    if re.fullmatch(r"S\d{3}", text):
        return text

    return ""


# 파일명 안전화
def sanitize_key(value):
    text = str(value or "unknown").strip()

    text = text.replace("/", "_")
    text = text.replace("\\", "_")
    text = text.replace("..", "_")

    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._-")

    if not text:
        text = "unknown"

    return text


# 사용자별 템플릿 파일 경로
def get_template_paths(user_key):
    safe_key = sanitize_key(user_key)

    embeddings_path = TEMPLATE_DIR / f"{safe_key}_embeddings.npy"
    metadata_path = TEMPLATE_DIR / f"{safe_key}_metadata.json"

    return embeddings_path, metadata_path


# JSON 저장
def save_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
        suffix=".tmp",
    ) as temp_file:
        json.dump(payload, temp_file, ensure_ascii=False, indent=2)
        temp_path = Path(temp_file.name)

    temp_path.replace(path)


# NPY 저장
def save_npy_atomic(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "wb",
        dir=str(path.parent),
        delete=False,
        suffix=".tmp",
    ) as temp_file:
        temp_path = Path(temp_file.name)
        np.save(temp_file, array)

    temp_path.replace(path)


# waveform JSON 패키지 로드
def load_waveform_package(waveform_json_path):
    waveform_json_path = Path(waveform_json_path)

    if not waveform_json_path.exists():
        raise FileNotFoundError(f"Waveform JSON not found: {waveform_json_path}")

    with open(waveform_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    values = extract_waveform_values_from_json(data)
    values = sanitize_waveform_values(values)

    sampling_rate = extract_sampling_rate_from_waveform_json(data)
    quality = extract_quality_flags_from_waveform_json(data)

    return {
        "data": data,
        "values": values,
        "sampling_rate": sampling_rate,
        "quality": quality,
        "path": waveform_json_path,
    }


# waveform JSON 로드
def load_waveform_json(waveform_json_path):
    package = load_waveform_package(waveform_json_path)
    return package["values"]


# sampling rate 추출
def extract_sampling_rate_from_waveform_json(data):
    if isinstance(data, dict):
        for key in ["sampling_rate", "fs", "sample_rate", "frequency"]:
            if key not in data:
                continue

            try:
                sampling_rate = int(float(data[key]))
                if sampling_rate > 0:
                    return sampling_rate
            except Exception:
                pass

    return DEFAULT_SAMPLING_RATE


# 품질 메타데이터 추출
def extract_quality_flags_from_waveform_json(data):
    if not isinstance(data, dict):
        return {
            "registration_eligible": True,
            "verification_eligible": True,
            "quality_status": "unknown",
            "quality_reason": "not_available",
            "flat_tail_detected": False,
            "sanity_check": {},
        }

    sanity_check = data.get("sanity_check")
    if not isinstance(sanity_check, dict):
        sanity_check = {}

    return {
        "registration_eligible": bool(data.get("registration_eligible", True)),
        "verification_eligible": bool(data.get("verification_eligible", True)),
        "quality_status": str(data.get("quality_status", "unknown")),
        "quality_reason": str(data.get("quality_reason", "not_available")),
        "flat_tail_detected": bool(data.get("flat_tail_detected", False)),
        "sample_count": data.get("sample_count"),
        "duration_seconds": data.get("duration_seconds"),
        "collection_completeness": data.get("collection_completeness"),
        "missing_sample_estimate": data.get("missing_sample_estimate"),
        "lead_off_count": data.get("lead_off_count"),
        "invalid_sample_count": data.get("invalid_sample_count"),
        "sanity_check": sanity_check,
    }


# JSON에서 waveform 값 추출
def extract_waveform_values_from_json(data):
    if isinstance(data, dict):
        for key in [
            "model_amplitude",
            "model_waveform",
            "amplitude",
            "waveform",
            "signal",
        ]:
            if key not in data:
                continue

            values = data[key]

            if isinstance(values, dict):
                for nested_key in ["amplitude", "waveform", "signal", "values"]:
                    if nested_key in values:
                        return values[nested_key]

                raise ValueError(
                    f"Nested waveform object in '{key}' must contain "
                    "amplitude/waveform/signal/values."
                )

            return values

        raise ValueError(
            "Waveform JSON must contain one of: "
            "'model_amplitude', 'model_waveform', 'amplitude', 'waveform', or 'signal'."
        )

    if isinstance(data, list):
        return data

    raise ValueError("Unsupported waveform JSON format.")


# waveform 값 정리
def sanitize_waveform_values(values):
    array = np.asarray(values, dtype=np.float32)

    if array.ndim != 1:
        array = array.reshape(-1)

    if len(array) < MIN_WAVEFORM_POINTS:
        raise ValueError(
            f"Waveform is too short: {len(array)} points. "
            f"At least {MIN_WAVEFORM_POINTS} points are required."
        )

    array = interpolate_nonfinite(array)

    if len(array) < MIN_WAVEFORM_POINTS:
        raise ValueError("Waveform became too short after sanitization.")

    if not np.all(np.isfinite(array)):
        raise ValueError("Waveform contains non-finite values after sanitization.")

    std = float(np.std(array))
    peak_to_peak = float(np.max(array) - np.min(array))

    if std < MIN_SIGNAL_STD:
        raise ValueError("Waveform signal variance is too small for authentication.")

    if peak_to_peak < MIN_PEAK_TO_PEAK:
        raise ValueError("Waveform amplitude range is too small for authentication.")

    return array.astype(np.float32)


# NaN/Inf 보간
def interpolate_nonfinite(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = np.isfinite(values)

    if np.all(finite):
        return values

    if np.sum(finite) < 2:
        raise ValueError("Waveform has too few finite values.")

    x = np.arange(len(values), dtype=np.float32)

    return np.interp(x, x[finite], values[finite]).astype(np.float32)


# 1D 리샘플링
def resample_1d(values, target_length):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    target_length = int(target_length)

    if target_length < 2:
        raise ValueError("target_length must be at least 2.")

    if len(values) == target_length:
        return values.astype(np.float32)

    if len(values) < 2:
        raise ValueError("Waveform must contain at least 2 points for resampling.")

    old_x = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)

    return np.interp(new_x, old_x, values).astype(np.float32)


# 이동 평균
def moving_average(values, window_size):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    window_size = int(window_size)

    if window_size <= 1 or len(values) < 3:
        return values.astype(np.float32)

    window_size = min(window_size, len(values))

    if window_size % 2 == 0:
        window_size += 1

    pad = window_size // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window_size, dtype=np.float32) / window_size

    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


# median filter
def median_filter_1d(values, window_size=3):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    window_size = int(window_size)

    if window_size <= 1 or len(values) < 3:
        return values.astype(np.float32)

    window_size = min(window_size, len(values))

    if window_size % 2 == 0:
        window_size += 1

    pad = window_size // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    result = np.empty_like(values, dtype=np.float32)

    for index in range(len(values)):
        result[index] = np.median(padded[index : index + window_size])

    return result.astype(np.float32)


# robust clip
def robust_clip(values, z=6.0):
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    median = np.median(values)
    mad = np.median(np.abs(values - median)) + EPS

    robust_z = 0.6745 * (values - median) / mad
    clipped_z = np.clip(robust_z, -z, z)

    return (clipped_z * mad / 0.6745 + median).astype(np.float32)


# z-score 정규화
def z_score_normalize(values, eps=EPS):
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    mean = np.mean(values)
    std = np.std(values)

    if std < eps:
        return (values - mean).astype(np.float32)

    return ((values - mean) / (std + eps)).astype(np.float32)


# robust z-score 정규화
def robust_z_score_normalize(values, eps=EPS):
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    median = np.median(values)
    mad = np.median(np.abs(values - median))

    if mad < eps:
        return z_score_normalize(values, eps=eps)

    scaled = (values - median) / (1.4826 * mad + eps)

    return np.clip(scaled, -6.0, 6.0).astype(np.float32)


# min-max 정규화
def minmax_normalize(values, eps=EPS):
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    min_v = np.min(values)
    max_v = np.max(values)

    if abs(max_v - min_v) < eps:
        return (values - min_v).astype(np.float32)

    return ((values - min_v) / (max_v - min_v + eps)).astype(np.float32)


# 모델 입력 정규화
def normalize_for_model(values):
    normalization = get_normalization_name()

    if normalization in ["z_score", "z-score", "standard"]:
        return z_score_normalize(values)

    if normalization in ["robust_z_score", "robust-z-score", "robust"]:
        return robust_z_score_normalize(values)

    if normalization in ["minmax", "min_max", "min-max"]:
        return minmax_normalize(values)

    if normalization in ["none", "raw"]:
        return np.asarray(values, dtype=np.float32).reshape(-1)

    raise ValueError(f"Unsupported normalization method: {normalization}")


# 모델 입력용 파형 안정화
def stabilize_waveform_for_model(waveform_values):
    x = np.asarray(waveform_values, dtype=np.float32).reshape(-1)
    x = interpolate_nonfinite(x)

    x = robust_clip(x, z=7.0)

    if len(x) >= 7:
        x = median_filter_1d(x, window_size=3)

    x = x - np.median(x)

    if np.std(x) < MIN_SIGNAL_STD:
        raise ValueError("Stabilized waveform variance is too small.")

    return x.astype(np.float32)


# 단일 segment 모델 입력 전처리
def preprocess_segment_for_model(segment_values):
    input_length = get_input_length()

    x = stabilize_waveform_for_model(segment_values)
    x = resample_1d(x, input_length)
    x = normalize_for_model(x)

    if not np.all(np.isfinite(x)):
        raise ValueError("Model input contains non-finite values after preprocessing.")

    if np.std(x) < MIN_SIGNAL_STD:
        raise ValueError("Model input variance is too small.")

    x = x.astype(np.float32)
    x = x.reshape(input_length, 1)

    return x


# 기존 호환용 전체 waveform 입력 전처리
def preprocess_waveform_for_model(waveform_values):
    x = preprocess_segment_for_model(waveform_values)
    return x.reshape(1, x.shape[0], 1)


# waveform 품질 진단
def diagnose_waveform_quality(waveform_values):
    values = np.asarray(waveform_values, dtype=np.float32).reshape(-1)
    values = interpolate_nonfinite(values)

    length = int(len(values))
    std = float(np.std(values))
    peak_to_peak = float(np.max(values) - np.min(values))
    robust_peak_to_peak = float(np.percentile(values, 99) - np.percentile(values, 1))
    finite_ratio = float(np.mean(np.isfinite(values)))

    quality_pass = (
        length >= MIN_WAVEFORM_POINTS
        and std >= MIN_SIGNAL_STD
        and max(peak_to_peak, robust_peak_to_peak) >= MIN_PEAK_TO_PEAK
        and finite_ratio >= 0.98
    )

    return {
        "length": length,
        "std": round(std, 6),
        "peak_to_peak": round(peak_to_peak, 6),
        "robust_peak_to_peak": round(robust_peak_to_peak, 6),
        "finite_ratio": round(finite_ratio, 6),
        "quality_pass": bool(quality_pass),
    }


# R-peak 검출용 신호 생성
def make_peak_detection_signal(values):
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    x = interpolate_nonfinite(x)

    x = x - np.median(x)
    mad = np.median(np.abs(x - np.median(x)))

    if mad < EPS:
        x = z_score_normalize(x)
    else:
        x = (x - np.median(x)) / (1.4826 * mad + EPS)

    x = np.clip(x, -8.0, 8.0)

    # R파가 양/음 어느 방향이어도 잡히도록 절댓값 기반 검출
    return np.abs(x).astype(np.float32)


# 간단 R-peak 후보 검출
def detect_r_peaks(waveform_values, sampling_rate):
    signal = make_peak_detection_signal(waveform_values)

    if len(signal) < MIN_WAVEFORM_POINTS:
        return np.array([], dtype=np.int64)

    min_distance = max(1, int(float(sampling_rate) * R_PEAK_MIN_DISTANCE_SECONDS))

    threshold = max(
        1.0,
        float(np.percentile(signal, 92)),
        float(np.mean(signal) + 1.0 * np.std(signal)),
    )

    candidate_indexes = []

    for index in range(1, len(signal) - 1):
        if signal[index] < threshold:
            continue

        if signal[index] >= signal[index - 1] and signal[index] >= signal[index + 1]:
            candidate_indexes.append(index)

    if not candidate_indexes:
        return np.array([], dtype=np.int64)

    # 강한 peak부터 선택하고, min_distance 이내 중복 제거
    candidate_indexes = sorted(
        candidate_indexes,
        key=lambda idx: float(signal[idx]),
        reverse=True,
    )

    selected = []

    for idx in candidate_indexes:
        if all(abs(idx - chosen) >= min_distance for chosen in selected):
            selected.append(idx)

    selected = sorted(selected)

    return np.asarray(selected, dtype=np.int64)


# R-centered segment 추출
def extract_r_centered_segments(waveform_values, sampling_rate):
    values = np.asarray(waveform_values, dtype=np.float32).reshape(-1)
    input_length = get_input_length()

    window_samples = int(float(sampling_rate) * SEGMENT_WINDOW_SECONDS)

    if window_samples < input_length:
        window_samples = input_length

    if window_samples % 2 != 0:
        window_samples += 1

    half_window = window_samples // 2

    r_peaks = detect_r_peaks(values, sampling_rate)

    if len(r_peaks) < MIN_R_PEAK_COUNT:
        return np.empty((0, input_length, 1), dtype=np.float32), {
            "r_peak_count": int(len(r_peaks)),
            "segment_count": 0,
            "window_samples": int(window_samples),
            "reason": "not_enough_r_peaks",
        }

    segments = []
    used_peaks = []

    for peak in r_peaks:
        start = int(peak) - half_window
        end = int(peak) + half_window

        if start < 0 or end > len(values):
            continue

        segment = values[start:end]

        if len(segment) < MIN_WAVEFORM_POINTS:
            continue

        try:
            model_input = preprocess_segment_for_model(segment)
        except Exception:
            continue

        segments.append(model_input)
        used_peaks.append(int(peak))

        if len(segments) >= MAX_SEGMENT_COUNT:
            break

    if len(segments) < MIN_SEGMENT_COUNT:
        return np.empty((0, input_length, 1), dtype=np.float32), {
            "r_peak_count": int(len(r_peaks)),
            "segment_count": int(len(segments)),
            "window_samples": int(window_samples),
            "reason": "not_enough_valid_segments",
        }

    return np.asarray(segments, dtype=np.float32), {
        "r_peak_count": int(len(r_peaks)),
        "segment_count": int(len(segments)),
        "used_peak_indexes": used_peaks,
        "window_samples": int(window_samples),
        "reason": "ok",
    }


# segment batch에서 embedding 추출
def extract_embeddings_from_segments(segments):
    if segments.ndim != 3:
        raise ValueError(f"segments must be 3D array, got shape={segments.shape}")

    if segments.shape[0] == 0:
        raise ValueError("No valid ECG segments for embedding extraction.")

    model = load_embedding_model()

    embeddings = model.predict(segments, verbose=0)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    if embeddings.ndim != 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)

    validate_embedding_matrix(embeddings, name="segment_embeddings")

    embeddings = l2_normalize_matrix(embeddings)

    return embeddings.astype(np.float32)


# waveform에서 multi-beat embedding 추출
def extract_embeddings_from_waveform(waveform_values, sampling_rate):
    segments, segmentation = extract_r_centered_segments(
        waveform_values=waveform_values,
        sampling_rate=sampling_rate,
    )

    if segments.shape[0] == 0:
        raise ValueError(
            "Failed to extract enough valid ECG beat segments: " f"{segmentation}"
        )

    embeddings = extract_embeddings_from_segments(segments)

    return embeddings, segmentation


# 기존 호환용 단일 embedding 추출
def extract_embedding_from_waveform(waveform_values):
    model = load_embedding_model()

    x = preprocess_waveform_for_model(waveform_values)

    embedding = model.predict(x, verbose=0)
    embedding = np.asarray(embedding, dtype=np.float32)

    if embedding.ndim == 2:
        embedding = embedding[0]

    embedding = embedding.reshape(-1)
    validate_embedding(embedding, name="extracted_embedding")

    return embedding


# waveform JSON에서 단일 embedding 추출
def extract_embedding_from_waveform_json(waveform_json_path):
    waveform_values = load_waveform_json(waveform_json_path)
    embedding = extract_embedding_from_waveform(waveform_values)

    return embedding


# embedding 검증
def validate_embedding(embedding, name="embedding"):
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)

    if embedding.size == 0:
        raise ValueError(f"{name} is empty.")

    if not np.all(np.isfinite(embedding)):
        raise ValueError(f"{name} contains non-finite values.")

    norm = np.linalg.norm(embedding)

    if norm < EPS:
        raise ValueError(f"{name} norm is too small.")

    return True


# embedding matrix 검증
def validate_embedding_matrix(embeddings, name="embeddings"):
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(f"{name} must be 2D array.")

    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError(f"{name} is empty.")

    if not np.all(np.isfinite(embeddings)):
        raise ValueError(f"{name} contains non-finite values.")

    norms = np.linalg.norm(embeddings, axis=1)

    if np.any(norms < EPS):
        raise ValueError(f"{name} contains near-zero vectors.")

    return True


# L2 정규화
def l2_normalize_vector(vector, eps=EPS):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    validate_embedding(vector, name="vector_before_l2")

    norm = np.linalg.norm(vector)

    if norm < eps:
        raise ValueError("Vector norm is too small for L2 normalization.")

    return (vector / (norm + eps)).astype(np.float32)


# embedding matrix L2 정규화
def l2_normalize_matrix(matrix, eps=EPS):
    matrix = np.asarray(matrix, dtype=np.float32)

    validate_embedding_matrix(matrix, name="matrix_before_l2")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)

    return (matrix / (norms + eps)).astype(np.float32)


# cosine similarity 계산
def cosine_similarity(a, b, eps=EPS):
    a = l2_normalize_vector(a, eps=eps)
    b = l2_normalize_vector(b, eps=eps)

    similarity = float(np.dot(a, b))
    similarity = float(np.clip(similarity, -1.0, 1.0))

    return similarity


# multi-beat similarity 계산
def robust_multibeat_similarity(
    template_embeddings, query_embeddings, top_ratio=TOP_MATCH_RATIO
):
    template_embeddings = l2_normalize_matrix(template_embeddings)
    query_embeddings = l2_normalize_matrix(query_embeddings)

    if template_embeddings.shape[1] != query_embeddings.shape[1]:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"template={template_embeddings.shape}, query={query_embeddings.shape}"
        )

    similarity_matrix = np.matmul(query_embeddings, template_embeddings.T)
    similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)

    flat_scores = similarity_matrix.reshape(-1)

    if flat_scores.size == 0:
        raise ValueError("No similarity scores available.")

    query_best_scores = np.max(similarity_matrix, axis=1)
    template_best_scores = np.max(similarity_matrix, axis=0)

    global_k = max(1, int(np.ceil(flat_scores.size * top_ratio)))
    query_k = max(1, int(np.ceil(query_best_scores.size * ROW_BEST_RATIO)))
    template_k = max(1, int(np.ceil(template_best_scores.size * COL_BEST_RATIO)))

    global_top_scores = np.sort(flat_scores)[-global_k:]
    query_top_scores = np.sort(query_best_scores)[-query_k:]
    template_top_scores = np.sort(template_best_scores)[-template_k:]

    global_median = float(np.median(global_top_scores))
    query_median = float(np.median(query_top_scores))
    template_median = float(np.median(template_top_scores))

    similarity = float(
        0.50 * global_median + 0.30 * query_median + 0.20 * template_median
    )
    similarity = float(np.clip(similarity, -1.0, 1.0))

    return similarity, {
        "query_segment_count": int(query_embeddings.shape[0]),
        "template_segment_count": int(template_embeddings.shape[0]),
        "score_count": int(flat_scores.size),
        "top_score_count": int(global_k),
        "top_score_mean": round(float(np.mean(global_top_scores)), 6),
        "top_score_median": round(global_median, 6),
        "top_score_min": round(float(np.min(global_top_scores)), 6),
        "top_score_max": round(float(np.max(global_top_scores)), 6),
        "query_best_median": round(query_median, 6),
        "template_best_median": round(template_median, 6),
        "blended_similarity": round(similarity, 6),
    }


# 템플릿 metadata 로드
def load_template_metadata(user_key):
    _, metadata_path = get_template_paths(user_key)

    if not metadata_path.exists():
        return None

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return metadata


# 사용자별 템플릿 등록
def register_template_from_waveform_json(
    waveform_json_path,
    user_key,
    user_id=None,
    name=None,
    birth_date=None,
):
    embeddings_path, metadata_path = get_template_paths(user_key)

    subject_id = normalize_subject_id(name) or normalize_subject_id(user_id) or ""

    package = load_waveform_package(waveform_json_path)
    waveform_values = package["values"]
    sampling_rate = package["sampling_rate"]
    quality = package["quality"]

    if not quality.get("registration_eligible", True):
        raise ValueError(
            "Template ECG is not eligible for registration: "
            f"quality_status={quality.get('quality_status')}, "
            f"quality_reason={quality.get('quality_reason')}, "
            f"flat_tail_detected={quality.get('flat_tail_detected')}"
        )

    diagnostics = diagnose_waveform_quality(waveform_values)

    if not diagnostics["quality_pass"]:
        raise ValueError(
            "Template waveform quality is not sufficient for registration."
        )

    embeddings, segmentation = extract_embeddings_from_waveform(
        waveform_values=waveform_values,
        sampling_rate=sampling_rate,
    )

    threshold = get_effective_threshold()
    input_length = get_input_length()
    normalization = get_normalization_name()

    save_npy_atomic(embeddings_path, embeddings)

    metadata = {
        "subject_id": subject_id,
        "user_key": user_key,
        "user_id": user_id or user_key,
        "name": name,
        "birth_date": birth_date,
        "model_name": get_model_name(),
        "embedding_model_file": str(EMBEDDING_MODEL_PATH.name),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_count": int(embeddings.shape[0]),
        "input_length": input_length,
        "normalization": normalization,
        "threshold": round(float(threshold), 6),
        "source_waveform_json": str(waveform_json_path),
        "template_embedding_file": str(embeddings_path),
        "template_metadata_file": str(metadata_path),
        "fingerprint_mode": "multibeat_r_centered_ecg_embedding",
        "diagnostics": diagnostics,
        "segmentation": segmentation,
        "quality": quality,
        "status": "registered",
    }

    save_json_atomic(metadata_path, metadata)

    return {
        "subject_id": subject_id,
        "user_key": user_key,
        "user_id": user_id or user_key,
        "name": name,
        "birth_date": birth_date,
        "model": metadata["model_name"],
        "embedding_dim": metadata["embedding_dim"],
        "embedding_count": metadata["embedding_count"],
        "threshold": metadata["threshold"],
        "status": "registered",
        "template_embedding_file": str(embeddings_path),
        "template_metadata_file": str(metadata_path),
        "fingerprint_mode": metadata["fingerprint_mode"],
        "waveform_quality": diagnostics,
        "segmentation": segmentation,
        "collection_quality": quality,
    }


# 인증 실패 응답 생성
def build_rejected_result(
    waveform_json_path,
    user_key,
    user_id=None,
    template_embedding_path=None,
    reason="Rejected.",
    diagnostics=None,
    segmentation=None,
    quality=None,
):
    threshold = get_effective_threshold()
    subject_id = normalize_subject_id(user_id) or ""

    return {
        "subject_id": subject_id,
        "user_key": user_key,
        "user_id": user_id or user_key,
        "model": get_model_name(),
        "embedding_dim": 0,
        "embedding_count": 0,
        "cosine_similarity": 0.0,
        "matching_score": 0.0,
        "threshold": round(float(threshold), 6),
        "decision": "Rejected",
        "decision_reason": reason,
        "quality_pass": False,
        "template_embedding_file": (
            str(template_embedding_path) if template_embedding_path else None
        ),
        "source_waveform_json": str(waveform_json_path),
        "fingerprint_mode": "multibeat_r_centered_ecg_embedding",
        "waveform_quality": diagnostics or {},
        "segmentation": segmentation or {},
        "collection_quality": quality or {},
    }


# 사용자별 템플릿 인증
def verify_waveform_json_against_template(
    waveform_json_path,
    user_key,
    template_embedding_path=None,
    user_id=None,
):
    if template_embedding_path is None:
        template_embedding_path, _ = get_template_paths(user_key)
    else:
        template_embedding_path = Path(template_embedding_path)

    subject_id = normalize_subject_id(user_id) or ""

    if not template_embedding_path.exists():
        raise FileNotFoundError(
            f"Template embedding not found: {template_embedding_path}. "
            "Register a template ECG first."
        )

    try:
        package = load_waveform_package(waveform_json_path)
        waveform_values = package["values"]
        sampling_rate = package["sampling_rate"]
        quality = package["quality"]

        if not quality.get("verification_eligible", True):
            return build_rejected_result(
                waveform_json_path=waveform_json_path,
                user_key=user_key,
                user_id=user_id,
                template_embedding_path=template_embedding_path,
                reason=(
                    "Rejected by collection quality gate: "
                    f"quality_status={quality.get('quality_status')}, "
                    f"quality_reason={quality.get('quality_reason')}, "
                    f"flat_tail_detected={quality.get('flat_tail_detected')}"
                ),
                diagnostics={},
                segmentation={},
                quality=quality,
            )

        diagnostics = diagnose_waveform_quality(waveform_values)

        if not diagnostics["quality_pass"]:
            return build_rejected_result(
                waveform_json_path=waveform_json_path,
                user_key=user_key,
                user_id=user_id,
                template_embedding_path=template_embedding_path,
                reason="Rejected by waveform quality gate.",
                diagnostics=diagnostics,
                quality=quality,
            )

        template_embeddings = np.load(template_embedding_path)

        if template_embeddings.ndim == 1:
            template_embeddings = template_embeddings.reshape(1, -1)

        validate_embedding_matrix(template_embeddings, name="template_embeddings")
        template_embeddings = l2_normalize_matrix(template_embeddings)

        query_embeddings, segmentation = extract_embeddings_from_waveform(
            waveform_values=waveform_values,
            sampling_rate=sampling_rate,
        )

        if template_embeddings.shape[1] != query_embeddings.shape[1]:
            return build_rejected_result(
                waveform_json_path=waveform_json_path,
                user_key=user_key,
                user_id=user_id,
                template_embedding_path=template_embedding_path,
                reason=(
                    "Rejected by embedding dimension mismatch: "
                    f"template={template_embeddings.shape}, "
                    f"query={query_embeddings.shape}."
                ),
                diagnostics=diagnostics,
                segmentation=segmentation,
                quality=quality,
            )

        threshold = get_effective_threshold()

        similarity, similarity_detail = robust_multibeat_similarity(
            template_embeddings=template_embeddings,
            query_embeddings=query_embeddings,
            top_ratio=TOP_MATCH_RATIO,
        )

        matching_score = max(0.0, similarity) * 100.0

        quality_pass = bool(diagnostics["quality_pass"])
        decision = (
            "Authenticated" if similarity >= threshold and quality_pass else "Rejected"
        )

        if not quality_pass:
            decision_reason = "Rejected by waveform quality gate."
        elif similarity < threshold:
            decision_reason = "Rejected by similarity threshold."
        else:
            decision_reason = "Authenticated."

        return {
            "subject_id": subject_id,
            "user_key": user_key,
            "user_id": user_id or user_key,
            "model": get_model_name(),
            "embedding_dim": int(query_embeddings.shape[1]),
            "embedding_count": int(query_embeddings.shape[0]),
            "template_embedding_count": int(template_embeddings.shape[0]),
            "cosine_similarity": round(float(similarity), 6),
            "matching_score": round(float(matching_score), 3),
            "threshold": round(float(threshold), 6),
            "decision": decision,
            "decision_reason": decision_reason,
            "quality_pass": bool(quality_pass),
            "template_embedding_file": str(template_embedding_path),
            "source_waveform_json": str(waveform_json_path),
            "fingerprint_mode": "multibeat_r_centered_ecg_embedding",
            "waveform_quality": diagnostics,
            "segmentation": segmentation,
            "similarity_detail": similarity_detail,
            "collection_quality": quality,
        }

    except Exception as error:
        return build_rejected_result(
            waveform_json_path=waveform_json_path,
            user_key=user_key,
            user_id=user_id,
            template_embedding_path=template_embedding_path,
            reason=f"Rejected by inference error: {str(error)}",
            diagnostics={},
            segmentation={},
            quality={},
        )


# 특정 사용자 템플릿 삭제
def delete_template_embedding(user_key):
    deleted = []

    embeddings_path, metadata_path = get_template_paths(user_key)

    for path in [embeddings_path, metadata_path]:
        if path.exists():
            path.unlink()
            deleted.append(str(path))

    return deleted


# 모든 템플릿 삭제
def reset_template_embedding():
    deleted = []

    for path in TEMPLATE_DIR.glob("*_embeddings.npy"):
        path.unlink()
        deleted.append(str(path))

    for path in TEMPLATE_DIR.glob("*_embedding.npy"):
        path.unlink()
        deleted.append(str(path))

    for path in TEMPLATE_DIR.glob("*_metadata.json"):
        path.unlink()
        deleted.append(str(path))

    legacy_files = [
        TEMPLATE_DIR / "template_embedding.npy",
        TEMPLATE_DIR / "template_embeddings.npy",
        TEMPLATE_DIR / "template_metadata.json",
        TEMPLATE_DIR / "USER-01_embedding.npy",
        TEMPLATE_DIR / "USER-01_embeddings.npy",
        TEMPLATE_DIR / "USER-01_embedding_metadata.json",
        TEMPLATE_DIR / "USER-01_metadata.json",
    ]

    for path in legacy_files:
        if path.exists():
            path.unlink()
            deleted.append(str(path))

    return deleted
