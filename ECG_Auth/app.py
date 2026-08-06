# 기본 설정
import hashlib
import re
import uuid
from datetime import datetime
from pathlib import Path
import json
import os
os.environ["TF_USE_LEGACY_KERAS"] = "0"

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from services.hrv_service import calculate_hrv_metrics
from services.inference_service import (
    register_template_from_waveform_json,
    verify_waveform_json_against_template,
    reset_template_embedding,
)
from services.db_service import (
    init_db,
    make_user_key,
    get_user_by_key,
    create_user,
    update_user_last_verified,
    create_auth_log,
    get_auth_logs_by_user,
    reset_database,
)
from services.raw_ecg_service import (
    build_raw_waveform_response,
    validate_registration_quality,
)

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent

RAW_UPLOAD_DIR = BASE_DIR / "data" / "raw_uploads"
RAW_EXTRACTED_DIR = BASE_DIR / "data" / "extracted" / "raw"
TEMPLATE_DIR = BASE_DIR / "data" / "templates"

REQUIRED_DIRS = [
    RAW_UPLOAD_DIR,
    RAW_EXTRACTED_DIR,
    TEMPLATE_DIR,
]


# Flask 설정
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024


# 세션 상태
session_state = {
    "last_user_key": None,
    "last_mode": None,
    "last_waveform": None,
    "latest_result": None,
    "latest_received_at": None,
    "current_subject": None,
}


# 앱 초기화
for directory in REQUIRED_DIRS:
    directory.mkdir(parents=True, exist_ok=True)

init_db()


# 공통 응답 함수
def success_response(payload: dict, status_code: int = 200):
    return (
        jsonify(
            {
                "success": True,
                **payload,
            }
        ),
        status_code,
    )


def error_response(message: str, status_code: int = 400):
    return (
        jsonify(
            {
                "success": False,
                "message": message,
            }
        ),
        status_code,
    )


# 파일 유틸
def allowed_raw_json(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "json"


def validate_uploaded_raw_json():
    if "file" not in request.files:
        return None, error_response("ECG JSON 파일이 없습니다.", 400)

    file = request.files["file"]

    if file.filename == "":
        return None, error_response("선택된 파일이 없습니다.", 400)

    if not allowed_raw_json(file.filename):
        return None, error_response("JSON 파일만 업로드할 수 있습니다.", 400)

    return file, None


def make_saved_filename(
    original_filename: str,
    role: str,
    user_key: str = "unknown",
) -> str:
    safe_name = secure_filename(original_filename)

    if not safe_name:
        safe_name = "ecg_data.json"

    safe_user_key = secure_filename(str(user_key)) or "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]

    return f"{role}_{safe_user_key}_{timestamp}_{unique_id}_{safe_name}"


def calculate_file_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_json_safely(path_value):
    if not path_value:
        return None

    path = Path(path_value)

    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as error:
        print(f"[JSON LOAD ERROR] {error}")
        return None


def save_payload_to_raw_dir(
    payload: dict,
    original_filename: str,
    role: str,
    user_key: str = "unknown",
) -> tuple[Path, str]:
    saved_filename = make_saved_filename(
        original_filename=original_filename,
        role=role,
        user_key=user_key,
    )
    saved_path = RAW_UPLOAD_DIR / saved_filename

    with open(saved_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return saved_path, saved_filename


def load_uploaded_json_file(file) -> dict:
    try:
        payload = json.load(file.stream)
    except Exception as error:
        raise ValueError(f"업로드된 ECG JSON을 읽지 못했습니다: {str(error)}")

    if not isinstance(payload, dict):
        raise ValueError("업로드된 ECG JSON 데이터가 올바르지 않습니다.")

    return payload


def _to_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def _to_float(value, default=None):
    try:
        number = float(value)
    except Exception:
        return default

    if number != number or number in (float("inf"), float("-inf")):
        return default

    return number


def sanitize_raw_ecg_payload(payload: dict) -> tuple[dict, dict]:
    if not isinstance(payload, dict):
        raise ValueError("ECG JSON payload 형식이 올바르지 않습니다.")

    if "ecg_mv" not in payload:
        raise ValueError("ECG 데이터 ecg_mv 항목이 없습니다.")

    ecg_values = payload.get("ecg_mv")

    if not isinstance(ecg_values, list):
        raise ValueError("ecg_mv는 배열이어야 합니다.")

    cleaned_samples = []

    for index, value in enumerate(ecg_values):
        number = _to_float(value, default=None)

        if number is None:
            raise ValueError(f"ecg_mv[{index}] 값이 숫자가 아닙니다.")

        cleaned_samples.append(number)

    if len(cleaned_samples) < 2:
        raise ValueError("ecg_mv 길이가 너무 짧습니다.")

    payload = dict(payload)
    payload["ecg_mv"] = cleaned_samples

    corrected_fields = []
    warnings = []
    actual_sample_count = len(cleaned_samples)

    sampling_rate = _to_int(payload.get("sampling_rate"), 500)
    if sampling_rate <= 0:
        sampling_rate = 500
        corrected_fields.append("sampling_rate")
        warnings.append("invalid_sampling_rate")
    payload["sampling_rate"] = sampling_rate

    sample_count = _to_int(payload.get("sample_count"), actual_sample_count)
    if sample_count != actual_sample_count:
        warnings.append("sample_count_mismatch")
        corrected_fields.append("sample_count")
    payload["sample_count"] = actual_sample_count

    valid_sample_count = _to_int(payload.get("valid_sample_count"), actual_sample_count)
    if valid_sample_count != actual_sample_count:
        warnings.append("valid_sample_count_mismatch")
        corrected_fields.append("valid_sample_count")
    payload["valid_sample_count"] = actual_sample_count

    total_sample_count = _to_int(payload.get("total_sample_count"), actual_sample_count)
    if total_sample_count < actual_sample_count:
        total_sample_count = actual_sample_count
        warnings.append("total_sample_count_lt_sample_count")
        corrected_fields.append("total_sample_count")
    payload["total_sample_count"] = total_sample_count

    expected_sample_count = _to_int(payload.get("expected_sample_count"), 15000)
    if expected_sample_count <= 0:
        expected_sample_count = max(actual_sample_count, 1)
        corrected_fields.append("expected_sample_count")
        warnings.append("invalid_expected_sample_count")
    payload["expected_sample_count"] = expected_sample_count

    sample_index = payload.get("sample_index")
    if sample_index is None:
        payload["sample_index"] = list(range(actual_sample_count))
        corrected_fields.append("sample_index")
        warnings.append("sample_index_missing")
    elif not isinstance(sample_index, list) or len(sample_index) != actual_sample_count:
        payload["sample_index"] = list(range(actual_sample_count))
        corrected_fields.append("sample_index")
        warnings.append("sample_index_length_mismatch")

    duration_seconds = actual_sample_count / float(sampling_rate)
    payload["duration_seconds"] = round(duration_seconds, 6)
    payload["duration_ms"] = int(round(duration_seconds * 1000))

    collection_completeness = actual_sample_count / float(expected_sample_count)
    payload["collection_completeness"] = min(1.0, max(0.0, collection_completeness))
    payload["missing_sample_estimate"] = max(0, expected_sample_count - actual_sample_count)

    sanity_check = {
        "status": "corrected" if corrected_fields else "ok",
        "warnings": sorted(set(warnings)),
        "corrected_fields": corrected_fields,
        "actual_sample_count": actual_sample_count,
        "actual_duration_seconds": round(duration_seconds, 6),
    }

    payload["sanity_check"] = sanity_check
    return payload, sanity_check


# subject_id / session_id 유틸
def normalize_subject_id(value):
    subject_id = str(value or "").strip().upper().replace(" ", "")

    if re.fullmatch(r"S\d{3}", subject_id):
        return subject_id

    return ""


def normalize_session_id(value):
    session_id = str(value or "").strip().upper().replace(" ", "")

    if re.fullmatch(r"T\d{2}", session_id):
        return session_id

    if re.fullmatch(r"\d+", session_id):
        return f"T{session_id.zfill(2)}"

    return ""


def extract_subject_id_from_filename(filename):
    match = re.match(r"^(S\d{3})_", str(filename or ""), re.IGNORECASE)

    if not match:
        return ""

    return match.group(1).upper()


def extract_session_id_from_filename(filename):
    match = re.match(r"^S\d{3}_(T\d{2})_", str(filename or ""), re.IGNORECASE)

    if not match:
        return ""

    return match.group(1).upper()


def is_unassigned_subject(value):
    text = str(value or "").strip().upper()

    return text in {
        "",
        "-",
        "USER-01",
        "USER-UNKNOWN",
        "UNASSIGNED",
        "UNKNOWN",
        "ECG 데이터",
        "ECG DATA",
    }


def resolve_subject_id(
    payload: dict,
    filename: str = "",
    form_subject_id: str = "",
    fallback_subject_id: str = "",
):
    subject_id = normalize_subject_id(form_subject_id)

    if subject_id:
        return subject_id

    subject_id = normalize_subject_id(payload.get("subject_id"))

    if subject_id:
        return subject_id

    subject_id = normalize_subject_id(payload.get("user_id"))

    if subject_id:
        return subject_id

    subject_id = normalize_subject_id(payload.get("patient_id"))

    if subject_id:
        return subject_id

    subject_id = extract_subject_id_from_filename(filename)

    if subject_id:
        return subject_id

    subject_id = normalize_subject_id(fallback_subject_id)

    if subject_id:
        return subject_id

    raise ValueError(
        "subject_id를 확인할 수 없습니다. "
        "파일명을 S001_T01_... 형식으로 바꾸거나 JSON 내부 subject_id를 S001 형식으로 수정하세요."
    )


def resolve_session_id(
    payload: dict,
    filename: str = "",
    form_session_id: str = "",
    fallback_session_id: str = "",
):
    session_id = normalize_session_id(form_session_id)

    if session_id:
        return session_id

    session_id = normalize_session_id(payload.get("session_id"))

    if session_id:
        return session_id

    session_id = normalize_session_id(payload.get("trial_id"))

    if session_id:
        return session_id

    session_id = normalize_session_id(payload.get("record_id"))

    if session_id:
        return session_id

    session_id = extract_session_id_from_filename(filename)

    if session_id:
        return session_id

    return normalize_session_id(fallback_session_id)


def resolve_subject_name(payload: dict, subject_id: str, form_subject_name: str = ""):
    candidates = [
        form_subject_name,
        payload.get("subject_name"),
        payload.get("real_name"),
        payload.get("display_name"),
        payload.get("name"),
    ]

    blocked_values = {
        "",
        "USER-01",
        "USER-UNKNOWN",
        "UNASSIGNED",
        "ECG 데이터",
        "ECG DATA",
        "ECG_JSON_SAMPLE",
        "RAWSAMPLE",
    }

    for candidate in candidates:
        value = str(candidate or "").strip()

        if value and value.upper() not in blocked_values:
            return value

    return subject_id


def parse_measured_datetime(payload: dict):
    measured_at = str(payload.get("measured_at") or "").strip()

    for fmt in [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
    ]:
        try:
            return datetime.strptime(measured_at, fmt)
        except ValueError:
            pass

    return datetime.now()


def normalize_device_tag(value):
    text = str(value or "").lower()

    if "watch 6" in text or "watch6" in text:
        return "watch6"

    if "watch" in text:
        return "watch"

    return "device"


def normalize_lead_tag(value):
    text = str(value or "").lower()

    if "lead i" in text or "leadi" in text:
        return "leadI"

    return "ecg"


def build_labeled_ecg_filename(payload: dict, subject_id: str, session_id: str):
    measured_dt = parse_measured_datetime(payload)
    date_tag = measured_dt.strftime("%Y%m%d")
    time_tag = measured_dt.strftime("%H%M%S")

    device_tag = normalize_device_tag(payload.get("device", "Samsung Galaxy Watch 6"))
    lead_tag = normalize_lead_tag(payload.get("lead", "Lead I ECG"))

    try:
        sampling_rate = int(float(payload.get("sampling_rate", 500)))
    except Exception:
        sampling_rate = 500

    clean_subject_id = normalize_subject_id(subject_id) or "S000"
    clean_session_id = normalize_session_id(session_id) or "T00"

    return (
        f"{clean_subject_id}_{clean_session_id}_{date_tag}_{time_tag}_"
        f"{device_tag}_{lead_tag}_{sampling_rate}Hz.json"
    )


def get_next_session_id_for_subject(subject_id: str):
    clean_subject_id = normalize_subject_id(subject_id)

    if not clean_subject_id:
        return "T00"

    try:
        logs = get_auth_logs_by_user(clean_subject_id, limit=500)
        next_index = len(logs) + 1
    except Exception:
        next_index = 1

    return f"T{next_index:02d}"


def apply_subject_metadata_to_payload(
    payload: dict,
    filename: str = "",
    form_subject_id: str = "",
    form_session_id: str = "",
    form_subject_name: str = "",
    fallback_subject_id: str = "",
    fallback_session_id: str = "",
    force_labeled_filename: bool = False,
):
    subject_id = resolve_subject_id(
        payload=payload,
        filename=filename,
        form_subject_id=form_subject_id,
        fallback_subject_id=fallback_subject_id,
    )
    session_id = resolve_session_id(
        payload=payload,
        filename=filename,
        form_session_id=form_session_id,
        fallback_session_id=fallback_session_id,
    )
    subject_name = resolve_subject_name(
        payload=payload,
        subject_id=subject_id,
        form_subject_name=form_subject_name,
    )

    if not session_id:
        session_id = "T00"

    if force_labeled_filename:
        final_filename = build_labeled_ecg_filename(
            payload=payload,
            subject_id=subject_id,
            session_id=session_id,
        )
    else:
        final_filename = (
            filename
            or payload.get("filename")
            or build_labeled_ecg_filename(
                payload=payload,
                subject_id=subject_id,
                session_id=session_id,
            )
        )

    payload["subject_id"] = subject_id
    payload["subject_name"] = subject_name
    payload["name"] = subject_id
    payload["session_id"] = session_id
    payload["filename"] = final_filename

    return payload


# 측정 대상자 설정
def normalize_subject_value(value):
    if value is None:
        return ""

    return str(value).strip()


def validate_subject_payload(payload: dict):
    subject_id = normalize_subject_id(payload.get("subject_id"))
    birth_date = normalize_subject_value(payload.get("birth_date"))
    name = normalize_subject_value(payload.get("name")) or subject_id

    if not subject_id:
        raise ValueError("사용자 ID는 S001 형식으로 입력하세요.")

    if not birth_date:
        raise ValueError("생년월일을 입력하세요.")

    birth_digits = "".join(ch for ch in birth_date if ch.isdigit())

    if len(birth_digits) != 8:
        raise ValueError("생년월일은 YYYYMMDD 형식으로 입력하세요.")

    return {
        "subject_id": subject_id,
        "name": name,
        "birth_date": birth_digits,
    }


def apply_current_subject_to_payload(payload: dict):
    current_subject = session_state.get("current_subject")

    if not current_subject:
        raise ValueError(
            "현재 측정 대상자가 설정되지 않았습니다. 웹에서 사용자 정보를 먼저 설정하세요."
        )

    subject_id = current_subject["subject_id"]
    subject_name = current_subject["name"]
    birth_date = current_subject["birth_date"]
    session_id = get_next_session_id_for_subject(subject_id)

    payload["subject_id"] = subject_id
    payload["subject_name"] = subject_name
    payload["name"] = subject_id
    payload["birth_date"] = birth_date
    payload["session_id"] = session_id
    payload["filename"] = build_labeled_ecg_filename(
        payload=payload,
        subject_id=subject_id,
        session_id=session_id,
    )

    return payload


# 사용자 정보
def get_report_value(report: dict, key: str, default="-"):
    value = report.get(key)

    if value is None or value == "":
        return default

    return value


def is_placeholder_name(value):
    text = str(value or "").strip()
    upper_text = text.upper()

    return upper_text in {
        "",
        "-",
        "USER-01",
        "USER-UNKNOWN",
        "UNASSIGNED",
        "ECG 데이터",
        "ECG DATA",
        "ECG_JSON_SAMPLE",
        "RAW_SAMPLE",
        "RAWSAMPLE",
    }


def resolve_display_name_from_report(report: dict, subject_id: str):
    candidates = [
        report.get("subject_name"),
        report.get("display_name"),
        report.get("real_name"),
        report.get("name"),
    ]

    for candidate in candidates:
        value = str(candidate or "").strip()

        if value and not is_placeholder_name(value):
            return value

    return subject_id or "ECG 데이터"


def resolve_user_from_report(report: dict):
    subject_id = get_report_value(report, "subject_id", None)
    name = resolve_display_name_from_report(report, subject_id)
    birth_date = get_report_value(report, "birth_date", "-")

    user_key_source = subject_id or name or "ECG 데이터"
    user_key = make_user_key(user_key_source, birth_date)
    existing_user = get_user_by_key(user_key)
    mode = "register" if existing_user is None else "verify"

    return {
        "subject_id": subject_id or user_key_source,
        "name": name,
        "birth_date": birth_date,
        "user_key": user_key,
        "existing_user": existing_user,
        "mode": mode,
    }


# 세션 상태
def update_session_state(user_key: str, mode: str, waveform_json_path: Path = None):
    session_state["last_user_key"] = user_key
    session_state["last_mode"] = mode

    if waveform_json_path:
        session_state["last_waveform"] = str(waveform_json_path)


def update_latest_result(response_tuple):
    try:
        response_obj = (
            response_tuple[0] if isinstance(response_tuple, tuple) else response_tuple
        )
        payload = response_obj.get_json(silent=True)

        if not payload or not payload.get("success"):
            return

        session_state["latest_result"] = payload
        session_state["latest_received_at"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    except Exception as error:
        print(f"[LATEST RESULT UPDATE ERROR] {error}")


def reset_session_state():
    session_state["last_user_key"] = None
    session_state["last_mode"] = None
    session_state["last_waveform"] = None
    session_state["latest_result"] = None
    session_state["latest_received_at"] = None
    session_state["current_subject"] = None


# ECG JSON 처리 컨텍스트 생성
def build_raw_ecg_context_from_saved_file(saved_path: Path, saved_filename: str):
    file_hash = calculate_file_sha256(saved_path)
    stem = saved_path.stem

    display_waveform_json_path = RAW_EXTRACTED_DIR / f"{stem}_display_waveform.json"
    model_waveform_json_path = RAW_EXTRACTED_DIR / f"{stem}_model_waveform.json"

    saved_payload = load_json_safely(saved_path) or {}

    raw_result = build_raw_waveform_response(
        json_path=saved_path,
        display_waveform_json_path=display_waveform_json_path,
        model_waveform_json_path=model_waveform_json_path,
    )

    sanity_check = saved_payload.get("sanity_check", {}) if isinstance(saved_payload, dict) else {}

    report = raw_result["report"]
    if sanity_check:
        report["sanity_check"] = sanity_check
        raw_result["raw_summary"]["sanity_check"] = sanity_check
    user_info = resolve_user_from_report(report)

    mode = user_info["mode"]
    user_key = user_info["user_key"]

    report["subject_id"] = user_info["subject_id"]
    report["subject_name"] = user_info["name"]
    report["name"] = user_info["name"]
    report["filename"] = saved_filename
    report["role"] = mode
    report["user_key"] = user_key

    model_waveform = raw_result["model_waveform"]
    sampling_rate = model_waveform.get("sampling_rate", 500)

    try:
        hrv_result = calculate_hrv_metrics(
            waveform_values=model_waveform.get("amplitude", []),
            sampling_rate=sampling_rate,
            pdf_heart_rate=report.get("heart_rate"),
        )
    except Exception as error:
        print(f"[ECG HRV ERROR] {error}")
        hrv_result = {
            "available": False,
            "quality_status": "unavailable",
            "quality_label": "분석 불가",
            "trust_level": "분석 불가",
            "quality_message": "ECG 신호 품질 분석을 수행하지 못했습니다.",
            "heart_rate_bpm": None,
            "ecg_estimated_heart_rate_bpm": None,
            "pdf_heart_rate_bpm": report.get("heart_rate"),
            "valid_rr_count": None,
            "rr_stability_score": None,
            "sdnn_ms": None,
            "rmssd_ms": None,
        }

    context = {
        "mode": mode,
        "report": report,
        "subject_id": user_info["subject_id"],
        "name": user_info["name"],
        "birth_date": user_info["birth_date"],
        "user_key": user_key,
        "existing_user": user_info["existing_user"],
        "saved_filename": saved_filename,
        "saved_path": saved_path,
        "file_hash": file_hash,
        "waveform": raw_result["waveform"],
        "model_waveform": raw_result["model_waveform"],
        "hrv": hrv_result,
        "display_waveform_json_path": display_waveform_json_path,
        "model_waveform_json_path": model_waveform_json_path,
        "raw_summary": raw_result["raw_summary"],
        "sanity_check": sanity_check,
    }

    update_session_state(
        user_key=user_key,
        mode=mode,
        waveform_json_path=display_waveform_json_path,
    )

    return context


def build_raw_ecg_context_from_upload(file):
    original_filename = request.form.get("filename") or file.filename
    form_subject_id = request.form.get("subject_id", "")
    form_session_id = request.form.get("session_id", "")
    form_subject_name = request.form.get("subject_name", "")

    payload = load_uploaded_json_file(file)
    payload, _ = sanitize_raw_ecg_payload(payload)

    current_subject = session_state.get("current_subject")
    fallback_subject_id = current_subject["subject_id"] if current_subject else ""
    fallback_session_id = (
        get_next_session_id_for_subject(fallback_subject_id)
        if fallback_subject_id
        else ""
    )

    payload = apply_subject_metadata_to_payload(
        payload=payload,
        filename=original_filename,
        form_subject_id=form_subject_id,
        form_session_id=form_session_id,
        form_subject_name=form_subject_name,
        fallback_subject_id=fallback_subject_id,
        fallback_session_id=fallback_session_id,
        force_labeled_filename=False,
    )

    subject_id = payload.get("subject_id") or "unknown"

    saved_path, saved_filename = save_payload_to_raw_dir(
        payload=payload,
        original_filename=payload.get("filename") or original_filename,
        role="manual",
        user_key=subject_id,
    )

    return build_raw_ecg_context_from_saved_file(
        saved_path=saved_path,
        saved_filename=saved_filename,
    )


def save_received_json_to_raw_dir(payload: dict) -> tuple[Path, str]:
    original_filename = (
        payload.get("filename")
        or request.headers.get("X-ECG-Filename")
        or build_labeled_ecg_filename(
            payload=payload,
            subject_id=payload.get("subject_id", "S000"),
            session_id=payload.get("session_id", "T00"),
        )
    )

    subject_id = payload.get("subject_id") or "unknown"

    return save_payload_to_raw_dir(
        payload=payload,
        original_filename=original_filename,
        role="watch",
        user_key=subject_id,
    )


def build_raw_ecg_context_from_received_json(payload: dict):
    saved_path, saved_filename = save_received_json_to_raw_dir(payload)

    return build_raw_ecg_context_from_saved_file(
        saved_path=saved_path,
        saved_filename=saved_filename,
    )


# 등록 처리
def register_raw_ecg_template(context: dict):
    validate_registration_quality(context["raw_summary"])

    template_result = register_template_from_waveform_json(
        waveform_json_path=context["model_waveform_json_path"],
        user_key=context["user_key"],
        user_id=context["user_key"],
        name=context["name"],
        birth_date=context["birth_date"],
    )

    user = create_user(
        user_key=context["user_key"],
        subject_id=context["subject_id"],
        name=context["name"],
        birth_date=context["birth_date"],
        template_embedding_path=template_result["template_embedding_file"],
        template_metadata_path=template_result["template_metadata_file"],
        template_waveform_path=context["display_waveform_json_path"],
        template_pdf_hash=context["file_hash"],
    )

    create_auth_log(
        user_key=context["user_key"],
        subject_id=context["subject_id"],
        name=context["name"],
        birth_date=context["birth_date"],
        filename=context["saved_filename"],
        measured_at=context["report"].get("measured_at"),
        heart_rate=context["report"].get("heart_rate"),
        mode="register",
        cosine_similarity=None,
        matching_score=None,
        threshold=template_result["threshold"],
        decision="Registered",
    )

    return template_result, user


def build_register_response(context: dict, template_result: dict, user: dict):
    return success_response(
        {
            "mode": "register",
            "message": "ECG template registered.",
            "report": context["report"],
            "waveform": context["waveform"],
            "template_waveform": context["waveform"],
            "hrv": context["hrv"],
            "raw_summary": context["raw_summary"],
            "sanity_check": context.get("sanity_check", {}),
            "template": {
                "subject_id": context["subject_id"],
                "user_key": context["user_key"],
                "user_id": context["user_key"],
                "name": context["name"],
                "birth_date": context["birth_date"],
                "model": template_result["model"],
                "embedding_dim": template_result["embedding_dim"],
                "embedding_count": template_result.get("embedding_count"),
                "threshold": template_result["threshold"],
                "status": template_result["status"],
                "fingerprint_mode": template_result.get("fingerprint_mode"),
                "segmentation": template_result.get("segmentation", {}),
                "collection_quality": template_result.get("collection_quality", {}),
                "waveform_quality": template_result.get("waveform_quality", {}),
            },
            "user": user,
            "artifacts": {
                "saved_json": str(context["saved_path"]),
                "display_waveform_json": str(context["display_waveform_json_path"]),
                "model_waveform_json": str(context["model_waveform_json_path"]),
                "template_embedding_file": template_result["template_embedding_file"],
                "template_metadata_file": template_result["template_metadata_file"],
            },
        }
    )


def handle_raw_register_flow(context: dict):
    try:
        template_result, user = register_raw_ecg_template(context)
    except ValueError as error:
        print(f"[ECG REGISTER VALIDATION ERROR] {error}")
        return error_response(
            f"템플릿 등록 조건을 만족하지 못했습니다: {str(error)}", 400
        )
    except Exception as error:
        print(f"[ECG REGISTER ERROR] {error}")
        return error_response(f"템플릿 등록에 실패했습니다: {str(error)}", 500)

    return build_register_response(context, template_result, user)


# 인증 처리
def verify_raw_ecg_template(context: dict):
    existing_user = context["existing_user"]

    inference_result = verify_waveform_json_against_template(
        waveform_json_path=context["model_waveform_json_path"],
        user_key=context["user_key"],
        template_embedding_path=existing_user["template_embedding_path"],
        user_id=context["user_key"],
    )

    update_user_last_verified(context["user_key"])

    auth_log = create_auth_log(
        user_key=context["user_key"],
        subject_id=context["subject_id"],
        name=context["name"],
        birth_date=context["birth_date"],
        filename=context["saved_filename"],
        measured_at=context["report"].get("measured_at"),
        heart_rate=context["report"].get("heart_rate"),
        mode="verify",
        cosine_similarity=inference_result["cosine_similarity"],
        matching_score=inference_result["matching_score"],
        threshold=inference_result["threshold"],
        decision=inference_result["decision"],
    )

    return inference_result, auth_log


def build_verify_response(context: dict, inference_result: dict, auth_log: dict):
    existing_user = context["existing_user"]
    template_waveform_path = existing_user.get("template_waveform_path")
    template_waveform = load_json_safely(template_waveform_path)

    return success_response(
        {
            "mode": "verify",
            "message": "ECG verification completed.",
            "report": context["report"],
            "waveform": context["waveform"],
            "template_waveform": template_waveform,
            "hrv": context["hrv"],
            "raw_summary": context["raw_summary"],
            "sanity_check": context.get("sanity_check", {}),
            "inference": {
                "subject_id": context["subject_id"],
                "user_key": context["user_key"],
                "user_id": context["user_key"],
                "name": context["name"],
                "birth_date": context["birth_date"],
                "model": inference_result["model"],
                "embedding_dim": inference_result["embedding_dim"],
                "embedding_count": inference_result.get("embedding_count"),
                "template_embedding_count": inference_result.get(
                    "template_embedding_count"
                ),
                "cosine_similarity": inference_result["cosine_similarity"],
                "matching_score": inference_result["matching_score"],
                "threshold": inference_result["threshold"],
                "decision": inference_result["decision"],
                "decision_reason": inference_result.get("decision_reason"),
                "quality_pass": inference_result.get("quality_pass"),
                "fingerprint_mode": inference_result.get("fingerprint_mode"),
                "segmentation": inference_result.get("segmentation", {}),
                "similarity_detail": inference_result.get("similarity_detail", {}),
                "collection_quality": inference_result.get("collection_quality", {}),
                "waveform_quality": inference_result.get("waveform_quality", {}),
            },
            "auth_log": auth_log,
            "artifacts": {
                "saved_json": str(context["saved_path"]),
                "display_waveform_json": str(context["display_waveform_json_path"]),
                "model_waveform_json": str(context["model_waveform_json_path"]),
                "template_embedding_file": inference_result["template_embedding_file"],
                "template_waveform_file": template_waveform_path,
            },
        }
    )


def handle_raw_verify_flow(context: dict):
    try:
        inference_result, auth_log = verify_raw_ecg_template(context)
    except Exception as error:
        print(f"[ECG VERIFY ERROR] {error}")
        return error_response(f"ECG 인증에 실패했습니다: {str(error)}", 500)

    return build_verify_response(context, inference_result, auth_log)


# 공통 ECG 처리 실행
def process_raw_ecg_context(context: dict):
    if context["mode"] == "register":
        response = handle_raw_register_flow(context)
    else:
        response = handle_raw_verify_flow(context)

    update_latest_result(response)

    return response


# 기본 페이지
@app.route("/")
def index():
    return render_template("index.html")


# ECG JSON 수동 업로드 처리
@app.route("/api/raw-ecg/process", methods=["POST"])
def process_raw_ecg():
    file, error = validate_uploaded_raw_json()

    if error:
        return error

    try:
        context = build_raw_ecg_context_from_upload(file)
    except ValueError as error:
        print(f"[ECG JSON VALIDATION ERROR] {error}")
        return error_response(str(error), 400)
    except Exception as error:
        print(f"[ECG JSON PROCESS ERROR] {error}")
        return error_response(
            f"ECG JSON 처리 중 오류가 발생했습니다: {str(error)}",
            500,
        )

    return process_raw_ecg_context(context)


# 측정 대상자 설정
@app.route("/api/subject/set", methods=["POST"])
def set_current_subject():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return error_response("사용자 정보가 올바르지 않습니다.", 400)

    try:
        subject = validate_subject_payload(payload)
    except Exception as error:
        return error_response(str(error), 400)

    session_state["current_subject"] = subject

    return success_response(
        {
            "message": "측정 대상자가 설정되었습니다.",
            "current_subject": subject,
        }
    )


# 현재 측정 대상자 조회
@app.route("/api/subject/current", methods=["GET"])
def get_current_subject():
    current_subject = session_state.get("current_subject")

    return success_response(
        {
            "has_subject": current_subject is not None,
            "current_subject": current_subject,
        }
    )


# 워치 ECG JSON 자동 수신 처리
@app.route("/api/ecg-json/receive", methods=["POST"])
def receive_ecg_json_from_watch():
    try:
        if request.files and "file" in request.files:
            file, error = validate_uploaded_raw_json()

            if error:
                return error

            payload = load_uploaded_json_file(file)
            payload, _ = sanitize_raw_ecg_payload(payload)

            try:
                payload = apply_current_subject_to_payload(payload)
            except Exception as error:
                return error_response(str(error), 409)

            context = build_raw_ecg_context_from_received_json(payload)
            return process_raw_ecg_context(context)

        payload = request.get_json(silent=True)

        if not isinstance(payload, dict):
            return error_response("전송된 ECG JSON 데이터가 올바르지 않습니다.", 400)

        payload, _ = sanitize_raw_ecg_payload(payload)

        try:
            payload = apply_current_subject_to_payload(payload)
        except Exception as error:
            return error_response(str(error), 409)

        context = build_raw_ecg_context_from_received_json(payload)

    except ValueError as error:
        print(f"[WATCH ECG VALIDATION ERROR] {error}")
        return error_response(str(error), 400)
    except Exception as error:
        print(f"[WATCH ECG RECEIVE ERROR] {error}")
        return error_response(
            f"워치 ECG JSON 수신 처리 중 오류가 발생했습니다: {str(error)}",
            500,
        )

    return process_raw_ecg_context(context)


# 최신 측정 결과 조회
@app.route("/api/latest-result", methods=["GET"])
def get_latest_result():
    latest_result = session_state.get("latest_result")

    if latest_result is None:
        return success_response(
            {
                "has_result": False,
                "message": "아직 수신된 ECG 측정 결과가 없습니다.",
                "latest_received_at": None,
                "result": None,
            }
        )

    return success_response(
        {
            "has_result": True,
            "latest_received_at": session_state.get("latest_received_at"),
            "result": latest_result,
        }
    )


# 인증 기록 조회
@app.route("/api/auth-logs", methods=["GET"])
def get_auth_logs():
    user_key = session_state.get("last_user_key")

    if not user_key:
        return error_response(
            "현재 선택된 사용자가 없습니다. ECG 데이터를 먼저 수신하세요.",
            400,
        )

    user = get_user_by_key(user_key)

    if user is None:
        return error_response("등록된 사용자 정보를 찾을 수 없습니다.", 404)

    logs = get_auth_logs_by_user(user_key, limit=50)

    return success_response(
        {
            "user": user,
            "logs": logs,
        }
    )


# 초기화
@app.route("/api/reset", methods=["POST"])
def reset():
    reset_session_state()

    try:
        deleted_files = reset_template_embedding()
        reset_database()
    except Exception as error:
        print(f"[RESET ERROR] {error}")
        deleted_files = []

    return success_response(
        {
            "message": "Session and database reset completed.",
            "deleted_files": deleted_files,
        }
    )


# 서버 실행
if __name__ == "__main__":
    app.run(
        debug=True,
        use_reloader=False,
        host="0.0.0.0",
        port=5000,
    )
