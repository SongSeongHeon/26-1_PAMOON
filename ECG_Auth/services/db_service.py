# 기본 설정
import re
import sqlite3
import unicodedata
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "app.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# DB 연결
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# 컬럼 존재 여부 확인
def column_exists(conn, table_name, column_name):
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row["name"] == column_name for row in rows)


# subject_id 정규화
def normalize_subject_id(value):
    text = str(value or "").strip().upper().replace(" ", "")

    if re.fullmatch(r"S\d{3}", text):
        return text

    return ""


# DB 초기화
def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_key TEXT PRIMARY KEY,
                subject_id TEXT,
                name TEXT NOT NULL,
                birth_date TEXT NOT NULL,
                template_embedding_path TEXT NOT NULL,
                template_metadata_path TEXT,
                template_waveform_path TEXT,
                template_pdf_hash TEXT,
                registered_at TEXT NOT NULL,
                last_verified_at TEXT
            )
            """)

        # 기존 app.db를 유지한 상태에서 새 컬럼만 추가
        if not column_exists(conn, "users", "subject_id"):
            conn.execute("""
                ALTER TABLE users
                ADD COLUMN subject_id TEXT
                """)

        if not column_exists(conn, "users", "template_metadata_path"):
            conn.execute("""
                ALTER TABLE users
                ADD COLUMN template_metadata_path TEXT
                """)

        if not column_exists(conn, "users", "template_waveform_path"):
            conn.execute("""
                ALTER TABLE users
                ADD COLUMN template_waveform_path TEXT
                """)

        if not column_exists(conn, "users", "template_pdf_hash"):
            conn.execute("""
                ALTER TABLE users
                ADD COLUMN template_pdf_hash TEXT
                """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS auth_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_key TEXT NOT NULL,
                subject_id TEXT,
                name TEXT,
                birth_date TEXT,
                filename TEXT,
                measured_at TEXT,
                heart_rate TEXT,
                mode TEXT NOT NULL,
                cosine_similarity REAL,
                matching_score REAL,
                threshold REAL,
                decision TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_key) REFERENCES users(user_key)
            )
            """)

        if not column_exists(conn, "auth_logs", "subject_id"):
            conn.execute("""
                ALTER TABLE auth_logs
                ADD COLUMN subject_id TEXT
                """)

        conn.commit()


# 문자열 정규화
def normalize_text(value):
    if value is None:
        return ""

    text = str(value).strip()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", "", text)

    return text


# 생년월일 정규화
def normalize_birth_date(value):
    if value is None:
        return ""

    text = str(value).strip()
    text = unicodedata.normalize("NFKC", text)

    # 예: 2001년 6월 11일
    match = re.search(r"(\d{4})\D+(\d{1,2})\D+(\d{1,2})", text)
    if match:
        year, month, day = match.groups()
        return f"{int(year):04d}{int(month):02d}{int(day):02d}"

    # 예: 2001-06-11, 2001.06.11, 2001/06/11
    match = re.search(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})", text)
    if match:
        year, month, day = match.groups()
        return f"{int(year):04d}{int(month):02d}{int(day):02d}"

    # 예: 20010611
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 8:
        return digits[:8]

    return digits


# 사용자 키 생성
def make_user_key(name, birth_date):
    clean_name = normalize_text(name)
    subject_id = normalize_subject_id(clean_name)

    # S001, S002처럼 라벨링된 데이터는 subject_id 자체를 user_key로 사용
    if subject_id:
        return subject_id

    clean_birth = normalize_birth_date(birth_date)

    if not clean_name:
        clean_name = "unknown"

    if not clean_birth:
        clean_birth = "unknownbirth"

    return f"{clean_name}_{clean_birth}"


# 사용자 조회
def get_user_by_key(user_key):
    init_db()

    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM users
            WHERE user_key = ?
            """,
            (user_key,),
        ).fetchone()

    if row is None:
        return None

    return dict(row)


# subject_id로 사용자 조회
def get_user_by_subject_id(subject_id):
    init_db()

    clean_subject_id = normalize_subject_id(subject_id)

    if not clean_subject_id:
        return None

    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM users
            WHERE subject_id = ?
               OR user_key = ?
            """,
            (clean_subject_id, clean_subject_id),
        ).fetchone()

    if row is None:
        return None

    return dict(row)


# 템플릿 PDF 해시로 사용자 조회
def get_user_by_template_pdf_hash(template_pdf_hash):
    init_db()

    if not template_pdf_hash:
        return None

    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM users
            WHERE template_pdf_hash = ?
            """,
            (template_pdf_hash,),
        ).fetchone()

    if row is None:
        return None

    return dict(row)


# 사용자 등록
def create_user(
    user_key,
    name,
    birth_date,
    template_embedding_path,
    template_metadata_path=None,
    template_waveform_path=None,
    template_pdf_hash=None,
    subject_id=None,
):
    init_db()

    now = datetime.now().isoformat(timespec="seconds")
    clean_subject_id = normalize_subject_id(subject_id) or normalize_subject_id(name)
    clean_name = normalize_text(name) or clean_subject_id or "ECG 데이터"
    clean_birth_date = normalize_birth_date(birth_date) or "unknownbirth"

    if clean_subject_id:
        user_key = clean_subject_id

    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO users (
                user_key,
                subject_id,
                name,
                birth_date,
                template_embedding_path,
                template_metadata_path,
                template_waveform_path,
                registered_at,
                last_verified_at,
                template_pdf_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_key,
                clean_subject_id,
                clean_name,
                clean_birth_date,
                str(template_embedding_path),
                str(template_metadata_path) if template_metadata_path else None,
                str(template_waveform_path) if template_waveform_path else None,
                now,
                None,
                template_pdf_hash,
            ),
        )

        conn.commit()

    return get_user_by_key(user_key)


# 사용자 인증 시간 갱신
def update_user_last_verified(user_key):
    init_db()

    now = datetime.now().isoformat(timespec="seconds")

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE users
            SET last_verified_at = ?
            WHERE user_key = ?
            """,
            (now, user_key),
        )

        conn.commit()


# 인증 로그 저장
def create_auth_log(
    user_key,
    name=None,
    birth_date=None,
    filename=None,
    measured_at=None,
    heart_rate=None,
    mode="verify",
    cosine_similarity=None,
    matching_score=None,
    threshold=None,
    decision=None,
    subject_id=None,
):
    init_db()

    now = datetime.now().isoformat(timespec="seconds")
    clean_subject_id = normalize_subject_id(subject_id) or normalize_subject_id(name)
    clean_name = normalize_text(name) or clean_subject_id or None
    clean_birth_date = normalize_birth_date(birth_date) or None

    if clean_subject_id:
        user_key = clean_subject_id

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO auth_logs (
                user_key,
                subject_id,
                name,
                birth_date,
                filename,
                measured_at,
                heart_rate,
                mode,
                cosine_similarity,
                matching_score,
                threshold,
                decision,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_key,
                clean_subject_id,
                clean_name,
                clean_birth_date,
                filename,
                measured_at,
                str(heart_rate) if heart_rate is not None else None,
                mode,
                cosine_similarity,
                matching_score,
                threshold,
                decision,
                now,
            ),
        )

        conn.commit()
        log_id = cursor.lastrowid

    return get_auth_log_by_id(log_id)


# 인증 로그 단건 조회
def get_auth_log_by_id(log_id):
    init_db()

    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM auth_logs
            WHERE id = ?
            """,
            (log_id,),
        ).fetchone()

    if row is None:
        return None

    return dict(row)


# 사용자별 인증 로그 조회
def get_auth_logs_by_user(user_key, limit=30):
    init_db()

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM auth_logs
            WHERE user_key = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_key, limit),
        ).fetchall()

    return [dict(row) for row in rows]


# subject_id별 인증 로그 조회
def get_auth_logs_by_subject_id(subject_id, limit=30):
    init_db()

    clean_subject_id = normalize_subject_id(subject_id)

    if not clean_subject_id:
        return []

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM auth_logs
            WHERE subject_id = ?
               OR user_key = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (clean_subject_id, clean_subject_id, limit),
        ).fetchall()

    return [dict(row) for row in rows]


# 전체 사용자 조회
def get_all_users():
    init_db()

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT *
            FROM users
            ORDER BY registered_at DESC
            """).fetchall()

    return [dict(row) for row in rows]


# DB 초기화
def reset_database():
    init_db()

    with get_connection() as conn:
        conn.execute("DELETE FROM auth_logs")
        conn.execute("DELETE FROM users")
        conn.commit()

    return {
        "success": True,
        "message": "Database reset completed.",
    }
