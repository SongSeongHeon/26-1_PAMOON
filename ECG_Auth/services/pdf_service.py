import re
from pathlib import Path

import fitz  # PyMuPDF


def extract_pdf_text(pdf_path: str | Path) -> str:
    """
    Extract text from a Samsung Health Monitor ECG PDF.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    text_chunks = []

    with fitz.open(pdf_path) as doc:
        for page in doc:
            text_chunks.append(page.get_text("text"))

    return "\n".join(text_chunks)


def parse_samsung_ecg_report(pdf_path: str | Path) -> dict:
    """
    Parse metadata from Samsung Health Monitor ECG PDF.
    Expected fields:
    - name
    - birth_date
    - rhythm
    - heart_rate
    - measured_at
    - device
    - sampling_rate
    - lead
    """

    text = extract_pdf_text(pdf_path)
    normalized = normalize_text(text)

    name = extract_name(normalized)
    birth_date = extract_birth_date(normalized)
    rhythm = extract_rhythm(normalized)
    heart_rate = extract_heart_rate(normalized)
    measured_at = extract_measured_time(normalized)
    sampling_rate = extract_sampling_rate(normalized)
    device = extract_device(normalized)
    lead = extract_lead(normalized)

    return {
        "name": name or "-",
        "birth_date": birth_date or "-",
        "rhythm": rhythm or "-",
        "measured_at": measured_at or "-",
        "heart_rate": heart_rate,
        "device": device or "-",
        "sampling_rate": sampling_rate or "-",
        "lead": lead or "Lead I-like ECG",
        "raw_text": text,
    }


def normalize_text(text: str) -> str:
    """
    Normalize common spacing and special colon characters.
    """
    if not text:
        return ""

    text = text.replace("∶", ":")
    text = text.replace("：", ":")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)

    return text.strip()


def extract_name(text: str) -> str | None:
    match = re.search(r"이름:\s*([^\n]+?)(?:\s+생년월일:|\n|$)", text)
    if match:
        return match.group(1).strip()
    return None


def extract_birth_date(text: str) -> str | None:
    match = re.search(r"생년월일:\s*([^\n]+)", text)
    if match:
        return match.group(1).strip()
    return None


def extract_rhythm(text: str) -> str | None:
    # Samsung ECG PDF usually contains a rhythm result such as "동리듬".
    candidates = ["동리듬", "심방세동", "판정 불가", "불규칙한 심장 리듬"]

    for candidate in candidates:
        if candidate in text:
            return candidate

    return None


def extract_heart_rate(text: str) -> int | None:
    patterns = [
        r"평균\s*심박수\s*:\s*(\d+)\s*bpm",
        r"심박수\s*:\s*(\d+)\s*bpm",
        r"(\d+)\s*bpm",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return int(match.group(1))

    return None


def extract_measured_time(text: str) -> str | None:
    patterns = [
        r"측정\s*일시\s*:\s*([^\n]+)",
        r"측정일시\s*:\s*([^\n]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            measured = match.group(1).strip()

            # Remove footer text if it is captured together.
            measured = re.split(r"\s*나타난\s*증상", measured)[0].strip()
            measured = re.split(r"\s*25\s*mm/s", measured)[0].strip()

            return measured

    return None


def extract_sampling_rate(text: str) -> str | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*Hz", text, re.IGNORECASE)
    if match:
        value = match.group(1)
        if value.endswith(".000"):
            value = value.replace(".000", "")
        return f"{value} Hz"
    return None


def extract_device(text: str) -> str | None:
    # Example:
    # 25 mm/s, 10 mm/mV, 500.000 Hz, Galaxy Watch7 (9XHE), Wear OS 16 ...
    match = re.search(r"Hz,\s*([^,\n]+Galaxy Watch[^,\n]*)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r"(Galaxy Watch[^\n,]*)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None


def extract_lead(text: str) -> str | None:
    if "Lead I" in text or "Lead I ECG" in text:
        return "Lead I-like ECG"

    if "Lead I ECG와 유사" in text:
        return "Lead I-like ECG"

    return "Lead I-like ECG"