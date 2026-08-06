from pathlib import Path
import json

import fitz  # PyMuPDF
import cv2
import numpy as np


def render_pdf_first_page(
    pdf_path: str | Path,
    output_path: str | Path,
    zoom: float = 2.0,
) -> Path:
    """
    Render the first page of a Samsung ECG PDF as PNG.
    """
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with fitz.open(pdf_path) as doc:
        page = doc[0]
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        pix.save(output_path)

    return output_path


def extract_orange_ecg_waveform(
    image_path: str | Path,
    output_json_path: str | Path | None = None,
) -> dict:
    """
    Extract an ECG waveform from a rendered Samsung ECG report image.

    Current purpose:
    - detect the orange ECG line displayed in the PDF
    - reconstruct a waveform that stays close to the PDF visual waveform
    - apply only minimal correction required for browser rendering

    This is not the original medically calibrated 500 Hz ECG signal.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    height, width = image_bgr.shape[:2]

    # ECG 그래프 영역만 사용해서 제목, 설명, 하단 문구 오검출을 줄임
    y_min = int(height * 0.18)
    y_max = int(height * 0.76)

    roi = image_bgr[y_min:y_max, :]

    mask = make_orange_mask(roi)
    rows = find_ecg_rows(mask)

    if len(rows) == 0:
        raise ValueError("No ECG waveform rows were detected.")

    rows = rows[:3]

    strip_signals = []

    for row_start, row_end in rows:
        row_mask = mask[row_start:row_end, :]

        # PDF 이미지에 표시된 ECG 선의 픽셀 y좌표를 1D 파형으로 변환
        raw_strip = extract_waveform_from_row(row_mask)

        # 완전 무보정은 결측/잡음 픽셀 때문에 불안정하므로 고립성 튐만 약하게 제거
        minimally_cleaned_strip = fill_outliers_by_local_median(
            raw_strip,
            window=9,
            z=6.0,
        )

        strip_signals.append(minimally_cleaned_strip)

    # 3개 strip 길이를 맞춘 뒤 연결
    min_len = min(len(strip) for strip in strip_signals)
    strip_signals = [resample_1d(strip, min_len) for strip in strip_signals]

    # 기존보다 약한 strip 연결 보정
    connected_signal = concatenate_strips_like_pdf(strip_signals)

    # 브라우저 표시 범위만 맞춤. 의료적 mV 보정은 아님
    display_ready_signal = normalize_for_display_preserve_shape(connected_signal)

    # 브라우저 렌더링 성능을 위한 길이 조정
    display_signal = resample_1d(display_ready_signal, target_len=1200)

    time = np.linspace(0, 30, len(display_signal)).tolist()

    result = {
        "time": time,
        "amplitude": display_signal.astype(float).tolist(),
        "source_image": str(image_path),
        "row_count": len(rows),
        "sample_count": len(display_signal),
        "signal_type": "pdf_image_extracted_minimal",
        "description": "PDF 이미지에서 검출한 ECG 표시 파형을 최소 보정으로 복원한 파형입니다.",
        "medical_calibration": False,
    }

    if output_json_path is not None:
        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def make_orange_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Detect Samsung ECG orange line using HSV color threshold.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Samsung ECG waveform is orange/salmon.
    # Wider range helps with PDF rendering differences.
    lower_orange_1 = np.array([3, 35, 90])
    upper_orange_1 = np.array([28, 255, 255])

    mask = cv2.inRange(hsv, lower_orange_1, upper_orange_1)

    # 작은 잡음만 제거하고 얇은 ECG 선 끊김만 약하게 보완
    open_kernel = np.ones((2, 2), np.uint8)
    close_kernel = np.ones((2, 2), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    return mask


def find_ecg_rows(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Find the 3 horizontal ECG strip regions.
    """
    row_density = np.sum(mask > 0, axis=1)

    if np.max(row_density) <= 0:
        return []

    # row 탐지용 밀도만 smoothing. 실제 출력 파형에는 직접 smoothing하지 않음
    density = moving_average(row_density.astype(np.float32), window_size=21)

    threshold = max(6, np.percentile(density, 82))
    active_rows = np.where(density > threshold)[0]

    if len(active_rows) == 0:
        return []

    groups = []
    start = active_rows[0]
    prev = active_rows[0]

    for y in active_rows[1:]:
        if y - prev > 18:
            groups.append((start, prev))
            start = y
        prev = y

    groups.append((start, prev))

    expanded = []

    for start, end in groups:
        if end - start < 12:
            continue

        pad = 55
        row_start = max(0, start - pad)
        row_end = min(mask.shape[0], end + pad)

        expanded.append((row_start, row_end))

    merged = merge_overlapping_ranges(expanded)

    if len(merged) > 3:
        merged = sorted(merged, key=lambda item: item[1] - item[0], reverse=True)[:3]
        merged.sort(key=lambda item: item[0])

    return [(int(s), int(e)) for s, e in merged]


def merge_overlapping_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []

    ranges = sorted(ranges, key=lambda item: item[0])
    merged = [[ranges[0][0], ranges[0][1]]]

    for start, end in ranges[1:]:
        last = merged[-1]

        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])

    return [(s, e) for s, e in merged]


def extract_waveform_from_row(row_mask: np.ndarray) -> np.ndarray:
    """
    Convert one ECG strip mask to 1D waveform.

    For each x coordinate:
    - collect orange pixels
    - remove extreme y values caused by line thickness or artifacts
    - use median y coordinate
    """
    row_height, row_width = row_mask.shape[:2]
    y_values = np.full(row_width, np.nan, dtype=np.float32)

    for x in range(row_width):
        ys = np.where(row_mask[:, x] > 0)[0]

        if len(ys) == 0:
            continue

        if len(ys) >= 5:
            low = np.percentile(ys, 20)
            high = np.percentile(ys, 80)
            ys = ys[(ys >= low) & (ys <= high)]

        y_values[x] = np.median(ys)

    y_values = interpolate_nan(y_values)

    center = np.nanmedian(y_values)
    amplitude = -(y_values - center) / max(row_height, 1)

    return amplitude.astype(np.float32)


def fill_outliers_by_local_median(
    signal: np.ndarray,
    window: int = 15,
    z: float = 4.5,
) -> np.ndarray:
    """
    Replace abrupt isolated spikes with local median.

    This is used only for severe pixel extraction artifacts.
    """
    signal = signal.copy().astype(np.float32)

    if len(signal) < window * 2 + 1:
        return signal

    diff = np.abs(np.diff(signal, prepend=signal[0]))
    mad = np.median(np.abs(diff - np.median(diff))) + 1e-8
    threshold = np.median(diff) + z * mad

    outlier_idx = np.where(diff > threshold)[0]

    for idx in outlier_idx:
        left = max(0, idx - window)
        right = min(len(signal), idx + window + 1)
        signal[idx] = np.median(signal[left:right])

    return signal


def concatenate_strips_like_pdf(strips: list[np.ndarray]) -> np.ndarray:
    """
    Concatenate ECG strips with minimal correction.

    Purpose:
    - keep the waveform close to the ECG line displayed in the PDF
    - reduce only severe strip boundary discontinuity
    """
    if not strips:
        return np.array([], dtype=np.float32)

    connected = [strips[0]]

    for strip in strips[1:]:
        prev = connected[-1]
        current = strip.copy()

        # strip 경계에서만 baseline을 약하게 맞춤
        tail_median = np.median(prev[-60:])
        head_median = np.median(current[:60])
        current = current + (tail_median - head_median)

        # 기존 40 point blend보다 훨씬 짧게 연결
        blend_len = min(8, len(prev), len(current))

        if blend_len > 0:
            prev_tail = prev[-blend_len:]
            current_head = current[:blend_len]
            alpha = np.linspace(0, 1, blend_len)

            blended = (1 - alpha) * prev_tail + alpha * current_head

            connected[-1] = np.concatenate([prev[:-blend_len], blended])
            current = current[blend_len:]

        connected.append(current)

    return np.concatenate(connected).astype(np.float32)


def normalize_for_display_preserve_shape(signal: np.ndarray) -> np.ndarray:
    """
    Normalize only for browser display scale while preserving the PDF waveform shape.

    This does not medically calibrate the signal.
    It only converts pixel-derived amplitude into a stable display range.
    """
    signal = signal.astype(np.float32)

    if len(signal) == 0:
        return signal

    signal = signal - np.median(signal)

    scale = np.percentile(np.abs(signal), 99)

    if scale < 1e-8:
        return np.zeros_like(signal, dtype=np.float32)

    signal = signal / scale
    signal = np.clip(signal, -1.2, 1.2)

    return signal.astype(np.float32)


def interpolate_nan(values: np.ndarray) -> np.ndarray:
    """
    Fill missing waveform points by linear interpolation.
    """
    x = np.arange(len(values))
    valid = ~np.isnan(values)

    if np.sum(valid) < 2:
        return np.zeros_like(values, dtype=np.float32)

    return np.interp(x, x[valid], values[valid]).astype(np.float32)


def moving_average(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Moving average smoothing.

    This function is retained for ECG row detection density,
    not for strong waveform smoothing in the final display signal.
    """
    signal = np.asarray(signal, dtype=np.float32)

    if window_size <= 1:
        return signal

    kernel = np.ones(window_size, dtype=np.float32) / window_size
    return np.convolve(signal, kernel, mode="same").astype(np.float32)


def resample_1d(signal: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample 1D signal to target length using linear interpolation.
    """
    signal = np.asarray(signal, dtype=np.float32)

    if len(signal) == target_len:
        return signal

    if len(signal) < 2:
        return np.zeros(target_len, dtype=np.float32)

    old_x = np.linspace(0, 1, len(signal))
    new_x = np.linspace(0, 1, target_len)

    return np.interp(new_x, old_x, signal).astype(np.float32)
