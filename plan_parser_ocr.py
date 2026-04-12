"""
OCR evidence extraction for floor-plan dimensions and labels.
"""

from __future__ import annotations

import os
import re
from typing import Any

from PIL import Image

try:
    import pytesseract
    from pytesseract import Output
except Exception:  # pragma: no cover - optional dependency path
    pytesseract = None
    Output = None


DIMENSION_RE = re.compile(r"\b(\d{1,3})\s*['’]\s*(?:[-\s]?(\d{1,2}))?\s*(?:\"|in)?\b")
ROOM_KEYWORDS = {
    "garage": "garage",
    "living": "living",
    "family": "family",
    "great": "great room",
    "kitchen": "kitchen",
    "bed": "bedroom",
    "bath": "bath",
    "closet": "closet",
    "utility": "utility",
    "laundry": "laundry",
    "study": "study",
    "office": "study",
    "porch": "porch",
    "patio": "patio",
    "entry": "entry",
    "hall": "corridor",
    "corridor": "corridor",
    "stair": "stair",
}


def ocr_available() -> bool:
    return pytesseract is not None


def _configure_tesseract_path() -> str | None:
    if pytesseract is None:
        return None
    candidates = [
        os.getenv("TESSERACT_CMD", ""),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return candidate
    return None


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = _normalize_text(value)
        if not normalized:
            continue
        if normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        ordered.append(normalized)
    return ordered


def dimension_string_to_feet(value: str) -> float | None:
    match = DIMENSION_RE.search(value or "")
    if not match:
        return None
    feet = float(match.group(1))
    inches = float(match.group(2) or 0.0)
    return round(feet + inches / 12.0, 3)


def _infer_room_type(label: str) -> str | None:
    lower = (label or "").lower()
    for key, room_type in ROOM_KEYWORDS.items():
        if key in lower:
            return room_type
    return None


def _words_from_native_text(native_text: str) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for token in re.findall(r"[A-Za-z0-9'\"-]+", native_text or ""):
        words.append({"text": token, "confidence": None, "bbox_px": None, "source": "native_pdf_text"})
    return words


def extract_ocr_evidence(pil_image: Image.Image, *, native_text: str = "") -> dict[str, Any]:
    configured_cmd = _configure_tesseract_path()
    words = _words_from_native_text(native_text)
    raw_fragments: list[str] = [native_text.strip()] if native_text.strip() else []
    source_parts: list[str] = ["native_pdf_text"] if native_text.strip() else []

    if pytesseract is not None and configured_cmd:
        try:
            data = pytesseract.image_to_data(pil_image, output_type=Output.DICT, config="--psm 11")
            total = len(data.get("text", []))
            for idx in range(total):
                text = _normalize_text(data["text"][idx])
                if not text:
                    continue
                conf_raw = str(data.get("conf", [""] * total)[idx]).strip()
                try:
                    confidence = float(conf_raw)
                except Exception:
                    confidence = None
                left = int(data["left"][idx])
                top = int(data["top"][idx])
                width = int(data["width"][idx])
                height = int(data["height"][idx])
                words.append(
                    {
                        "text": text,
                        "confidence": confidence,
                        "bbox_px": [left, top, left + width, top + height],
                        "source": "tesseract",
                    }
                )
                raw_fragments.append(text)
            source_parts.append("tesseract")
        except Exception:
            source_parts.append("tesseract_failed")
    else:
        source_parts.append("tesseract_unavailable")

    raw_text = _normalize_text(" ".join(fragment for fragment in raw_fragments if fragment))
    dimension_strings = _unique_strings(
        [match.group(0) for match in DIMENSION_RE.finditer(raw_text)] + [word["text"] for word in words if DIMENSION_RE.search(word["text"])]
    )
    dimension_values_ft = [value for value in (dimension_string_to_feet(item) for item in dimension_strings) if value is not None]

    room_labels = []
    for word in words:
        room_type = _infer_room_type(word["text"])
        if not room_type:
            continue
        room_labels.append(
            {
                "label": word["text"],
                "room_type": room_type,
                "bbox_px": word.get("bbox_px"),
                "confidence": word.get("confidence"),
                "source": word.get("source"),
            }
        )

    notes = []
    if native_text.strip():
        notes.append("Found text-native PDF content.")
    if pytesseract is None:
        notes.append("pytesseract not installed; OCR limited to native PDF text if available.")
    elif not configured_cmd:
        notes.append("Tesseract executable not found; OCR limited to native PDF text if available.")

    return {
        "source": "+".join(source_parts),
        "tesseract_cmd": configured_cmd,
        "raw_text": raw_text,
        "words": words,
        "dimension_strings": dimension_strings,
        "dimension_values_ft": dimension_values_ft,
        "room_labels": room_labels,
        "notes": notes,
    }
