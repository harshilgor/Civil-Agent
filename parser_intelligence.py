"""
Parser-intelligence layer between OCR/CV evidence and structural meaning.
"""

from __future__ import annotations

import math
import re
from typing import Any


DIMENSION_RE = re.compile(r"\b(\d{1,3})\s*['’]\s*(?:[-\s]?(\d{1,2}))?\s*(?:\"|in)?\b")
NOTE_KEYWORDS = {
    "clg": "ceiling_note",
    "ceiling": "ceiling_note",
    "cathedral": "ceiling_note",
    "flat": "ceiling_note",
    "vault": "ceiling_note",
    "reserved": "irrelevant",
    "copyright": "irrelevant",
    "rights": "irrelevant",
}
ROOM_KEYWORDS = {
    "garage": "garage",
    "living": "living",
    "family": "family",
    "great": "great room",
    "kitchen": "kitchen",
    "dining": "dining",
    "bed": "bedroom",
    "bath": "bath",
    "suite": "primary suite",
    "closet": "closet",
    "utility": "utility",
    "mud": "utility",
    "laundry": "utility",
    "study": "study",
    "office": "study",
    "porch": "porch",
    "patio": "patio",
    "entry": "entry",
    "foyer": "entry",
    "hall": "corridor",
    "corridor": "corridor",
    "stair": "stair",
}


def _normalize_text(value: str) -> str:
    text = (value or "").strip()
    replacements = {
        "â€™": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "O": "0" if re.fullmatch(r"[0O]+", text.upper()) else "O",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" - ", "-")
    return text.strip()


def _token_bucket(text: str) -> str:
    lower = text.lower()
    if not text:
        return "irrelevant"
    if DIMENSION_RE.search(text):
        return "dimension_candidate"
    if any(keyword in lower for keyword in ROOM_KEYWORDS):
        return "room_label_candidate"
    if any(keyword in lower for keyword in NOTE_KEYWORDS):
        return "note_annotation"
    if re.fullmatch(r"[\W_]+", text):
        return "irrelevant"
    if len(text) <= 1 and not text.isdigit():
        return "unknown"
    return "unknown"


def _phrase_room_type(text: str) -> str | None:
    lower = text.lower()
    for keyword, room_type in ROOM_KEYWORDS.items():
        if keyword in lower:
            return room_type
    return None


def _dimension_to_feet(text: str) -> float | None:
    match = DIMENSION_RE.search(text or "")
    if not match:
        return None
    feet = float(match.group(1))
    inches = float(match.group(2) or 0.0)
    return round(feet + inches / 12.0, 3)


def normalize_and_classify_text(ocr_evidence: dict[str, Any]) -> dict[str, Any]:
    tokens = []
    for idx, word in enumerate(ocr_evidence.get("words", []), start=1):
        raw_text = str(word.get("text", ""))
        normalized = _normalize_text(raw_text)
        bucket = _token_bucket(normalized)
        tokens.append(
            {
                "id": f"T{idx:04d}",
                "raw_text": raw_text,
                "text": normalized,
                "confidence": word.get("confidence"),
                "bbox_px": word.get("bbox_px"),
                "source": word.get("source"),
                "bucket": bucket,
                "room_type": _phrase_room_type(normalized) if bucket == "room_label_candidate" else None,
                "dimension_ft": _dimension_to_feet(normalized) if bucket == "dimension_candidate" else None,
            }
        )
    return {
        "tokens": tokens,
        "summary": {
            "token_count": len(tokens),
            "dimension_candidates": sum(1 for item in tokens if item["bucket"] == "dimension_candidate"),
            "room_label_candidates": sum(1 for item in tokens if item["bucket"] == "room_label_candidate"),
            "note_annotations": sum(1 for item in tokens if item["bucket"] == "note_annotation"),
        },
    }


def assemble_text_phrases(text_artifacts: dict[str, Any]) -> dict[str, Any]:
    sortable = [item for item in text_artifacts.get("tokens", []) if item.get("bbox_px")]
    sortable.sort(key=lambda item: (item["bbox_px"][1], item["bbox_px"][0]))
    phrases: list[dict[str, Any]] = []

    for token in sortable:
        left, top, right, bottom = token["bbox_px"]
        placed = False
        for phrase in phrases:
            p_left, p_top, p_right, p_bottom = phrase["bbox_px"]
            same_line = abs(top - p_top) <= max(12, (p_bottom - p_top))
            close = left - p_right <= max(24, (right - left) * 1.6)
            if same_line and close:
                phrase["tokens"].append(token["id"])
                phrase["texts"].append(token["text"])
                phrase["bbox_px"] = [min(p_left, left), min(p_top, top), max(p_right, right), max(p_bottom, bottom)]
                phrase["max_confidence"] = max(float(phrase["max_confidence"] or 0), float(token.get("confidence") or 0))
                placed = True
                break
        if not placed:
            phrases.append(
                {
                    "id": f"P{len(phrases)+1:03d}",
                    "tokens": [token["id"]],
                    "texts": [token["text"]],
                    "bbox_px": list(token["bbox_px"]),
                    "max_confidence": float(token.get("confidence") or 0),
                }
            )

    for phrase in phrases:
        phrase["text"] = _normalize_text(" ".join(phrase.pop("texts")))
        phrase["bucket"] = _token_bucket(phrase["text"])
        phrase["room_type"] = _phrase_room_type(phrase["text"]) if phrase["bucket"] == "room_label_candidate" else None
        phrase["dimension_ft"] = _dimension_to_feet(phrase["text"]) if phrase["bucket"] == "dimension_candidate" else None
        phrase["center_px"] = [
            round((phrase["bbox_px"][0] + phrase["bbox_px"][2]) / 2.0, 1),
            round((phrase["bbox_px"][1] + phrase["bbox_px"][3]) / 2.0, 1),
        ]
    return {
        **text_artifacts,
        "phrases": phrases,
        "phrase_summary": {
            "phrase_count": len(phrases),
            "dimension_phrases": sum(1 for item in phrases if item["bucket"] == "dimension_candidate"),
            "room_label_phrases": sum(1 for item in phrases if item["bucket"] == "room_label_candidate"),
        },
    }


def infer_scale_candidates(geometry: dict[str, Any], text_artifacts: dict[str, Any]) -> dict[str, Any]:
    bbox = geometry.get("bbox_px", {})
    width_px = max(1.0, float(bbox.get("x_max", 1) - bbox.get("x_min", 0)))
    height_px = max(1.0, float(bbox.get("y_max", 1) - bbox.get("y_min", 0)))
    phrases = text_artifacts.get("phrases", [])
    candidates = []

    for phrase in phrases:
        if phrase.get("bucket") != "dimension_candidate":
            continue
        value_ft = phrase.get("dimension_ft")
        if not value_ft:
            continue
        left, top, right, bottom = phrase["bbox_px"]
        cx, cy = phrase["center_px"]
        width = max(1.0, right - left)
        height = max(1.0, bottom - top)
        near_top = abs(cy - bbox.get("y_min", 0)) <= height_px * 0.12
        near_bottom = abs(cy - bbox.get("y_max", 0)) <= height_px * 0.12
        near_left = abs(cx - bbox.get("x_min", 0)) <= width_px * 0.12
        near_right = abs(cx - bbox.get("x_max", 0)) <= width_px * 0.12
        note_penalty = 0.0
        text_lower = phrase["text"].lower()
        if any(keyword in text_lower for keyword in ("clg", "ceiling", "cathedral", "flat")):
            note_penalty += 0.4

        if width >= height:
            inferred_length = value_ft
            inferred_width = round(height_px * (value_ft / width_px), 3)
            perimeter_bonus = 0.25 if near_top or near_bottom else 0.0
        else:
            inferred_width = value_ft
            inferred_length = round(width_px * (value_ft / height_px), 3)
            perimeter_bonus = 0.25 if near_left or near_right else 0.0

        realism_bonus = 0.0
        if 28.0 <= inferred_length <= 140.0:
            realism_bonus += 0.15
        if 18.0 <= inferred_width <= 110.0:
            realism_bonus += 0.15
        if inferred_length >= inferred_width:
            realism_bonus += 0.05

        score = 0.35 + perimeter_bonus + realism_bonus - note_penalty
        score -= 0.20 if inferred_length < 26.0 or inferred_width < 18.0 else 0.0
        score = round(max(0.0, min(1.0, score)), 3)
        candidates.append(
            {
                "source_phrase": phrase["text"],
                "phrase_id": phrase["id"],
                "length_ft": round(inferred_length, 3),
                "width_ft": round(inferred_width, 3),
                "score": score,
                "reason": "perimeter_aligned" if perimeter_bonus else "local_dimension",
            }
        )

    candidates.sort(key=lambda item: (-item["score"], -item["length_ft"], -item["width_ft"]))
    best = candidates[0] if candidates else {"length_ft": 0.0, "width_ft": 0.0, "score": 0.0, "reason": "none"}
    return {
        "candidates": candidates[:12],
        "best_candidate": best,
    }


def _find_neighbor_bounds(value: float, sorted_positions: list[float], min_value: float, max_value: float) -> tuple[float, float]:
    lower = min_value
    upper = max_value
    for position in sorted_positions:
        if position < value:
            lower = position
        elif position > value:
            upper = position
            break
    return lower, upper


def build_region_artifacts(geometry_model: dict[str, Any], text_artifacts: dict[str, Any]) -> dict[str, Any]:
    length_ft = float(geometry_model["length_ft"])
    width_ft = float(geometry_model["width_ft"])
    x_positions = sorted({0.0, length_ft, *[float(item["position_ft"]) for item in geometry_model.get("wall_lines", []) if item["orientation"] == "vertical"]})
    y_positions = sorted({0.0, width_ft, *[float(item["position_ft"]) for item in geometry_model.get("wall_lines", []) if item["orientation"] == "horizontal"]})

    labeled_regions = []
    for phrase in text_artifacts.get("phrases", []):
        if phrase.get("bucket") != "room_label_candidate":
            continue
        bbox_px = phrase.get("bbox_px")
        if not bbox_px:
            continue
        cx_px, cy_px = phrase["center_px"]
        cx_ft = round((cx_px - geometry_model["bbox_px"]["x_min"]) * geometry_model["scale_ft_per_px_x"], 3)
        cy_ft = round((cy_px - geometry_model["bbox_px"]["y_min"]) * geometry_model["scale_ft_per_px_y"], 3)
        x0, x1 = _find_neighbor_bounds(cx_ft, x_positions, 0.0, length_ft)
        y0, y1 = _find_neighbor_bounds(cy_ft, y_positions, 0.0, width_ft)
        area = max(0.1, (x1 - x0) * (y1 - y0))
        labeled_regions.append(
            {
                "id": f"R{len(labeled_regions)+1:03d}",
                "label": phrase["text"],
                "room_type": phrase.get("room_type") or "space",
                "rect_ft": [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)],
                "center_ft": [cx_ft, cy_ft],
                "area_sqft": round(area, 2),
                "confidence": "high" if (phrase.get("max_confidence") or 0) >= 70 else "medium",
            }
        )

    return {
        "labeled_regions": labeled_regions,
        "wall_grid_ft": {"x": x_positions, "y": y_positions},
        "summary": {
            "region_count": len(labeled_regions),
        },
    }


def infer_semantic_zones(
    geometry_model: dict[str, Any],
    region_artifacts: dict[str, Any],
    *,
    existing_zones: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    length_ft = float(geometry_model["length_ft"])
    width_ft = float(geometry_model["width_ft"])
    zones = list(existing_zones or [])
    used_types = {str(item.get("type", "")).lower() for item in zones}

    for region in region_artifacts.get("labeled_regions", []):
        room_type = str(region.get("room_type", "space")).lower()
        if room_type in used_types:
            continue
        x0, y0, x1, y1 = region["rect_ft"]
        zones.append(
            {
                "name": region["label"],
                "type": room_type,
                "x": round(x0 / max(length_ft, 1e-6), 4),
                "y": round(y0 / max(width_ft, 1e-6), 4),
                "w": round((x1 - x0) / max(length_ft, 1e-6), 4),
                "h": round((y1 - y0) / max(width_ft, 1e-6), 4),
                "confidence": region["confidence"],
            }
        )
        used_types.add(room_type)

    zone_notes = []
    if not zones:
        zone_notes.append("No deterministic zones could be assembled from OCR phrases.")
    elif len(zones) < 3:
        zone_notes.append("Only a partial zone map could be inferred; user confirmation is still important.")

    return {
        "zones": zones,
        "notes": zone_notes,
        "summary": {
            "zone_count": len(zones),
            "open_zone_count": sum(1 for item in zones if item["type"] in {"garage", "living", "family", "great room", "patio", "porch", "entry"}),
            "support_friendly_count": sum(1 for item in zones if item["type"] in {"bedroom", "bath", "corridor", "closet", "utility", "study"}),
        },
    }
