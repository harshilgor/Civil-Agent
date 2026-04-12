"""
Semantic interpretation layer for floor plans.
Uses optional vision OCR/room parsing with a deterministic fallback.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from drawing_processor import image_to_base64


SEMANTIC_PROMPT = """
You are parsing a residential floor plan for structural concept design.
Return ONLY JSON with this structure:
{
  "overall_dimensions_ft": {
    "length_ft": 0,
    "width_ft": 0
  },
  "detected_dimension_strings": ["40'-0\\""],
  "zones": [
    {
      "name": "Garage",
      "type": "garage",
      "x": 0.0,
      "y": 0.0,
      "w": 0.25,
      "h": 0.20,
      "confidence": "medium"
    }
  ],
  "ceiling_notes": [],
  "open_zone_hints": ["garage", "great room"],
  "support_hints": ["long interior wall near center"],
  "notes": "uncertainties"
}

Rules:
- x,y,w,h are normalized 0-1 relative to the main building rectangle.
- Focus on major spaces only: garage, living/family/great room, kitchen, dining,
  bedroom, bath, corridor, utility/laundry, porch/patio, closet, study, stair.
- If dimensions are unclear, use 0 and explain in notes.
- Confidence must be high, medium, or low.
"""


def _dimension_string_to_feet(value: str) -> float | None:
    match = re.search(r"(\d{1,3})\s*['’]\s*(?:[-\s]?(\d{1,2}))?\s*(?:\"|in)?", value or "")
    if not match:
        return None
    feet = float(match.group(1))
    inches = float(match.group(2) or 0.0)
    return round(feet + inches / 12.0, 3)


def _infer_dimensions_from_strings(strings: list[str]) -> tuple[float, float]:
    values = sorted({_dimension_string_to_feet(item) for item in strings if _dimension_string_to_feet(item) is not None}, reverse=True)
    if len(values) >= 2:
        return float(values[0]), float(values[1])
    if len(values) == 1:
        return float(values[0]), 0.0
    return 0.0, 0.0


def extract_plan_semantics(pil_image, ocr_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return fallback_semantics()

    try:
        client = OpenAI(api_key=api_key)
        base64_image = image_to_base64(pil_image)
        evidence_text = ""
        if ocr_evidence:
            evidence_text = (
                "\nOCR evidence:\n"
                f"- dimension strings: {ocr_evidence.get('dimension_strings', [])[:10]}\n"
                f"- room labels: {[item.get('label') for item in ocr_evidence.get('room_labels', [])[:10]]}\n"
            )
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1800,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": SEMANTIC_PROMPT + evidence_text,
                        },
                    ],
                }
            ],
        )
        raw = (response.choices[0].message.content or "").strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        parsed = json.loads(raw)
        return normalize_semantics(parsed)
    except Exception:
        return fallback_semantics()


def normalize_semantics(parsed: dict[str, Any]) -> dict[str, Any]:
    parsed = parsed or {}
    dims = parsed.get("overall_dimensions_ft") or {}
    zones = []
    for zone in parsed.get("zones", []) or []:
        try:
            zones.append(
                {
                    "name": str(zone.get("name", zone.get("type", "Space"))),
                    "type": str(zone.get("type", "space")).lower(),
                    "x": max(0.0, min(1.0, float(zone.get("x", 0.0)))),
                    "y": max(0.0, min(1.0, float(zone.get("y", 0.0)))),
                    "w": max(0.02, min(1.0, float(zone.get("w", 0.1)))),
                    "h": max(0.02, min(1.0, float(zone.get("h", 0.1)))),
                    "confidence": str(zone.get("confidence", "medium")).lower(),
                }
            )
        except Exception:
            continue
    return {
        "overall_dimensions_ft": {
            "length_ft": float(dims.get("length_ft", 0) or 0),
            "width_ft": float(dims.get("width_ft", 0) or 0),
        },
        "detected_dimension_strings": [str(item) for item in parsed.get("detected_dimension_strings", []) or []],
        "zones": zones,
        "ceiling_notes": [str(item) for item in parsed.get("ceiling_notes", []) or []],
        "open_zone_hints": [str(item) for item in parsed.get("open_zone_hints", []) or []],
        "support_hints": [str(item) for item in parsed.get("support_hints", []) or []],
        "notes": str(parsed.get("notes", "")),
    }


def merge_semantic_evidence(semantics: dict[str, Any], ocr_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = normalize_semantics(semantics or {})
    if not ocr_evidence:
        return merged

    merged["detected_dimension_strings"] = list(
        dict.fromkeys([*merged.get("detected_dimension_strings", []), *ocr_evidence.get("dimension_strings", [])])
    )
    dims = merged.get("overall_dimensions_ft", {})
    length_ft = float(dims.get("length_ft", 0) or 0)
    width_ft = float(dims.get("width_ft", 0) or 0)
    if length_ft <= 0 or width_ft <= 0:
        inferred_length, inferred_width = _infer_dimensions_from_strings(merged["detected_dimension_strings"])
        merged["overall_dimensions_ft"] = {
            "length_ft": length_ft or inferred_length,
            "width_ft": width_ft or inferred_width,
        }

    if ocr_evidence.get("room_labels") and not merged.get("zones"):
        hints = []
        for item in ocr_evidence["room_labels"][:10]:
            label = item.get("label", "")
            room_type = item.get("room_type", "")
            hints.append(f"{label} ({room_type})")
        merged["notes"] = (merged.get("notes", "") + " OCR labels detected: " + ", ".join(hints)).strip()
    return merged


def fallback_semantics() -> dict[str, Any]:
    return {
        "overall_dimensions_ft": {"length_ft": 0.0, "width_ft": 0.0},
        "detected_dimension_strings": [],
        "zones": [],
        "ceiling_notes": [],
        "open_zone_hints": [],
        "support_hints": [],
        "notes": "Vision semantics unavailable; please confirm dimensions and zones manually.",
    }


def infer_scale_from_semantics(semantics: dict[str, Any], geometry: dict[str, Any]) -> dict[str, Any]:
    bbox = geometry.get("bbox_px", {})
    width_px = max(1.0, float(bbox.get("x_max", 1) - bbox.get("x_min", 0)))
    height_px = max(1.0, float(bbox.get("y_max", 1) - bbox.get("y_min", 0)))
    dims = semantics.get("overall_dimensions_ft") or {}

    length_ft = float(dims.get("length_ft", 0) or 0)
    width_ft = float(dims.get("width_ft", 0) or 0)
    if length_ft <= 0 or width_ft <= 0:
        inferred_length, inferred_width = _infer_dimensions_from_strings(semantics.get("detected_dimension_strings", []))
        length_ft = length_ft or inferred_length
        width_ft = width_ft or inferred_width
    scale_candidates = []
    if length_ft > 0:
        scale_candidates.append(length_ft / width_px)
    if width_ft > 0:
        scale_candidates.append(width_ft / height_px)
    scale_ft_per_px = sum(scale_candidates) / len(scale_candidates) if scale_candidates else 0.0

    return {
        "length_ft": length_ft,
        "width_ft": width_ft,
        "scale_ft_per_px": round(scale_ft_per_px, 5),
        "confidence": "medium" if scale_ft_per_px > 0 else "low",
    }
