"""
OpenAI vision extraction for structural drawing review.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled in the Streamlit UI
    OpenAI = None


print("API key present:", bool(os.getenv("OPENAI_API_KEY")))

LAST_DEBUG: dict[str, Any] = {
    "api_key_present": bool(os.getenv("OPENAI_API_KEY")),
    "image_size_chars": None,
    "raw_response": None,
    "error": None,
}


EXTRACTION_PROMPT = """
You are analyzing a structural engineering drawing. Extract all structural
member information you can find.

Look for:
1. BEAM SCHEDULE or FRAMING PLAN
   - Beam marks (B1, B2, FB1, etc.)
   - Section sizes (W21x44, W18x35, etc.)
   - Spans in feet
   - Spacing in feet
   - Unbraced length Lb in feet, if shown
   - Whether composite (look for "COMP" or shear stud notes)

2. COLUMN SCHEDULE
   - Column marks (C1, C2, etc.)
   - Section sizes (W14x82, etc.)
   - Heights or floor ranges
   - Factored axial load Pu in kips, if shown
   - Unbraced length or effective length factor K, if shown

3. GIRDER SCHEDULE
   - Girder marks (G1, G2, etc.)
   - Section sizes
   - Spans
   - Unbraced length Lb in feet, if shown

4. LOADS (if shown)
   - Dead load in psf
   - Live load in psf

Return ONLY a JSON object, nothing else:
{
  "beams": [
    {
      "mark": "B1",
      "section": "W21x44",
      "span_ft": 25.0,
      "spacing_ft": 10.0,
      "Lb_ft": 5.83,
      "composite": true,
      "confidence": "high"
    }
  ],
  "girders": [
    {
      "mark": "G1",
      "section": "W27x84",
      "span_ft": 30.0,
      "Lb_ft": 10.0,
      "confidence": "high"
    }
  ],
  "columns": [
    {
      "mark": "C1",
      "section": "W14x82",
      "height_ft": 14.0,
      "unbraced_ft": 14.0,
      "Pu_kips": 1000.0,
      "K_factor": 1.0,
      "confidence": "high"
    }
  ],
  "loads": {
    "dead_psf": 50,
    "live_psf": 80,
    "confidence": "medium"
  },
  "notes": "Any relevant notes or uncertainties found"
}

Use confidence levels:
  "high" = clearly readable
  "medium" = readable but some uncertainty
  "low" = guessed or partially readable

If a field is not found, use null.
For section sizes use standard format: W21x44
"""


def _empty_extraction() -> dict[str, Any]:
    return {
        "beams": [],
        "girders": [],
        "columns": [],
        "loads": {},
        "notes": "",
    }


def _get_client():
    if OpenAI is None:
        return None
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    return OpenAI()


def _clean_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.IGNORECASE | re.MULTILINE).strip()
    return raw


def extract_from_image(base64_image: str) -> dict[str, Any] | None:
    """
    Send an image to OpenAI vision and extract structural member information.
    """
    client = _get_client()
    LAST_DEBUG["api_key_present"] = bool(os.getenv("OPENAI_API_KEY"))
    LAST_DEBUG["image_size_chars"] = len(base64_image)
    LAST_DEBUG["raw_response"] = None
    LAST_DEBUG["error"] = None
    if client is None:
        LAST_DEBUG["error"] = "OPENAI_API_KEY is not set or openai is not installed"
        print(f"Extraction error: {LAST_DEBUG['error']}")
        return None

    try:
        print(f"Image size (base64 chars): {len(base64_image)}")
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2000,
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
                            "text": EXTRACTION_PROMPT,
                        },
                    ],
                }
            ],
        )

        raw = response.choices[0].message.content or ""
        print("GPT-4V raw response:")
        print(raw)
        print("---")
        LAST_DEBUG["raw_response"] = raw
        return json.loads(_clean_json(raw))
    except Exception as exc:
        LAST_DEBUG["error"] = str(exc)
        print(f"Extraction error: {exc}")
        return None


def merge_extractions(extractions: list[dict[str, Any] | None]) -> dict[str, Any]:
    """
    Merge results from multiple pages into one combined dataset.

    Members are deduplicated by mark within each member type.
    """
    merged = _empty_extraction()
    merged["notes"] = []

    seen_marks = {
        "beams": set(),
        "girders": set(),
        "columns": set(),
    }

    for ext in extractions:
        if not ext:
            continue

        for member_type in ("beams", "girders", "columns"):
            for member in ext.get(member_type, []) or []:
                mark = str(member.get("mark") or "").strip()
                dedupe_key = mark or json.dumps(member, sort_keys=True)
                if dedupe_key not in seen_marks[member_type]:
                    merged[member_type].append(member)
                    seen_marks[member_type].add(dedupe_key)

        loads = ext.get("loads") or {}
        if not merged["loads"] and loads:
            merged["loads"] = loads

        notes = ext.get("notes")
        if notes:
            merged["notes"].append(notes)

    return merged
