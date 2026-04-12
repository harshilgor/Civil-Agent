"""
Persist user corrections to extracted drawing data for later review.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any


LOG_DIR = "data/corrections"


def _safe_filename(name: str) -> str:
    stem = os.path.basename(name or "unknown")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def log_correction(
    original: dict[str, Any],
    corrected: dict[str, Any],
    drawing_name: str = "unknown",
) -> str:
    """
    Save original vision output and user corrections for future training review.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "drawing": drawing_name,
        "original": original,
        "corrected": corrected,
        "corrections": find_differences(original, corrected),
    }

    filename = os.path.join(
        LOG_DIR,
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_safe_filename(drawing_name)}.json",
    )
    with open(filename, "w", encoding="utf-8") as handle:
        json.dump(log_entry, handle, indent=2)

    return filename


def find_differences(
    original: dict[str, Any],
    corrected: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Find fields the user changed.
    """
    diffs: list[dict[str, Any]] = []

    for member_type in ("beams", "girders", "columns"):
        orig_members = {
            member.get("mark"): member
            for member in original.get(member_type, []) or []
        }
        corr_members = {
            member.get("mark"): member
            for member in corrected.get(member_type, []) or []
        }

        for mark, corr in corr_members.items():
            if mark in orig_members:
                orig = orig_members[mark]
                for key, value in corr.items():
                    if value != orig.get(key):
                        diffs.append(
                            {
                                "type": member_type,
                                "mark": mark,
                                "field": key,
                                "was": orig.get(key),
                                "became": value,
                            }
                        )
            else:
                diffs.append(
                    {
                        "type": member_type,
                        "mark": mark,
                        "action": "added",
                    }
                )

        for mark in orig_members:
            if mark not in corr_members:
                diffs.append(
                    {
                        "type": member_type,
                        "mark": mark,
                        "action": "deleted",
                    }
                )

    orig_loads = original.get("loads", {}) or {}
    corr_loads = corrected.get("loads", {}) or {}
    for key, value in corr_loads.items():
        if value != orig_loads.get(key):
            diffs.append(
                {
                    "type": "loads",
                    "field": key,
                    "was": orig_loads.get(key),
                    "became": value,
                }
            )

    return diffs
