"""
Architectural constraint helpers for generative structural layouts.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ZONE_TYPES = {
    "no_columns": "No columns",
    "no_braces": "No braces",
    "preferred_core": "Preferred core",
    "clear_span": "Required clear span",
}


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def sanitize_constraint_zones(
    zones: list[dict[str, Any]] | None,
    *,
    length_ft: float,
    width_ft: float,
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for idx, raw in enumerate(zones or []):
        try:
            zone_type = str(raw.get("zone_type") or raw.get("type") or "").strip().lower()
            if zone_type not in ZONE_TYPES:
                continue
            x_ft = float(raw.get("x_ft", 0.0))
            y_ft = float(raw.get("y_ft", 0.0))
            zone_width = float(raw.get("width_ft", 0.0))
            zone_height = float(raw.get("height_ft", 0.0))
            if zone_width <= 0 or zone_height <= 0:
                continue
            x_ft = _clamp(x_ft, 0.0, max(0.0, length_ft))
            y_ft = _clamp(y_ft, 0.0, max(0.0, width_ft))
            zone_width = _clamp(zone_width, 0.0, max(0.0, length_ft - x_ft))
            zone_height = _clamp(zone_height, 0.0, max(0.0, width_ft - y_ft))
            if zone_width <= 0 or zone_height <= 0:
                continue
            cleaned.append(
                {
                    "id": raw.get("id") or f"zone_{idx + 1}",
                    "name": str(raw.get("name") or ZONE_TYPES[zone_type]).strip() or ZONE_TYPES[zone_type],
                    "zone_type": zone_type,
                    "x_ft": round(x_ft, 3),
                    "y_ft": round(y_ft, 3),
                    "width_ft": round(zone_width, 3),
                    "height_ft": round(zone_height, 3),
                    "note": str(raw.get("note") or "").strip(),
                }
            )
        except (TypeError, ValueError):
            continue
    return cleaned


def _point_in_zone(x: float, y: float, zone: dict[str, Any]) -> bool:
    return (
        zone["x_ft"] < x < zone["x_ft"] + zone["width_ft"]
        and zone["y_ft"] < y < zone["y_ft"] + zone["height_ft"]
    )


def _rects_intersect(a: dict[str, float], b: dict[str, float]) -> bool:
    return not (
        a["x1"] <= b["x0"]
        or a["x0"] >= b["x1"]
        or a["y1"] <= b["y0"]
        or a["y0"] >= b["y1"]
    )


def evaluate_layout_constraints(
    candidate: dict[str, Any],
    brief: dict[str, Any],
    *,
    lateral_system: str | None = None,
) -> dict[str, Any]:
    length_ft = float(brief["length_ft"])
    width_ft = float(brief["width_ft"])
    zones = sanitize_constraint_zones(
        brief.get("architectural_constraints"),
        length_ft=length_ft,
        width_ft=width_ft,
    )
    if not zones:
        return {
            "passes": True,
            "hard_fail_reasons": [],
            "penalty": 0.0,
            "bonus": 0.0,
            "notes": [],
            "zones": [],
        }

    bays_x = int(candidate["bays_x"])
    bays_y = int(candidate["bays_y"])
    span_x = float(candidate["span_x"])
    span_y = float(candidate["span_y"])
    x_positions = [i * span_x for i in range(bays_x + 1)]
    y_positions = [j * span_y for j in range(bays_y + 1)]
    interior_columns = [(x, y) for x in x_positions[1:-1] for y in y_positions[1:-1]]

    hard_fail_reasons: list[str] = []
    notes: list[str] = []
    penalty = 0.0
    bonus = 0.0

    for zone in zones:
        zone_type = zone["zone_type"]
        columns_in_zone = [(x, y) for x, y in interior_columns if _point_in_zone(x, y, zone)]

        if zone_type in {"no_columns", "clear_span"} and columns_in_zone:
            hard_fail_reasons.append(
                f"{zone['name']} contains {len(columns_in_zone)} interior column point(s)."
            )
            continue

        if zone_type == "clear_span":
            clear_span_needed = max(float(zone["width_ft"]), float(zone["height_ft"]))
            available_clear_span = max(span_x, span_y)
            if available_clear_span + 1e-6 < clear_span_needed:
                hard_fail_reasons.append(
                    f"{zone['name']} needs about {clear_span_needed:.1f} ft clear span but the scheme only provides {available_clear_span:.1f} ft."
                )
            else:
                bonus += 0.18
                notes.append(
                    f"{zone['name']} is respected with a clear span of about {available_clear_span:.1f} ft."
                )

        if zone_type == "preferred_core":
            x_core = any(zone["x_ft"] <= x <= zone["x_ft"] + zone["width_ft"] for x in x_positions)
            y_core = any(zone["y_ft"] <= y <= zone["y_ft"] + zone["height_ft"] for y in y_positions)
            if x_core and y_core:
                bonus += 0.22
                notes.append(f"{zone['name']} aligns with the structural grid and can work as a core/support corridor.")
            else:
                penalty += 0.10
                notes.append(f"{zone['name']} is not strongly aligned with the current support grid.")

        if zone_type == "no_braces" and lateral_system == "braced":
            center_bay = max(0, min(bays_x - 1, bays_x // 2))
            brace_rects = [
                {"x0": center_bay * span_x, "x1": (center_bay + 1) * span_x, "y0": 0.0, "y1": span_y},
                {"x0": center_bay * span_x, "x1": (center_bay + 1) * span_x, "y0": width_ft - span_y, "y1": width_ft},
            ]
            zone_rect = {
                "x0": zone["x_ft"],
                "x1": zone["x_ft"] + zone["width_ft"],
                "y0": zone["y_ft"],
                "y1": zone["y_ft"] + zone["height_ft"],
            }
            if any(_rects_intersect(brace_rect, zone_rect) for brace_rect in brace_rects):
                hard_fail_reasons.append(
                    f"{zone['name']} blocks the default braced-bay location."
                )
            else:
                notes.append(f"{zone['name']} stays clear of the default braced-frame locations.")

    return {
        "passes": len(hard_fail_reasons) == 0,
        "hard_fail_reasons": hard_fail_reasons,
        "penalty": round(penalty, 3),
        "bonus": round(bonus, 3),
        "notes": notes,
        "zones": zones,
    }


def make_constraint_map_figure(brief: dict[str, Any]) -> plt.Figure:
    length_ft = float(brief["length_ft"])
    width_ft = float(brief["width_ft"])
    zones = sanitize_constraint_zones(
        brief.get("architectural_constraints"),
        length_ft=length_ft,
        width_ft=width_ft,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.add_patch(Rectangle((0, 0), length_ft, width_ft, fill=False, edgecolor="#111827", linewidth=2.0))
    colors = {
        "no_columns": ("#ef4444", 0.18),
        "no_braces": ("#f59e0b", 0.18),
        "preferred_core": ("#22c55e", 0.20),
        "clear_span": ("#3b82f6", 0.18),
    }

    for zone in zones:
        color, alpha = colors.get(zone["zone_type"], ("#6b7280", 0.15))
        ax.add_patch(
            Rectangle(
                (zone["x_ft"], zone["y_ft"]),
                zone["width_ft"],
                zone["height_ft"],
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                linewidth=2.0,
            )
        )
        ax.text(
            zone["x_ft"] + zone["width_ft"] / 2.0,
            zone["y_ft"] + zone["height_ft"] / 2.0,
            zone["name"],
            ha="center",
            va="center",
            fontsize=9,
            color="#111827",
            weight="bold",
        )

    ax.set_title("Architectural Constraint Map")
    ax.set_xlabel("Building length (ft)")
    ax.set_ylabel("Building width (ft)")
    ax.set_xlim(-2, length_ft + 2)
    ax.set_ylim(-2, width_ft + 2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig
