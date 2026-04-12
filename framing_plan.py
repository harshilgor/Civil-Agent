"""
2D framing plan sketches for generative structural layouts.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from architectural_constraints import sanitize_constraint_zones


def _candidate_member_names(candidate: dict[str, Any]) -> tuple[str, str, str]:
    beam = str(candidate.get("beam_name") or candidate.get("beam") or "Beam")
    girder = str(candidate.get("girder_name") or candidate.get("girder") or "Girder")
    column = str(candidate.get("col_name") or candidate.get("column") or "Column")
    return beam, girder, column


def make_framing_plan_figure(
    candidate: dict[str, Any],
    brief: dict[str, Any],
    *,
    title: str | None = None,
) -> plt.Figure:
    length_ft = float(brief["length_ft"])
    width_ft = float(brief["width_ft"])
    bays_x = int(candidate["bays_x"])
    bays_y = int(candidate["bays_y"])
    span_x = float(candidate["span_x"])
    span_y = float(candidate["span_y"])
    beam_spacing = float(candidate.get("beam_spacing", 10.0))
    framing_dir = str(candidate.get("framing_dir") or candidate.get("beam_dir") or "x")
    beam_label, girder_label, column_label = _candidate_member_names(candidate)
    zones = sanitize_constraint_zones(
        brief.get("architectural_constraints"),
        length_ft=length_ft,
        width_ft=width_ft,
    )

    fig, ax = plt.subplots(figsize=(9.5, 7))
    ax.add_patch(Rectangle((0, 0), length_ft, width_ft, fill=False, edgecolor="#111827", linewidth=2.2))

    for zone in zones:
        zone_colors = {
            "no_columns": "#ef4444",
            "no_braces": "#f59e0b",
            "preferred_core": "#22c55e",
            "clear_span": "#3b82f6",
        }
        color = zone_colors.get(zone["zone_type"], "#6b7280")
        ax.add_patch(
            Rectangle(
                (zone["x_ft"], zone["y_ft"]),
                zone["width_ft"],
                zone["height_ft"],
                facecolor=color,
                edgecolor=color,
                alpha=0.12,
                linewidth=1.5,
            )
        )
        ax.text(
            zone["x_ft"] + zone["width_ft"] / 2,
            zone["y_ft"] + zone["height_ft"] / 2,
            zone["name"],
            ha="center",
            va="center",
            fontsize=8,
            color="#374151",
        )

    x_positions = [i * span_x for i in range(bays_x + 1)]
    y_positions = [j * span_y for j in range(bays_y + 1)]

    if framing_dir in {"x", "x_beams"}:
        for x in x_positions:
            ax.plot([x, x], [0, width_ft], color="#6d28d9", linewidth=2.4, alpha=0.85)

        for bay_y in range(bays_y):
            y0 = bay_y * span_y
            n_spaces = max(1, int(round(span_y / beam_spacing)))
            actual_spacing = span_y / n_spaces
            for k in range(1, n_spaces):
                y = y0 + k * actual_spacing
                ax.plot([0, length_ft], [y, y], color="#f59e0b", linewidth=1.5, alpha=0.75)
    else:
        for y in y_positions:
            ax.plot([0, length_ft], [y, y], color="#6d28d9", linewidth=2.4, alpha=0.85)

        for bay_x in range(bays_x):
            x0 = bay_x * span_x
            n_spaces = max(1, int(round(span_x / beam_spacing)))
            actual_spacing = span_x / n_spaces
            for k in range(1, n_spaces):
                x = x0 + k * actual_spacing
                ax.plot([x, x], [0, width_ft], color="#f59e0b", linewidth=1.5, alpha=0.75)

    for x in x_positions:
        for y in y_positions:
            ax.add_patch(Circle((x, y), radius=max(length_ft, width_ft) * 0.0055, facecolor="#dc2626", edgecolor="#7f1d1d"))

    ax.text(length_ft * 0.01, width_ft + 3.0, f"Beams: {beam_label}", color="#92400e", fontsize=10, weight="bold")
    ax.text(length_ft * 0.36, width_ft + 3.0, f"Girders: {girder_label}", color="#5b21b6", fontsize=10, weight="bold")
    ax.text(length_ft * 0.74, width_ft + 3.0, f"Columns: {column_label}", color="#991b1b", fontsize=10, weight="bold")

    ax.set_title(title or f"Framing Sketch - {candidate.get('candidate_id', 'Selected scheme')}")
    ax.set_xlabel("Plan X (ft)")
    ax.set_ylabel("Plan Y (ft)")
    ax.set_xlim(-2, length_ft + 2)
    ax.set_ylim(-2, width_ft + 8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    return fig
