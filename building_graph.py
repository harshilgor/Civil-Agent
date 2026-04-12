"""
Internal building graph construction for plan intelligence.
"""

from __future__ import annotations

from typing import Any


OPEN_ZONE_TYPES = {"garage", "living", "family", "great room", "patio", "porch", "entry"}
SUPPORT_FRIENDLY_TYPES = {"bedroom", "bath", "corridor", "closet", "utility", "laundry", "study", "hall"}


def _overlaps_rect_line(rect_ft: list[float], orientation: str, position_ft: float) -> bool:
    x0, y0, x1, y1 = rect_ft
    if orientation == "vertical":
        return x0 <= position_ft <= x1
    return y0 <= position_ft <= y1


def _classify_boundary(
    orientation: str,
    position_ft: float,
    length_ft: float,
    thickness_px: float,
    building_length_ft: float,
    building_width_ft: float,
    spaces: list[dict[str, Any]],
) -> dict[str, Any]:
    edge_tol = 1.0
    thickness_score = min(1.0, thickness_px / 8.0)
    ref_length = building_width_ft if orientation == "vertical" else building_length_ft
    normalized_length = min(1.0, length_ft / max(ref_length, 1.0))

    if orientation == "vertical":
        is_exterior = abs(position_ft) < edge_tol or abs(position_ft - building_length_ft) < edge_tol
    else:
        is_exterior = abs(position_ft) < edge_tol or abs(position_ft - building_width_ft) < edge_tol
    if is_exterior:
        return {
            "classification": "exterior",
            "support_score": 1.0,
            "open_zone_overlap": 0,
            "support_zone_overlap": 0,
            "reasons": ["Exterior wall"],
        }

    open_overlap = 0
    support_overlap = 0
    uncertain_overlap = 0
    for space in spaces:
        if not _overlaps_rect_line(space["rect_ft"], orientation, position_ft):
            continue
        if space["open_zone_candidate"]:
            open_overlap += 1
        if space["support_friendly"]:
            support_overlap += 1
        if space["confidence"] == "low":
            uncertain_overlap += 1

    score = 0.45 * normalized_length + 0.35 * thickness_score + 0.15 * min(1.0, support_overlap / 2.0) - 0.20 * min(1.0, open_overlap / 2.0)
    score = max(0.0, min(1.0, score))

    if score >= 0.62 and open_overlap == 0:
        classification = "probable_bearing"
    elif score >= 0.42:
        classification = "possible_support"
    else:
        classification = "partition"

    reasons = []
    if normalized_length > 0.65:
        reasons.append("Long continuous wall line")
    if thickness_score > 0.75:
        reasons.append("Relatively thick linework")
    if support_overlap:
        reasons.append(f"Passes through support-friendly zones ({support_overlap})")
    if open_overlap:
        reasons.append(f"Crosses open-zone spaces ({open_overlap})")
    if uncertain_overlap:
        reasons.append("Touches low-confidence space interpretation")
    if not reasons:
        reasons.append("Limited structural evidence")

    return {
        "classification": classification,
        "support_score": round(score, 3),
        "open_zone_overlap": open_overlap,
        "support_zone_overlap": support_overlap,
        "reasons": reasons,
    }


def build_building_graph(
    geometry: dict[str, Any],
    semantics: dict[str, Any],
    *,
    length_ft: float,
    width_ft: float,
    num_floors: int = 1,
    floor_height_ft: float = 10.0,
    occupancy: str = "residential",
) -> dict[str, Any]:
    bbox = geometry["bbox_px"]
    width_px = max(1.0, float(bbox["x_max"] - bbox["x_min"]))
    height_px = max(1.0, float(bbox["y_max"] - bbox["y_min"]))
    scale_x = length_ft / width_px if width_px else 0.0
    scale_y = width_ft / height_px if height_px else 0.0

    spaces = []
    for idx, zone in enumerate(semantics.get("zones", []) or [], start=1):
        x0 = bbox["x_min"] + zone["x"] * width_px
        y0 = bbox["y_min"] + zone["y"] * height_px
        x1 = x0 + zone["w"] * width_px
        y1 = y0 + zone["h"] * height_px
        spaces.append(
            {
                "id": f"S{idx:02d}",
                "name": zone["name"],
                "type": zone["type"],
                "confidence": zone["confidence"],
                "rect_px": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                "rect_ft": [
                    round((x0 - bbox["x_min"]) * scale_x, 2),
                    round((y0 - bbox["y_min"]) * scale_y, 2),
                    round((x1 - bbox["x_min"]) * scale_x, 2),
                    round((y1 - bbox["y_min"]) * scale_y, 2),
                ],
                "open_zone_candidate": zone["type"] in OPEN_ZONE_TYPES,
                "support_friendly": zone["type"] in SUPPORT_FRIENDLY_TYPES,
            }
        )

    boundaries = []
    for line in geometry.get("major_vertical_lines", []):
        position_ft = round((line["position_px"] - bbox["x_min"]) * scale_x, 2)
        length_ft_line = round((line["end_px"] - line["start_px"]) * scale_y, 2)
        classification = _classify_boundary(
            "vertical",
            position_ft,
            length_ft_line,
            line["thickness_px"],
            float(length_ft),
            float(width_ft),
            spaces,
        )
        boundaries.append(
            {
                "type": "wall_line",
                "orientation": "vertical",
                "position_px": line["position_px"],
                "position_ft": position_ft,
                "length_ft": length_ft_line,
                "thickness_px": line["thickness_px"],
                "source": "geometry",
                **classification,
            }
        )
    for line in geometry.get("major_horizontal_lines", []):
        position_ft = round((line["position_px"] - bbox["y_min"]) * scale_y, 2)
        length_ft_line = round((line["end_px"] - line["start_px"]) * scale_x, 2)
        classification = _classify_boundary(
            "horizontal",
            position_ft,
            length_ft_line,
            line["thickness_px"],
            float(length_ft),
            float(width_ft),
            spaces,
        )
        boundaries.append(
            {
                "type": "wall_line",
                "orientation": "horizontal",
                "position_px": line["position_px"],
                "position_ft": position_ft,
                "length_ft": length_ft_line,
                "thickness_px": line["thickness_px"],
                "source": "geometry",
                **classification,
            }
        )

    return {
        "bbox_px": bbox,
        "outline_px": geometry.get("outline_px", []),
        "length_ft": float(length_ft),
        "width_ft": float(width_ft),
        "num_floors": int(num_floors),
        "floor_height_ft": float(floor_height_ft),
        "occupancy": occupancy,
        "scale_ft_per_px_x": round(scale_x, 5),
        "scale_ft_per_px_y": round(scale_y, 5),
        "spaces": spaces,
        "boundaries": boundaries,
        "annotations": {
            "detected_dimensions": semantics.get("detected_dimension_strings", []),
            "ceiling_notes": semantics.get("ceiling_notes", []),
            "notes": semantics.get("notes", ""),
        },
    }
