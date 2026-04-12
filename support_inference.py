"""
Support and constraint inference layer.
"""

from __future__ import annotations

from typing import Any


def _exterior_supports(bbox: dict[str, float], length_ft: float, width_ft: float) -> list[dict[str, Any]]:
    return [
        {
            "name": "Exterior West",
            "orientation": "vertical",
            "position_px": float(bbox["x_min"]),
            "position_ft": 0.0,
            "source": "exterior",
            "classification": "exterior",
            "score": 1.0,
            "active": True,
            "reasons": ["Primary exterior wall"],
        },
        {
            "name": "Exterior East",
            "orientation": "vertical",
            "position_px": float(bbox["x_max"]),
            "position_ft": round(length_ft, 2),
            "source": "exterior",
            "classification": "exterior",
            "score": 1.0,
            "active": True,
            "reasons": ["Primary exterior wall"],
        },
        {
            "name": "Exterior South",
            "orientation": "horizontal",
            "position_px": float(bbox["y_min"]),
            "position_ft": 0.0,
            "source": "exterior",
            "classification": "exterior",
            "score": 1.0,
            "active": True,
            "reasons": ["Primary exterior wall"],
        },
        {
            "name": "Exterior North",
            "orientation": "horizontal",
            "position_px": float(bbox["y_max"]),
            "position_ft": round(width_ft, 2),
            "source": "exterior",
            "classification": "exterior",
            "score": 1.0,
            "active": True,
            "reasons": ["Primary exterior wall"],
        },
    ]


def _candidate_name(boundary: dict[str, Any]) -> str:
    prefix = "Vertical" if boundary["orientation"] == "vertical" else "Horizontal"
    return f"{prefix} candidate @ {boundary['position_ft']:.1f} ft"


def _select_interior_candidates(
    candidates: list[dict[str, Any]],
    *,
    orientation: str,
    dimension_ft: float,
    preferred_count: int,
) -> list[dict[str, Any]]:
    interior = [item for item in candidates if item["orientation"] == orientation and item["classification"] != "exterior"]
    interior.sort(key=lambda item: (item["classification"] != "probable_bearing", -item["score"], -item["length_ft"]))
    selected: list[dict[str, Any]] = []
    min_separation_ft = max(8.0, 0.18 * dimension_ft)
    for item in interior:
        if item["classification"] == "partition":
            continue
        if item["open_zone_overlap"] >= 2 and item["classification"] != "probable_bearing":
            continue
        if any(abs(other["position_ft"] - item["position_ft"]) < min_separation_ft for other in selected):
            continue
        selected.append(item)
        if len(selected) >= preferred_count:
            break
    return selected


def infer_support_and_constraints(building_graph: dict[str, Any]) -> dict[str, Any]:
    bbox = building_graph["bbox_px"]
    length_ft = float(building_graph["length_ft"])
    width_ft = float(building_graph["width_ft"])
    support_candidates = _exterior_supports(bbox, length_ft, width_ft)

    for boundary in building_graph.get("boundaries", []):
        if boundary["source"] != "geometry":
            continue
        if boundary["classification"] == "partition":
            continue
        reference_length = width_ft if boundary["orientation"] == "vertical" else length_ft
        if boundary["length_ft"] < 0.38 * reference_length:
            continue
        edge_distance = (
            min(abs(boundary["position_ft"]), abs(boundary["position_ft"] - length_ft))
            if boundary["orientation"] == "vertical"
            else min(abs(boundary["position_ft"]), abs(boundary["position_ft"] - width_ft))
        )
        if edge_distance < 1.5:
            continue
        support_candidates.append(
            {
                "name": _candidate_name(boundary),
                "orientation": boundary["orientation"],
                "position_px": float(boundary["position_px"]),
                "position_ft": float(boundary["position_ft"]),
                "source": "plan_inference",
                "classification": boundary["classification"],
                "score": float(boundary["support_score"]),
                "length_ft": float(boundary["length_ft"]),
                "open_zone_overlap": int(boundary["open_zone_overlap"]),
                "support_zone_overlap": int(boundary["support_zone_overlap"]),
                "reasons": list(boundary["reasons"]),
                "active": False,
            }
        )

    blocked_zones = []
    support_friendly_zones = []
    uncertain_regions = []
    for space in building_graph.get("spaces", []):
        record = {
            "name": space["name"],
            "type": space["type"],
            "rect_px": space["rect_px"],
            "rect_ft": space["rect_ft"],
        }
        if space["open_zone_candidate"]:
            blocked_zones.append(record)
        if space["support_friendly"]:
            support_friendly_zones.append(record)
        if space["confidence"] == "low":
            uncertain_regions.append(record)

    vertical_count = max(1, round(length_ft / 32.0) - 1)
    horizontal_count = max(1, round(width_ft / 28.0) - 1)
    selected_vertical = _select_interior_candidates(
        support_candidates,
        orientation="vertical",
        dimension_ft=length_ft,
        preferred_count=vertical_count,
    )
    selected_horizontal = _select_interior_candidates(
        support_candidates,
        orientation="horizontal",
        dimension_ft=width_ft,
        preferred_count=horizontal_count,
    )

    active_keys = {
        (item["orientation"], round(float(item["position_ft"]), 2), item["name"])
        for item in [*selected_vertical, *selected_horizontal]
    }
    for candidate in support_candidates:
        if candidate["classification"] == "exterior":
            candidate["active"] = True
            continue
        key = (candidate["orientation"], round(float(candidate["position_ft"]), 2), candidate["name"])
        candidate["active"] = key in active_keys

    support_lines = [
        {
            "name": candidate["name"],
            "orientation": candidate["orientation"],
            "position_px": candidate["position_px"],
            "position_ft": candidate["position_ft"],
            "source": candidate["source"],
            "classification": candidate["classification"],
            "score": candidate["score"],
            "reasons": candidate["reasons"],
        }
        for candidate in support_candidates
        if candidate.get("active")
    ]

    return {
        "support_candidates": sorted(
            support_candidates,
            key=lambda item: (
                item["classification"] != "exterior",
                item["classification"] != "probable_bearing",
                -float(item["score"]),
                item["orientation"],
                float(item["position_ft"]),
            ),
        ),
        "support_lines": support_lines,
        "blocked_zones": blocked_zones,
        "support_friendly_zones": support_friendly_zones,
        "uncertain_regions": uncertain_regions,
        "notes": [
            "Exterior walls treated as primary supports.",
            "Only the highest-confidence interior support candidates are activated by default.",
            "Open rooms, garages, and patios are treated as support-avoidance zones unless wall evidence is strong.",
            "Minor partitions remain visible to the parser but are not promoted into supports automatically.",
        ],
    }
