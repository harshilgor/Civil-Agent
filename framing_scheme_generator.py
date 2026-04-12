"""
Candidate framing generation from building graph + support inference.
"""

from __future__ import annotations

from typing import Any

from layout_generator import build_brief, run_generative_design


def _base_loads_for_occupancy(occupancy: str) -> tuple[float, float]:
    occupancy = (occupancy or "residential").lower()
    if occupancy == "residential":
        return 40.0, 40.0
    if occupancy == "office":
        return 50.0, 50.0
    if occupancy == "retail":
        return 60.0, 100.0
    return 40.0, 40.0


def _count_candidates(
    support_model: dict[str, Any],
    *,
    orientation: str | None = None,
    classification: str | None = None,
) -> int:
    items = support_model.get("support_candidates") or support_model.get("support_lines") or []
    total = 0
    for item in items:
        if orientation and item.get("orientation") != orientation:
            continue
        if classification and item.get("classification") != classification:
            continue
        total += 1
    return total


def _top_support_names(support_model: dict[str, Any], limit: int = 3) -> list[str]:
    items = support_model.get("support_candidates") or []
    filtered = [item for item in items if item.get("classification") in {"probable_bearing", "possible_support"}]
    filtered.sort(key=lambda item: (-float(item.get("score", 0.0)), item.get("orientation", ""), float(item.get("position_ft", 0.0))))
    return [item["name"] for item in filtered[:limit]]


def generate_framing_schemes(
    building_graph: dict[str, Any],
    support_model: dict[str, Any],
    *,
    city: str = "Chicago",
    occupancy: str = "residential",
) -> list[dict[str, Any]]:
    length_ft = float(building_graph["length_ft"])
    width_ft = float(building_graph["width_ft"])
    num_floors = int(building_graph.get("num_floors", 1))
    floor_height_ft = float(building_graph.get("floor_height_ft", 10.0))
    dead_psf, live_psf = _base_loads_for_occupancy(occupancy)

    open_zone_count = len(support_model.get("blocked_zones", []))
    strong_supports = _count_candidates(support_model, classification="probable_bearing")
    possible_supports = _count_candidates(support_model, classification="possible_support")
    support_line_count = strong_supports + possible_supports
    wing_support_note = ", ".join(_top_support_names(support_model)) or "perimeter-led support logic"

    scheme_inputs = [
        {
            "scheme_name": "Wing-wall assisted",
            "description": "Uses likely bedroom and bath wing support corridors while keeping the more open rooms less interrupted.",
            "priority": "min_steel",
            "max_span_ft": 22.0 if strong_supports >= 2 else 26.0,
            "min_span_ft": 12.0,
            "allow_interior_cols": True,
            "composite": False,
            "max_aspect_ratio": 1.8,
            "alignment_mode": "bearing_wall",
            "structural_story": f"Leans on the strongest inferred support bands first: {wing_support_note}.",
        },
        {
            "scheme_name": "Regularized builder grid",
            "description": "Regularizes the framing into a cleaner, more repetitive grid for constructability and erection simplicity.",
            "priority": "balanced",
            "max_span_ft": 30.0,
            "allow_interior_cols": True,
            "composite": True,
            "min_span_ft": 15.0,
            "max_aspect_ratio": 1.35,
            "alignment_mode": "regularized",
            "structural_story": "Prioritizes repetitive bays, stacked columns, and a cleaner builder-friendly framing rhythm.",
        },
        {
            "scheme_name": "Open-zone preserving",
            "description": "Pushes support away from the garage and open living zones, relying more on perimeter support and stronger collectors.",
            "priority": "few_columns" if open_zone_count else "balanced",
            "max_span_ft": 36.0 if open_zone_count else 32.0,
            "min_span_ft": 18.0 if open_zone_count else 15.0,
            "allow_interior_cols": False if open_zone_count else True,
            "composite": True,
            "max_aspect_ratio": 2.0,
            "alignment_mode": "open_zone",
            "structural_story": "Treats garages, family rooms, and other open zones as support-avoidance regions wherever practical.",
        },
    ]

    schemes = []
    for scheme in scheme_inputs:
        brief = build_brief(
            length_ft=length_ft,
            width_ft=width_ft,
            num_floors=num_floors,
            floor_height_ft=floor_height_ft,
            occupancy=occupancy,
            city=city,
            priority=scheme["priority"],
            max_span_ft=scheme["max_span_ft"],
            min_span_ft=scheme["min_span_ft"],
            allow_interior_cols=scheme["allow_interior_cols"],
            dead_psf=dead_psf,
            live_psf=live_psf,
            composite=scheme["composite"],
            max_aspect_ratio=scheme["max_aspect_ratio"],
        )
        result = run_generative_design(brief)
        recommended = result.get("recommended")
        if recommended:
            schemes.append(
                {
                    **scheme,
                    "brief": brief,
                    "result": result,
                    "recommended": recommended,
                    "total_cost": recommended["total_cost"],
                    "total_steel_lbs": recommended["total_steel_lbs"],
                    "floor_depth_in": recommended["floor_depth_in"],
                    "num_columns": recommended["num_columns"],
                    "alignment_mode": scheme["alignment_mode"],
                    "structural_story": scheme["structural_story"],
                    "support_emphasis": wing_support_note,
                }
            )

    return schemes
