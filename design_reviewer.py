"""
Design-review helpers for drawing-extracted member schedules.
"""

from __future__ import annotations

from typing import Any

from beams_data import BEAMS_DF, COLUMN_SECTIONS
from beam_physics import check_beam_design
from civil_agent import CivilAgent
from column_physics import check_column_design


STEEL_COST_PER_LB = 1.50
_agent: CivilAgent | None = None


def get_agent() -> CivilAgent:
    """Create the CivilAgent once and reuse it across Drawing Review checks."""
    global _agent
    if _agent is None:
        _agent = CivilAgent()
    return _agent


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _section_weight(section: str, *, columns: bool = False) -> float:
    df = COLUMN_SECTIONS if columns else BEAMS_DF
    row = df[df["name"] == section]
    if row.empty and columns:
        row = BEAMS_DF[BEAMS_DF["name"] == section]
    if row.empty:
        return 0.0
    return float(row.iloc[0]["weight"])


def _beam_props(section: str) -> dict[str, Any] | None:
    row = BEAMS_DF[BEAMS_DF["name"] == section]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def find_optimal_section(
    span_ft: float,
    dead_load: float,
    live_load: float,
    current_section: str,
    Lb_ft: float | None = None,
    spacing_ft: float = 10,
    composite: bool = False,
) -> dict[str, Any]:
    """
    Use CivilAgent to find the optimal beam section.

    This reuses the RL agent plus greedy fallback already used by the main beam
    design page.
    """
    if Lb_ft is None:
        Lb_ft = span_ft

    agent = get_agent()
    result = agent.find_optimal_beam(
        span_ft=span_ft,
        dead_load=dead_load,
        live_load=live_load,
        Lb_ft=Lb_ft,
        point_load=0,
        defl_limit=360,
        composite=composite,
        beam_spacing_ft=spacing_ft,
    )

    current_weight = _section_weight(current_section)
    if current_weight <= 0:
        current_weight = 999.0

    current_props = _beam_props(current_section)
    current_passes = None
    current_details: dict[str, Any] = {}
    if current_props is not None:
        current_passes, _, _, current_details = check_beam_design(
            span_ft,
            dead_load,
            live_load,
            current_props,
            Lb_ft=Lb_ft,
            point_load=0,
            defl_limit=360,
        )

    optimal_weight = float(result.get("weight", 0.0) or 0.0)
    if optimal_weight <= 0:
        optimal_section = current_section
        optimal_weight = current_weight
        weight_saved = 0.0
        status = "Unknown"
    elif optimal_weight > current_weight and current_passes:
        optimal_section = current_section
        optimal_weight = current_weight
        weight_saved = 0.0
        status = "Already optimal"
    elif optimal_weight > current_weight:
        optimal_section = result.get("beam_name") or current_section
        weight_saved = 0.0
        status = "Under-designed"
    else:
        optimal_section = result.get("beam_name") or current_section
        weight_saved = current_weight - optimal_weight
        status = "Over-designed" if weight_saved > 0 else "Optimal"

    if weight_saved > 0:
        status = "Over-designed"

    return {
        "current_section": current_section,
        "current_weight": current_weight,
        "optimal_section": optimal_section,
        "optimal_weight": optimal_weight,
        "weight_saved_per_ft": weight_saved,
        "passes": result.get("passes", False),
        "status": status,
        "details": result.get("details", {}),
        "current_passes": current_passes,
        "current_details": current_details,
        "explanation": result.get("explanation", ""),
        "spacing_ft": spacing_ft,
        "composite": composite,
    }


def _review_linear_member(
    member: dict[str, Any],
    member_type: str,
    dead_psf: float,
    live_psf: float,
    default_spacing_ft: float,
    Lb_ft: float | None = None,
) -> tuple[dict[str, Any], float, float]:
    span = _safe_float(member.get("span_ft"), 20.0)
    spacing = _safe_float(member.get("spacing_ft"), default_spacing_ft)
    section = str(member.get("section") or "").strip()

    dead_line = dead_psf * spacing / 1000.0
    live_line = live_psf * spacing / 1000.0

    result = find_optimal_section(
        span,
        dead_line,
        live_line,
        section,
        Lb_ft=Lb_ft,
        spacing_ft=spacing,
        composite=bool(member.get("composite", False)),
    )
    result["mark"] = member.get("mark")
    result["span_ft"] = span
    result["spacing_ft"] = spacing
    result["Lb_ft"] = Lb_ft
    result["type"] = member_type
    result["dead_line_kip_ft"] = dead_line
    result["live_line_kip_ft"] = live_line
    result["dead_line"] = dead_line
    result["live_line"] = live_line

    current_total = span * float(result.get("current_weight", 0.0))
    if result.get("status") == "Already optimal":
        optimal_total = current_total
    else:
        optimal_total = span * float(result.get("optimal_weight", 0.0))
    return result, current_total, optimal_total


def _beam_end_reactions(beam_result: dict[str, Any]) -> tuple[float, float]:
    span = float(beam_result.get("span_ft", 0.0) or 0.0)
    dead_line = float(beam_result.get("dead_line_kip_ft", 0.0) or 0.0)
    live_line = float(beam_result.get("live_line_kip_ft", 0.0) or 0.0)
    return dead_line * span / 2.0, live_line * span / 2.0


def _review_girder(
    girder: dict[str, Any],
    dead_psf: float,
    live_psf: float,
    beam_results: list[dict[str, Any]],
    default_beam_spacing_ft: float,
) -> tuple[dict[str, Any], float, float]:
    span = _safe_float(girder.get("span_ft"), 40.0)
    beam_spacing = _safe_float(girder.get("spacing_ft"), default_beam_spacing_ft)
    Lb_ft = _safe_float(girder.get("Lb_ft"), beam_spacing)
    section = str(girder.get("section") or "").strip()

    if beam_results:
        max_dead_reaction, max_live_reaction = max(
            (_beam_end_reactions(beam_result) for beam_result in beam_results),
            key=lambda pair: pair[0] + pair[1],
        )
    else:
        fallback_dead = dead_psf * beam_spacing / 1000.0
        fallback_live = live_psf * beam_spacing / 1000.0
        max_dead_reaction = fallback_dead * beam_spacing / 2.0
        max_live_reaction = fallback_live * beam_spacing / 2.0

    n_beams = max(1, int(round(span / beam_spacing)) - 1) if beam_spacing > 0 else 1
    dead_line = (n_beams * max_dead_reaction * 2.0) / span if span > 0 else 0.0
    live_line = (n_beams * max_live_reaction * 2.0) / span if span > 0 else 0.0

    result = find_optimal_section(
        span,
        dead_line,
        live_line,
        section,
        Lb_ft=Lb_ft,
        spacing_ft=beam_spacing,
    )
    result["mark"] = girder.get("mark")
    result["span_ft"] = span
    result["spacing_ft"] = beam_spacing
    result["Lb_ft"] = Lb_ft
    result["type"] = "girder"
    result["dead_line_kip_ft"] = dead_line
    result["live_line_kip_ft"] = live_line
    result["dead_line"] = dead_line
    result["live_line"] = live_line
    result["beam_reaction_dead_kips"] = round(max_dead_reaction, 2)
    result["beam_reaction_live_kips"] = round(max_live_reaction, 2)
    result["framing_beams"] = n_beams

    current_total = span * float(result.get("current_weight", 0.0))
    if result.get("status") == "Already optimal":
        optimal_total = current_total
    else:
        optimal_total = span * float(result.get("optimal_weight", 0.0))
    return result, current_total, optimal_total


def review_design(members: dict[str, Any], loads: dict[str, Any]) -> dict[str, Any]:
    """
    Review extracted/corrected members and estimate material waste.
    """
    dead_psf = _safe_float(loads.get("dead_psf"), 50.0)
    live_psf = _safe_float(loads.get("live_psf"), 80.0)

    results: dict[str, Any] = {
        "beams": [],
        "girders": [],
        "columns": [],
    }

    total_current_weight = 0.0
    total_optimal_weight = 0.0

    for beam in members.get("beams", []) or []:
        beam_span = _safe_float(beam.get("span_ft"), 20.0)
        beam_Lb = _safe_float(beam.get("Lb_ft"), 0.0)
        if beam.get("Lb_ft") is None:
            beam_Lb = 0.0
        result, current_total, optimal_total = _review_linear_member(
            beam,
            "beam",
            dead_psf,
            live_psf,
            default_spacing_ft=10.0,
            Lb_ft=min(beam_Lb, beam_span) if beam_Lb > 0 else 0.0,
        )
        total_current_weight += current_total
        total_optimal_weight += optimal_total
        results["beams"].append(result)

    for girder in members.get("girders", []) or []:
        result, current_total, optimal_total = _review_girder(
            girder,
            dead_psf,
            live_psf,
            results["beams"],
            default_beam_spacing_ft=10.0,
        )
        total_current_weight += current_total
        total_optimal_weight += optimal_total
        results["girders"].append(result)

    for col in members.get("columns", []) or []:
        section = str(col.get("section") or "").strip()
        physical_height = _safe_float(col.get("height_ft"), 14.0)
        height = _safe_float(col.get("unbraced_ft"), physical_height)
        Pu_est = _safe_float(col.get("Pu_kips"), dead_psf * 30.0 * 30.0 / 1000.0)
        K_factor = _safe_float(col.get("K_factor"), 1.0)
        current_weight = _section_weight(section, columns=True)
        current_total = current_weight * physical_height

        column_result: dict[str, Any] = {
            "mark": col.get("mark"),
            "current_section": section,
            "current_weight": current_weight,
            "height_ft": physical_height,
            "unbraced_ft": height,
            "estimated_Pu_kips": round(Pu_est, 2),
            "K_factor": K_factor,
            "note": "Column check uses explicit Pu when provided; column optimization is not included in savings.",
        }

        if section:
            col_row = COLUMN_SECTIONS[COLUMN_SECTIONS["name"] == section]
            if not col_row.empty:
                passes, weight, worst, details = check_column_design(
                    height,
                    Pu_est,
                    0.0,
                    col_row.iloc[0].to_dict(),
                    K=K_factor,
                )
                column_result.update(
                    {
                        "passes": passes,
                        "weight": float(weight),
                        "worst_check": worst,
                        "details": details,
                    }
                )

        total_current_weight += current_total
        total_optimal_weight += current_total
        results["columns"].append(column_result)

    weight_saved = max(0.0, total_current_weight - total_optimal_weight)
    cost_saved = weight_saved * STEEL_COST_PER_LB
    savings_pct = (
        weight_saved / total_current_weight * 100.0
        if total_current_weight > 0
        else 0.0
    )

    over_designed = sum(
        1
        for group in (results["beams"], results["girders"])
        for member in group
        if float(member.get("weight_saved_per_ft", 0.0)) > 0
    )

    results["summary"] = {
        "total_members": len(results["beams"]) + len(results["girders"]) + len(results["columns"]),
        "over_designed": over_designed,
        "total_weight_saved_lbs": round(weight_saved, 0),
        "total_cost_saved": round(cost_saved, 0),
        "savings_percent": round(savings_pct, 1),
        "current_weight_lbs": round(total_current_weight, 0),
        "optimal_weight_lbs": round(total_optimal_weight, 0),
    }
    return results
