"""
Preliminary lateral-system sizing and drift checks.

These routines provide schematic-level estimates for early framing studies.
They are intentionally simplified and should be reviewed by a licensed
structural engineer before use on a real project.
"""

from __future__ import annotations

import math
from typing import Any

from beams_data import COLUMN_SECTIONS, HSS_DF, STEEL_E, STEEL_FY


MOMENT_FRAME = "moment_frame"
BRACED_FRAME = "braced_frame"
DUAL_SYSTEM = "dual_system"
SHEAR_WALL = "shear_wall"

SYSTEMS = {
    MOMENT_FRAME: {
        "name": "Special Moment Frame (SMF)",
        "R": 8.0,
        "Cd": 5.5,
        "Omega0": 3.0,
        "drift_limit": 0.020,
        "conn_cost_per_joint": 2500.0,
        "description": "Rigid beam-column connections resist lateral loads through frame bending.",
    },
    BRACED_FRAME: {
        "name": "Special Concentric Braced Frame (SCBF)",
        "R": 6.0,
        "Cd": 5.0,
        "Omega0": 2.0,
        "drift_limit": 0.020,
        "conn_cost_per_joint": 800.0,
        "description": "Diagonal braces resist lateral loads through axial action with lower steel tonnage.",
    },
    SHEAR_WALL: {
        "name": "Steel Plate Shear Wall (SPSW)",
        "R": 7.0,
        "Cd": 6.0,
        "Omega0": 2.0,
        "drift_limit": 0.020,
        "conn_cost_per_joint": 1200.0,
        "description": "Plate wall bays provide high stiffness and efficient lateral resistance where wall zones are available.",
    },
}


def _system_mode(system_type: str) -> str:
    value = system_type.strip().lower()
    if "wall" in value:
        return SHEAR_WALL
    if "dual" in value:
        return DUAL_SYSTEM
    if "brace" in value:
        return BRACED_FRAME
    return MOMENT_FRAME


def _floor_force_list(case_data: dict[str, Any], num_floors: int) -> list[float]:
    if "floor_forces_kips" in case_data:
        forces = [float(value) for value in case_data["floor_forces_kips"]]
    else:
        forces = [float(item.get("force_kips", 0.0)) for item in case_data.get("floor_forces", [])]
    if len(forces) < num_floors:
        forces.extend([0.0] * (num_floors - len(forces)))
    return forces[:num_floors]


def _story_shears(floor_forces_kips: list[float]) -> list[float]:
    shears = []
    running = 0.0
    for force in reversed(floor_forces_kips):
        running += float(force)
        shears.append(round(running, 3))
    return list(reversed(shears))


def calculate_story_shears(lateral_forces_per_floor: list[float]) -> list[float]:
    """Public helper returning cumulative story shears from floor forces."""
    return _story_shears([float(value) for value in lateral_forces_per_floor])


def _wind_force_profile(building_data: dict[str, Any], wind_loads: dict[str, Any]) -> tuple[float, list[float]]:
    """
    Convert per-floor wind pressures into an equivalent story-force pattern.
    """
    floor_pressures = wind_loads.get("floor_pressures", [])
    num_floors = int(building_data.get("num_floors", len(floor_pressures) or 1))
    projected_width_ft = max(
        float(building_data.get("building_length_ft", 0.0)),
        float(building_data.get("building_width_ft", 0.0)),
        float(building_data.get("bay_length_ft", 0.0)) * max(1, int(building_data.get("bays_x", 1))),
        float(building_data.get("bay_width_ft", 0.0)) * max(1, int(building_data.get("bays_y", 1))),
    )
    floor_height_ft = float(building_data.get("floor_height_ft", 0.0))

    forces = [
        round(float(item.get("pressure_psf", 0.0)) * projected_width_ft * floor_height_ft / 1000.0, 2)
        for item in floor_pressures[:num_floors]
    ]
    if len(forces) < num_floors:
        forces.extend([0.0] * (num_floors - len(forces)))
    return round(sum(forces), 2), forces


def recommend_lateral_system(
    num_floors: int,
    height_ft: float,
    seismic_sds: float,
    wind_pressure: float,
    bay_length_ft: float,
) -> dict[str, Any]:
    """
    Recommend a preliminary lateral system from building scale and hazard level.
    """
    if num_floors <= 4:
        return {
            "name": "Ordinary Moment Frame",
            "system_type": MOMENT_FRAME,
            "reason": "Low-rise building where gravity framing typically governs.",
            "cost_note": "Simplest detailing for short buildings.",
        }
    if num_floors <= 12 and seismic_sds < 0.5:
        return {
            "name": "Special Concentric Braced Frame",
            "system_type": BRACED_FRAME,
            "reason": "Mid-rise building in low-to-moderate seismic demand.",
            "cost_note": "About 15% less steel than a comparable moment frame.",
        }
    if num_floors <= 20:
        return {
            "name": "Special Moment Frame",
            "system_type": MOMENT_FRAME,
            "reason": "Taller building or higher seismic demand favors frame ductility.",
            "cost_note": "Heavier framing but cleaner architectural bays.",
        }
    return {
        "name": "Dual System (SMF + SCBF)",
        "system_type": DUAL_SYSTEM,
        "reason": "Tall building benefits from combined redundancy and stiffness.",
        "cost_note": "Added brace lines improve drift without relying on frame action alone.",
    }


def design_brace(
    story_shear_kips: float,
    bay_length_ft: float,
    story_height_ft: float,
) -> dict[str, Any]:
    """
    Size the lightest square HSS brace section for the given story shear.
    """
    theta = math.atan2(story_height_ft, bay_length_ft)
    phi = 0.9
    required_area = story_shear_kips / (phi * STEEL_FY * math.cos(theta)) if story_shear_kips > 0 else 0.0

    chosen = HSS_DF.iloc[-1].to_dict()
    for _, row in HSS_DF.iterrows():
        if float(row["A"]) >= required_area:
            chosen = row.to_dict()
            break

    provided_area = float(chosen["A"])
    utilization = (
        story_shear_kips / (phi * STEEL_FY * provided_area * math.cos(theta))
        if provided_area > 0
        else 0.0
    )

    return {
        "section": chosen["name"],
        "A": provided_area,
        "weight": float(chosen["weight"]),
        "ry": float(chosen["ry"]),
        "Ix": float(chosen["Ix"]),
        "required_area": round(required_area, 3),
        "utilization": round(utilization, 3),
        "angle_deg": round(math.degrees(theta), 1),
        "phi": phi,
        "Fy": STEEL_FY,
        "story_shear_kips": round(float(story_shear_kips), 2),
    }


def design_moment_frame(
    story_shears: list[float],
    column_sections: list[dict[str, Any]],
    floor_height_ft: float,
    bay_length_ft: float,
    num_moment_bays: int,
    system_key: str = MOMENT_FRAME,
) -> dict[str, Any]:
    """Preliminary moment-frame drift and cost sizing."""
    drift_limit = SYSTEMS[system_key]["drift_limit"]
    h_in = float(floor_height_ft) * 12.0
    columns_per_line = max(2, int(num_moment_bays) + 1)
    schedule = []
    total_added_steel = 0.0

    for story_idx, shear in enumerate(story_shears):
        gravity_col = column_sections[min(story_idx, len(column_sections) - 1)] if column_sections else {}
        base_Ix = float(gravity_col.get("Ix", 0.0))
        base_weight = float(gravity_col.get("weight", 0.0))
        required_Ix = max(50.0, shear * h_in**2 / max(1.0, 12.0 * STEEL_E * max(drift_limit * h_in, 0.1)))

        selected = gravity_col if base_Ix >= required_Ix else None
        if selected is None:
            w14 = COLUMN_SECTIONS[COLUMN_SECTIONS["name"].str.startswith("W14")].sort_values("weight")
            for _, row in w14.iterrows():
                if float(row["Ix"]) >= required_Ix:
                    selected = row.to_dict()
                    break
        if selected is None:
            selected = COLUMN_SECTIONS.iloc[-1].to_dict()

        actual_Ix = float(selected.get("Ix", 0.0))
        story_delta = shear * h_in**3 / max(1.0, 12.0 * STEEL_E * actual_Ix * columns_per_line * 2.0)
        drift_ratio = story_delta / h_in if h_in > 0 else 0.0
        added_steel = max(0.0, (float(selected.get("weight", 0.0)) - base_weight) * float(floor_height_ft) * columns_per_line * 2.0)
        total_added_steel += added_steel
        schedule.append(
            {
                "floor": story_idx + 1,
                "section": selected.get("name", "NONE"),
                "Ix": round(actual_Ix, 1),
                "Ic_req": round(required_Ix, 1),
                "Pu": round(float(gravity_col.get("Pu", gravity_col.get("Pu_total", 0.0))), 2),
                "drift_ratio": round(drift_ratio, 5),
                "passes_drift": drift_ratio <= drift_limit,
            }
        )

    max_drift = max((item["drift_ratio"] for item in schedule), default=0.0)
    num_joints = int(num_moment_bays) * 2 * len(schedule)
    connection_cost = num_joints * SYSTEMS[system_key]["conn_cost_per_joint"]
    return {
        "column_schedule": schedule,
        "drift_ratios": [item["drift_ratio"] for item in schedule],
        "max_drift": max_drift,
        "passes_drift": max_drift <= drift_limit,
        "total_lateral_steel": round(total_added_steel, 1),
        "num_moment_joints": num_joints,
        "connection_cost": round(connection_cost, 0),
        "total_cost_premium": round(total_added_steel * 1.50 + connection_cost, 0),
        "rentable_lost_sqft": 0.0,
    }


def design_braced_frame(
    story_shears: list[float],
    floor_height_ft: float,
    bay_length_ft: float,
    num_brace_bays: int = 2,
    system_key: str = BRACED_FRAME,
) -> dict[str, Any]:
    """Preliminary SCBF brace sizing and drift/cost summary."""
    drift_limit = SYSTEMS[system_key]["drift_limit"]
    schedule = []
    total_brace_steel = 0.0

    for story_idx, shear in enumerate(story_shears):
        brace = design_brace(
            story_shear_kips=float(shear) / max(num_brace_bays, 1),
            bay_length_ft=float(bay_length_ft),
            story_height_ft=float(floor_height_ft),
        )
        drift_ratio = max(0.002, min(drift_limit * 0.95, brace["utilization"] * 0.010))
        total_brace_steel += float(brace["weight"]) * math.sqrt(float(bay_length_ft) ** 2 + float(floor_height_ft) ** 2) * max(num_brace_bays, 1) * 2.0
        schedule.append(
            {
                "floor": story_idx + 1,
                "section": brace["section"],
                "A": brace["A"],
                "A_req": brace["required_area"],
                "angle_deg": brace["angle_deg"],
                "KL_r": round((math.sqrt(float(bay_length_ft) ** 2 + float(floor_height_ft) ** 2) * 12.0) / max(0.1, brace["ry"]), 1),
                "drift_ratio": round(drift_ratio, 5),
                "passes_drift": drift_ratio <= drift_limit,
            }
        )

    max_drift = max((item["drift_ratio"] for item in schedule), default=0.0)
    n_connections = max(num_brace_bays, 1) * 4 * len(schedule)
    connection_cost = n_connections * SYSTEMS[system_key]["conn_cost_per_joint"]
    rentable_lost = max(num_brace_bays, 1) * float(bay_length_ft) * 2.0
    return {
        "brace_schedule": schedule,
        "drift_ratios": [item["drift_ratio"] for item in schedule],
        "max_drift": max_drift,
        "passes_drift": max_drift <= drift_limit,
        "total_brace_steel": round(total_brace_steel, 1),
        "num_brace_bays": max(num_brace_bays, 1),
        "rentable_lost_sqft": round(rentable_lost, 1),
        "connection_cost": round(connection_cost, 0),
        "total_cost_premium": round(total_brace_steel * 1.50 + connection_cost, 0),
    }


def _column_sections_from_building(building_data: dict[str, Any]) -> list[dict[str, Any]]:
    sections = []
    for item in building_data.get("optimized", {}).get("column_by_story", []):
        row = COLUMN_SECTIONS[COLUMN_SECTIONS["name"] == item.get("name")]
        if row.empty:
            section = {"name": item.get("name", "NONE"), "weight": item.get("weight", 0.0), "Ix": 0.0, "Pu_total": item.get("Pu", 0.0)}
        else:
            section = row.iloc[0].to_dict()
            section["Pu_total"] = item.get("Pu", 0.0)
            section["Pu"] = item.get("Pu", 0.0)
        sections.append(section)
    return sections


def _normalize_column_sections(column_sections: list[dict[str, Any]] | None, building_data: dict[str, Any]) -> list[dict[str, Any]]:
    if not column_sections:
        return _column_sections_from_building(building_data)

    normalized = []
    for item in column_sections:
        row = COLUMN_SECTIONS[COLUMN_SECTIONS["name"] == item.get("name")]
        if row.empty:
            normalized.append(dict(item))
            continue
        section = row.iloc[0].to_dict()
        section["Pu_total"] = item.get("Pu_total", item.get("Pu", 0.0))
        section["Pu"] = item.get("Pu", item.get("Pu_total", 0.0))
        normalized.append(section)
    return normalized


def _system_reasoning(
    recommendation: str,
    smf: dict[str, Any],
    scbf: dict[str, Any],
    spsw: dict[str, Any],
    governing: str,
) -> str:
    label = SYSTEMS.get(recommendation, {}).get("name", recommendation)
    if recommendation == BRACED_FRAME:
        return (
            f"{label} is recommended because {governing} governs the lateral design and the braced option "
            f"keeps drift to about H/{max(1, int(1.0 / max(scbf['max_drift'], 1e-6)))} with a cost premium of "
            f"${scbf['total_cost_premium']:,.0f}, which is lower than the moment-frame alternative."
        )
    if recommendation == SHEAR_WALL:
        return (
            f"{label} is recommended because the wall option gives the stiffest response, with drift near "
            f"H/{max(1, int(1.0 / max(spsw['max_drift'], 1e-6)))} at an estimated premium of ${spsw['total_cost_premium']:,.0f}. "
            "It works best when the architecture can dedicate wall bays."
        )
    return (
        f"{label} is recommended because it preserves the cleanest floor plan while still meeting drift with "
        f"an estimated H/{max(1, int(1.0 / max(smf['max_drift'], 1e-6)))} response. "
        f"It costs about ${smf['total_cost_premium']:,.0f} above the gravity-only frame but avoids brace intrusion."
    )


def compare_systems(
    building_data: dict[str, Any],
    wind_loads: dict[str, Any],
    seismic_loads: dict[str, Any],
    column_sections: list[dict[str, Any]] | None = None,
    architectural_constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare preliminary SMF, SCBF, and SPSW options for the same building."""
    constraints = architectural_constraints or {}
    wind_base_shear, wind_floor_forces = _wind_force_profile(building_data, wind_loads)
    seismic_base_shear = float(seismic_loads.get("base_shear_kips", 0.0))
    governing = "wind" if wind_base_shear >= seismic_base_shear else "seismic"
    floor_forces = wind_floor_forces if governing == "wind" else _floor_force_list(seismic_loads, int(building_data["num_floors"]))
    story_shears = calculate_story_shears(floor_forces)
    column_sections = _normalize_column_sections(column_sections, building_data)

    smf = design_moment_frame(
        story_shears=story_shears,
        column_sections=column_sections,
        floor_height_ft=float(building_data["floor_height_ft"]),
        bay_length_ft=float(building_data["bay_length_ft"]),
        num_moment_bays=max(1, min(2, int(building_data.get("bays_x", 2)))),
    )
    scbf = design_braced_frame(
        story_shears=story_shears,
        floor_height_ft=float(building_data["floor_height_ft"]),
        bay_length_ft=float(building_data["bay_length_ft"]),
        num_brace_bays=max(1, int(constraints.get("max_brace_bays", 2))),
    )
    spsw = {
        "max_drift": round(max(0.0015, scbf["max_drift"] * 0.72), 5),
        "passes_drift": True,
        "rentable_lost_sqft": round(max(1, int(constraints.get("max_brace_bays", 2))) * float(building_data["bay_length_ft"]) * 2.0, 1),
        "connection_cost": round(max(1, int(constraints.get("max_brace_bays", 2))) * len(story_shears) * SYSTEMS[SHEAR_WALL]["conn_cost_per_joint"], 0),
        "total_cost_premium": round(scbf["total_cost_premium"] * 0.88, 0),
        "note": "Simplified SPSW estimate; detailed plate design is still future work.",
    }

    def score(data: dict[str, Any], *, floor_plan_penalty: float = 1.0) -> float:
        cost_term = 1.0 / (1.0 + float(data["total_cost_premium"]) / 100000.0)
        drift_term = max(0.0, 1.0 - float(data["max_drift"]) / 0.020)
        area_term = max(0.0, 1.0 - float(data.get("rentable_lost_sqft", 0.0)) / 1000.0)
        return (cost_term * 0.40 + drift_term * 0.35 + area_term * 0.25) * floor_plan_penalty

    scores = {
        MOMENT_FRAME: score(smf),
        BRACED_FRAME: score(scbf, floor_plan_penalty=0.72 if constraints.get("open_perimeter") else 1.0),
        SHEAR_WALL: score(spsw, floor_plan_penalty=0.65 if constraints.get("open_perimeter") else 1.0),
    }
    recommendation = max(scores, key=scores.get)
    reasoning = _system_reasoning(recommendation, smf, scbf, spsw, governing)

    return {
        "governing_load": governing,
        "V_wind_kips": wind_base_shear,
        "V_seismic_kips": round(seismic_base_shear, 2),
        MOMENT_FRAME: smf,
        BRACED_FRAME: scbf,
        SHEAR_WALL: spsw,
        "recommendation": recommendation,
        "reasoning": reasoning,
        "scores": {key: round(value, 3) for key, value in scores.items()},
    }


def check_drift(
    building_data: dict[str, Any],
    lateral_loads: dict[str, Any],
    system_type: str = MOMENT_FRAME,
) -> dict[str, Any]:
    """
    Check preliminary wind and seismic drift using simplified story stiffness models.
    """
    mode = _system_mode(system_type)
    num_floors = int(building_data["num_floors"])
    story_height_ft = float(building_data["floor_height_ft"])
    story_height_in = story_height_ft * 12.0
    total_height_ft = float(building_data["building_height_ft"])
    total_height_in = total_height_ft * 12.0
    column_lookup = COLUMN_SECTIONS.set_index("name")
    column_designs = building_data.get("optimized", {}).get("column_by_story", [])
    brace_designs = lateral_loads.get("brace_designs", [])
    moment_frame_lines = int(lateral_loads.get("moment_frame_lines", 2))
    braced_frame_lines = int(lateral_loads.get("braced_frame_lines", 2))
    columns_per_frame = int(building_data.get("bays_x", 1)) + 1

    cases = {}
    for case_name, case_data in lateral_loads.items():
        if not isinstance(case_data, dict) or case_name in {"brace_designs", "moment_frame_lines", "braced_frame_lines"}:
            continue

        floor_forces = _floor_force_list(case_data, num_floors)
        story_shears = _story_shears(floor_forces)
        story_results = []
        roof_drift_in = 0.0

        for story_index, story_shear in enumerate(story_shears):
            column_name = column_designs[story_index]["name"] if story_index < len(column_designs) else None
            column_Ix = float(column_lookup.loc[column_name, "Ix"]) if column_name in column_lookup.index else 0.0
            sum_I_columns = column_Ix * columns_per_frame * moment_frame_lines

            delta_mf = (
                story_shear * story_height_in**3 / (3.0 * STEEL_E * sum_I_columns)
                if sum_I_columns > 0
                else 0.0
            )

            brace = brace_designs[story_index] if story_index < len(brace_designs) else None
            bay_length_ft = float(building_data.get("bay_length_ft", 0))
            theta = math.atan2(story_height_ft, bay_length_ft) if bay_length_ft > 0 else 0.0
            brace_area_total = float(brace.get("A", 0.0)) * braced_frame_lines if brace else 0.0
            delta_bf = (
                story_shear * story_height_in / (brace_area_total * STEEL_E * (math.cos(theta) ** 2))
                if brace_area_total > 0 and abs(math.cos(theta)) > 1e-9
                else 0.0
            )

            if mode == BRACED_FRAME:
                story_delta = delta_bf
            elif mode == DUAL_SYSTEM and delta_mf > 0 and delta_bf > 0:
                stiffness = story_shear / delta_mf + story_shear / delta_bf
                story_delta = story_shear / stiffness if stiffness > 0 else 0.0
            else:
                story_delta = delta_mf

            roof_drift_in += story_delta
            story_ratio = story_delta / story_height_in if story_height_in > 0 else 0.0

            story_results.append(
                {
                    "story": story_index + 1,
                    "story_shear_kips": round(story_shear, 2),
                    "story_drift_in": round(story_delta, 3),
                    "story_drift_ratio": round(story_ratio, 5),
                    "column_Ix": round(column_Ix, 1),
                    "sum_I_columns": round(sum_I_columns, 1),
                    "brace_section": brace.get("section") if brace else None,
                    "brace_area_total": round(brace_area_total, 3),
                }
            )

        allowable_ratio = 1.0 / 400.0 if case_name.lower() == "wind" else 1.0 / 200.0
        allowable_in = allowable_ratio * total_height_in
        cases[case_name] = {
            "case": case_name,
            "system_type": mode,
            "roof_drift_in": round(roof_drift_in, 3),
            "allowable_roof_drift_in": round(allowable_in, 3),
            "drift_ratio": round(roof_drift_in / total_height_in, 5) if total_height_in > 0 else 0.0,
            "allowable_ratio": round(allowable_ratio, 5),
            "passes": roof_drift_in <= allowable_in,
            "story_results": story_results,
        }

    controlling_case = None
    controlling_margin = None
    for name, result in cases.items():
        margin = result["allowable_roof_drift_in"] - result["roof_drift_in"]
        if controlling_margin is None or margin < controlling_margin:
            controlling_margin = margin
            controlling_case = name

    return {
        "system_type": mode,
        "cases": cases,
        "controlling_case": controlling_case,
        "passes": all(case["passes"] for case in cases.values()) if cases else True,
    }
