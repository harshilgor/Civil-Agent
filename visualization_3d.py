"""
3D steel frame visualization helpers for Streamlit.
"""

from __future__ import annotations

import json
import math
from typing import Any

from beams_data import COLUMN_SECTIONS, HSS_DF
from column_physics import check_column_design
from floor_system import FloorSystem
from lateral_system import check_drift, design_brace, recommend_lateral_system
from load_flow_calculator import compute_gravity_load_flow, compute_lateral_load_flow


OVERDESIGNED_BLUE = "#3498db"
GREEN = "#2ecc71"
AMBER = "#f39c12"
ORANGE = "#e67e22"
RED = "#e74c3c"
PURPLE = "#8e44ad"


def _utilization_from_details(details: dict[str, Any] | None) -> float:
    details = details or {}
    candidate_keys = (
        "construction_ratio",
        "moment_ratio",
        "shear_ratio",
        "defl_ratio",
        "deflection_ratio",
        "ltb_ratio",
        "flange_ratio",
        "web_ratio",
        "axial_ratio",
        "pm_ratio",
    )
    values = []
    for key in candidate_keys:
        value = details.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return round(max(values), 3) if values else 0.0


def _safe_round(value: Any, digits: int = 3) -> float:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def _line_wu(dead_line: float, live_line: float) -> float:
    return round(1.2 * dead_line + 1.6 * live_line, 3)


def _column_design_for_load(height_ft: float, Pu_kips: float, Mu_kipft: float, K: float = 1.0) -> dict[str, Any]:
    w14 = COLUMN_SECTIONS[COLUMN_SECTIONS["name"].str.startswith("W14")].reset_index(drop=True)
    for _, row in w14.iterrows():
        section = row.to_dict()
        passes, weight, worst, details = check_column_design(height_ft, Pu_kips, Mu_kipft, section, K)
        if passes:
            return {
                "name": row["name"],
                "weight": float(weight),
                "passes": True,
                "worst": worst,
                "details": details,
                "Pu": round(Pu_kips, 2),
                "Mu": round(Mu_kipft, 2),
                "K": K,
            }
    return {
        "name": "NONE",
        "weight": 0.0,
        "passes": False,
        "worst": "none",
        "details": {},
        "Pu": round(Pu_kips, 2),
        "Mu": round(Mu_kipft, 2),
        "K": K,
    }


def _get_hss_row(section_name: str) -> dict[str, Any] | None:
    match = HSS_DF[HSS_DF["name"] == section_name]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def _next_heavier_hss(section_name: str) -> dict[str, Any]:
    current = HSS_DF[HSS_DF["name"] == section_name]
    if current.empty:
        return HSS_DF.iloc[-1].to_dict()
    current_area = float(current.iloc[0]["A"])
    heavier = HSS_DF[HSS_DF["A"] > current_area]
    if heavier.empty:
        return current.iloc[0].to_dict()
    return heavier.iloc[0].to_dict()


def _member_template(
    *,
    member_id: int,
    member_type: str,
    mark: str,
    floor_label: str,
    story_index: int,
    x_start: float,
    y_start: float,
    z_start: float,
    x_end: float,
    y_end: float,
    z_end: float,
    baseline_section: str,
    baseline_weight: float,
    baseline_utilization: float,
    optimized_section: str,
    optimized_weight: float,
    optimized_utilization: float,
    weight_saved_per_ft: float,
    span_ft: float,
    tributary_ft: float,
    dead_line: float,
    live_line: float,
    Wu: float,
    baseline_details: dict[str, Any],
    optimized_details: dict[str, Any],
    baseline_extras: dict[str, Any] | None = None,
    optimized_extras: dict[str, Any] | None = None,
    load_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    length_ft = (
        (x_end - x_start) ** 2
        + (y_end - y_start) ** 2
        + (z_end - z_start) ** 2
    ) ** 0.5
    total_saved_lbs = max(0.0, weight_saved_per_ft) * length_ft
    can_be_lighter = bool(
        weight_saved_per_ft > 0
        and optimized_section
        and baseline_section
        and optimized_section != baseline_section
    )
    return {
        "id": member_id,
        "type": member_type,
        "mark": mark,
        "floor_label": floor_label,
        "story_index": story_index,
        "x_start": round(x_start, 3),
        "y_start": round(y_start, 3),
        "z_start": round(z_start, 3),
        "x_end": round(x_end, 3),
        "y_end": round(y_end, 3),
        "z_end": round(z_end, 3),
        "length_ft": round(length_ft, 3),
        "section": baseline_section,
        "baseline_section": baseline_section,
        "baseline_weight": round(baseline_weight, 3),
        "optimized_weight": round(optimized_weight, 3),
        "utilization": round(baseline_utilization, 3),
        "baseline_utilization": round(baseline_utilization, 3),
        "optimized_section": optimized_section,
        "optimized_utilization": round(optimized_utilization, 3),
        "weight_saved": round(weight_saved_per_ft, 3),
        "total_saved_lbs": round(total_saved_lbs, 1),
        "can_be_lighter": can_be_lighter,
        "span_ft": round(span_ft, 3),
        "tributary_ft": round(tributary_ft, 3),
        "dead_line": round(dead_line, 3),
        "live_line": round(live_line, 3),
        "Wu": round(Wu, 3),
        "baseline_details": baseline_details or {},
        "optimized_details": optimized_details or {},
        "baseline_extras": baseline_extras or {},
        "optimized_extras": optimized_extras or {},
        "load_summary": load_summary or {},
    }


def member_schedule_rows(building_data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for member in building_data.get("members", []):
        rows.append(
            {
                "ID": member["id"],
                "Mark": member["mark"],
                "Type": member["type"],
                "Floor": member["floor_label"],
                "Baseline Section": member["baseline_section"],
                "Optimized Section": member["optimized_section"],
                "Length (ft)": member["length_ft"],
                "Span (ft)": member["span_ft"],
                "Tributary (ft)": member["tributary_ft"],
                "Dead Line (kip/ft)": member["dead_line"],
                "Live Line (kip/ft)": member["live_line"],
                "Wu": member["Wu"],
                "Baseline Utilization": member["baseline_utilization"],
                "Optimized Utilization": member["optimized_utilization"],
                "Saved (lb/ft)": member["weight_saved"],
                "Total Saved (lb)": member["total_saved_lbs"],
            }
        )
    return rows


def generate_building_data(
    num_floors: int,
    floor_height: float,
    bay_length: float,
    bay_width: float,
    bays_x: int,
    bays_y: int,
    dead_psf: float,
    live_psf: float,
    composite: bool = True,
    beam_spacing_ft: float = 10.0,
    composite_ratio: float = 0.5,
    hazards: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a 3D member list from the existing floor-system design engine.
    """
    optimized_fs = FloorSystem(
        bay_length_ft=bay_length,
        bay_width_ft=bay_width,
        dead_load_psf=dead_psf,
        live_load_psf=live_psf,
        beam_spacing_ft=beam_spacing_ft,
        num_floors=num_floors,
        floor_height_ft=floor_height,
        composite_beams=composite,
        composite_ratio=composite_ratio,
    )
    optimized = optimized_fs.design_all()

    baseline_fs = FloorSystem(
        bay_length_ft=bay_length,
        bay_width_ft=bay_width,
        dead_load_psf=dead_psf,
        live_load_psf=live_psf,
        beam_spacing_ft=beam_spacing_ft,
        num_floors=num_floors,
        floor_height_ft=floor_height,
        composite_beams=False,
    )
    baseline = baseline_fs.design_all()

    beam_opt = optimized["beams"]
    girder_opt = optimized["girder"]
    column_baseline = baseline["column"]
    beam_base = baseline["beams"]
    girder_base = baseline["girder"]

    building_length = bays_x * bay_length
    building_width = bays_y * bay_width
    building_height = num_floors * floor_height
    trib_area = bay_length * bay_width
    # Use a representative gravity-frame column tributary of half a bay.
    # This keeps the 3D optimizer aligned with the one-bay floor-system model
    # instead of assigning every column the demand of a full interior bay.
    column_tributary_area = trib_area / 2.0
    dead_kips_per_floor = dead_psf * column_tributary_area / 1000.0
    live_kips_per_floor = live_psf * column_tributary_area / 1000.0
    wu_per_floor = 1.2 * dead_kips_per_floor + 1.6 * live_kips_per_floor

    beam_spacing = float(beam_opt.get("spacing", 10) or 10)
    n_beams_per_bay = int(beam_opt.get("num_beams", 0) or 0)
    beam_weight_saved = max(0.0, float(beam_base.get("weight", 0)) - float(beam_opt.get("weight", 0)))
    girder_weight_saved = max(0.0, float(girder_base.get("weight", 0)) - float(girder_opt.get("weight", 0)))

    members: list[dict[str, Any]] = []
    member_id = 0
    beam_count = 0
    girder_count = 0
    column_count = 0

    optimized_column_designs: list[dict[str, Any]] = []
    baseline_column_designs: list[dict[str, Any]] = []
    for story_index in range(num_floors):
        floors_above = num_floors - story_index
        Pu = wu_per_floor * floors_above + 0.5 * floors_above
        Mu = 0.01 * Pu * floor_height
        optimized_column_designs.append(
            _column_design_for_load(floor_height, Pu, Mu, K=1.0)
        )
        baseline_column_designs.append(
            {
                "name": column_baseline.get("name", "NONE"),
                "weight": float(column_baseline.get("weight", 0.0)),
                "passes": column_baseline.get("passes", False),
                "worst": "baseline",
                "details": column_baseline.get("details", {}),
                "Pu": round(float(column_baseline.get("Pu", Pu)), 2),
                "Mu": round(float(column_baseline.get("Mu", Mu)), 2),
                "K": float(column_baseline.get("K", 1.0)),
            }
        )

    lateral_recommendation = None
    lateral_loads: dict[str, Any] = {}
    lateral_results = None
    brace_designs: list[dict[str, Any]] = []
    braced_bays: list[dict[str, Any]] = []
    if hazards:
        lateral_recommendation = recommend_lateral_system(
            num_floors=num_floors,
            height_ft=building_height,
            seismic_sds=float(hazards.get("seismic", {}).get("sds", 0.0)),
            wind_pressure=float(hazards.get("wind", {}).get("roof_pressure_psf", 0.0)),
            bay_length_ft=bay_length,
        )

        wind_story_forces = []
        projected_width_ft = max(bay_length * bays_x, bay_width * bays_y)
        for item in hazards.get("wind", {}).get("floor_pressures", []):
            wind_story_forces.append(
                round(float(item.get("pressure_psf", 0.0)) * projected_width_ft * floor_height / 1000.0, 2)
            )
        wind_story_forces = wind_story_forces[:num_floors]
        if len(wind_story_forces) < num_floors:
            wind_story_forces.extend([0.0] * (num_floors - len(wind_story_forces)))

        seismic_story_forces = [
            round(float(item.get("force_kips", 0.0)), 2)
            for item in hazards.get("seismic", {}).get("floor_forces", [])
        ][:num_floors]
        if len(seismic_story_forces) < num_floors:
            seismic_story_forces.extend([0.0] * (num_floors - len(seismic_story_forces)))

        if lateral_recommendation["system_type"] in ("braced_frame", "dual_system"):
            x_bay_index = max(0, min(bays_x - 1, bays_x // 2))
            braced_bays = [
                {"x0": x_bay_index * bay_length, "x1": (x_bay_index + 1) * bay_length, "y": 0.0, "label": "South frame"},
                {"x0": x_bay_index * bay_length, "x1": (x_bay_index + 1) * bay_length, "y": building_width, "label": "North frame"},
            ]

            running_shear = 0.0
            story_shears = []
            for force in reversed(seismic_story_forces if any(seismic_story_forces) else wind_story_forces):
                running_shear += force
                story_shears.append(running_shear)
            story_shears = list(reversed(story_shears))

            for shear in story_shears:
                brace = design_brace(
                    story_shear_kips=float(shear) / max(len(braced_bays), 1),
                    bay_length_ft=bay_length,
                    story_height_ft=floor_height,
                )
                heavier = _next_heavier_hss(brace["section"])
                brace_designs.append(
                    {
                        **brace,
                        "baseline_section": heavier["name"],
                        "baseline_weight": float(heavier["weight"]),
                        "baseline_area": float(heavier["A"]),
                        "baseline_utilization": round(
                            brace["story_shear_kips"]
                            / (brace["phi"] * brace["Fy"] * float(heavier["A"]) * max(0.001, math.cos(math.radians(brace["angle_deg"])))),
                            3,
                        ),
                    }
                )

        lateral_loads = {
            "wind": {"floor_forces_kips": wind_story_forces},
            "seismic": {"floor_forces_kips": seismic_story_forces},
            "brace_designs": brace_designs,
            "moment_frame_lines": 2,
            "braced_frame_lines": max(len(braced_bays), 2) if braced_bays else 2,
        }

    for story_index in range(num_floors):
        z0 = story_index * floor_height
        z1 = z0 + floor_height
        floor_label = f"Floor {story_index + 1}"
        col_opt = optimized_column_designs[story_index]
        col_base = baseline_column_designs[story_index]
        col_weight_saved = max(0.0, col_base["weight"] - col_opt["weight"])

        for ix in range(bays_x + 1):
            x = ix * bay_length
            for iy in range(bays_y + 1):
                y = iy * bay_width
                column_count += 1
                members.append(
                    _member_template(
                        member_id=member_id,
                        member_type="column",
                        mark=f"C-{column_count:03d}",
                        floor_label=floor_label,
                        story_index=story_index + 1,
                        x_start=x,
                        y_start=y,
                        z_start=z0,
                        x_end=x,
                        y_end=y,
                        z_end=z1,
                        baseline_section=col_base["name"],
                        baseline_weight=col_base["weight"],
                        baseline_utilization=_utilization_from_details(col_base["details"]),
                        optimized_section=col_opt["name"],
                        optimized_weight=col_opt["weight"],
                        optimized_utilization=_utilization_from_details(col_opt["details"]),
                        weight_saved_per_ft=col_weight_saved,
                        span_ft=floor_height,
                        tributary_ft=column_tributary_area,
                        dead_line=0.0,
                        live_line=0.0,
                        Wu=round(col_opt["Pu"], 3),
                        baseline_details=col_base["details"],
                        optimized_details=col_opt["details"],
                        load_summary={
                            "dead_kips_per_floor": round(dead_kips_per_floor, 2),
                            "live_kips_per_floor": round(live_kips_per_floor, 2),
                            "wu_per_floor": round(wu_per_floor, 2),
                            "floors_above": num_floors - story_index,
                            "tributary_area_ft2": round(column_tributary_area, 1),
                            "Pu": round(col_opt["Pu"], 2),
                            "Mu": round(col_opt["Mu"], 2),
                        },
                    )
                )
                member_id += 1

        level_z = z1
        for ix in range(bays_x):
            bay_x0 = ix * bay_length
            bay_x1 = bay_x0 + bay_length

            for iy in range(bays_y + 1):
                y = iy * bay_width
                girder_count += 1
                members.append(
                    _member_template(
                        member_id=member_id,
                        member_type="girder",
                        mark=f"G-{girder_count:03d}",
                        floor_label=floor_label,
                        story_index=story_index + 1,
                        x_start=bay_x0,
                        y_start=y,
                        z_start=level_z,
                        x_end=bay_x1,
                        y_end=y,
                        z_end=level_z,
                        baseline_section=girder_base.get("name", girder_opt.get("name", "NONE")),
                        baseline_weight=float(girder_base.get("weight", 0)),
                        baseline_utilization=_utilization_from_details(girder_base.get("details")),
                        optimized_section=girder_opt.get("name", "NONE"),
                        optimized_weight=float(girder_opt.get("weight", 0)),
                        optimized_utilization=_utilization_from_details(girder_opt.get("details")),
                        weight_saved_per_ft=girder_weight_saved,
                        span_ft=bay_length,
                        tributary_ft=float(girder_opt.get("n_point_loads", 0)) * beam_spacing,
                        dead_line=float(girder_opt.get("equiv_DL", 0)),
                        live_line=float(girder_opt.get("equiv_LL", 0)),
                        Wu=_line_wu(float(girder_opt.get("equiv_DL", 0)), float(girder_opt.get("equiv_LL", 0))),
                        baseline_details=girder_base.get("details", {}),
                        optimized_details=girder_opt.get("details", {}),
                        baseline_extras={
                            "point_load_kips": float(girder_base.get("point_load_kips", 0)),
                            "n_point_loads": int(girder_base.get("n_point_loads", 0)),
                        },
                        optimized_extras={
                            "point_load_kips": float(girder_opt.get("point_load_kips", 0)),
                            "n_point_loads": int(girder_opt.get("n_point_loads", 0)),
                            "stud_count": int(girder_opt.get("stud_count", 0)),
                            "studs_per_side": int(girder_opt.get("studs_per_side", 0)),
                            "Ieff": _safe_round(girder_opt.get("Ieff", 0), 1),
                        },
                    )
                )
                member_id += 1

            for iy in range(bays_y):
                bay_y0 = iy * bay_width
                for beam_idx in range(1, n_beams_per_bay + 1):
                    x = bay_x0 + beam_idx * beam_spacing
                    if x >= bay_x1 - 1e-6:
                        continue
                    beam_count += 1
                    members.append(
                        _member_template(
                            member_id=member_id,
                            member_type="beam",
                            mark=f"B-{beam_count:03d}",
                            floor_label=floor_label,
                            story_index=story_index + 1,
                            x_start=x,
                            y_start=bay_y0,
                            z_start=level_z,
                            x_end=x,
                            y_end=bay_y0 + bay_width,
                            z_end=level_z,
                            baseline_section=beam_base.get("name", beam_opt.get("name", "NONE")),
                            baseline_weight=float(beam_base.get("weight", 0)),
                            baseline_utilization=_utilization_from_details(beam_base.get("details")),
                            optimized_section=beam_opt.get("name", "NONE"),
                            optimized_weight=float(beam_opt.get("weight", 0)),
                            optimized_utilization=_utilization_from_details(beam_opt.get("details")),
                            weight_saved_per_ft=beam_weight_saved,
                            span_ft=bay_width,
                            tributary_ft=beam_spacing,
                            dead_line=float(beam_opt.get("DL_line", 0)),
                            live_line=float(beam_opt.get("LL_line", 0)),
                            Wu=_line_wu(float(beam_opt.get("DL_line", 0)), float(beam_opt.get("LL_line", 0))),
                            baseline_details=beam_base.get("details", {}),
                            optimized_details=beam_opt.get("details", {}),
                            optimized_extras={
                                "stud_count": int(beam_opt.get("stud_count", 0)),
                                "studs_per_side": int(beam_opt.get("studs_per_side", 0)),
                                "Ieff": _safe_round(beam_opt.get("Ieff", 0), 1),
                                "composite": bool(beam_opt.get("composite", False)),
                            },
                        )
                    )
                    member_id += 1

        if brace_designs and braced_bays:
            brace_design = brace_designs[story_index]
            brace_baseline_row = _get_hss_row(brace_design["baseline_section"]) or _next_heavier_hss(brace_design["section"])
            for bay in braced_bays:
                brace_count_label = f"BR-{story_index + 1:02d}-{int(bay['y']):03d}"
                brace_length = ((bay["x1"] - bay["x0"]) ** 2 + (z1 - z0) ** 2) ** 0.5
                weight_saved = max(0.0, float(brace_baseline_row["weight"]) - float(brace_design["weight"]))
                common_kwargs = dict(
                    floor_label=floor_label,
                    story_index=story_index + 1,
                    baseline_section=brace_design["baseline_section"],
                    baseline_weight=float(brace_baseline_row["weight"]),
                    baseline_utilization=float(brace_design["baseline_utilization"]),
                    optimized_section=brace_design["section"],
                    optimized_weight=float(brace_design["weight"]),
                    optimized_utilization=float(brace_design["utilization"]),
                    weight_saved_per_ft=weight_saved,
                    span_ft=brace_length,
                    tributary_ft=float(brace_design["story_shear_kips"]),
                    dead_line=0.0,
                    live_line=0.0,
                    Wu=float(brace_design["story_shear_kips"]),
                    baseline_details={"axial_ratio": float(brace_design["baseline_utilization"])},
                    optimized_details={"axial_ratio": float(brace_design["utilization"])},
                    baseline_extras={
                        "required_area": float(brace_design["required_area"]),
                        "brace_angle_deg": float(brace_design["angle_deg"]),
                        "story_shear_kips": float(brace_design["story_shear_kips"]),
                    },
                    optimized_extras={
                        "required_area": float(brace_design["required_area"]),
                        "brace_angle_deg": float(brace_design["angle_deg"]),
                        "story_shear_kips": float(brace_design["story_shear_kips"]),
                        "provided_area": float(brace_design["A"]),
                    },
                    load_summary={
                        "story_shear_kips": float(brace_design["story_shear_kips"]),
                        "brace_angle_deg": float(brace_design["angle_deg"]),
                        "required_area": float(brace_design["required_area"]),
                    },
                )
                members.append(
                    _member_template(
                        member_id=member_id,
                        member_type="brace",
                        mark=f"{brace_count_label}A",
                        x_start=bay["x0"],
                        y_start=bay["y"],
                        z_start=z0,
                        x_end=bay["x1"],
                        y_end=bay["y"],
                        z_end=z1,
                        **common_kwargs,
                    )
                )
                member_id += 1
                members.append(
                    _member_template(
                        member_id=member_id,
                        member_type="brace",
                        mark=f"{brace_count_label}B",
                        x_start=bay["x0"],
                        y_start=bay["y"],
                        z_start=z1,
                        x_end=bay["x1"],
                        y_end=bay["y"],
                        z_end=z0,
                        **common_kwargs,
                    )
                )
                member_id += 1

    before_weight = sum(member["baseline_weight"] * member["length_ft"] for member in members)
    after_weight = sum(member["optimized_weight"] * member["length_ft"] for member in members)
    saved_weight = max(0.0, before_weight - after_weight)
    savings_pct = (saved_weight / before_weight * 100.0) if before_weight > 0 else 0.0
    total_members = len(members)

    interim_building_data = {
        "num_floors": num_floors,
        "floor_height_ft": floor_height,
        "bays_x": bays_x,
        "bays_y": bays_y,
        "bay_length_ft": bay_length,
        "bay_width_ft": bay_width,
        "building_length_ft": building_length,
        "building_width_ft": building_width,
        "building_height_ft": building_height,
        "optimized": {
            "column_by_story": optimized_column_designs,
        },
    }
    if lateral_loads and lateral_recommendation:
        lateral_results = check_drift(
            interim_building_data,
            lateral_loads,
            system_type=lateral_recommendation["system_type"],
        )

    building_data = {
        "num_floors": num_floors,
        "floor_height_ft": floor_height,
        "bays_x": bays_x,
        "bays_y": bays_y,
        "bay_length_ft": bay_length,
        "bay_width_ft": bay_width,
        "beam_spacing_ft": beam_spacing,
        "composite_ratio": composite_ratio,
        "building_length_ft": building_length,
        "building_width_ft": building_width,
        "building_height_ft": building_height,
        "building_x_min": 0.0,
        "building_x_max": building_length,
        "building_center_y": building_width / 2.0,
        "dead_psf": dead_psf,
        "live_psf": live_psf,
        "members": members,
        "total_members": total_members,
        "total_weight": round(after_weight, 0),
        "baseline_total_weight": round(before_weight, 0),
        "weight_saved": round(saved_weight, 0),
        "savings_pct": round(savings_pct, 1),
        "cost_savings": round(saved_weight * 1.50, 0),
        "passes": all(bool(member["optimized_section"] != "NONE") for member in members),
        "composite": composite,
        "before_after": {
            "before_lbs": round(before_weight, 0),
            "after_lbs": round(after_weight, 0),
            "saved_lbs": round(saved_weight, 0),
            "saved_pct": round(savings_pct, 1),
            "cost_savings": round(saved_weight * 1.50, 0),
        },
        "lateral": {
            "recommendation": lateral_recommendation,
            "loads": lateral_loads,
            "results": lateral_results,
            "brace_designs": brace_designs,
            "braced_bays": braced_bays,
        },
        "baseline": {
            "beams": beam_base,
            "girder": girder_base,
            "column": column_baseline,
            "total_weight": round(before_weight, 0),
        },
        "optimized": {
            "beams": beam_opt,
            "girder": girder_opt,
            "column_by_story": optimized_column_designs,
            "total_weight": round(after_weight, 0),
        },
    }
    building_data["load_flow"] = compute_gravity_load_flow(
        building_data,
        {"dead_psf": dead_psf, "live_psf": live_psf},
    )
    building_data["lateral_flow"] = compute_lateral_load_flow(building_data)
    return building_data


def generate_3d_frame_html(
    building_data: dict[str, Any],
    width: int = 800,
    height: int = 600,
) -> str:
    """
    Generate embedded Three.js HTML for the frame model.
    """
    payload = json.dumps(building_data)
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Civil Agent 3D Frame</title>
  <style>
    body {{
      margin: 0;
      overflow: hidden;
      background: #0d1117;
      font-family: Arial, sans-serif;
      color: #e6edf3;
    }}
    #container {{
      position: relative;
      width: {width}px;
      height: {height}px;
      background: #0d1117;
    }}
    #tooltip {{
      position: absolute;
      display: none;
      pointer-events: none;
      background: rgba(13, 17, 23, 0.92);
      color: #f0f6fc;
      padding: 8px 10px;
      border: 1px solid #30363d;
      border-radius: 8px;
      font-size: 12px;
      white-space: pre-line;
      z-index: 10;
      max-width: 220px;
    }}
    #legend {{
      position: absolute;
      top: 16px;
      left: 16px;
      background: rgba(13, 17, 23, 0.88);
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 12px 14px;
      font-size: 12px;
      line-height: 1.6;
      z-index: 10;
    }}
    .legend-row {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .swatch {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
    }}
    #controls {{
      position: absolute;
      right: 16px;
      top: 16px;
      z-index: 10;
      display: flex;
      gap: 8px;
    }}
    .btn {{
      background: #415a77;
      color: white;
      border: 0;
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 12px;
      cursor: pointer;
    }}
    #summary {{
      position: absolute;
      right: 16px;
      bottom: 16px;
      z-index: 10;
      background: rgba(13, 17, 23, 0.88);
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 12px;
      line-height: 1.5;
      text-align: right;
    }}
  </style>
</head>
<body>
  <div id="container">
    <div id="tooltip"></div>
    <div id="legend">
      <div><strong>Civil Agent Legend</strong></div>
      <div class="legend-row"><span class="swatch" style="background:#2ecc71;"></span>Green: well optimized</div>
      <div class="legend-row"><span class="swatch" style="background:#f39c12;"></span>Amber: acceptable</div>
      <div class="legend-row"><span class="swatch" style="background:#e67e22;"></span>Orange: efficient</div>
      <div class="legend-row"><span class="swatch" style="background:{OVERDESIGNED_BLUE};"></span>Blue: over-designed</div>
      <div class="legend-row"><span class="swatch" style="background:#8e44ad;"></span>Purple: failing</div>
    </div>
    <div id="controls">
      <button class="btn" onclick="resetToBaseline()">Reset</button>
      <button class="btn" onclick="playOptimization()">Play Optimization</button>
    </div>
    <div id="summary"></div>
  </div>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script>
    const buildingData = {payload};
    const tooltip = document.getElementById("tooltip");
    const summary = document.getElementById("summary");
    const container = document.getElementById("container");

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d1117);

    const camera = new THREE.PerspectiveCamera(45, {width} / {height}, 0.1, 5000);
    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize({width}, {height});
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    container.appendChild(renderer.domElement);

    const ambient = new THREE.AmbientLight(0xffffff, 0.75);
    scene.add(ambient);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.85);
    dirLight.position.set(1, 2, 2);
    scene.add(dirLight);

    const buildingLength = buildingData.building_length_ft;
    const buildingWidth = buildingData.building_width_ft;
    const buildingHeight = buildingData.building_height_ft;
    const gridSize = Math.max(buildingLength, buildingWidth) + 30;
    const grid = new THREE.GridHelper(gridSize, Math.max(10, Math.round(gridSize / 10)), 0x415a77, 0x1f2937);
    grid.position.set(buildingLength / 2, 0, buildingWidth / 2);
    scene.add(grid);

    camera.position.set(buildingLength * 1.1, buildingHeight * 0.9 + 20, buildingWidth * 1.4 + 20);
    const target = new THREE.Vector3(buildingLength / 2, buildingHeight / 2, buildingWidth / 2);
    camera.lookAt(target);

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const memberMeshes = new Map();

    let autoRotate = true;
    let isDragging = false;
    let previousMouse = null;
    let spherical = new THREE.Spherical().setFromVector3(camera.position.clone().sub(target));

    function colorForUtilization(utilization, canBeLighter, beforeState) {{
      if (beforeState && canBeLighter) return "{OVERDESIGNED_BLUE}";
      if (utilization > 1.0) return "#8e44ad";
      if (utilization > 0.9) return "#e74c3c";
      if (utilization > 0.75) return "#e67e22";
      if (utilization >= 0.5) return "#f39c12";
      return "#2ecc71";
    }}

    function memberRadius(type) {{
      if (type === "column") return 0.3;
      if (type === "girder") return 0.25;
      return 0.2;
    }}

    function midpoint(member) {{
      return new THREE.Vector3(
        (member.x_start + member.x_end) / 2,
        (member.z_start + member.z_end) / 2,
        (member.y_start + member.y_end) / 2
      );
    }}

    function lengthVector(member) {{
      return new THREE.Vector3(
        member.x_end - member.x_start,
        member.z_end - member.z_start,
        member.y_end - member.y_start
      );
    }}

    function tooltipText(data) {{
      const saved = Number(data.weight_saved || 0).toFixed(1);
      const util = Number(data.optimized_utilization ?? data.utilization ?? 0).toFixed(2);
      return `${{data.section}} → ${{data.optimized_section}}\\nSaves ${{saved}} lb/ft\\nUtilization: ${{util}}`;
    }}

    function updateSummary() {{
      summary.innerHTML = `
        <div><strong>${{buildingData.num_floors}} floors, ${{buildingData.bays_x}}x${{buildingData.bays_y}} bays</strong></div>
        <div>Total members: ${{buildingData.total_members}}</div>
        <div>Optimized steel: ${{Number(buildingData.total_weight).toLocaleString()}} lb</div>
        <div>Baseline steel: ${{Number(buildingData.baseline_total_weight).toLocaleString()}} lb</div>
        <div>Savings: ${{buildingData.savings_pct.toFixed(1)}}% ($${{Number(buildingData.cost_savings).toLocaleString()}})</div>
      `;
    }}

    function addMember(member) {{
      const dir = lengthVector(member);
      const length = dir.length();
      const geometry = new THREE.CylinderGeometry(memberRadius(member.type), memberRadius(member.type), length, 12);
      const material = new THREE.MeshStandardMaterial({{
        color: colorForUtilization(member.utilization, member.can_be_lighter, true),
        metalness: 0.55,
        roughness: 0.45
      }});
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.copy(midpoint(member));

      const up = new THREE.Vector3(0, 1, 0);
      mesh.quaternion.setFromUnitVectors(up, dir.clone().normalize());
      mesh.userData = JSON.parse(JSON.stringify(member));
      scene.add(mesh);
      memberMeshes.set(member.id, mesh);
    }}

    buildingData.members.forEach(addMember);
    updateSummary();

    window.updateMember = function(memberId, newSection, newUtilization, weightSaved) {{
      const mesh = memberMeshes.get(memberId);
      if (!mesh) return;
      const data = mesh.userData;
      data.optimized_section = newSection;
      data.optimized_utilization = newUtilization;
      data.weight_saved = weightSaved;
      const finalColor = colorForUtilization(newUtilization, false, false);
      mesh.material.color.set("#ffffff");
      setTimeout(() => {{
        mesh.material.color.set(finalColor);
      }}, 220);
    }};

    window.resetToBaseline = function() {{
      buildingData.members.forEach((member) => {{
        const mesh = memberMeshes.get(member.id);
        if (!mesh) return;
        mesh.userData = JSON.parse(JSON.stringify(member));
        mesh.material.color.set(colorForUtilization(member.utilization, member.can_be_lighter, true));
      }});
    }};

    window.playOptimization = function() {{
      let delay = 0;
      buildingData.members.forEach((member) => {{
        if (!member.can_be_lighter) return;
        setTimeout(() => {{
          updateMember(
            member.id,
            member.optimized_section,
            member.optimized_utilization,
            member.weight_saved
          );
        }}, delay);
        delay += 70;
      }});
    }};

    renderer.domElement.addEventListener("mousedown", (event) => {{
      isDragging = true;
      autoRotate = false;
      previousMouse = {{ x: event.clientX, y: event.clientY }};
    }});

    window.addEventListener("mouseup", () => {{
      isDragging = false;
      previousMouse = null;
    }});

    window.addEventListener("mousemove", (event) => {{
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(Array.from(memberMeshes.values()));
      if (intersects.length > 0) {{
        const data = intersects[0].object.userData;
        tooltip.style.display = "block";
        tooltip.style.left = `${{event.clientX - rect.left + 14}}px`;
        tooltip.style.top = `${{event.clientY - rect.top + 14}}px`;
        tooltip.textContent = tooltipText(data);
      }} else {{
        tooltip.style.display = "none";
      }}

      if (!isDragging || !previousMouse) return;
      const dx = event.clientX - previousMouse.x;
      const dy = event.clientY - previousMouse.y;
      previousMouse = {{ x: event.clientX, y: event.clientY }};
      spherical.theta -= dx * 0.008;
      spherical.phi -= dy * 0.008;
      spherical.phi = Math.max(0.2, Math.min(Math.PI - 0.2, spherical.phi));
      const nextPos = new THREE.Vector3().setFromSpherical(spherical).add(target);
      camera.position.copy(nextPos);
      camera.lookAt(target);
    }});

    renderer.domElement.addEventListener("wheel", (event) => {{
      event.preventDefault();
      autoRotate = false;
      spherical.radius *= event.deltaY > 0 ? 1.06 : 0.94;
      spherical.radius = Math.max(20, Math.min(2000, spherical.radius));
      const nextPos = new THREE.Vector3().setFromSpherical(spherical).add(target);
      camera.position.copy(nextPos);
      camera.lookAt(target);
    }}, {{ passive: false }});

    function animate() {{
      requestAnimationFrame(animate);
      if (autoRotate) {{
        spherical.theta += 0.002;
        const nextPos = new THREE.Vector3().setFromSpherical(spherical).add(target);
        camera.position.copy(nextPos);
        camera.lookAt(target);
      }}
      renderer.render(scene, camera);
    }}

    animate();
  </script>
</body>
</html>
"""


def generate_3d_frame_html(
    building_data: dict[str, Any],
    width: int = 800,
    height: int = 600,
    load_mode: str = "Off",
) -> str:
    """
    Upgraded viewer with baseline/optimized color states, click details,
    optimization playback, and load-flow overlays.
    """
    payload = json.dumps({**building_data, "default_load_mode": load_mode})
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Civil Agent 3D Frame</title>
  <style>
    body {{ margin: 0; overflow: hidden; background: #0d1117; font-family: Arial, sans-serif; color: #e6edf3; }}
    #container {{ position: relative; width: {width}px; height: {height}px; background: #0d1117; }}
    #tooltip {{ position: absolute; display: none; pointer-events: none; background: rgba(13,17,23,0.94); color: #f0f6fc; padding: 8px 10px; border: 1px solid #30363d; border-radius: 8px; font-size: 12px; white-space: pre-line; z-index: 20; max-width: 240px; }}
    #legend, #statusBox, #detailPanel {{ background: rgba(13,17,23,0.90); border: 1px solid #30363d; border-radius: 10px; z-index: 10; }}
    #legend {{ position: absolute; top: 16px; left: 16px; padding: 12px 14px; font-size: 12px; line-height: 1.55; width: 190px; }}
    #statusBox {{ position: absolute; top: 16px; right: 16px; width: 290px; padding: 12px 14px; font-size: 12px; line-height: 1.55; text-align: right; }}
    #detailPanel {{ position: absolute; right: 16px; top: 195px; width: 310px; max-height: calc({height}px - 235px); overflow: auto; padding: 14px; font-size: 12px; line-height: 1.5; }}
    .legend-row {{ display: flex; align-items: center; gap: 8px; }}
    .swatch {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
    #controls {{ position: absolute; left: 220px; top: 16px; z-index: 10; display: flex; gap: 8px; }}
    #loadControls {{ position: absolute; bottom: 16px; left: 50%; transform: translateX(-50%); display: flex; gap: 8px; z-index: 12; }}
    .btn {{ background: #415a77; color: white; border: 0; border-radius: 8px; padding: 8px 10px; font-size: 12px; cursor: pointer; }}
    .detail-title {{ font-size: 15px; font-weight: 700; margin-bottom: 8px; }}
    .detail-rule {{ border-top: 1px solid #30363d; margin: 8px 0 10px; }}
    .muted {{ color: #9ba7b4; }}
    pre {{ white-space: pre-wrap; margin: 0; font-family: Consolas, monospace; }}
  </style>
</head>
<body>
  <div id="container">
    <div id="tooltip"></div>
    <div id="legend">
      <div><strong>Civil Agent Legend</strong></div>
      <div class="legend-row"><span class="swatch" style="background:{GREEN};"></span>Green: optimized</div>
      <div class="legend-row"><span class="swatch" style="background:{AMBER};"></span>Amber: acceptable</div>
      <div class="legend-row"><span class="swatch" style="background:{ORANGE};"></span>Orange: efficient / near limit</div>
      <div class="legend-row"><span class="swatch" style="background:{RED};"></span>Red: very near limit</div>
      <div class="legend-row"><span class="swatch" style="background:{OVERDESIGNED_BLUE};"></span>Blue: can be lighter</div>
      <div class="legend-row"><span class="swatch" style="background:{PURPLE};"></span>Purple: failing</div>
    </div>
    <div id="controls">
      <button class="btn" onclick="resetToBaseline()">Reset</button>
      <button class="btn" onclick="playOptimization()">Play Optimization</button>
    </div>
    <div id="loadControls">
      <button class="btn" onclick="clearLoadViz()">Members only</button>
      <button class="btn" onclick="drawGravityLoadFlow(buildingData, 'all')">Gravity loads</button>
      <button class="btn" onclick="animateLoadFlow()">Animate flow</button>
      <button class="btn" onclick="drawLateralLoadFlow(buildingData)">Wind loads</button>
    </div>
    <div id="statusBox"></div>
    <div id="detailPanel"><div class="detail-title">Member Details</div><div class="muted">Click any member to inspect baseline and optimized sections.</div></div>
  </div>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script>
    const buildingData = {payload};
    const tooltip = document.getElementById("tooltip");
    const statusBox = document.getElementById("statusBox");
    const detailPanel = document.getElementById("detailPanel");
    const container = document.getElementById("container");
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d1117);
    const camera = new THREE.PerspectiveCamera(45, {width} / {height}, 0.1, 6000);
    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize({width}, {height});
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    container.appendChild(renderer.domElement);
    scene.add(new THREE.AmbientLight(0xffffff, 0.76));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
    dirLight.position.set(2, 3, 1.5);
    scene.add(dirLight);
    const buildingLength = buildingData.building_length_ft;
    const buildingWidth = buildingData.building_width_ft;
    const buildingHeight = buildingData.building_height_ft;
    const gridSize = Math.max(buildingLength, buildingWidth) + 40;
    const grid = new THREE.GridHelper(gridSize, Math.max(10, Math.round(gridSize / 10)), 0x415a77, 0x1f2937);
    grid.position.set(buildingLength / 2, 0, buildingWidth / 2);
    scene.add(grid);
    const target = new THREE.Vector3(buildingLength / 2, buildingHeight / 2, buildingWidth / 2);
    camera.position.set(buildingLength * 1.05, buildingHeight * 0.82 + 18, buildingWidth * 1.25 + 24);
    camera.lookAt(target);
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const memberMeshes = new Map();
    const loadVizGroup = new THREE.Group();
    const memberObjects = [];
    let loadObjects = [];
    scene.add(loadVizGroup);
    let autoRotate = true;
    let isDragging = false;
    let previousMouse = null;
    let spherical = new THREE.Spherical().setFromVector3(camera.position.clone().sub(target));
    function beforeColor(member) {{
      const u = Number(member.baseline_utilization || member.utilization || 0);
      if (member.can_be_lighter || u < 0.6) return "{OVERDESIGNED_BLUE}";
      if (u > 1.0) return "{PURPLE}";
      if (u >= 0.9) return "{RED}";
      if (u >= 0.75) return "{ORANGE}";
      return "{AMBER}";
    }}
    function afterColor(member) {{
      const u = Number(member.optimized_utilization || 0);
      if (u > 1.0) return "{PURPLE}";
      if (u >= 0.97) return "{RED}";
      if (u >= 0.95) return "{ORANGE}";
      if (u >= 0.75) return "{GREEN}";
      return "{AMBER}";
    }}
    function memberRadius(type) {{ if (type === "column") return 0.3; if (type === "girder") return 0.25; if (type === "brace") return 0.16; return 0.2; }}
    function midpoint(member) {{ return new THREE.Vector3((member.x_start + member.x_end) / 2, (member.z_start + member.z_end) / 2, (member.y_start + member.y_end) / 2); }}
    function lengthVector(member) {{ return new THREE.Vector3(member.x_end - member.x_start, member.z_end - member.z_start, member.y_end - member.y_start); }}
    function tooltipText(data) {{ return `${{data.baseline_section}} -> ${{data.optimized_section}}\\nSaves ${{Number(data.weight_saved || 0).toFixed(1)}} lb/ft\\nUtilization: ${{Number(data.optimized_utilization || 0).toFixed(2)}}`; }}
    function loadTooltipText(data) {{ return data.label || data.tooltip || "Load path"; }}
    function detailHtml(data) {{
      const checks = data.optimized_details || {{}};
      const extras = data.optimized_extras || {{}};
      const load = data.load_summary || {{}};
      const optimizedLabel = extras.composite ? `${{data.optimized_section}} composite` : data.optimized_section;
      const lineA = data.type === "column"
        ? `Tributary: ${{Number(load.tributary_area_ft2 || data.tributary_ft || 0).toFixed(0)}} sf\\nPu:        ${{Number(load.Pu || 0).toFixed(1)}} kips\\nMu:        ${{Number(load.Mu || 0).toFixed(1)}} kip-ft`
        : data.type === "brace"
        ? `Story shear: ${{Number(load.story_shear_kips || data.Wu || 0).toFixed(1)}} kips\\nBrace angle: ${{Number(load.brace_angle_deg || extras.brace_angle_deg || 0).toFixed(1)}} deg\\nRequired A: ${{Number(load.required_area || extras.required_area || 0).toFixed(2)}} in^2`
        : `Span:      ${{Number(data.span_ft || 0).toFixed(1)}} ft\\nTributary: ${{Number(data.tributary_ft || 0).toFixed(1)}} ft\\nDL:        ${{Number(data.dead_line || 0).toFixed(2)}} kip/ft\\nLL:        ${{Number(data.live_line || 0).toFixed(2)}} kip/ft\\nWu:        ${{Number(data.Wu || 0).toFixed(2)}}`;
      const checkRows = data.type === "column"
        ? `Axial:      ${{Number(checks.axial_ratio || 0).toFixed(3)}}\\nP-M:        ${{Number(checks.pm_ratio || 0).toFixed(3)}}\\nFlange:     ${{Number(checks.flange_ratio || 0).toFixed(3)}}\\nWeb:        ${{Number(checks.web_ratio || 0).toFixed(3)}}`
        : data.type === "brace"
        ? `Axial:      ${{Number(checks.axial_ratio || 0).toFixed(3)}}\\nProvided A: ${{Number(extras.provided_area || 0).toFixed(2)}} in^2`
        : `Moment:     ${{Number(checks.moment_ratio || 0).toFixed(3)}}\\nShear:      ${{Number(checks.shear_ratio || 0).toFixed(3)}}\\nDeflection: ${{Number(checks.defl_ratio || checks.deflection_ratio || 0).toFixed(3)}}\\nLTB:        ${{Number(checks.ltb_ratio || checks.construction_ratio || 0).toFixed(3)}}`;
      const studLine = extras.stud_count ? `Studs: ${{extras.studs_per_side || Math.floor(extras.stud_count / 2)}} per side (${{extras.stud_count}} total)\\nIeff:  ${{Number(extras.Ieff || 0).toFixed(0)}} in^4` : "";
      return `<div class="detail-title">${{data.type.toUpperCase()}} ${{data.mark}} (${{data.floor_label}})</div><div class="detail-rule"></div><pre>Baseline:   ${{data.baseline_section}}\\nOptimized:  ${{optimizedLabel}}\\n\\n${{lineA}}\\n\\nChecks (optimized):\\n${{checkRows}}\\n\\n${{studLine ? studLine + "\\n\\n" : ""}}Savings: ${{Number(data.weight_saved || 0).toFixed(1)}} lb/ft x ${{Number(data.length_ft || 0).toFixed(1)}} ft = ${{Number(data.total_saved_lbs || 0).toFixed(0)}} lbs</pre>`;
    }}
    function updateStatus(html) {{ statusBox.innerHTML = html; }}
    function baselineStatus() {{
      const comp = buildingData.before_after;
      const lateral = buildingData.lateral || {{}};
      const rec = lateral.recommendation || {{}};
      const wind = lateral.results && lateral.results.cases ? lateral.results.cases.wind : null;
      const seismic = lateral.results && lateral.results.cases ? lateral.results.cases.seismic : null;
      updateStatus(`<div><strong>Baseline frame</strong></div><div>${{buildingData.num_floors}} floors, ${{buildingData.bays_x}}x${{buildingData.bays_y}} bays</div><div>${{rec.name || 'Gravity frame'}}</div><div class="detail-rule"></div><div>BEFORE: ${{Number(comp.before_lbs).toLocaleString()}} lbs</div><div>AFTER:  ${{Number(comp.after_lbs).toLocaleString()}} lbs</div><div>SAVED:  ${{Number(comp.saved_lbs).toLocaleString()}} lbs (-${{Number(comp.saved_pct).toFixed(1)}}%)</div><div>COST:   $${{Number(comp.cost_savings).toLocaleString()}}</div><div class="detail-rule"></div><div>Wind drift: ${{wind ? wind.roof_drift_in.toFixed(2) : '--'}} in</div><div>Seismic drift: ${{seismic ? seismic.roof_drift_in.toFixed(2) : '--'}} in</div>`);
    }}
    function finalStatus() {{
      const comp = buildingData.before_after;
      const lateral = buildingData.lateral || {{}};
      const wind = lateral.results && lateral.results.cases ? lateral.results.cases.wind : null;
      const seismic = lateral.results && lateral.results.cases ? lateral.results.cases.seismic : null;
      updateStatus(`<div><strong>Optimization complete</strong></div><div>Saved ${{Number(comp.saved_lbs).toLocaleString()}} lbs</div><div class="detail-rule"></div><div>BEFORE: ${{Number(comp.before_lbs).toLocaleString()}} lbs</div><div>AFTER:  ${{Number(comp.after_lbs).toLocaleString()}} lbs</div><div>SAVED:  ${{Number(comp.saved_lbs).toLocaleString()}} lbs (-${{Number(comp.saved_pct).toFixed(1)}}%)</div><div>COST:   $${{Number(comp.cost_savings).toLocaleString()}}</div><div class="detail-rule"></div><div>Wind drift: ${{wind ? wind.roof_drift_in.toFixed(2) : '--'}} / ${{wind ? wind.allowable_roof_drift_in.toFixed(2) : '--'}} in</div><div>Seismic drift: ${{seismic ? seismic.roof_drift_in.toFixed(2) : '--'}} / ${{seismic ? seismic.allowable_roof_drift_in.toFixed(2) : '--'}} in</div>`);
    }}
    function lerpColor(startHex, endHex, t) {{
      const a = new THREE.Color(startHex);
      const b = new THREE.Color(endHex);
      return a.lerp(b, Math.max(0, Math.min(1, t)));
    }}
    function rememberLoadObject(object) {{
      loadObjects.push(object);
      loadVizGroup.add(object);
    }}
    function clearLoadViz() {{
      while (loadVizGroup.children.length) {{
        loadVizGroup.remove(loadVizGroup.children[0]);
      }}
      loadObjects = [];
      baselineStatus();
    }}
    function drawArrow(from, to, thickness, color, label) {{
      const direction = new THREE.Vector3().subVectors(to, from).normalize();
      const length = from.distanceTo(to);
      const material = new THREE.MeshBasicMaterial({{ color: color }});
      const group = new THREE.Group();
      const shaftGeo = new THREE.CylinderGeometry(thickness, thickness, length * 0.8, 8);
      const shaft = new THREE.Mesh(shaftGeo, material);
      shaft.position.lerpVectors(from, to, 0.4);
      shaft.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
      shaft.userData = {{ label: label, isLoadObject: true }};
      group.add(shaft);
      const headGeo = new THREE.ConeGeometry(thickness * 2.0, Math.max(length * 0.2, thickness * 3.0), 8);
      const head = new THREE.Mesh(headGeo, material);
      head.position.lerpVectors(from, to, 0.9);
      head.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
      head.userData = {{ label: label, isLoadObject: true }};
      group.add(head);
      group.userData = {{ label: label, isLoadObject: true }};
      rememberLoadObject(group);
    }}
    function drawLoadBar(x, y, zTop, zBot, thickness, loadKips, label) {{
      const heightFt = Math.max(0.25, Math.abs(zTop - zBot));
      const geometry = new THREE.CylinderGeometry(thickness, thickness, heightFt, 10);
      const material = new THREE.MeshBasicMaterial({{ color: 0x2ecc71, transparent: true, opacity: 0.55 }});
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(x, (zTop + zBot) / 2, y);
      mesh.userData = {{ label: label || `Column load: ${{loadKips.toFixed(1)}} kips`, isLoadObject: true }};
      rememberLoadObject(mesh);
    }}
    function drawGravityLoadFlow(data, mode) {{
      clearLoadViz();
      const loads = data.load_flow || {{}};
      (loads.slab_panels || []).forEach((panel) => {{
        if (!['slab', 'all'].includes(mode)) return;
        const geometry = new THREE.PlaneGeometry(panel.width_ft, panel.length_ft);
        const intensity = (loads.max_load_psf || 1) > 0 ? panel.load_psf / loads.max_load_psf : 0;
        const color = lerpColor(0x1a4a7a, 0xe74c3c, intensity);
        const material = new THREE.MeshBasicMaterial({{
          color: color,
          transparent: true,
          opacity: 0.35,
          side: THREE.DoubleSide
        }});
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(panel.center_x, panel.floor_z, panel.center_y);
        mesh.rotation.x = -Math.PI / 2;
        mesh.userData = {{
          label: `Slab: ${{panel.load_psf}} psf\\nArea: ${{panel.area_sqft}} sq ft\\nLoad: ${{panel.total_load_kips}} kips`,
          isLoadObject: true
        }};
        rememberLoadObject(mesh);
      }});
      (loads.beam_arrows || []).forEach((arrow) => {{
        if (!['beam', 'all'].includes(mode)) return;
        const thickness = Math.max(0.05, arrow.load_kip_ft * 0.08);
        drawArrow(
          new THREE.Vector3(arrow.x, arrow.floor_z + 1.0, arrow.y),
          new THREE.Vector3(arrow.x, arrow.floor_z, arrow.y),
          thickness,
          0x3498db,
          arrow.label || `To beam: ${{arrow.load_kip_ft.toFixed(2)}} kip/ft`
        );
      }});
      (loads.beam_reactions || []).forEach((rxn) => {{
        if (!['girder', 'all'].includes(mode)) return;
        const thickness = Math.max(0.08, rxn.reaction_kips * 0.01);
        drawArrow(
          new THREE.Vector3(rxn.from_x, rxn.z, rxn.from_y),
          new THREE.Vector3(rxn.to_x, rxn.z, rxn.to_y),
          thickness,
          0x2ecc71,
          `Beam reaction: ${{rxn.reaction_kips.toFixed(1)}} kips`
        );
      }});
      (loads.column_loads || []).forEach((col) => {{
        if (!['column', 'all'].includes(mode)) return;
        (col.floors || []).forEach((floor) => {{
          const thickness = Math.max(0.15, floor.cumulative_kips * 0.002);
          drawLoadBar(
            col.x,
            col.y,
            floor.z_top,
            floor.z_bot,
            thickness,
            floor.cumulative_kips,
            `Column load at floor ${{floor.num}}: ${{floor.cumulative_kips.toFixed(0)}} kips`
          );
        }});
      }});
      updateStatus(`<div><strong>Gravity load flow</strong></div><div>Mode: ${{mode}}</div><div class="detail-rule"></div><div>Arrow thickness scales with load magnitude.</div>`);
    }}
    function drawLateralLoadFlow(data) {{
      clearLoadViz();
      const lateral = data.lateral_flow || {{}};
      (lateral.wind_panels || []).forEach((panel) => {{
        const nArrows = Math.max(1, Math.ceil(panel.height_ft / 5.0));
        for (let i = 0; i < nArrows; i += 1) {{
          const z = panel.z_bot + (i + 0.5) * panel.height_ft / nArrows;
          drawArrow(
            new THREE.Vector3(panel.x - 15, z, panel.y_center),
            new THREE.Vector3(panel.x, z, panel.y_center),
            Math.max(0.1, panel.pressure_psf * 0.003),
            0xf39c12,
            `Wind: ${{panel.pressure_psf.toFixed(1)}} psf`
          );
        }}
      }});
      (lateral.story_shears || []).forEach((story) => {{
        drawArrow(
          new THREE.Vector3(lateral.building_x_min || 0, story.z, lateral.building_center_y || 0),
          new THREE.Vector3(lateral.building_x_max || buildingLength, story.z, lateral.building_center_y || 0),
          Math.max(0.2, story.shear_kips * 0.003),
          0xe74c3c,
          `Story shear: ${{story.shear_kips.toFixed(0)}} kips`
        );
      }});
      updateStatus(`<div><strong>Lateral load flow</strong></div><div>Wind pressure and cumulative story shear shown.</div>`);
    }}
    function animateLoadFlow() {{
      const steps = [
        {{ name: 'Floor loads', duration: 1400, fn: () => drawGravityLoadFlow(buildingData, 'slab') }},
        {{ name: 'Into beams', duration: 1400, fn: () => drawGravityLoadFlow(buildingData, 'beam') }},
        {{ name: 'Into girders', duration: 1400, fn: () => drawGravityLoadFlow(buildingData, 'girder') }},
        {{ name: 'Into columns', duration: 1400, fn: () => drawGravityLoadFlow(buildingData, 'column') }},
        {{ name: 'Full path', duration: 1600, fn: () => drawGravityLoadFlow(buildingData, 'all') }},
      ];
      let delay = 0;
      steps.forEach((step) => {{
        setTimeout(() => {{
          clearLoadViz();
          step.fn();
          updateStatus(`<div><strong>Load flow animation</strong></div><div>${{step.name}}</div>`);
        }}, delay);
        delay += step.duration;
      }});
    }}
    function addMember(member) {{
      const dir = lengthVector(member);
      const geometry = new THREE.CylinderGeometry(memberRadius(member.type), memberRadius(member.type), dir.length(), 12);
      const material = new THREE.MeshStandardMaterial({{ color: beforeColor(member), metalness: 0.55, roughness: 0.45 }});
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.copy(midpoint(member));
      mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
      mesh.userData = JSON.parse(JSON.stringify(member));
      scene.add(mesh);
      memberMeshes.set(member.id, mesh);
      memberObjects.push(mesh);
    }}
    buildingData.members.forEach(addMember);
    baselineStatus();
    window.updateMember = function(memberId, newSection, newUtilization, weightSaved) {{
      const mesh = memberMeshes.get(memberId);
      if (!mesh) return;
      mesh.userData.optimized_section = newSection;
      mesh.userData.optimized_utilization = newUtilization;
      mesh.userData.weight_saved = weightSaved;
      mesh.material.color.set("#ffffff");
      setTimeout(() => {{ mesh.material.color.set(afterColor(mesh.userData)); }}, 200);
    }};
    window.resetToBaseline = function() {{
      detailPanel.innerHTML = '<div class="detail-title">Member Details</div><div class="muted">Click any member to inspect baseline and optimized sections.</div>';
      buildingData.members.forEach((member) => {{
        const mesh = memberMeshes.get(member.id);
        if (!mesh) return;
        mesh.userData = JSON.parse(JSON.stringify(member));
        mesh.material.color.set(beforeColor(member));
      }});
      baselineStatus();
    }};
    window.playOptimization = function() {{
      window.resetToBaseline();
      const ordered = [...buildingData.members].sort((a, b) => a.story_index - b.story_index || a.id - b.id);
      let savedSoFar = 0;
      ordered.forEach((member, index) => {{
        setTimeout(() => {{
          const mesh = memberMeshes.get(member.id);
          if (!mesh) return;
          mesh.material.color.set("#ffffff");
          setTimeout(() => {{ mesh.material.color.set(afterColor(member)); }}, 200);
          savedSoFar += Number(member.total_saved_lbs || 0);
          updateStatus(`<div><strong>Optimizing member ${{index + 1}}/${{ordered.length}}...</strong></div><div>Steel saved so far: ${{Math.round(savedSoFar).toLocaleString()}} lbs</div><div class="detail-rule"></div><div>BEFORE: ${{Number(buildingData.before_after.before_lbs).toLocaleString()}} lbs</div><div>AFTER:  ${{Number(buildingData.before_after.after_lbs).toLocaleString()}} lbs</div>`);
          if (index === ordered.length - 1) setTimeout(finalStatus, 250);
        }}, index * 150);
      }});
    }};
    function pickMember(event) {{
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      return raycaster.intersectObjects([...memberObjects, ...loadObjects], true);
    }}
    renderer.domElement.addEventListener("click", (event) => {{
      const intersects = pickMember(event);
      if (!intersects.length) return;
      const picked = intersects[0].object.userData && intersects[0].object.userData.type
        ? intersects[0].object.userData
        : (intersects[0].object.parent && intersects[0].object.parent.userData ? intersects[0].object.parent.userData : null);
      if (picked && picked.type) {{
        detailPanel.innerHTML = detailHtml(picked);
      }}
    }});
    renderer.domElement.addEventListener("mousedown", (event) => {{ isDragging = true; autoRotate = false; previousMouse = {{ x: event.clientX, y: event.clientY }}; }});
    window.addEventListener("mouseup", () => {{ isDragging = false; previousMouse = null; }});
    window.addEventListener("mousemove", (event) => {{
      const rect = renderer.domElement.getBoundingClientRect();
      const intersects = pickMember(event);
      if (intersects.length > 0) {{
        const data = intersects[0].object.userData && Object.keys(intersects[0].object.userData).length
          ? intersects[0].object.userData
          : (intersects[0].object.parent && intersects[0].object.parent.userData ? intersects[0].object.parent.userData : {{}});
        tooltip.style.display = "block";
        tooltip.style.left = `${{event.clientX - rect.left + 14}}px`;
        tooltip.style.top = `${{event.clientY - rect.top + 14}}px`;
        tooltip.textContent = data.isLoadObject ? loadTooltipText(data) : tooltipText(data);
      }} else {{
        tooltip.style.display = "none";
      }}
      if (!isDragging || !previousMouse) return;
      const dx = event.clientX - previousMouse.x;
      const dy = event.clientY - previousMouse.y;
      previousMouse = {{ x: event.clientX, y: event.clientY }};
      spherical.theta -= dx * 0.008;
      spherical.phi -= dy * 0.008;
      spherical.phi = Math.max(0.2, Math.min(Math.PI - 0.2, spherical.phi));
      const nextPos = new THREE.Vector3().setFromSpherical(spherical).add(target);
      camera.position.copy(nextPos);
      camera.lookAt(target);
    }});
    renderer.domElement.addEventListener("wheel", (event) => {{
      event.preventDefault();
      autoRotate = false;
      spherical.radius *= event.deltaY > 0 ? 1.06 : 0.94;
      spherical.radius = Math.max(20, Math.min(4000, spherical.radius));
      const nextPos = new THREE.Vector3().setFromSpherical(spherical).add(target);
      camera.position.copy(nextPos);
      camera.lookAt(target);
    }}, {{ passive: false }});
    function animate() {{
      requestAnimationFrame(animate);
      if (autoRotate) {{
        spherical.theta += 0.002;
        const nextPos = new THREE.Vector3().setFromSpherical(spherical).add(target);
        camera.position.copy(nextPos);
        camera.lookAt(target);
      }}
      renderer.render(scene, camera);
    }}
    const defaultMode = buildingData.default_load_mode || "Off";
    if (defaultMode === "Gravity (static)") {{
      drawGravityLoadFlow(buildingData, "all");
    }} else if (defaultMode === "Gravity (animated)") {{
      animateLoadFlow();
    }} else if (defaultMode === "Lateral (wind)") {{
      drawLateralLoadFlow(buildingData);
    }}
    animate();
  </script>
</body>
</html>
"""
