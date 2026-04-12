"""
Load-flow data builders for 3D visualization.
"""

from __future__ import annotations

from typing import Any


def compute_gravity_load_flow(building_data: dict[str, Any], brief: dict[str, Any]) -> dict[str, Any]:
    """
    Compute slab, beam, girder, and column gravity load-path data.
    """
    bays_x = int(building_data["bays_x"])
    bays_y = int(building_data["bays_y"])
    bay_l = float(building_data["bay_length_ft"])
    bay_w = float(building_data["bay_width_ft"])
    spacing = float(building_data.get("beam_spacing_ft", 10.0) or 10.0)
    num_floors = int(building_data["num_floors"])
    floor_h = float(building_data["floor_height_ft"])

    dead_psf = float(brief["dead_psf"])
    live_psf = float(brief["live_psf"])
    total_psf = dead_psf + live_psf
    wu_psf = 1.2 * dead_psf + 1.6 * live_psf

    slab_panels = []
    beam_arrows = []
    beam_reactions = []
    column_loads = []

    n_strips = max(1, int(round(bay_l / spacing)))
    strip_w = bay_l / n_strips
    beam_line_load = total_psf * spacing / 1000.0
    beam_factored_line = wu_psf * spacing / 1000.0
    beam_end_reaction = beam_factored_line * bay_w / 2.0
    Pu_floor = wu_psf * bay_l * bay_w / 1000.0

    for floor in range(num_floors):
        z = (floor + 1) * floor_h
        for bx in range(bays_x):
            for by in range(bays_y):
                x_start = bx * bay_l
                y_start = by * bay_w

                for strip in range(n_strips):
                    panel_x = x_start + strip * strip_w
                    panel_w = strip_w
                    panel_l = bay_w
                    trib_area = panel_w * panel_l
                    total_load = total_psf * trib_area / 1000.0
                    arrow_load = total_psf * panel_w / 1000.0

                    slab_panels.append(
                        {
                            "center_x": round(panel_x + panel_w / 2.0, 3),
                            "center_y": round(y_start + panel_l / 2.0, 3),
                            "floor_z": round(z, 3),
                            "width_ft": round(panel_w, 3),
                            "length_ft": round(panel_l, 3),
                            "load_psf": round(total_psf, 2),
                            "area_sqft": round(trib_area, 2),
                            "total_load_kips": round(total_load, 3),
                        }
                    )
                    beam_arrows.append(
                        {
                            "x": round(panel_x + panel_w / 2.0, 3),
                            "y": round(y_start + panel_l / 2.0, 3),
                            "floor_z": round(z, 3),
                            "load_kip_ft": round(arrow_load, 3),
                            "label": f"Slab strip load: {arrow_load:.2f} kip/ft",
                        }
                    )

                n_beams = max(1, int(bay_l / spacing) - 1)
                for beam_index in range(n_beams):
                    beam_x = x_start + (beam_index + 1) * spacing
                    beam_reactions.append(
                        {
                            "from_x": round(beam_x, 3),
                            "from_y": round(y_start + bay_w / 2.0, 3),
                            "to_x": round(beam_x, 3),
                            "to_y": round(y_start, 3),
                            "z": round(z, 3),
                            "reaction_kips": round(beam_end_reaction, 3),
                            "direction": "to_girder",
                        }
                    )
                    beam_reactions.append(
                        {
                            "from_x": round(beam_x, 3),
                            "from_y": round(y_start + bay_w / 2.0, 3),
                            "to_x": round(beam_x, 3),
                            "to_y": round(y_start + bay_w, 3),
                            "z": round(z, 3),
                            "reaction_kips": round(beam_end_reaction, 3),
                            "direction": "to_girder",
                        }
                    )

    for col_x in range(bays_x + 1):
        for col_y in range(bays_y + 1):
            floors_data = []
            cumulative = 0.0
            for floor in range(num_floors, 0, -1):
                cumulative += Pu_floor
                floors_data.append(
                    {
                        "num": floor,
                        "z_top": round(floor * floor_h, 3),
                        "z_bot": round((floor - 1) * floor_h, 3),
                        "added_kips": round(Pu_floor, 3),
                        "cumulative_kips": round(cumulative, 3),
                    }
                )
            column_loads.append(
                {
                    "x": round(col_x * bay_l, 3),
                    "y": round(col_y * bay_w, 3),
                    "floors": floors_data,
                    "total_kips": round(cumulative, 3),
                }
            )

    return {
        "slab_panels": slab_panels,
        "beam_arrows": beam_arrows,
        "beam_reactions": beam_reactions,
        "column_loads": column_loads,
        "max_load_psf": round(total_psf, 2),
        "max_col_load": round(max((col["total_kips"] for col in column_loads), default=0.0), 2),
        "beam_line_load_kip_ft": round(beam_line_load, 3),
    }


def compute_lateral_load_flow(building_data: dict[str, Any]) -> dict[str, Any]:
    """
    Build simple wind-load visualization data from existing hazard/lateral summaries.
    """
    hazards = building_data.get("hazards") or {}
    wind = hazards.get("wind") or {}
    floor_pressures = wind.get("floor_pressures") or []

    building_length = float(building_data.get("building_length_ft", 0.0))
    building_width = float(building_data.get("building_width_ft", 0.0))
    center_y = building_width / 2.0
    x_face = 0.0

    wind_panels = []
    story_shears = []
    cumulative = 0.0

    force_lookup = {}
    lateral = building_data.get("lateral") or {}
    wind_forces = ((lateral.get("loads") or {}).get("wind") or {}).get("floor_forces_kips") or []
    for idx, force in enumerate(wind_forces, start=1):
        force_lookup[idx] = float(force)

    for item in floor_pressures:
        floor = int(item.get("floor", 0))
        z_top = float(item.get("elevation_ft", 0.0))
        z_bot = max(0.0, z_top - float(building_data.get("floor_height_ft", 14.0)))
        pressure = float(item.get("pressure_psf", 0.0))
        wind_panels.append(
            {
                "floor": floor,
                "x": x_face,
                "y_center": center_y,
                "z_bot": round(z_bot, 3),
                "height_ft": round(z_top - z_bot, 3),
                "pressure_psf": round(pressure, 2),
            }
        )

    for item in reversed(floor_pressures):
        floor = int(item.get("floor", 0))
        cumulative += force_lookup.get(floor, 0.0)
        story_shears.append(
            {
                "floor": floor,
                "z": round(float(item.get("elevation_ft", 0.0)), 3),
                "shear_kips": round(cumulative, 2),
            }
        )
    story_shears.reverse()

    return {
        "wind_panels": wind_panels,
        "story_shears": story_shears,
        "building_x_min": 0.0,
        "building_x_max": building_length,
        "building_center_y": center_y,
    }
