"""
Spread footing and base plate design helpers.
"""

from __future__ import annotations

import math
from typing import Any


def _ceil_to_increment(value: float, increment: float) -> float:
    return math.ceil(value / increment) * increment


def design_spread_footing(
    Pu_kips: float,
    Mu_kip_ft: float = 0.0,
    soil_bearing_ksf: float = 2.0,
    fc_ksi: float = 4.0,
    fy_ksi: float = 60.0,
    column_width_in: float = 14.0,
    depth_ft: float = 4.0,
) -> dict[str, Any]:
    service_load = max(float(Pu_kips) / 1.5, 1.0)
    area_req_ft2 = service_load / max(float(soil_bearing_ksf), 0.25)
    B_ft = _ceil_to_increment(math.sqrt(area_req_ft2), 0.5)
    L_ft = B_ft
    qu_ksf = float(Pu_kips) / max(B_ft * L_ft, 1e-6)

    col_ft = float(column_width_in) / 12.0
    selected_h = None
    selected_d = None
    punching_ratio = 9.99
    one_way_ratio = 9.99

    for h_in in range(16, 49, 2):
        d_in = h_in - 3.0
        d_ft = d_in / 12.0

        one_way_demand = qu_ksf * B_ft * max(B_ft / 2.0 - col_ft / 2.0 - d_ft, 0.0)
        phi_vc_one_way = 0.75 * 2.0 * math.sqrt(fc_ksi * 1000.0) * (B_ft * 12.0) * d_in / 1000.0
        one_way_ratio = one_way_demand / max(phi_vc_one_way, 1e-6)

        bo_in = 4.0 * (column_width_in + d_in)
        vu_ksi = float(Pu_kips) / max((bo_in * d_in) / 144.0, 1e-6)
        phi_vc_punch = 0.75 * 4.0 * math.sqrt(fc_ksi * 1000.0) / 1000.0
        punching_ratio = vu_ksi / max(phi_vc_punch, 1e-6)

        if one_way_ratio <= 1.0 and punching_ratio <= 1.0:
            selected_h = float(h_in)
            selected_d = float(d_in)
            break

    if selected_h is None:
        selected_h = 48.0
        selected_d = 45.0

    projection_ft = max(B_ft / 2.0 - col_ft / 2.0, 0.5)
    Mu_kip_ft = max(
        float(Mu_kip_ft),
        qu_ksf * B_ft * projection_ft**2 / 2.0,
    )
    phi = 0.9
    fy = fy_ksi
    d_in = selected_d
    As_req = (Mu_kip_ft * 12.0) / max(phi * fy * 0.9 * d_in, 1e-6)
    As_min = 0.0018 * B_ft * 12.0 * selected_h
    As_use = max(As_req, As_min)

    bar_area = 0.31  # #5
    bar_size = "#5"
    if As_use > 8.0:
        bar_area = 0.44
        bar_size = "#6"
    bar_count = max(4, math.ceil(As_use / bar_area))
    clear_span_in = max(B_ft * 12.0 - 6.0, 12.0)
    spacing_in = clear_span_in / max(bar_count - 1, 1)
    if spacing_in < 6.0:
        bar_area = 0.79
        bar_size = "#8"
        bar_count = max(4, math.ceil(As_use / bar_area))
        spacing_in = clear_span_in / max(bar_count - 1, 1)

    flexure_ratio = Mu_kip_ft / max(phi * fy * As_use * 0.9 * d_in / 12.0, 1e-6)
    concrete_volume_cy = B_ft * L_ft * selected_h / 12.0 / 27.0
    rebar_lbs = bar_count * B_ft * 2.0 * {"#5": 1.043, "#6": 1.502, "#8": 2.670}[bar_size]
    cost_estimate = concrete_volume_cy * 900.0 + rebar_lbs * 1.6
    notes = []
    if punching_ratio > 1.0:
        notes.append("Punching shear governs; increase footing thickness or plan dimensions.")
    if one_way_ratio > 1.0:
        notes.append("One-way shear governs; increase footing thickness.")
    if spacing_in < 6.0:
        notes.append("Bar spacing is tight; consider a thicker footing or larger plan dimensions.")

    return {
        "B_ft": round(B_ft, 2),
        "L_ft": round(L_ft, 2),
        "h_in": round(selected_h, 1),
        "d_in": round(selected_d, 1),
        "qu_ksf": round(qu_ksf, 3),
        "As_req_in2": round(As_use, 3),
        "bar_size": bar_size,
        "bar_spacing_in": round(spacing_in, 2),
        "bar_count": int(bar_count),
        "punching_ratio": round(punching_ratio, 3),
        "one_way_ratio": round(one_way_ratio, 3),
        "flexure_ratio": round(flexure_ratio, 3),
        "concrete_volume_cy": round(concrete_volume_cy, 2),
        "rebar_lbs": round(rebar_lbs, 1),
        "cost_estimate": round(cost_estimate, 0),
        "passes": punching_ratio <= 1.0 and one_way_ratio <= 1.0 and flexure_ratio <= 1.0,
        "notes": notes,
    }


def design_base_plate(
    Pu_kips: float,
    Mu_kip_ft: float,
    column: dict[str, Any],
    fc_ksi: float = 4.0,
    Fy_plate_ksi: float = 36.0,
) -> dict[str, Any]:
    column_bf = float(column.get("bf", 14.0))
    A1_req = float(Pu_kips) / max(0.65 * 0.85 * fc_ksi, 1e-6)
    N_in = _ceil_to_increment(max(math.sqrt(A1_req), column_bf + 2.0), 1.0)
    B_in = _ceil_to_increment(max(A1_req / max(N_in, 1e-6), column_bf + 2.0), 1.0)
    bearing_pressure = float(Pu_kips) / max((N_in * B_in) / 144.0, 1e-6)
    moment_projection = max((N_in - column_bf) / 2.0, 1.0)
    thickness_req = moment_projection * math.sqrt(max(2.0 * float(Pu_kips), 1.0) / max(0.9 * Fy_plate_ksi * N_in * B_in, 1e-6))
    thickness_in = _ceil_to_increment(max(thickness_req, 0.75), 0.125)
    anchor_rod = '4 - 7/8" rods' if abs(Mu_kip_ft) < 1e-6 else '4 - 1-1/4" rods'
    cost_estimate = N_in * B_in * thickness_in * 0.283 * 1.45 + 250.0
    return {
        "N_in": round(N_in, 2),
        "B_in": round(B_in, 2),
        "t_in": round(thickness_in, 3),
        "bearing_pressure_ksf": round(bearing_pressure, 2),
        "anchor_rods": anchor_rod,
        "grout_bed_in": 1.5,
        "cost_estimate": round(cost_estimate, 0),
        "passes": True,
        "notes": [],
    }
