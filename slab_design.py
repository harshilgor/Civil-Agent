"""
Concrete slab on metal deck design helpers.

These routines are schematic-level checks intended to close the gravity-system
loop in Civil Agent. They are not a substitute for SDI catalog verification or
project-specific slab design by a licensed engineer.
"""

from __future__ import annotations

import math
from typing import Any


DECK_PROFILES = {
    "1.5B": {"height_in": 1.5, "weight_psf": 2.0, "max_unshored_span_ft": 10.0},
    "2B": {"height_in": 2.0, "weight_psf": 2.2, "max_unshored_span_ft": 11.0},
    "3B": {"height_in": 3.0, "weight_psf": 2.4, "max_unshored_span_ft": 12.0},
    "3W": {"height_in": 3.0, "weight_psf": 2.6, "max_unshored_span_ft": 14.0},
}

WWF_OPTIONS = [
    {"designation": "6x6-W1.4xW1.4", "area_in2_per_ft": 0.028},
    {"designation": "6x6-W2.1xW2.1", "area_in2_per_ft": 0.042},
    {"designation": "6x6-W2.9xW2.9", "area_in2_per_ft": 0.058},
    {"designation": "6x6-W4.0xW4.0", "area_in2_per_ft": 0.080},
    {"designation": "4x4-W1.4xW1.4", "area_in2_per_ft": 0.042},
    {"designation": "4x4-W2.1xW2.1", "area_in2_per_ft": 0.063},
]


def check_deck_span(
    beam_spacing_ft: float,
    deck_type: str = "3B",
    shored: bool = False,
) -> dict[str, Any]:
    profile = DECK_PROFILES[deck_type]
    max_span = float(profile["max_unshored_span_ft"])
    if shored:
        max_span *= 1.35
    passes = float(beam_spacing_ft) <= max_span
    note = (
        f"{deck_type} deck span of {beam_spacing_ft:.1f} ft is within the "
        f"{'shored' if shored else 'unshored'} construction limit of {max_span:.1f} ft."
        if passes
        else f"{deck_type} deck span of {beam_spacing_ft:.1f} ft exceeds the "
        f"{'shored' if shored else 'unshored'} construction limit of {max_span:.1f} ft."
    )
    return {
        "passes": passes,
        "max_span_ft": round(max_span, 2),
        "actual_span_ft": round(float(beam_spacing_ft), 2),
        "note": note,
    }


def get_fire_rating(
    total_thickness_in: float,
    concrete_above_deck_in: float,
    deck_type: str,
) -> dict[str, Any]:
    above = float(concrete_above_deck_in)
    if above >= 4.0:
        rating = 3.0
        min_cover = 4.0
    elif above >= 3.0:
        rating = 2.0
        min_cover = 3.0
    elif above >= 2.5:
        rating = 1.5
        min_cover = 2.5
    elif above >= 2.0:
        rating = 1.0
        min_cover = 2.0
    else:
        rating = 0.5
        min_cover = 2.0
    return {
        "fire_rating_hrs": rating,
        "min_cover_in": min_cover,
        "deck_type": deck_type,
        "total_thickness_in": round(float(total_thickness_in), 2),
    }


def _phi_mn_for_width(As_in2_per_ft: float, d_in: float, fc_ksi: float, fy_ksi: float) -> float:
    """
    Return design flexural capacity per foot width in lb-in.
    """
    b = 12.0
    fc_psi = fc_ksi * 1000.0
    fy_psi = fy_ksi * 1000.0
    a = As_in2_per_ft * fy_psi / max(0.85 * fc_psi * b, 1e-6)
    Mn = As_in2_per_ft * fy_psi * max(d_in - a / 2.0, 0.0)
    return 0.90 * Mn


def design_composite_slab(
    beam_spacing_ft: float,
    total_slab_thickness_in: float,
    dead_load_psf: float,
    live_load_psf: float,
    deck_type: str = "3B",
    fc_ksi: float = 4.0,
    fy_rebar_ksi: float = 60.0,
) -> dict[str, Any]:
    profile = DECK_PROFILES[deck_type]
    deck_height_in = float(profile["height_in"])
    total_thickness_in = float(total_slab_thickness_in)
    concrete_above_deck = total_thickness_in - deck_height_in
    self_weight_psf = total_thickness_in * 150.0 / 12.0 + float(profile["weight_psf"])
    wu_psf = 1.2 * (float(dead_load_psf) + self_weight_psf) + 1.6 * float(live_load_psf)
    span_ft = float(beam_spacing_ft)
    Mu_kip_ft_per_ft = wu_psf * span_ft**2 / 8000.0
    Mu_lb_in = Mu_kip_ft_per_ft * 12000.0

    cover_in = 0.75
    bar_radius_in = 0.25
    d_in = max(total_thickness_in - cover_in - bar_radius_in, 1.0)
    thickness_min = deck_height_in + 3.5
    passes_thickness = total_thickness_in >= thickness_min

    As_min = max(0.0018 * 12.0 * total_thickness_in, 0.0014 * 12.0 * total_thickness_in)
    selected_wwf = None
    phi_mn = 0.0
    for option in WWF_OPTIONS:
        area = max(float(option["area_in2_per_ft"]), As_min)
        capacity = _phi_mn_for_width(area, d_in, fc_ksi, fy_rebar_ksi)
        if area >= As_min and capacity >= Mu_lb_in:
            selected_wwf = option
            phi_mn = capacity
            break
    if selected_wwf is None:
        selected_wwf = WWF_OPTIONS[-1]
        phi_mn = _phi_mn_for_width(float(selected_wwf["area_in2_per_ft"]), d_in, fc_ksi, fy_rebar_ksi)

    deck_span = check_deck_span(span_ft, deck_type=deck_type, shored=False)
    fire = get_fire_rating(total_thickness_in, concrete_above_deck, deck_type)
    passes_flexure = phi_mn >= Mu_lb_in
    passes_fire = fire["fire_rating_hrs"] >= 1.0

    notes = []
    if not passes_thickness:
        notes.append(
            f"Increase total slab thickness to at least {thickness_min:.1f} in to provide 3.5 in of concrete above the deck ribs."
        )
    if not deck_span["passes"]:
        notes.append(deck_span["note"])
    if not passes_flexure:
        notes.append("Selected WWF is insufficient for the one-way slab flexural demand; increase slab thickness or add reinforcement.")
    if fire["fire_rating_hrs"] < 2.0:
        notes.append("Concrete above deck is below the typical 2-hour fire-rating threshold.")

    return {
        "deck_type": deck_type,
        "total_thickness_in": round(total_thickness_in, 2),
        "concrete_above_deck": round(concrete_above_deck, 2),
        "self_weight_psf": round(self_weight_psf, 2),
        "wu_psf": round(wu_psf, 2),
        "Mu_kip_ft_per_ft": round(Mu_kip_ft_per_ft, 3),
        "As_req_in2_per_ft": round(max(As_min, float(selected_wwf["area_in2_per_ft"])), 3),
        "WWF_designation": selected_wwf["designation"],
        "rebar_size": "WWF",
        "fire_rating_hrs": fire["fire_rating_hrs"],
        "deck_span_check": deck_span,
        "passes_thickness": passes_thickness,
        "passes_flexure": passes_flexure,
        "passes_fire": passes_fire,
        "passes": passes_thickness and passes_flexure and passes_fire and deck_span["passes"],
        "notes": notes,
    }
