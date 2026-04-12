"""
Generative layout engine for Civil Agent.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from asce7_loads import auto_load_brief
from architectural_constraints import evaluate_layout_constraints, sanitize_constraint_zones
from asce7_loads import list_supported_cities
from beams_data import BEAMS_DF
from floor_system import FloorSystem

SUPPORTED_CITIES = list_supported_cities()

OCCUPANCY_LOADS = {
    "office": {"dead_psf": 50.0, "live_psf": 50.0},
    "retail": {"dead_psf": 60.0, "live_psf": 100.0},
    "warehouse": {"dead_psf": 80.0, "live_psf": 125.0},
    "residential": {"dead_psf": 40.0, "live_psf": 40.0},
    "hospital": {"dead_psf": 80.0, "live_psf": 80.0},
}

PRIORITY_LABELS = {
    "Minimum total cost": "low_cost",
    "Minimum steel weight": "min_steel",
    "Shallowest floor depth": "shallow_depth",
    "Fewest columns (open plan)": "few_columns",
    "Best balance": "balanced",
}

PRIORITY_WEIGHTS = {
    "low_cost": {"cost": 0.35, "steel": 0.15, "depth": 0.10, "cols": 0.15, "rentable": 0.10, "simplicity": 0.15},
    "min_steel": {"cost": 0.15, "steel": 0.40, "depth": 0.10, "cols": 0.10, "rentable": 0.10, "simplicity": 0.15},
    "shallow_depth": {"cost": 0.15, "steel": 0.10, "depth": 0.40, "cols": 0.10, "rentable": 0.05, "simplicity": 0.20},
    "few_columns": {"cost": 0.10, "steel": 0.10, "depth": 0.05, "cols": 0.45, "rentable": 0.10, "simplicity": 0.20},
    "balanced": {"cost": 0.10, "steel": 0.10, "depth": 0.05, "cols": 0.35, "rentable": 0.10, "simplicity": 0.30},
}


def _occupancy_defaults(occupancy: str) -> dict[str, float]:
    return OCCUPANCY_LOADS.get((occupancy or "office").lower(), OCCUPANCY_LOADS["office"]).copy()


def build_brief(
    *,
    length_ft: float,
    width_ft: float,
    num_floors: int,
    occupancy: str,
    city: str,
    priority: str = "balanced",
    max_span_ft: float = 40.0,
    min_span_ft: float = 15.0,
    floor_height_ft: float = 14.0,
    allow_interior_cols: bool = True,
    dead_psf: float | None = None,
    live_psf: float | None = None,
    max_aspect_ratio: float = 2.0,
    composite: bool = True,
    architectural_constraints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    defaults = _occupancy_defaults(occupancy)
    brief = {
        "length_ft": float(length_ft),
        "width_ft": float(width_ft),
        "num_floors": int(num_floors),
        "floor_height_ft": float(floor_height_ft),
        "occupancy": (occupancy or "office").lower(),
        "city": city if city in SUPPORTED_CITIES else "Chicago",
        "priority": priority if priority in PRIORITY_WEIGHTS else "balanced",
        "max_span_ft": float(max_span_ft),
        "min_span_ft": float(min_span_ft),
        "max_aspect_ratio": float(max_aspect_ratio),
        "allow_interior_cols": bool(allow_interior_cols),
        "open_perimeter": False,
        "no_transfer_beams": True,
        "dead_psf": float(defaults["dead_psf"] if dead_psf is None else dead_psf),
        "live_psf": float(defaults["live_psf"] if live_psf is None else live_psf),
        "composite": bool(composite),
        "architectural_constraints": sanitize_constraint_zones(
            architectural_constraints,
            length_ft=float(length_ft),
            width_ft=float(width_ft),
        ),
    }

    try:
        auto_loads = auto_load_brief(
            city=brief["city"],
            num_floors=brief["num_floors"],
            floor_height_ft=brief["floor_height_ft"],
            bay_length_ft=min(float(length_ft), max(float(width_ft), 1.0)),
            bay_width_ft=max(1.0, min(float(width_ft), float(length_ft))),
            occupancy=brief["occupancy"],
        )
        brief.update(auto_loads)
        if dead_psf is not None:
            brief["dead_psf"] = float(dead_psf)
        if live_psf is not None:
            brief["live_psf"] = float(live_psf)
    except Exception:
        brief["loads_auto_calculated"] = False

    return brief


def parse_design_brief(user_text: str) -> dict[str, Any]:
    """
    Extract a generative-design brief from natural language.
    Uses Claude when available and falls back to local rules.
    """
    fallback = _parse_design_brief_fallback(user_text)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return fallback

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=700,
            system=(
                "Extract building design parameters from the user's description. "
                "Return ONLY a JSON object with fields: "
                "length_ft, width_ft, num_floors, floor_height_ft, occupancy, city, "
                "priority, max_span_ft, min_span_ft, allow_interior_cols, dead_psf, live_psf. "
                "Priority must be one of: low_cost, min_steel, shallow_depth, few_columns, balanced. "
                "If not stated use defaults: floor_height_ft=14, max_span_ft=40, min_span_ft=15, "
                "allow_interior_cols=true. Infer priority from language like open plan/few columns, "
                "minimum cost, minimum steel, shallow depth. Occupancy must be one of office, retail, "
                "warehouse, residential, hospital."
            ),
            messages=[{"role": "user", "content": user_text}],
        )
        raw = re.sub(r"```json|```", "", message.content[0].text).strip()
        parsed = json.loads(raw)
        return build_brief(
            length_ft=float(parsed.get("length_ft", fallback["length_ft"])),
            width_ft=float(parsed.get("width_ft", fallback["width_ft"])),
            num_floors=int(parsed.get("num_floors", fallback["num_floors"])),
            floor_height_ft=float(parsed.get("floor_height_ft", fallback["floor_height_ft"])),
            occupancy=str(parsed.get("occupancy", fallback["occupancy"])).lower(),
            city=str(parsed.get("city", fallback["city"])),
            priority=str(parsed.get("priority", fallback["priority"])),
            max_span_ft=float(parsed.get("max_span_ft", fallback["max_span_ft"])),
            min_span_ft=float(parsed.get("min_span_ft", fallback["min_span_ft"])),
            allow_interior_cols=bool(parsed.get("allow_interior_cols", fallback["allow_interior_cols"])),
            dead_psf=float(parsed.get("dead_psf", fallback["dead_psf"])),
            live_psf=float(parsed.get("live_psf", fallback["live_psf"])),
            composite=bool(parsed.get("composite", fallback.get("composite", True))),
        )
    except Exception:
        return fallback


def _parse_design_brief_fallback(user_text: str) -> dict[str, Any]:
    text = (user_text or "").strip()
    lower = text.lower()

    size_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:ft|feet)?\s*(?:x|by)\s*(\d+(?:\.\d+)?)", lower)
    length_ft = float(size_match.group(1)) if size_match else 120.0
    width_ft = float(size_match.group(2)) if size_match else 90.0

    floors_match = re.search(r"(\d+)\s*(?:story|storey|floor)", lower)
    num_floors = int(floors_match.group(1)) if floors_match else 8

    floor_height_match = re.search(r"floor\s*height\s*(?:of)?\s*(\d+(?:\.\d+)?)", lower)
    floor_height_ft = float(floor_height_match.group(1)) if floor_height_match else 14.0

    occupancy = "office"
    for option in OCCUPANCY_LOADS:
        if option in lower:
            occupancy = option
            break

    city = "Chicago"
    for option in SUPPORTED_CITIES:
        if option.lower() in lower:
            city = option
            break

    max_span_match = re.search(r"max(?:imum)?\s*span\s*(?:of)?\s*(\d+(?:\.\d+)?)", lower)
    min_span_match = re.search(r"min(?:imum)?\s*span\s*(?:of)?\s*(\d+(?:\.\d+)?)", lower)
    max_span_ft = float(max_span_match.group(1)) if max_span_match else 40.0
    min_span_ft = float(min_span_match.group(1)) if min_span_match else 15.0

    priority = "balanced"
    if any(term in lower for term in ("open plan", "few columns", "fewer columns", "open floor", "long spans")):
        priority = "few_columns"
    elif any(term in lower for term in ("shallow", "low floor depth", "minimum depth", "headroom")):
        priority = "shallow_depth"
    elif any(term in lower for term in ("minimum steel", "min steel", "lightest", "least steel")):
        priority = "min_steel"
    elif any(term in lower for term in ("minimize cost", "minimum cost", "low cost", "cheapest", "budget")):
        priority = "low_cost"

    allow_interior_cols = not any(term in lower for term in ("no interior columns", "without interior columns"))
    brief = build_brief(
        length_ft=length_ft,
        width_ft=width_ft,
        num_floors=num_floors,
        floor_height_ft=floor_height_ft,
        occupancy=occupancy,
        city=city,
        priority=priority,
        max_span_ft=max_span_ft,
        min_span_ft=min_span_ft,
        allow_interior_cols=allow_interior_cols,
        composite=True,
    )
    return brief


def generate_candidates(brief: dict[str, Any]) -> list[dict[str, Any]]:
    length = float(brief["length_ft"])
    width = float(brief["width_ft"])
    max_span = float(brief.get("max_span_ft", 40.0))
    min_span = float(brief.get("min_span_ft", 15.0))
    max_aspect = float(brief.get("max_aspect_ratio", 2.0))

    candidates = []
    for bays_x in range(2, 9):
        for bays_y in range(2, 7):
            span_x = length / bays_x
            span_y = width / bays_y

            if span_x > max_span or span_y > max_span:
                continue
            if span_x < min_span or span_y < min_span:
                continue

            aspect = max(span_x, span_y) / min(span_x, span_y)
            if aspect > max_aspect:
                continue

            beam_dir = "x" if span_x <= span_y else "y"
            beam_span = min(span_x, span_y)
            girder_span = max(span_x, span_y)

            candidates.append(
                {
                    "id": f"{bays_x}x{bays_y}",
                    "candidate_id": f"{bays_x}x{bays_y}",
                    "bays_x": bays_x,
                    "bays_y": bays_y,
                    "span_x": round(span_x, 3),
                    "span_y": round(span_y, 3),
                    "beam_span": round(beam_span, 3),
                    "girder_span": round(girder_span, 3),
                    "beam_dir": beam_dir,
                    "num_columns": int((bays_x + 1) * (bays_y + 1)),
                    "interior_columns": int(max(0, bays_x - 1) * max(0, bays_y - 1)),
                    "aspect_ratio": round(aspect, 3),
                }
            )
            constraint_eval = evaluate_layout_constraints(candidates[-1], brief)
            if not constraint_eval["passes"]:
                candidates.pop()
                continue
            candidates[-1]["constraint_penalty"] = constraint_eval["penalty"]
            candidates[-1]["constraint_bonus"] = constraint_eval["bonus"]
            candidates[-1]["constraint_notes"] = constraint_eval["notes"]
    return candidates


def try_spacing(spacing_ft: float, candidate: dict[str, Any], brief: dict[str, Any]) -> dict[str, Any] | None:
    beam_span = float(candidate["beam_span"])
    girder_span = float(candidate["girder_span"])
    if spacing_ft < 8.0 or spacing_ft > 15.0:
        return None
    if spacing_ft >= girder_span:
        return None

    system = FloorSystem(
        bay_length_ft=girder_span,
        bay_width_ft=beam_span,
        dead_load_psf=float(brief["dead_psf"]),
        live_load_psf=float(brief["live_psf"]),
        beam_spacing_ft=float(spacing_ft),
        num_floors=int(brief["num_floors"]),
        floor_height_ft=float(brief["floor_height_ft"]),
        composite_beams=bool(brief.get("composite", True)),
    )
    design = system.design_all()
    if not design.get("passes"):
        return None

    beam = design["beams"]
    girder = design["girder"]
    column = design["column"]

    n_beams_per_bay = max(1, int(round(girder_span / spacing_ft)) - 1)
    if candidate["beam_dir"] == "x":
        beams_per_floor = n_beams_per_bay * int(candidate["bays_x"]) * int(candidate["bays_y"])
        girders_per_floor = (int(candidate["bays_x"]) + 1) * int(candidate["bays_y"])
    else:
        beams_per_floor = n_beams_per_bay * int(candidate["bays_x"]) * int(candidate["bays_y"])
        girders_per_floor = int(candidate["bays_x"]) * (int(candidate["bays_y"]) + 1)

    n_columns = int(candidate["num_columns"])
    total_beams = beams_per_floor * int(brief["num_floors"])
    total_girders = girders_per_floor * int(brief["num_floors"])
    total_columns = n_columns * int(brief["num_floors"])

    beam_steel = float(beam["weight"]) * beam_span * total_beams
    girder_steel = float(girder["weight"]) * girder_span * total_girders
    col_steel = float(column["weight"]) * float(brief["floor_height_ft"]) * total_columns
    total_steel = beam_steel + girder_steel + col_steel

    steel_cost = total_steel * 1.50
    connections = beams_per_floor * 2 * int(brief["num_floors"])
    conn_cost = connections * 350.0
    found_cost = n_columns * 8500.0
    total_cost = steel_cost + conn_cost + found_cost

    beam_row = BEAMS_DF[BEAMS_DF["name"] == beam["name"]]
    beam_depth = float(beam_row.iloc[0]["d"]) if not beam_row.empty else 20.0
    floor_depth = beam_depth + 5.5

    gross_area = float(brief["length_ft"]) * float(brief["width_ft"])
    col_footprint = n_columns * 1.5
    rentable_area = gross_area - col_footprint

    if candidate["beam_dir"] == "x":
        render_bay_length = float(candidate["span_y"])
        render_bay_width = float(candidate["span_x"])
        render_bays_x = int(candidate["bays_y"])
        render_bays_y = int(candidate["bays_x"])
    else:
        render_bay_length = float(candidate["span_x"])
        render_bay_width = float(candidate["span_y"])
        render_bays_x = int(candidate["bays_x"])
        render_bays_y = int(candidate["bays_y"])

    return {
        "candidate_id": candidate["candidate_id"],
        "bays_x": int(candidate["bays_x"]),
        "bays_y": int(candidate["bays_y"]),
        "span_x": float(candidate["span_x"]),
        "span_y": float(candidate["span_y"]),
        "beam_dir": candidate["beam_dir"],
        "beam_span": beam_span,
        "girder_span": girder_span,
        "beam_spacing": float(spacing_ft),
        "beam": beam["name"],
        "girder": girder["name"],
        "column": column["name"],
        "beam_studs": int(beam.get("studs_per_side", 0)),
        "total_steel_lbs": round(total_steel, 0),
        "steel_psf": round(total_steel / gross_area, 2),
        "floor_depth_in": round(floor_depth, 1),
        "num_columns": n_columns,
        "interior_columns": int(candidate["interior_columns"]),
        "rentable_sqft": round(rentable_area, 1),
        "steel_cost": round(steel_cost, 0),
        "conn_cost": round(conn_cost, 0),
        "found_cost": round(found_cost, 0),
        "total_cost": round(total_cost, 0),
        "passes": True,
        "beam_weight_lbft": float(beam["weight"]),
        "girder_weight_lbft": float(girder["weight"]),
        "column_weight_lbft": float(column["weight"]),
        "beam_reaction_kips": float(beam.get("reaction", 0)),
        "girder_reaction_kips": float(girder.get("reaction", 0)),
        "beam_details": beam.get("details", {}),
        "girder_details": girder.get("details", {}),
        "column_details": column.get("details", {}),
        "constraint_penalty": float(candidate.get("constraint_penalty", 0.0)),
        "constraint_bonus": float(candidate.get("constraint_bonus", 0.0)),
        "constraint_notes": list(candidate.get("constraint_notes", [])),
        "render_bay_length": render_bay_length,
        "render_bay_width": render_bay_width,
        "render_bays_x": render_bays_x,
        "render_bays_y": render_bays_y,
    }


def design_candidate(candidate: dict[str, Any], brief: dict[str, Any]) -> dict[str, Any] | None:
    beam_span = float(candidate["beam_span"])
    best = None
    for n_spaces in (2, 3, 4):
        spacing = beam_span / n_spaces
        result = try_spacing(spacing, candidate, brief)
        if result and (best is None or result["total_cost"] < best["total_cost"]):
            best = result
    return best


def rank_candidates(results: list[dict[str, Any]], brief: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    df = pd.DataFrame(results)
    if df.empty:
        return df, None

    def normalize(series: pd.Series, ascending: bool = True) -> pd.Series:
        rng = float(series.max() - series.min())
        if rng == 0:
            base = pd.Series([0.5] * len(series), index=series.index)
        else:
            base = (series - series.min()) / rng
        return base if ascending else 1.0 - base

    df["score_cost"] = normalize(df["total_cost"])
    df["score_steel"] = normalize(df["total_steel_lbs"])
    df["score_depth"] = normalize(df["floor_depth_in"])
    df["score_cols"] = normalize(df["num_columns"])
    df["score_rentable"] = normalize(df["rentable_sqft"], ascending=False)
    grid_deviation = (df["span_x"] - 30.0).abs() + (df["span_y"] - 30.0).abs()
    df["score_simplicity"] = normalize(grid_deviation)

    weights = PRIORITY_WEIGHTS.get(brief.get("priority", "balanced"), PRIORITY_WEIGHTS["balanced"])
    df["overall_score"] = (
        df["score_cost"] * weights["cost"]
        + df["score_steel"] * weights["steel"]
        + df["score_depth"] * weights["depth"]
        + df["score_cols"] * weights["cols"]
        + df["score_rentable"] * weights["rentable"]
        + df["score_simplicity"] * weights["simplicity"]
    )

    if not brief.get("allow_interior_cols", True):
        df["overall_score"] += normalize(df["interior_columns"]) * 0.35
    if "constraint_penalty" in df.columns:
        df["overall_score"] += df["constraint_penalty"] * 0.35
    if "constraint_bonus" in df.columns:
        df["overall_score"] -= df["constraint_bonus"] * 0.25

    df = df.sort_values(["overall_score", "total_cost", "total_steel_lbs"]).reset_index(drop=True)
    return df, df.iloc[0].to_dict()


def explain_recommendation(top: dict[str, Any], second: dict[str, Any] | None, brief: dict[str, Any]) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            second_layout = second["candidate_id"] if second else "N/A"
            second_cost = f"${second['total_cost']:,.0f}" if second else "N/A"
            second_steel = f"{second['total_steel_lbs']:,.0f} lbs" if second else "N/A"
            second_depth = f"{second['floor_depth_in']:.1f} in" if second else "N/A"
            prompt = f"""
You are a structural engineer explaining a design recommendation to another engineer.

Building: {brief['length_ft']}x{brief['width_ft']} ft, {brief['num_floors']} floors, {brief['occupancy']}
City: {brief['city']}
Priority: {brief['priority']}

Recommended layout: {top['candidate_id']}
  Bay size: {top['span_x']} ft x {top['span_y']} ft
  Beam: {top['beam']} composite @ {top['beam_spacing']} ft
  Girder: {top['girder']}
  Column: {top['column']}
  Total cost: ${top['total_cost']:,.0f}
  Steel: {top['total_steel_lbs']:,.0f} lbs
  Floor depth: {top['floor_depth_in']:.1f} in
  Columns: {top['num_columns']}

Second option:
  Layout: {second_layout}
  Cost: {second_cost}
  Steel: {second_steel}
  Depth: {second_depth}

Write 3-4 sentences:
1. Why the recommended layout is best
2. The key tradeoff versus the next option
3. One engineering watch-out
Use the actual numbers.
"""
            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=260,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception:
            pass

    sentences = [
        (
            f"The recommended layout is {top['candidate_id']} with {top['span_x']:.1f} ft x "
            f"{top['span_y']:.1f} ft bays because it best fits the '{brief['priority']}' objective "
            f"while keeping total estimated cost at ${top['total_cost']:,.0f}."
        ),
        (
            f"It uses {top['beam']} beams, {top['girder']} girders, and {top['column']} columns for "
            f"{top['total_steel_lbs']:,.0f} lb of steel with a {top['floor_depth_in']:.1f} in floor depth."
        ),
    ]
    if second:
        cost_delta = float(second["total_cost"]) - float(top["total_cost"])
        steel_delta = float(second["total_steel_lbs"]) - float(top["total_steel_lbs"])
        sentences.append(
            f"Compared with the next option ({second['candidate_id']}), it saves ${cost_delta:,.0f} and {steel_delta:,.0f} lb while using {top['num_columns']} columns."
        )
    sentences.append(
        f"One thing to watch is vibration and serviceability on the {top['beam_span']:.1f} ft beam spans; the selected spacing of {top['beam_spacing']:.1f} ft is the main lever if you want a stiffer floor."
    )
    return " ".join(sentences)


def make_tradeoff_chart(df: pd.DataFrame, recommended: dict[str, Any]):
    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224, polar=True)

    ax1.scatter(df["total_cost"], df["total_steel_lbs"], c="#778da9", s=55, alpha=0.8)
    ax1.scatter([recommended["total_cost"]], [recommended["total_steel_lbs"]], c="#2ecc71", s=100, label="Recommended")
    ax1.set_title("Cost vs Steel")
    ax1.set_xlabel("Total cost ($)")
    ax1.set_ylabel("Steel weight (lb)")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="best")

    ax2.scatter(df["floor_depth_in"], df["num_columns"], c="#415a77", s=55, alpha=0.8)
    ax2.scatter([recommended["floor_depth_in"]], [recommended["num_columns"]], c="#2ecc71", s=100)
    ax2.set_title("Depth vs Columns")
    ax2.set_xlabel("Floor depth (in)")
    ax2.set_ylabel("Columns")
    ax2.grid(alpha=0.3)

    top5 = df.nsmallest(5, "total_cost")
    ax3.bar(top5["candidate_id"], top5["total_cost"], color=["#2ecc71" if cid == recommended["candidate_id"] else "#9fb3c8" for cid in top5["candidate_id"]])
    ax3.set_title("Top 5 by Total Cost")
    ax3.set_ylabel("Total cost ($)")
    ax3.tick_params(axis="x", rotation=20)

    categories = ["Cost", "Weight", "Depth", "Columns", "Simplicity"]
    values = [
        1.0 - float(recommended.get("score_cost", 0.5)),
        1.0 - float(recommended.get("score_steel", 0.5)),
        1.0 - float(recommended.get("score_depth", 0.5)),
        1.0 - float(recommended.get("score_cols", 0.5)),
        1.0 - float(recommended.get("score_simplicity", 0.5)),
    ]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    ax4.plot(angles, values, color="#2ecc71", linewidth=2)
    ax4.fill(angles, values, color="#2ecc71", alpha=0.25)
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_yticklabels([])
    ax4.set_title("Recommended Layout Score")

    fig.tight_layout()
    return fig


def run_generative_design(brief: dict[str, Any]) -> dict[str, Any]:
    candidates = generate_candidates(brief)
    results = []
    for candidate in candidates:
        designed = design_candidate(candidate, brief)
        if designed:
            combined = {**candidate, **designed}
            results.append(combined)

    ranked_df, best = rank_candidates(results, brief)
    if best is None:
        return {
            "all_results": pd.DataFrame(),
            "top_5": [],
            "recommended": None,
            "explanation": "No candidate layouts passed the current structural and span constraints.",
            "num_candidates_tried": len(candidates),
            "num_candidates_passed": 0,
        }

    top_5 = ranked_df.head(5).to_dict("records")
    second = top_5[1] if len(top_5) > 1 else None
    explanation = explain_recommendation(best, second, brief)

    return {
        "all_results": ranked_df,
        "top_5": top_5,
        "recommended": best,
        "explanation": explanation,
        "num_candidates_tried": len(candidates),
        "num_candidates_passed": len(ranked_df),
    }
