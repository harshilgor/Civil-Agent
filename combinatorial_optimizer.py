"""
Full combinatorial optimizer for Civil Agent.
"""

from __future__ import annotations

import copy
import itertools
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from architectural_constraints import evaluate_layout_constraints
from asce7_loads import get_wind_loads
from beams_data import BEAMS_DF
from column_physics import check_column_design
from composite_beam import design_composite_beam


@lru_cache(maxsize=1024)
def _cached_composite_design(
    span_ft: float,
    dead_load: float,
    live_load: float,
    spacing_ft: float,
    composite_ratio: float,
) -> dict[str, Any] | None:
    result = design_composite_beam(
        span_ft,
        dead_load,
        live_load,
        spacing_ft,
        composite_ratio=composite_ratio,
    )
    return copy.deepcopy(result) if result else None


@lru_cache(maxsize=1024)
def _cached_column_design(
    family: str,
    floor_height_ft: float,
    Pu_total: float,
) -> dict[str, Any] | None:
    col_df = BEAMS_DF[BEAMS_DF["name"].str.startswith(family)].sort_values("weight")
    for _, row in col_df.iterrows():
        section = row.to_dict()
        passes, weight, worst, details = check_column_design(floor_height_ft, Pu_total, 0.0, section)
        if passes:
            return {
                "name": row["name"],
                "weight": float(weight),
                "worst": worst,
                "details": details,
                "section": section,
            }
    return None


@lru_cache(maxsize=128)
def _cached_wind_roof_pressure(city: str, total_height_ft: float) -> float:
    try:
        return float(get_wind_loads(city, total_height_ft)["roof_pressure_psf"])
    except Exception:
        return 0.0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class CombinatorialOptimizer:
    def __init__(self, brief: dict[str, Any]):
        self.brief = brief
        self.results: list[dict[str, Any]] = []

    def generate_candidates(self) -> list[dict[str, Any]]:
        """
        Generate all valid design-variable combinations with fast pre-filters.
        """
        length = float(self.brief["length_ft"])
        width = float(self.brief["width_ft"])
        max_span = float(self.brief.get("max_span_ft", 40.0))
        min_span = float(self.brief.get("min_span_ft", 15.0))

        bays_x_options = range(3, 7)
        bays_y_options = range(3, 7)
        spacing_options = [5.0, 6.0, 7.5, 8.0, 10.0, 12.0]
        col_options = ["W10", "W12", "W14"]
        system_options = ["moment", "braced", "wall"]
        grouping_options = ["none", "by_zone", "by_floor"]
        framing_options = ["x_beams", "y_beams"]

        candidates = []
        for bx, by in itertools.product(bays_x_options, bays_y_options):
            span_x = length / bx
            span_y = width / by

            if not (min_span <= span_x <= max_span):
                continue
            if not (min_span <= span_y <= max_span):
                continue
            if max(span_x, span_y) / min(span_x, span_y) > 2.0:
                continue

            for framing_dir in framing_options:
                beam_span = span_x if framing_dir == "x_beams" else span_y
                girder_span = span_y if framing_dir == "x_beams" else span_x

                valid_spacings = []
                for spacing in spacing_options:
                    if spacing >= beam_span:
                        continue
                    if spacing < beam_span / 5.0:
                        continue
                    n_spaces = beam_span / spacing
                    if not (2.0 <= n_spaces <= 4.0):
                        continue
                    valid_spacings.append((abs(n_spaces - round(n_spaces)), spacing))
                if not valid_spacings:
                    continue
                valid_spacings = [spacing for _, spacing in sorted(valid_spacings)[:1]]

                for spacing in valid_spacings:
                    if spacing >= 10.0:
                        comp_options = [0.75, 1.00]
                    elif spacing >= 8.0:
                        comp_options = [0.50, 0.75]
                    else:
                        comp_options = [0.25, 0.50]

                    active_col_options = col_options

                    for comp, col_fam, system, grouping in itertools.product(
                        comp_options,
                        active_col_options,
                        system_options,
                        grouping_options,
                    ):
                        # Fast filters to keep the search focused and within runtime.
                        if self.brief["num_floors"] >= 12 and system == "moment":
                            continue
                        if grouping == "by_zone" and (bx * by) < 9:
                            continue
                        if grouping == "by_zone" and min(bx, by) < 4:
                            continue
                        if grouping == "by_floor" and int(self.brief["num_floors"]) < 4:
                            continue
                        if grouping == "by_floor" and (bx * by) < 12:
                            continue
                        if self.brief["num_floors"] >= 8 and system == "moment" and col_fam != "W14":
                            continue
                        if self.brief["num_floors"] >= 8 and system == "braced" and col_fam == "W12" and max(span_x, span_y) > 30:
                            continue
                        if self.brief["num_floors"] >= 8 and system == "wall" and col_fam == "W10" and max(span_x, span_y) > 24:
                            continue

                        if framing_dir == "x_beams":
                            render_bays_x = by
                            render_bays_y = bx
                        else:
                            render_bays_x = bx
                            render_bays_y = by

                        candidate = {
                            "bays_x": bx,
                            "bays_y": by,
                            "span_x": round(span_x, 3),
                            "span_y": round(span_y, 3),
                            "aspect_ratio": round(max(span_x, span_y) / min(span_x, span_y), 3),
                            "beam_span": round(beam_span, 3),
                            "girder_span": round(girder_span, 3),
                            "beam_spacing": float(spacing),
                            "framing_dir": framing_dir,
                            "composite": float(comp),
                            "col_family": col_fam,
                            "lateral": system,
                            "grouping": grouping,
                            "num_columns": int((bx + 1) * (by + 1)),
                            "candidate_id": f"{bx}x{by}-{framing_dir}-{spacing:g}ft-{int(comp * 100)}c-{col_fam}-{system}-{grouping}",
                            "render_bays_x": render_bays_x,
                            "render_bays_y": render_bays_y,
                        }
                        constraint_eval = evaluate_layout_constraints(candidate, self.brief, lateral_system=system)
                        if not constraint_eval["passes"]:
                            continue
                        candidate["constraint_penalty"] = constraint_eval["penalty"]
                        candidate["constraint_bonus"] = constraint_eval["bonus"]
                        candidate["constraint_notes"] = constraint_eval["notes"]
                        candidates.append(candidate)

        return candidates

    def evaluate_candidate(self, candidate: dict[str, Any]) -> dict[str, Any] | None:
        """
        Full physics + code check for one candidate. Thread-safe.
        """
        try:
            brief = self.brief
            lateral_result = self._check_lateral(candidate, brief)
            if not lateral_result["passes"]:
                return None

            spacing = float(candidate["beam_spacing"])
            dl_line = float(brief["dead_psf"]) * spacing / 1000.0
            ll_line = float(brief["live_psf"]) * spacing / 1000.0

            beam = _cached_composite_design(
                float(candidate["beam_span"]),
                round(dl_line, 4),
                round(ll_line, 4),
                round(spacing, 4),
                float(candidate["composite"]),
            )
            if not beam or not beam.get("passes"):
                return None
            beam = copy.deepcopy(beam)

            if candidate["grouping"] in {"by_zone", "by_floor"}:
                beam = self._apply_zone_grouping(beam, candidate, brief)

            n_beams = max(1, int(candidate["girder_span"] / spacing) - 1)
            Wu = 1.2 * dl_line + 1.6 * ll_line
            beam_rxn = Wu * float(candidate["beam_span"]) / 2.0
            total_rxn = n_beams * beam_rxn * 2.0
            w_equiv = total_rxn / float(candidate["girder_span"])

            total_area_load = float(brief["dead_psf"]) + float(brief["live_psf"])
            dead_r = float(brief["dead_psf"]) / total_area_load if total_area_load else 0.5
            live_r = 1.0 - dead_r

            girder = _cached_composite_design(
                float(candidate["girder_span"]),
                round(w_equiv * dead_r, 4),
                round(w_equiv * live_r, 4),
                round(spacing, 4),
                float(candidate["composite"]),
            )
            if not girder or not girder.get("passes"):
                return None
            girder = copy.deepcopy(girder)

            trib_area = float(candidate["span_x"]) * float(candidate["span_y"])
            Pu_floor = ((1.2 * float(brief["dead_psf"]) + 1.6 * float(brief["live_psf"])) * trib_area / 1000.0)
            Pu_total = Pu_floor * int(brief["num_floors"])

            column = _cached_column_design(
                candidate["col_family"],
                float(brief["floor_height_ft"]),
                round(Pu_total, 3),
            )
            if column is None:
                return None
            column = copy.deepcopy(column)

            return self._compute_metrics(candidate, brief, beam, girder, column, n_beams, lateral_result)
        except Exception:
            return None

    def _apply_zone_grouping(self, beam: dict[str, Any], candidate: dict[str, Any], brief: dict[str, Any]) -> dict[str, Any]:
        adjusted = copy.deepcopy(beam)
        premium = 1.03 if candidate["grouping"] == "by_zone" else 1.025
        adjusted["weight"] = round(float(adjusted["weight"]) * premium, 3)
        adjusted.setdefault("details", {})
        grouping_label = "Zone" if candidate["grouping"] == "by_zone" else "Floor"
        adjusted["details"]["grouping_note"] = f"{grouping_label} grouping assumed; standardization premium applied."
        return adjusted

    def _check_lateral(self, candidate: dict[str, Any], brief: dict[str, Any]) -> dict[str, Any]:
        city = brief.get("city")
        if not city:
            return {"passes": True, "drift_ratio": 0.0, "roof_pressure_psf": 0.0}

        total_height = float(brief["floor_height_ft"]) * int(brief["num_floors"])
        roof_pressure = _cached_wind_roof_pressure(city, total_height)
        plan_width = max(float(candidate["span_x"]) * int(candidate["bays_x"]), float(candidate["span_y"]) * int(candidate["bays_y"]))
        base_shear = roof_pressure * plan_width * total_height * 0.65 / 1000.0

        system_factor = {"moment": 0.9, "braced": 2.0, "wall": 2.8}[candidate["lateral"]]
        family_factor = {"W10": 0.80, "W12": 0.95, "W14": 1.10}[candidate["col_family"]]
        spacing_factor = max(0.7, min(1.15, 9.0 / float(candidate["beam_spacing"])))
        stiffness = max(1.0, int(candidate["num_columns"])) * system_factor * family_factor * spacing_factor
        drift_proxy_ft = base_shear * total_height / (2100.0 * stiffness)
        allowable_ft = total_height / 400.0
        return {
            "passes": drift_proxy_ft <= allowable_ft,
            "drift_proxy_ft": round(drift_proxy_ft, 4),
            "allowable_ft": round(allowable_ft, 4),
            "base_shear_kips": round(base_shear, 2),
            "roof_pressure_psf": round(roof_pressure, 2),
        }

    def _compute_metrics(
        self,
        candidate: dict[str, Any],
        brief: dict[str, Any],
        beam: dict[str, Any],
        girder: dict[str, Any],
        column: dict[str, Any],
        n_beams: int,
        lateral_result: dict[str, Any],
    ) -> dict[str, Any]:
        bx = int(candidate["bays_x"])
        by = int(candidate["bays_y"])
        floors = int(brief["num_floors"])

        if candidate["framing_dir"] == "x_beams":
            beams_per_floor = n_beams * bx * by
            girders_per_floor = (bx + 1) * by
        else:
            beams_per_floor = n_beams * bx * by
            girders_per_floor = bx * (by + 1)

        n_cols = int(candidate["num_columns"])
        beam_steel = float(beam["weight"]) * float(candidate["beam_span"]) * beams_per_floor * floors
        girder_steel = float(girder["weight"]) * float(candidate["girder_span"]) * girders_per_floor * floors
        col_steel = float(column["weight"]) * float(brief["floor_height_ft"]) * n_cols * floors
        total_steel = beam_steel + girder_steel + col_steel

        beam_row = BEAMS_DF[BEAMS_DF["name"] == beam["name"]]
        girder_row = BEAMS_DF[BEAMS_DF["name"] == girder["name"]]
        beam_d = float(beam_row.iloc[0]["d"]) if not beam_row.empty else 20.0
        girder_d = float(girder_row.iloc[0]["d"]) if not girder_row.empty else beam_d
        floor_depth = beam_d + 5.5

        steel_cost = total_steel * 1.50
        conn_cost = beams_per_floor * 2 * floors * 350.0
        found_cost = n_cols * 8500.0

        distinct = len({beam["name"], girder["name"], column["name"]})
        aspect_penalty = max(float(candidate["span_x"]), float(candidate["span_y"])) / min(float(candidate["span_x"]), float(candidate["span_y"]))
        unique_section_penalty = max(0.0, distinct - 3.0)
        grid_irregularity = max(0.0, aspect_penalty - 1.0)
        framing_density = max(0.0, beams_per_floor / max(1, bx * by))
        framing_repetition_penalty = {"none": 1.0, "by_zone": 0.6, "by_floor": 0.35}[candidate["grouping"]]
        brace_disruption_penalty = {"moment": 0.10, "braced": 0.75, "wall": 0.55}[candidate["lateral"]]
        floor_depth_variation = abs(girder_d - beam_d)
        spacing_irregularity = abs(round(float(candidate["beam_spacing"])) - float(candidate["beam_spacing"]))
        complexity = (
            distinct * 0.35
            + framing_density * 0.60
            + grid_irregularity * 2.8
            + framing_repetition_penalty * 0.9
            + brace_disruption_penalty * 0.8
            + (floor_depth_variation / 12.0) * 0.6
            + spacing_irregularity * 0.8
            + float(candidate.get("constraint_penalty", 0.0)) * 1.6
        )

        constructability_penalty = (
            unique_section_penalty * 14.0
            + grid_irregularity * 22.0
            + framing_repetition_penalty * 12.0
            + brace_disruption_penalty * 10.0
            + _clamp(floor_depth_variation, 0.0, 24.0) * 1.1
            + spacing_irregularity * 12.0
            + float(candidate.get("constraint_penalty", 0.0)) * 25.0
        )
        constructability_score = round(
            _clamp(100.0 - constructability_penalty + float(candidate.get("constraint_bonus", 0.0)) * 18.0, 5.0, 100.0),
            1,
        )
        repetition_score = round(_clamp(100.0 - (framing_repetition_penalty * 35.0 + spacing_irregularity * 18.0), 5.0, 100.0), 1)
        grid_regularity_score = round(_clamp(100.0 - grid_irregularity * 35.0, 5.0, 100.0), 1)
        section_standardization_score = round(_clamp(100.0 - unique_section_penalty * 18.0, 5.0, 100.0), 1)

        carbon_kg = total_steel * 0.454 * 0.89
        rentable = float(brief["length_ft"]) * float(brief["width_ft"]) - n_cols * 1.5

        render_bay_length = float(candidate["girder_span"])
        render_bay_width = float(candidate["beam_span"])

        return {
            **candidate,
            "beam_name": beam["name"],
            "girder_name": girder["name"],
            "col_name": column["name"],
            "beam_studs": int(beam.get("studs_per_side", 0)),
            "beam_Ieff": float(beam.get("Ieff", 0)),
            "total_steel": round(total_steel, 0),
            "steel_psf": round(total_steel / (float(brief["length_ft"]) * float(brief["width_ft"])), 2),
            "floor_depth": round(floor_depth, 1),
            "num_columns": n_cols,
            "rentable_sqft": round(rentable, 1),
            "steel_cost": round(steel_cost, 0),
            "conn_cost": round(conn_cost, 0),
            "found_cost": round(found_cost, 0),
            "total_cost": round(steel_cost + conn_cost + found_cost, 0),
            "complexity": round(complexity, 3),
            "carbon_kg": round(carbon_kg, 0),
            "carbon_tonnes": round(carbon_kg / 1000.0, 2),
            "distinct_sections": distinct,
            "unique_sections": distinct,
            "grid_regularity_score": grid_regularity_score,
            "repetition_score": repetition_score,
            "section_standardization_score": section_standardization_score,
            "constructability_score": constructability_score,
            "constructability_penalty": round(constructability_penalty, 2),
            "brace_disruption_penalty": round(brace_disruption_penalty, 3),
            "floor_depth_variation": round(floor_depth_variation, 2),
            "constraint_penalty": round(float(candidate.get("constraint_penalty", 0.0)), 3),
            "constraint_bonus": round(float(candidate.get("constraint_bonus", 0.0)), 3),
            "constraint_notes": list(candidate.get("constraint_notes", [])),
            "passes": True,
            "beam_details": beam.get("details", {}),
            "girder_details": girder.get("details", {}),
            "column_details": column.get("details", {}),
            "lateral_ok": lateral_result["passes"],
            "lateral_base_shear_kips": lateral_result["base_shear_kips"],
            "lateral_drift_ft": lateral_result["drift_proxy_ft"],
            "lateral_allowable_ft": lateral_result["allowable_ft"],
            "render_bay_length": render_bay_length,
            "render_bay_width": render_bay_width,
        }

    def score_candidates(self, results_df: pd.DataFrame, objectives: dict[str, float]) -> pd.DataFrame:
        df = results_df.copy()
        normalize_cols = {
            "total_cost": True,
            "total_steel": True,
            "floor_depth": True,
            "num_columns": True,
            "complexity": True,
            "lateral_drift_ft": True,
            "carbon_kg": True,
            "rentable_sqft": False,
            "aspect_ratio": True,
            "constructability_score": False,
            "grid_regularity_score": False,
            "repetition_score": False,
            "section_standardization_score": False,
            "floor_depth_variation": True,
            "constraint_penalty": True,
            "constraint_bonus": False,
        }

        for col, lower_better in normalize_cols.items():
            if col not in df.columns:
                continue
            rng = float(df[col].max() - df[col].min())
            if rng == 0:
                df[f"norm_{col}"] = 0.5
                continue
            normalized = (df[col] - df[col].min()) / rng
            df[f"norm_{col}"] = normalized if lower_better else 1.0 - normalized

        total_weight = sum(max(0.0, float(weight)) for weight in objectives.values())
        normalized_objectives = (
            {metric: float(weight) / total_weight for metric, weight in objectives.items()}
            if total_weight > 0
            else {
                "total_cost": 0.40,
                "total_steel": 0.20,
                "floor_depth": 0.15,
                "num_columns": 0.15,
                "complexity": 0.10,
            }
        )

        df["score"] = 0.0
        for metric, weight in normalized_objectives.items():
            norm_col = f"norm_{metric}"
            if norm_col in df.columns:
                df["score"] += df[norm_col] * float(weight)
        if "norm_aspect_ratio" in df.columns:
            df["score"] += df["norm_aspect_ratio"] * 0.12
        if "norm_constructability_score" in df.columns:
            df["score"] += df["norm_constructability_score"] * 0.08
        if "norm_constraint_penalty" in df.columns:
            df["score"] += df["norm_constraint_penalty"] * 0.10
        if "norm_constraint_bonus" in df.columns:
            df["score"] -= df["norm_constraint_bonus"] * 0.06

        df = df.sort_values("score").reset_index(drop=True)
        df["rank"] = df.index + 1
        return df

    def _compute_pareto(self, df: pd.DataFrame, obj1: str, obj2: str) -> list[dict[str, Any]]:
        pareto = []
        for i, row in df.iterrows():
            dominated = False
            for j, other in df.iterrows():
                if i == j:
                    continue
                if (
                    other[obj1] <= row[obj1]
                    and other[obj2] <= row[obj2]
                    and (other[obj1] < row[obj1] or other[obj2] < row[obj2])
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(row.to_dict())
        return pareto

    def _comparison_note(self, candidate: dict[str, Any], reference: dict[str, Any] | None) -> str:
        if not reference:
            return (
                f"This scheme uses {candidate['beam_name']} beams at {candidate['beam_spacing']:.1f} ft spacing, "
                f"with a {candidate['lateral']} lateral system and a constructability score of "
                f"{candidate['constructability_score']:.1f}."
            )

        notes: list[str] = []
        spacing_delta = float(candidate["beam_spacing"]) - float(reference["beam_spacing"])
        if abs(spacing_delta) >= 0.4:
            if spacing_delta < 0:
                notes.append(
                    f"Beam spacing tightens from {reference['beam_spacing']:.1f} ft to {candidate['beam_spacing']:.1f} ft, "
                    "which reduces tributary demand on the beam lines and helps the gravity frame."
                )
            else:
                notes.append(
                    f"Beam spacing opens from {reference['beam_spacing']:.1f} ft to {candidate['beam_spacing']:.1f} ft, "
                    "which reduces piece count but pushes more demand into the primary framing."
                )

        if candidate["lateral"] != reference["lateral"]:
            notes.append(
                f"The lateral system shifts from {reference['lateral']} to {candidate['lateral']}, changing the drift-versus-constructability tradeoff."
            )

        steel_delta = float(candidate["total_steel"]) - float(reference["total_steel"])
        if abs(steel_delta) >= 1000:
            direction = "less" if steel_delta < 0 else "more"
            notes.append(f"It uses {abs(steel_delta):,.0f} lb {direction} total steel than the reference scheme.")

        constructability_delta = float(candidate["constructability_score"]) - float(reference["constructability_score"])
        if abs(constructability_delta) >= 4:
            if constructability_delta > 0:
                notes.append(
                    f"Constructability improves by {constructability_delta:.1f} points through more regular framing or fewer unique members."
                )
            else:
                notes.append(
                    f"Constructability drops by {abs(constructability_delta):.1f} points because the framing is less repetitive or more disruptive."
                )

        drift_delta = float(candidate["lateral_drift_ft"]) - float(reference["lateral_drift_ft"])
        if abs(drift_delta) >= 0.05:
            direction = "reducing" if drift_delta < 0 else "increasing"
            notes.append(f"It changes lateral drift by {abs(drift_delta):.2f} ft, {direction} serviceability demand.")

        if not notes:
            notes.append(
                f"This option stays close to the baseline but shifts the balance among cost, steel, and constructability."
            )
        return " ".join(notes[:3])

    def _build_design_alternatives(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []

        selectors = [
            ("recommended", "Best balanced", "Best weighted balance across the chosen objectives.", ["score", "total_cost", "total_steel"], [True, True, True]),
            ("lowest_cost", "Lowest cost", "Least total estimated project cost.", ["total_cost", "score"], [True, True]),
            ("lightest", "Lightest design", "Minimum total steel tonnage among passing schemes.", ["total_steel", "score"], [True, True]),
            ("lowest_drift", "Lowest drift", "Strongest lateral response among passing schemes.", ["lateral_drift_ft", "score"], [True, True]),
            ("constructability", "Most construction-friendly", "Highest constructability score with simpler repetition and fewer awkward framing decisions.", ["constructability_score", "score"], [False, True]),
        ]

        alternatives: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        reference = df.iloc[0].to_dict()

        for key, title, subtitle, sort_cols, ascending in selectors:
            ranked = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
            selected = None
            for _, row in ranked.iterrows():
                candidate = row.to_dict()
                if candidate["candidate_id"] not in seen_ids or key == "recommended":
                    selected = candidate
                    break
            if selected is None:
                selected = ranked.iloc[0].to_dict()
            seen_ids.add(selected["candidate_id"])

            alternatives.append(
                {
                    "key": key,
                    "title": title,
                    "subtitle": subtitle,
                    "candidate_id": selected["candidate_id"],
                    "candidate": selected,
                    "summary_metrics": {
                        "total_cost": selected["total_cost"],
                        "total_steel": selected["total_steel"],
                        "lateral_drift_ft": selected["lateral_drift_ft"],
                        "constructability_score": selected["constructability_score"],
                        "floor_depth": selected["floor_depth"],
                        "num_columns": selected["num_columns"],
                        "unique_sections": selected["unique_sections"],
                    },
                    "narrative": self._comparison_note(selected, None if key == "recommended" else reference),
                }
            )

        return alternatives

    def _warm_caches(self, candidates: list[dict[str, Any]], max_workers: int) -> None:
        beam_keys = set()
        girder_keys = set()
        column_keys = set()

        for candidate in candidates:
            spacing = float(candidate["beam_spacing"])
            dl_line = float(self.brief["dead_psf"]) * spacing / 1000.0
            ll_line = float(self.brief["live_psf"]) * spacing / 1000.0
            beam_keys.add(
                (
                    round(float(candidate["beam_span"]), 3),
                    round(dl_line, 4),
                    round(ll_line, 4),
                    round(spacing, 4),
                    float(candidate["composite"]),
                )
            )

            n_beams = max(1, int(candidate["girder_span"] / spacing) - 1)
            Wu = 1.2 * dl_line + 1.6 * ll_line
            beam_rxn = Wu * float(candidate["beam_span"]) / 2.0
            total_rxn = n_beams * beam_rxn * 2.0
            w_equiv = total_rxn / float(candidate["girder_span"])
            total_area_load = float(self.brief["dead_psf"]) + float(self.brief["live_psf"])
            dead_r = float(self.brief["dead_psf"]) / total_area_load if total_area_load else 0.5
            live_r = 1.0 - dead_r
            girder_keys.add(
                (
                    round(float(candidate["girder_span"]), 3),
                    round(w_equiv * dead_r, 4),
                    round(w_equiv * live_r, 4),
                    round(spacing, 4),
                    float(candidate["composite"]),
                )
            )

            trib_area = float(candidate["span_x"]) * float(candidate["span_y"])
            Pu_total = ((1.2 * float(self.brief["dead_psf"]) + 1.6 * float(self.brief["live_psf"])) * trib_area / 1000.0) * int(self.brief["num_floors"])
            column_keys.add((candidate["col_family"], float(self.brief["floor_height_ft"]), round(Pu_total, 3)))

        def warm_beam(key):
            _cached_composite_design(*key)

        def warm_column(key):
            _cached_column_design(*key)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(warm_beam, beam_keys))
            list(executor.map(warm_beam, girder_keys))
            list(executor.map(warm_column, column_keys))

    def _explain(self, top: dict[str, Any], second: dict[str, Any] | None, objectives: dict[str, float] | None = None) -> str:
        active_objectives = objectives or {
            "total_cost": 0.40,
            "total_steel": 0.20,
            "floor_depth": 0.15,
            "num_columns": 0.15,
            "complexity": 0.10,
        }
        ranked_objectives = sorted(active_objectives.items(), key=lambda item: item[1], reverse=True)
        objective_labels = {
            "total_cost": "cost",
            "total_steel": "steel tonnage",
            "floor_depth": "framing depth",
            "num_columns": "column count",
            "complexity": "simplicity",
            "constructability_score": "constructability",
            "lateral_drift_ft": "drift",
        }
        top_drivers = [objective_labels.get(metric, metric) for metric, weight in ranked_objectives[:3] if weight > 0]

        parts = [
            (
                f"The full search recommends {top['candidate_id']} because it delivered the best weighted balance for "
                f"{', '.join(top_drivers)}."
            ),
            (
                f"It uses {top['beam_name']} beams at {top['beam_spacing']:.1f} ft spacing, {top['girder_name']} girders, "
                f"and {top['col_name']} columns for ${top['total_cost']:,.0f} total cost, {top['total_steel']:,.0f} lb of steel, "
                f"{top['floor_depth']:.1f} in floor depth, and a constructability score of {top['constructability_score']:.1f}."
            ),
        ]
        if second:
            parts.append(self._comparison_note(top, second))
            parts.append(
                f"The closest alternative is {second['candidate_id']}, which shifts the tradeoff to "
                f"${second['total_cost']:,.0f}, {second['total_steel']:,.0f} lb, and {second['lateral_drift_ft']:.2f} ft drift."
            )
        else:
            parts.append(
                f"The selected scheme keeps drift to {top['lateral_drift_ft']:.2f} ft and limits detailing complexity with "
                f"{top['unique_sections']} unique member families."
            )
        return " ".join(parts)

    def run(
        self,
        objectives: dict[str, float] | None = None,
        max_workers: int = 8,
        progress_callback=None,
    ) -> dict[str, Any]:
        if objectives is None:
            objectives = {
                "total_cost": 0.40,
                "total_steel": 0.20,
                "floor_depth": 0.15,
                "num_columns": 0.15,
                "complexity": 0.10,
            }

        candidates = self.generate_candidates()
        n_tried = len(candidates)
        start = time.time()
        results = []
        self._warm_caches(candidates, max_workers=max_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.evaluate_candidate, candidate) for candidate in candidates]
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                completed += 1
                if progress_callback:
                    progress_callback(completed, n_tried)

        search_time = time.time() - start

        if not results:
            return {
                "error": "No valid designs found",
                "n_tried": n_tried,
                "n_passed": 0,
                "search_time_s": round(search_time, 1),
            }

        df = pd.DataFrame(results)
        df = self.score_candidates(df, objectives)
        pareto = self._compute_pareto(df, "total_cost", "total_steel")
        top = df.iloc[0].to_dict()
        second = df.iloc[1].to_dict() if len(df) > 1 else None
        explanation = self._explain(top, second, objectives)
        alternatives = self._build_design_alternatives(df)
        alternative_map = {item["key"]: item for item in alternatives}

        return {
            "all_results": df,
            "top_10": df.head(10).to_dict("records"),
            "recommended": top,
            "design_alternatives": alternatives,
            "alternative_map": alternative_map,
            "pareto_front": pareto,
            "n_tried": n_tried,
            "n_passed": len(results),
            "search_time_s": round(search_time, 1),
            "explanation": explanation,
        }


def make_pareto_chart(
    all_results: pd.DataFrame,
    pareto_front: list[dict[str, Any]],
    recommended: dict[str, Any],
    highlighted: list[dict[str, Any]] | None = None,
):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(all_results["total_cost"], all_results["total_steel"], c="#9ba7b4", alpha=0.55, s=30, label="Passing candidates")
    if pareto_front:
        pareto_df = pd.DataFrame(pareto_front)
        ax.scatter(pareto_df["total_cost"], pareto_df["total_steel"], c="#3498db", s=48, label="Pareto front")
    if highlighted:
        highlight_df = pd.DataFrame([item["candidate"] if "candidate" in item else item for item in highlighted])
        if not highlight_df.empty:
            ax.scatter(
                highlight_df["total_cost"],
                highlight_df["total_steel"],
                c="#8e44ad",
                s=90,
                marker="D",
                label="Alternatives",
            )
    if recommended:
        ax.scatter([recommended["total_cost"]], [recommended["total_steel"]], c="#2ecc71", marker="*", s=230, label="Recommended")
    ax.set_xlabel("Total cost ($)")
    ax.set_ylabel("Total steel (lb)")
    ax.set_title("Pareto Front: Cost vs Steel")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
