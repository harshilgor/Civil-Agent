"""
End-to-end staged floor-plan pipeline orchestration.
"""

from __future__ import annotations

from typing import Any

from building_graph import build_building_graph
from framing_scheme_generator import generate_framing_schemes
from geometry_builder import build_geometry_model
from parser_intelligence import (
    assemble_text_phrases,
    build_region_artifacts,
    infer_scale_candidates,
    infer_semantic_zones,
    normalize_and_classify_text,
)
from plan_geometry import extract_plan_geometry, render_plan_overlay
from plan_ingestion import prepare_plan_for_pipeline
from plan_parser_ocr import extract_ocr_evidence
from plan_semantics import extract_plan_semantics, infer_scale_from_semantics, merge_semantic_evidence
from structural_graph import build_plan_structural_graph
from support_inference import infer_support_and_constraints


def analyze_uploaded_plan(uploaded_file) -> dict[str, Any]:
    document = prepare_plan_for_pipeline(uploaded_file)
    image = document["preprocessed_page"]
    geometry = extract_plan_geometry(image)
    ocr_evidence = extract_ocr_evidence(document["selected_page"], native_text=document.get("selected_page_text", ""))
    text_artifacts = assemble_text_phrases(normalize_and_classify_text(ocr_evidence))
    scale_artifacts = infer_scale_candidates(geometry, text_artifacts)
    semantics = extract_plan_semantics(document["selected_page"], ocr_evidence=ocr_evidence)
    semantics = merge_semantic_evidence(semantics, ocr_evidence)
    scale = infer_scale_from_semantics(semantics, geometry)
    best_scale = scale_artifacts.get("best_candidate", {})
    if (
        best_scale.get("score", 0) >= 0.55
        and (
            scale.get("length_ft", 0) <= 0
            or scale.get("width_ft", 0) <= 0
            or min(float(scale.get("length_ft", 0) or 0), float(scale.get("width_ft", 0) or 0)) < 26.0
        )
    ):
        scale = {
            "length_ft": float(best_scale.get("length_ft", 0.0)),
            "width_ft": float(best_scale.get("width_ft", 0.0)),
            "scale_ft_per_px": round(
                (
                    float(best_scale.get("length_ft", 0.0)) / max(1.0, float(geometry["bbox_px"]["x_max"] - geometry["bbox_px"]["x_min"]))
                    + float(best_scale.get("width_ft", 0.0)) / max(1.0, float(geometry["bbox_px"]["y_max"] - geometry["bbox_px"]["y_min"]))
                )
                / 2.0,
                5,
            ),
            "confidence": "medium",
            "source": "parser_intelligence",
        }
    elif (
        min(float(scale.get("length_ft", 0) or 0), float(scale.get("width_ft", 0) or 0)) < 26.0
        and best_scale.get("score", 0) < 0.55
    ):
        scale = {
            "length_ft": 0.0,
            "width_ft": 0.0,
            "scale_ft_per_px": 0.0,
            "confidence": "low",
            "source": "needs_confirmation",
        }
    return {
        "document": document,
        "geometry": geometry,
        "ocr_evidence": ocr_evidence,
        "text_artifacts": text_artifacts,
        "scale_artifacts": scale_artifacts,
        "semantics": semantics,
        "scale": scale,
    }


def build_confirmed_plan_state(
    analysis: dict[str, Any],
    *,
    length_ft: float,
    width_ft: float,
    num_floors: int,
    floor_height_ft: float,
    occupancy: str,
) -> dict[str, Any]:
    geometry_model = build_geometry_model(
        analysis["geometry"],
        analysis["semantics"],
        length_ft=length_ft,
        width_ft=width_ft,
        ocr_evidence=analysis.get("ocr_evidence"),
    )
    region_artifacts = build_region_artifacts(geometry_model, analysis.get("text_artifacts", {}))
    semantic_artifacts = infer_semantic_zones(
        geometry_model,
        region_artifacts,
        existing_zones=analysis["semantics"].get("zones", []),
    )
    enriched_semantics = {**analysis["semantics"], "zones": semantic_artifacts["zones"]}
    if semantic_artifacts.get("notes"):
        enriched_semantics["notes"] = (str(enriched_semantics.get("notes", "")) + " " + " ".join(semantic_artifacts["notes"])).strip()
    graph = build_building_graph(
        analysis["geometry"],
        enriched_semantics,
        length_ft=length_ft,
        width_ft=width_ft,
        num_floors=num_floors,
        floor_height_ft=floor_height_ft,
        occupancy=occupancy,
    )
    support_model = infer_support_and_constraints(graph)
    structural_graph = build_plan_structural_graph(graph, support_model, geometry_model)
    overlay = render_plan_overlay(
        analysis["document"]["selected_page"],
        analysis["geometry"],
        support_lines=support_model["support_lines"],
        blocked_zones=support_model["blocked_zones"],
        spaces=graph["spaces"],
    )
    return {
        **analysis,
        "geometry_model": geometry_model,
        "region_artifacts": region_artifacts,
        "semantic_artifacts": semantic_artifacts,
        "semantics": enriched_semantics,
        "building_graph": graph,
        "support_model": support_model,
        "structural_graph": structural_graph,
        "overlay_image": overlay,
    }


def generate_plan_schemes(plan_state: dict[str, Any], *, city: str) -> list[dict[str, Any]]:
    return generate_framing_schemes(
        plan_state["building_graph"],
        plan_state["support_model"],
        city=city,
        occupancy=plan_state["building_graph"]["occupancy"],
    )
