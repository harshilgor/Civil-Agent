import unittest

from PIL import Image, ImageDraw

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
from plan_geometry import extract_plan_geometry, render_plan_overlay, render_plan_structure_alignment
from plan_parser_ocr import extract_ocr_evidence
from plan_semantics import merge_semantic_evidence
from structural_graph import build_plan_structural_graph
from support_inference import infer_support_and_constraints


class PlanPipelineTests(unittest.TestCase):
    def _sample_plan(self):
        img = Image.new("RGB", (600, 400), "white")
        draw = ImageDraw.Draw(img)
        draw.rectangle((40, 40, 560, 360), outline="black", width=8)
        draw.line((220, 40, 220, 360), fill="black", width=6)
        draw.line((40, 210, 560, 210), fill="black", width=6)
        return img

    def test_geometry_extracts_outline_and_major_lines(self):
        geometry = extract_plan_geometry(self._sample_plan().convert("L"))
        self.assertGreater(geometry["bbox_px"]["x_max"] - geometry["bbox_px"]["x_min"], 400)
        self.assertGreaterEqual(len(geometry["major_vertical_lines"]), 1)
        self.assertGreaterEqual(len(geometry["major_horizontal_lines"]), 1)

    def test_graph_and_support_inference(self):
        geometry = extract_plan_geometry(self._sample_plan().convert("L"))
        semantics = {
            "overall_dimensions_ft": {"length_ft": 60.0, "width_ft": 40.0},
            "detected_dimension_strings": ["60'-0\"", "40'-0\""],
            "zones": [
                {"name": "Garage", "type": "garage", "x": 0.0, "y": 0.0, "w": 0.35, "h": 0.45, "confidence": "high"},
                {"name": "Bedroom Wing", "type": "bedroom", "x": 0.55, "y": 0.0, "w": 0.35, "h": 0.45, "confidence": "medium"},
            ],
            "ceiling_notes": [],
            "open_zone_hints": ["garage"],
            "support_hints": [],
            "notes": "",
        }
        graph = build_building_graph(
            geometry,
            semantics,
            length_ft=60.0,
            width_ft=40.0,
            num_floors=1,
            floor_height_ft=10.0,
            occupancy="residential",
        )
        support_model = infer_support_and_constraints(graph)
        overlay = render_plan_overlay(self._sample_plan(), geometry, support_model["support_lines"], support_model["blocked_zones"], graph["spaces"])
        self.assertEqual(graph["occupancy"], "residential")
        self.assertGreaterEqual(len(support_model["support_lines"]), 4)
        self.assertIn("support_candidates", support_model)
        self.assertTrue(any(item["classification"] in {"probable_bearing", "possible_support"} for item in graph["boundaries"]))
        self.assertGreaterEqual(len(support_model["blocked_zones"]), 1)
        self.assertEqual(overlay.size, (600, 400))

    def test_geometry_model_and_structural_graph_foundations(self):
        geometry = extract_plan_geometry(self._sample_plan().convert("L"))
        semantics = merge_semantic_evidence(
            {
                "overall_dimensions_ft": {"length_ft": 60.0, "width_ft": 40.0},
                "detected_dimension_strings": [],
                "zones": [
                    {"name": "Garage", "type": "garage", "x": 0.0, "y": 0.0, "w": 0.35, "h": 0.45, "confidence": "high"},
                    {"name": "Primary Bed", "type": "bedroom", "x": 0.55, "y": 0.0, "w": 0.30, "h": 0.40, "confidence": "medium"},
                ],
                "ceiling_notes": [],
                "open_zone_hints": [],
                "support_hints": [],
                "notes": "",
            },
            {
                "dimension_strings": ["60'-0\"", "40'-0\""],
                "room_labels": [{"label": "Garage", "room_type": "garage", "bbox_px": [50, 50, 130, 90], "confidence": 95}],
            },
        )
        geometry_model = build_geometry_model(
            geometry,
            semantics,
            length_ft=60.0,
            width_ft=40.0,
            ocr_evidence=extract_ocr_evidence(self._sample_plan(), native_text="GARAGE 60'-0\" 40'-0\""),
        )
        graph = build_building_graph(
            geometry,
            semantics,
            length_ft=60.0,
            width_ft=40.0,
            num_floors=1,
            floor_height_ft=10.0,
            occupancy="residential",
        )
        support_model = infer_support_and_constraints(graph)
        structural_graph = build_plan_structural_graph(graph, support_model, geometry_model)
        self.assertEqual(geometry_model["length_ft"], 60.0)
        self.assertGreaterEqual(len(geometry_model["wall_lines"]), 2)
        self.assertGreater(structural_graph["summary"]["node_count"], 0)
        self.assertGreater(structural_graph["summary"]["support_nodes"], 0)

    def test_parser_intelligence_artifacts(self):
        geometry = extract_plan_geometry(self._sample_plan().convert("L"))
        ocr = {
            "words": [
                {"text": "GARAGE", "confidence": 95, "bbox_px": [40, 40, 120, 65], "source": "test"},
                {"text": '60\'-0"', "confidence": 90, "bbox_px": [250, 16, 320, 40], "source": "test"},
                {"text": '40\'-0"', "confidence": 90, "bbox_px": [5, 150, 45, 190], "source": "test"},
                {"text": "BATH", "confidence": 85, "bbox_px": [300, 200, 350, 225], "source": "test"},
            ],
            "dimension_strings": ['60\'-0"', '40\'-0"'],
            "room_labels": [{"label": "GARAGE", "room_type": "garage", "bbox_px": [40, 40, 120, 65], "confidence": 95, "source": "test"}],
            "source": "test",
            "notes": [],
            "raw_text": 'GARAGE 60\'-0" 40\'-0" BATH',
        }
        text_artifacts = assemble_text_phrases(normalize_and_classify_text(ocr))
        scale_artifacts = infer_scale_candidates(geometry, text_artifacts)
        semantics = merge_semantic_evidence(
            {
                "overall_dimensions_ft": {"length_ft": 0.0, "width_ft": 0.0},
                "detected_dimension_strings": [],
                "zones": [],
                "ceiling_notes": [],
                "open_zone_hints": [],
                "support_hints": [],
                "notes": "",
            },
            ocr,
        )
        geometry_model = build_geometry_model(geometry, semantics, length_ft=60.0, width_ft=40.0, ocr_evidence=ocr)
        region_artifacts = build_region_artifacts(geometry_model, text_artifacts)
        semantic_artifacts = infer_semantic_zones(geometry_model, region_artifacts, existing_zones=[])
        self.assertGreater(text_artifacts["summary"]["dimension_candidates"], 0)
        self.assertGreater(text_artifacts["phrase_summary"]["phrase_count"], 0)
        self.assertGreaterEqual(len(scale_artifacts["candidates"]), 1)
        self.assertGreaterEqual(region_artifacts["summary"]["region_count"], 1)
        self.assertGreaterEqual(semantic_artifacts["summary"]["zone_count"], 1)

    def test_scheme_generation_returns_multiple_concepts(self):
        geometry = extract_plan_geometry(self._sample_plan().convert("L"))
        semantics = {
            "overall_dimensions_ft": {"length_ft": 60.0, "width_ft": 40.0},
            "detected_dimension_strings": [],
            "zones": [
                {"name": "Garage", "type": "garage", "x": 0.0, "y": 0.0, "w": 0.35, "h": 0.45, "confidence": "high"},
                {"name": "Living", "type": "living", "x": 0.35, "y": 0.0, "w": 0.35, "h": 0.45, "confidence": "medium"},
                {"name": "Bed", "type": "bedroom", "x": 0.0, "y": 0.45, "w": 0.4, "h": 0.35, "confidence": "medium"},
            ],
            "ceiling_notes": [],
            "open_zone_hints": [],
            "support_hints": [],
            "notes": "",
        }
        graph = build_building_graph(
            geometry,
            semantics,
            length_ft=60.0,
            width_ft=40.0,
            num_floors=1,
            floor_height_ft=10.0,
            occupancy="residential",
        )
        support_model = infer_support_and_constraints(graph)
        schemes = generate_framing_schemes(graph, support_model, city="Chicago", occupancy="residential")
        self.assertGreaterEqual(len(schemes), 2)
        self.assertIn("recommended", schemes[0])
        self.assertIn("alignment_mode", schemes[0])

    def test_alignment_overlay_renders_selected_scheme(self):
        geometry = extract_plan_geometry(self._sample_plan().convert("L"))
        semantics = {
            "overall_dimensions_ft": {"length_ft": 60.0, "width_ft": 40.0},
            "detected_dimension_strings": [],
            "zones": [
                {"name": "Garage", "type": "garage", "x": 0.0, "y": 0.0, "w": 0.35, "h": 0.45, "confidence": "high"},
                {"name": "Family", "type": "family", "x": 0.35, "y": 0.0, "w": 0.35, "h": 0.45, "confidence": "high"},
                {"name": "Beds", "type": "bedroom", "x": 0.0, "y": 0.45, "w": 0.45, "h": 0.35, "confidence": "medium"},
            ],
            "ceiling_notes": [],
            "open_zone_hints": [],
            "support_hints": [],
            "notes": "",
        }
        graph = build_building_graph(
            geometry,
            semantics,
            length_ft=60.0,
            width_ft=40.0,
            num_floors=1,
            floor_height_ft=10.0,
            occupancy="residential",
        )
        support_model = infer_support_and_constraints(graph)
        scheme = generate_framing_schemes(graph, support_model, city="Chicago", occupancy="residential")[0]
        aligned = render_plan_structure_alignment(
            self._sample_plan(),
            geometry,
            graph,
            support_model,
            scheme,
            scheme["recommended"],
        )
        self.assertEqual(aligned.size, (600, 400))


if __name__ == "__main__":
    unittest.main()
