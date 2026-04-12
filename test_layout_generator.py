import unittest

from layout_generator import (
    build_brief,
    generate_candidates,
    parse_design_brief,
    run_generative_design,
)
from framing_plan import make_framing_plan_figure


class LayoutGeneratorTests(unittest.TestCase):
    def test_parse_design_brief_open_plan(self):
        brief = parse_design_brief("8-story office building, Chicago, 120 by 90 feet, open floor plan, minimize cost")
        self.assertEqual(brief["city"], "Chicago")
        self.assertEqual(brief["occupancy"], "office")
        self.assertEqual(brief["priority"], "few_columns")
        self.assertEqual(brief["length_ft"], 120.0)
        self.assertEqual(brief["width_ft"], 90.0)

    def test_generate_candidates_includes_square_grid(self):
        brief = build_brief(
            length_ft=120,
            width_ft=90,
            num_floors=8,
            floor_height_ft=14,
            occupancy="office",
            city="Chicago",
            priority="balanced",
        )
        candidate_ids = {candidate["candidate_id"] for candidate in generate_candidates(brief)}
        self.assertIn("4x3", candidate_ids)
        self.assertGreaterEqual(len(candidate_ids), 10)

    def test_balanced_recommends_30_by_30_grid_for_sample(self):
        brief = build_brief(
            length_ft=120,
            width_ft=90,
            num_floors=8,
            floor_height_ft=14,
            occupancy="office",
            city="Chicago",
            priority="balanced",
        )
        results = run_generative_design(brief)
        self.assertEqual(results["recommended"]["candidate_id"], "4x3")
        self.assertEqual(results["recommended"]["span_x"], 30.0)
        self.assertEqual(results["recommended"]["span_y"], 30.0)

    def test_no_column_constraint_filters_central_grid(self):
        brief = build_brief(
            length_ft=120,
            width_ft=90,
            num_floors=8,
            floor_height_ft=14,
            occupancy="office",
            city="Chicago",
            priority="balanced",
            architectural_constraints=[
                {
                    "name": "Open center",
                    "zone_type": "no_columns",
                    "x_ft": 50.0,
                    "y_ft": 45.0,
                    "width_ft": 20.0,
                    "height_ft": 20.0,
                }
            ],
        )
        candidate_ids = {candidate["candidate_id"] for candidate in generate_candidates(brief)}
        self.assertNotIn("4x3", candidate_ids)

    def test_framing_plan_figure_renders_selected_layout(self):
        brief = build_brief(
            length_ft=120,
            width_ft=90,
            num_floors=8,
            floor_height_ft=14,
            occupancy="office",
            city="Chicago",
            priority="balanced",
        )
        results = run_generative_design(brief)
        fig = make_framing_plan_figure(results["recommended"], brief)
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.axes), 0)


if __name__ == "__main__":
    unittest.main()
