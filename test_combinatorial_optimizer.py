import unittest

from combinatorial_optimizer import CombinatorialOptimizer
from layout_generator import build_brief
from visualization_3d import generate_3d_frame_html, generate_building_data


class CombinatorialOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.brief = build_brief(
            length_ft=120,
            width_ft=90,
            num_floors=8,
            floor_height_ft=14,
            occupancy="office",
            city="Chicago",
            priority="balanced",
            max_span_ft=40,
            allow_interior_cols=True,
            composite=True,
        )

    def test_candidate_count_in_expected_band(self):
        optimizer = CombinatorialOptimizer(self.brief)
        count = len(optimizer.generate_candidates())
        self.assertGreaterEqual(count, 800)
        self.assertLessEqual(count, 1200)

    def test_full_run_returns_ranked_results(self):
        optimizer = CombinatorialOptimizer(self.brief)
        results = optimizer.run(max_workers=8)
        self.assertEqual(results["n_tried"], len(optimizer.generate_candidates()))
        self.assertGreater(results["n_passed"], 0)
        self.assertIn("recommended", results)
        self.assertIn("pareto_front", results)
        self.assertIn("design_alternatives", results)
        self.assertTrue(results["recommended"]["passes"])
        self.assertGreaterEqual(len(results["design_alternatives"]), 3)
        self.assertIn("constructability_score", results["recommended"])
        self.assertIn("unique_sections", results["recommended"])
        self.assertIn("lateral_drift_ft", results["recommended"])

    def test_design_alternatives_have_named_categories(self):
        optimizer = CombinatorialOptimizer(self.brief)
        results = optimizer.run(max_workers=8)
        alternative_keys = {item["key"] for item in results["design_alternatives"]}
        self.assertIn("recommended", alternative_keys)
        self.assertIn("lowest_cost", alternative_keys)
        self.assertIn("lightest", alternative_keys)
        self.assertIn("lowest_drift", alternative_keys)
        self.assertIn("constructability", alternative_keys)
        for item in results["design_alternatives"]:
            self.assertIn("narrative", item)
            self.assertTrue(item["narrative"])

    def test_3d_building_data_contains_load_flow(self):
        building_data = generate_building_data(
            num_floors=8,
            floor_height=14,
            bay_length=40,
            bay_width=30,
            bays_x=3,
            bays_y=2,
            dead_psf=50,
            live_psf=80,
            composite=True,
        )
        self.assertIn("load_flow", building_data)
        self.assertIn("lateral_flow", building_data)
        self.assertGreater(len(building_data["load_flow"]["slab_panels"]), 0)
        html = generate_3d_frame_html(building_data, load_mode="Gravity (static)")
        self.assertIn("drawGravityLoadFlow", html)
        self.assertIn("drawLateralLoadFlow", html)
        self.assertIn("animateLoadFlow", html)


if __name__ == "__main__":
    unittest.main()
