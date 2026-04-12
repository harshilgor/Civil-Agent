
import unittest

from beams_data import BEAMS_DF
from beam_physics import (
    calculate_demands,
    calculate_moment_capacity,
    calculate_shear_capacity,
    check_beam_design,
)
from code_checker import (
    check_deflection_aisc360,
    check_local_buckling_aisc360,
    check_ltb_aisc360,
    check_moment_aisc360,
    check_shear_aisc360,
)
from report import generate_beam_report_pdf


def beam_row(name):
    return BEAMS_DF[BEAMS_DF["name"] == name].iloc[0].to_dict()


class BeamPhysicsTests(unittest.TestCase):
    def test_demands_match_known_formula(self):
        Mu, Vu, Wu = calculate_demands(20, 0.5, 0.8)
        self.assertAlmostEqual(Wu, 1.88, places=2)
        self.assertAlmostEqual(Mu, 94.0, places=1)
        self.assertAlmostEqual(Vu, 18.8, places=1)

    def test_raw_capacities_are_positive(self):
        beam = beam_row("W14x53")
        self.assertGreater(calculate_moment_capacity(beam), 0)
        self.assertGreater(calculate_shear_capacity(beam), 0)

    def test_passing_beam_keeps_compatibility_shape(self):
        passes, weight, worst_ratio, details = check_beam_design(25, 0.5, 0.8, beam_row("W14x53"), Lb_ft=25)
        self.assertTrue(passes)
        self.assertIsInstance(weight, float)
        self.assertGreaterEqual(worst_ratio, 0)
        for key in ["moment_ratio", "shear_ratio", "defl_ratio", "ltb_ratio", "flange_ratio", "web_ratio"]:
            self.assertIn(key, details)
        self.assertIn("full_report", details)
        self.assertEqual(details["full_report"]["overall"], "PASS")

    def test_ltb_failure_gets_fix_suggestion(self):
        passes, _, _, details = check_beam_design(25, 0.5, 0.8, beam_row("W21x44"), Lb_ft=25)
        self.assertFalse(passes)
        self.assertEqual(details["controlling_check"], "ltb")
        self.assertIn("Upgrade to", details["fix_suggestion"])
        self.assertIn("AISC 360-22 Eq. F2", details["full_report"]["checks"][1]["equation"])

    def test_deflection_failure_gets_fix_suggestion(self):
        passes, _, _, details = check_beam_design(30, 2.0, 3.0, beam_row("W8x31"), Lb_ft=30)
        self.assertFalse(passes)
        self.assertEqual(details["controlling_check"], "deflection")
        self.assertIn("Upgrade to", details["fix_suggestion"])

    def test_code_checker_functions_return_expected_schema(self):
        moment = check_moment_aisc360(100, 150, "W14x53", 20, 4.5, 12.9, "elastic")
        shear = check_shear_aisc360(20, 60, "W14x53", 0.4, 14)
        deflection = check_deflection_aisc360(0.9, 30, 360)
        ltb = check_ltb_aisc360(100, 20, 5, 12, 200, 120, "W14x53")
        local_buckling = check_local_buckling_aisc360(12, 70, 8, 60, "Fake")
        for result in [moment, shear, deflection, ltb, local_buckling]:
            self.assertIn("check", result)
            self.assertIn("passes", result)
            self.assertIn("ratio", result)
            self.assertIn("equation", result)
        self.assertFalse(local_buckling["passes"])

    def test_beam_pdf_contains_equation_reference(self):
        beam = beam_row("W21x44")
        passes, _, _, details = check_beam_design(25, 0.5, 0.8, beam, Lb_ft=25)
        result = {
            "beam_name": beam["name"],
            "passes": passes,
            "details": details,
            "weight": beam["weight"],
            "connection": None,
        }
        Mu, Vu, Wu = calculate_demands(25, 0.5, 0.8)
        pdf_bytes = generate_beam_report_pdf(
            result,
            span_ft=25,
            dead_load=0.5,
            live_load=0.8,
            point_load=0,
            defl_limit=360,
            lb_display=25,
            Wu=Wu,
            Mu=Mu,
            Vu=Vu,
        )
        text = pdf_bytes.decode("latin-1", errors="ignore")
        self.assertIn("AISC 360-22", text)


if __name__ == "__main__":
    unittest.main()
