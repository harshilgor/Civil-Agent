import unittest

from concrete_design import run_rc_beam_checks


class ConcreteDesignTests(unittest.TestCase):
    def test_rc_beam_checks_wrapper(self):
        result = run_rc_beam_checks(
            b_in=14,
            h_in=24,
            d_in=21.5,
            fc_psi=4000,
            fy_psi=60000,
            tension_bar_count=3,
            tension_bar_size=9,
            stirrup_bar_size=3,
            stirrup_spacing_in=8.0,
            stirrup_legs=2,
            Mu_kip_ft=200.0,
            Vu_kips=20.0,
        )

        self.assertIn("flexure", result)
        self.assertIn("shear", result)
        self.assertEqual(result["overall_status"], "PASS")
        self.assertGreater(result["flexure"].capacity, result["flexure"].demand)
        self.assertGreater(result["shear"].capacity, result["shear"].demand)


if __name__ == "__main__":
    unittest.main()
