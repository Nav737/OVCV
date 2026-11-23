import json
import unittest
from datetime import date
from pathlib import Path

from example_32076vae3 import (
    CONFIG_PATH,
    accrued_interest_30_360,
    build_inputs,
    next_coupon_dates,
)


class Test32076VAE3Example(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_instrument_terms_and_coupon_schedule(self):
        sec = self.cfg["security"]
        self.assertEqual(sec["cusip"], "32076VAE3")
        self.assertAlmostEqual(sec["coupon_rate"], 0.00125)
        self.assertAlmostEqual(sec["conversion_ratio_per_1000"], 44.7227)
        coupons = next_coupon_dates(date(2026, 11, 20), date(2031, 1, 15))
        self.assertEqual(len(coupons), 9)
        self.assertEqual(coupons[0], date(2027, 1, 15))
        self.assertEqual(coupons[-1], date(2031, 1, 15))

    def test_accrued_interest_30_360(self):
        ai = accrued_interest_30_360(1000.0, 0.00125, date(2026, 7, 15), date(2026, 11, 20))
        self.assertAlmostEqual(ai, 0.4340277778, places=8)

    def test_market_placeholders_flagged(self):
        self.assertIn("ILLUSTRATIVE", self.cfg["market_inputs"]["market_data_status"])
        self.assertIsNone(self.cfg["market_inputs"]["market_clean_price_per_1000"])

    def test_model_inputs_build(self):
        built = build_inputs(self.cfg)
        bond, _, _, _, hazard, jd_vol, jd_params, bs_params, coupon_dates, maturity = built
        self.assertAlmostEqual(bond.face, 1000.0)
        self.assertAlmostEqual(bond.conversion_ratio, 44.7227)
        self.assertEqual(len(coupon_dates), 9)
        self.assertGreater(maturity, 4.0)
        self.assertGreater(hazard.at(1.0), 0.0)
        self.assertGreater(jd_vol.at(1.0), 0.0)
        self.assertEqual(jd_params.model, "jump_diffusion")
        self.assertEqual(bs_params.model, "black_scholes")


if __name__ == "__main__":
    unittest.main()
