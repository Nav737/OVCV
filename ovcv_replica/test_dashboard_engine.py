import json
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from dashboard_engine import run_model

HERE = Path(__file__).resolve().parent


class TestDashboardEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = json.loads((HERE / "bond_32076vae3_config.json").read_text())
        sec, mi = cfg["security"], cfg["market_inputs"]
        cls.base = {
            "bond_name": sec["name"],
            "cusip": sec["cusip"],
            "valuation_date": date.fromisoformat(sec["valuation_date"]),
            "maturity_date": date.fromisoformat(sec["maturity_date"]),
            "face": 1000.0,
            "coupon_rate_pct": 0.125,
            "conversion_ratio": sec["conversion_ratio_per_1000"],
            "spot": sec["underlying_spot"],
            "cds_recovery_pct": 40.0,
            "bond_recovery_pct": 40.0,
            "jtd_pct": 60.0,
            "borrow_cost_pct": 0.0,
            "rate_input_type": "Zero rates",
            "vol_df": pd.DataFrame({
                "Tenor (years)": mi["ovdv_bs_vol_maturities_years"],
                "Implied vol (%)": [v * 100 for v in mi["ovdv_bs_vols"]],
                "Calibration strike (USD)": mi["jd_vol_calibration_strikes"],
            }),
            "rate_df": pd.DataFrame({
                "Tenor (years)": mi["risk_free_forward_times_years"][1:],
                "Rate (%)": [v * 100 for v in mi["risk_free_forward_rates"][1:]],
            }),
            "cds_df": pd.DataFrame({
                "Tenor (years)": mi["cds_maturities_years"],
                "CDS spread (bps)": mi["cds_par_spreads_bps"],
            }),
            "dividend_df": pd.DataFrame({
                "Tenor (years)": [1.0, 3.0, 5.0],
                "Dividend yield (%)": [0.0, 0.0, 0.0],
            }),
            "n_stock": 201,
            "n_time": 300,
        }

    def test_base_case_returns_risky_and_riskless_prices(self):
        result = run_model(self.base)
        self.assertGreater(result["risky_dirty"], 0)
        self.assertGreater(result["riskless_dirty"], 0)
        self.assertEqual(len(result["vol_diagnostic"]), len(self.base["vol_df"]))
        self.assertEqual(len(result["hazard_diagnostic"]), len(self.base["cds_df"]))

    def test_recovery_controls_are_separate_inputs(self):
        base_result = run_model(self.base)
        bond_rec_case = dict(self.base, bond_recovery_pct=55.0)
        cds_rec_case = dict(self.base, cds_recovery_pct=55.0)
        bond_rec_result = run_model(bond_rec_case)
        cds_rec_result = run_model(cds_rec_case)
        self.assertNotAlmostEqual(base_result["risky_dirty"], bond_rec_result["risky_dirty"], places=5)
        self.assertNotAlmostEqual(base_result["risky_dirty"], cds_rec_result["risky_dirty"], places=5)

    def test_zero_rate_curve_and_forward_rate_curve_are_supported(self):
        zero_result = run_model(self.base)
        forward_case = dict(self.base, rate_input_type="Instantaneous forward rates")
        forward_result = run_model(forward_case)
        self.assertGreater(zero_result["risky_dirty"], 0)
        self.assertGreater(forward_result["risky_dirty"], 0)


if __name__ == "__main__":
    unittest.main()
