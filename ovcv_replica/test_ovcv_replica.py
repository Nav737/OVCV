import math
import unittest
from ovcv_replica import (
    Curve, ConvertibleBond, OVCVParameters, price_convertible,
    solve_oas, price_european_call_jump_diffusion, bootstrap_hazard_from_cds,
    compare_risky_riskless,
)

class TestOVCVReplica(unittest.TestCase):
    def setUp(self):
        self.r = Curve.flat(0.04)
        self.q = Curve.flat(0.01)
        self.vol = Curve.flat(0.25)
        self.h = Curve.flat(0.03)
        self.bond = ConvertibleBond(100.0, 3.0, 5.0, coupon_schedule={1.0: 2.0, 2.0: 2.0, 3.0: 2.0})

    def test_bs_zero_hazard_price_positive(self):
        p = OVCVParameters("black_scholes", self.r, self.q, self.vol)
        out = price_convertible(self.bond, 18.0, p, n_stock=201, n_time=100)
        self.assertGreater(out.price, 0.0)
        self.assertTrue(math.isfinite(out.price))

    def test_jd_risky_price_finite(self):
        p = OVCVParameters("jump_diffusion", self.r, self.q, self.vol, self.h, 0.4, 0.6)
        out = price_convertible(self.bond, 18.0, p, n_stock=201, n_time=100)
        self.assertGreater(out.price, 0.0)
        self.assertTrue(math.isfinite(out.price))

    def test_default_free_call_reduces_to_bs(self):
        call = price_european_call_jump_diffusion(100, 100, 2, self.r, self.q, self.vol, Curve.flat(0.0), 0.6)
        # Compare against direct BS value through zero-hazard limit of the same helper.
        call2 = price_european_call_jump_diffusion(100, 100, 2, self.r, self.q, self.vol, Curve.flat(0.03), 0.0)
        self.assertAlmostEqual(call, call2, places=6)

    def test_risky_riskless_outputs(self):
        result = compare_risky_riskless(self.bond, 18.0, self.r, self.q, self.vol, self.h, 0.4, 0.6, 201, 100)
        self.assertGreater(result["riskless_price"], 0)
        self.assertGreater(result["risky_price"], 0)

    def test_cds_bootstrap_positive_hazards(self):
        h = bootstrap_hazard_from_cds([1, 3, 5], [0.01, 0.012, 0.015], 0.4, self.r)
        self.assertTrue(all(x > 0 for x in h.values))

    def test_oas_solver(self):
        p0 = OVCVParameters("black_scholes", self.r, self.q, self.vol)
        target = price_convertible(self.bond, 18.0, p0, n_stock=201, n_time=100).price
        solved = solve_oas(self.bond, 18.0, target, p0, n_stock=201, n_time=100)
        self.assertAlmostEqual(solved, 0.0, places=5)

    def test_maturity_coupon_not_added_to_conversion_value(self):
        # At maturity a holder electing conversion receives shares, not the
        # final coupon on converted principal. With a very high spot, value
        # must approach conversion ratio * spot, not that amount + coupon.
        bond = ConvertibleBond(1000.0, 1.0, 50.0, coupon_schedule={1.0: 5.0})
        p = OVCVParameters("black_scholes", self.r, self.q, self.vol)
        out = price_convertible(bond, 100.0, p, n_stock=201, n_time=100)
        self.assertLess(out.price, 5005.0)
        self.assertGreater(out.price, 4990.0)

if __name__ == "__main__":
    unittest.main()
