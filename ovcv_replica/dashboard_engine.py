"""Calculation helpers for the Streamlit OVCV dashboard.

This module is separate from the UI so the pricing workflow can be tested
without launching Streamlit.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from ovcv_replica import (
    Curve, ConvertibleBond, OVCVParameters, bootstrap_hazard_from_cds,
    calibrate_jump_diffusion_vols, price_convertible,
)


def _float_list(df: pd.DataFrame, col: str, label: str) -> list[float]:
    vals = pd.to_numeric(df[col], errors="coerce")
    if vals.isna().any():
        raise ValueError(f"{label}: please enter numeric values in every row.")
    return [float(x) for x in vals]


def _validate_term_structure(
    df: pd.DataFrame,
    tenor_col: str,
    value_col: str,
    label: str,
    strictly_positive_value: bool = False,
) -> tuple[list[float], list[float]]:
    if df is None or len(df) < 1:
        raise ValueError(f"{label}: enter at least one tenor point.")
    tenors = _float_list(df, tenor_col, f"{label} tenor")
    values = _float_list(df, value_col, f"{label} value")
    if any(t <= 0 for t in tenors):
        raise ValueError(f"{label}: tenors must be greater than zero years.")
    if any(b <= a for a, b in zip(tenors, tenors[1:])):
        raise ValueError(f"{label}: tenors must be strictly increasing.")
    if strictly_positive_value and any(v <= 0 for v in values):
        raise ValueError(f"{label}: values must be greater than zero.")
    return tenors, values


def zero_rates_to_forward_curve(tenors: list[float], zero_rates: list[float]) -> Curve:
    return Curve.from_zero_rates(tenors, zero_rates, "Implied instantaneous forwards from zero curve")


def make_coupon_schedule(as_of: date, maturity_date: date, face: float, coupon_rate: float) -> dict[float, float]:
    dates: list[date] = []
    for year in range(as_of.year, maturity_date.year + 1):
        for month in (1, 7):
            d = date(year, month, 15)
            if as_of < d <= maturity_date:
                dates.append(d)
    return {(d - as_of).days / 365.0: face * coupon_rate / 2.0 for d in dates}


def thirty_360_us(start: date, end: date) -> int:
    d1, d2 = start.day, end.day
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30
    return 360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)


def accrued_interest(face: float, coupon_rate: float, as_of: date) -> float:
    prior_candidates = [date(as_of.year, 1, 15), date(as_of.year, 7, 15)]
    prior = max((d for d in prior_candidates if d <= as_of), default=date(as_of.year - 1, 7, 15))
    days = thirty_360_us(prior, as_of)
    return face * coupon_rate * days / 360.0


def build_market_curves(
    vol_df: pd.DataFrame,
    rate_df: pd.DataFrame,
    cds_df: pd.DataFrame,
    dividend_df: pd.DataFrame,
    rate_input_type: str,
    cds_recovery_pct: float,
    spot: float,
    jtd_pct: float,
) -> tuple[Curve, Curve, Curve, Curve, Curve, pd.DataFrame, pd.DataFrame]:
    vol_t, vol_pct = _validate_term_structure(vol_df, "Tenor (years)", "Implied vol (%)", "Volatility", True)
    strikes = _float_list(vol_df, "Calibration strike (USD)", "Volatility calibration strike")
    if any(k <= 0 for k in strikes):
        raise ValueError("Volatility calibration strikes must be positive.")

    rate_t, rate_pct = _validate_term_structure(rate_df, "Tenor (years)", "Rate (%)", "Interest-rate curve")
    if rate_input_type == "Zero rates":
        rate_curve = zero_rates_to_forward_curve(rate_t, [x / 100.0 for x in rate_pct])
    else:
        rate_curve = Curve(tuple([0.0] + rate_t[:-1]), tuple([rate_pct[0] / 100.0] + [x / 100.0 for x in rate_pct[1:]]), "step", "Instantaneous forward rates")

    cds_t, cds_bps = _validate_term_structure(cds_df, "Tenor (years)", "CDS spread (bps)", "CDS spread")
    if any(x < 0 for x in cds_bps):
        raise ValueError("CDS spreads cannot be negative.")
    cds_recovery = cds_recovery_pct / 100.0
    hazard = bootstrap_hazard_from_cds(cds_t, [x / 10000.0 for x in cds_bps], cds_recovery, rate_curve)

    if dividend_df is None or len(dividend_df) == 0:
        dividend_curve = Curve.flat(0.0, "Zero dividend curve")
    else:
        div_t, div_pct = _validate_term_structure(dividend_df, "Tenor (years)", "Dividend yield (%)", "Dividend curve")
        dividend_curve = Curve(tuple([0.0] + div_t[:-1]), tuple([div_pct[0] / 100.0] + [x / 100.0 for x in div_pct[1:]]), "step", "Dividend forwards (proxy)")

    bs_vol = Curve(tuple([0.0] + vol_t[:-1]), tuple([x / 100.0 for x in vol_pct]), "step", "User-entered OVDV implied volatility term structure")
    jd_vol = calibrate_jump_diffusion_vols(
        spot=spot,
        maturities=vol_t,
        black_scholes_vols=[x / 100.0 for x in vol_pct],
        strikes=strikes,
        risk_free_forward=rate_curve,
        dividend_forward=dividend_curve,
        hazard_rate=hazard,
        jtd_factor=jtd_pct / 100.0,
    )

    diagnostic = pd.DataFrame({
        "Expiry (years)": vol_t,
        "Calibration strike (USD)": strikes,
        "OVDV implied vol (%)": vol_pct,
        "Calibrated JD diffusion vol (%)": [v * 100.0 for v in jd_vol.values],
    })
    hazard_diag = pd.DataFrame({
        "CDS tenor (years)": cds_t,
        "CDS spread (bps)": cds_bps,
        "Bootstrapped hazard (%)": [h * 100.0 for h in hazard.values],
    })
    return rate_curve, dividend_curve, bs_vol, jd_vol, hazard, diagnostic, hazard_diag


def run_model(inputs: dict[str, Any]) -> dict[str, Any]:
    as_of = inputs["valuation_date"]
    maturity_date = inputs["maturity_date"]
    if maturity_date <= as_of:
        raise ValueError("Maturity date must be after valuation date.")
    maturity = (maturity_date - as_of).days / 365.0
    face = inputs["face"]
    coupon_rate = inputs["coupon_rate_pct"] / 100.0
    spot = inputs["spot"]
    conversion_ratio = inputs["conversion_ratio"]
    if min(face, spot, conversion_ratio) <= 0:
        raise ValueError("Face value, underlying spot and conversion ratio must be positive.")

    (rate_curve, dividend_curve, bs_vol, jd_vol, hazard,
     vol_diagnostic, hazard_diagnostic) = build_market_curves(
        inputs["vol_df"], inputs["rate_df"], inputs["cds_df"], inputs["dividend_df"],
        inputs["rate_input_type"], inputs["cds_recovery_pct"], spot, inputs["jtd_pct"],
    )

    bond = ConvertibleBond(
        face=face,
        maturity=maturity,
        conversion_ratio=conversion_ratio,
        coupon_schedule=make_coupon_schedule(as_of, maturity_date, face, coupon_rate),
        call_schedule={},
        put_schedule={},
        allow_early_conversion=True,
        name=inputs["bond_name"],
    )

    n_stock, n_time = inputs["n_stock"], inputs["n_time"]
    jd_params = OVCVParameters(
        model="jump_diffusion",
        risk_free_forward=rate_curve,
        dividend_forward=dividend_curve,
        volatility=jd_vol,
        hazard_rate=hazard,
        bond_recovery=inputs["bond_recovery_pct"] / 100.0,
        jtd_factor=inputs["jtd_pct"] / 100.0,
        borrow_cost=inputs["borrow_cost_pct"] / 100.0,
    )
    bs_params = OVCVParameters(
        model="black_scholes",
        risk_free_forward=rate_curve,
        dividend_forward=dividend_curve,
        volatility=bs_vol,
        borrow_cost=inputs["borrow_cost_pct"] / 100.0,
    )

    risky = price_convertible(bond, spot, jd_params, n_stock=n_stock, n_time=n_time)
    riskless = price_convertible(bond, spot, bs_params, n_stock=n_stock, n_time=n_time)
    ai = accrued_interest(face, coupon_rate, as_of)
    conversion_price = face / conversion_ratio
    moneyness = spot / conversion_price

    return {
        "risky_dirty": risky.price,
        "risky_clean": risky.price - ai,
        "riskless_dirty": riskless.price,
        "riskless_clean": riskless.price - ai,
        "risky_per_100": risky.price / face * 100.0,
        "riskless_per_100": riskless.price / face * 100.0,
        "riskless_minus_risky": riskless.price - risky.price,
        "accrued_interest": ai,
        "conversion_price": conversion_price,
        "moneyness": moneyness,
        "maturity_years": maturity,
        "vol_diagnostic": vol_diagnostic,
        "hazard_diagnostic": hazard_diagnostic,
        "hazard_curve": hazard,
        "jd_vol_curve": jd_vol,
        "notes": list(risky.notes) + list(riskless.notes),
        "bond": bond,
        "spot": spot,
    }


