"""Run the OVCV research replica with CUSIP 32076VAE3 as a worked example.

All contractual fields are sourced from the issuer's indenture / offering
release. The share spot is a dated public close. Market curves and model
parameters in the accompanying JSON are explicitly illustrative placeholders;
replace them with Bloomberg OVDV / curve / OVCV settings before validation.

Usage:
    python example_32076vae3.py

Outputs are written to ./outputs_32076vae3/.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ovcv_replica import (
    Curve,
    ConvertibleBond,
    OVCVParameters,
    bootstrap_hazard_from_cds,
    calibrate_jump_diffusion_vols,
    price_convertible,
    solve_oas,
)

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "bond_32076vae3_config.json"
OUT = HERE / "outputs_32076vae3"


def parse_date(s: str) -> date:
    return date.fromisoformat(s)


def year_fraction_act365(start: date, end: date) -> float:
    return (end - start).days / 365.0


def next_coupon_dates(as_of: date, maturity: date) -> list[date]:
    dates: list[date] = []
    for year in range(as_of.year, maturity.year + 1):
        for month in (1, 7):
            candidate = date(year, month, 15)
            if as_of < candidate <= maturity:
                dates.append(candidate)
    return sorted(dates)


def thirty_360_us(start: date, end: date) -> int:
    """30/360 US day count for the coupon accrued-interest illustration."""
    d1, d2 = start.day, end.day
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30
    return 360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)


def accrued_interest_30_360(
    face: float, coupon_rate: float, last_coupon: date, settlement: date
) -> float:
    days = thirty_360_us(last_coupon, settlement)
    return face * coupon_rate * days / 360.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def shifted_curve(curve: Curve, bump: float, name: str) -> Curve:
    return Curve(curve.times, [x + bump for x in curve.values], curve.interpolation, name)


def build_inputs(cfg: dict[str, Any]):
    sec = cfg["security"]
    mi = cfg["market_inputs"]
    as_of = parse_date(sec["valuation_date"])
    maturity_date = parse_date(sec["maturity_date"])
    maturity = year_fraction_act365(as_of, maturity_date)
    face = float(sec["face_per_note"])
    coupon_rate = float(sec["coupon_rate"])
    coupon_amount = face * coupon_rate / 2.0

    dates = next_coupon_dates(as_of, maturity_date)
    coupon_schedule = {
        year_fraction_act365(as_of, d): coupon_amount for d in dates
    }

    bond = ConvertibleBond(
        face=face,
        maturity=maturity,
        conversion_ratio=float(sec["conversion_ratio_per_1000"]),
        coupon_schedule=coupon_schedule,
        # The indenture's issuer call is conditional on a 20-of-30 trading-day
        # stock-price test and begins in 2029. It is intentionally NOT mapped
        # to a deterministic hard call in this single-factor research PDE.
        call_schedule={},
        # The Fundamental Change repurchase is a contingent event, not a
        # deterministic scheduled put. It is recorded in the term sheet only.
        put_schedule={},
        allow_early_conversion=True,
        name=sec["name"],
    )

    r = Curve(
        mi["risk_free_forward_times_years"],
        mi["risk_free_forward_rates"],
        "linear",
        "Illustrative instantaneous risk-free forwards",
    )
    q = Curve(
        mi["dividend_forward_times_years"],
        mi["dividend_forward_rates"],
        "linear",
        "Illustrative instantaneous dividend forwards",
    )
    bs_vol = Curve(
        [0.0] + mi["ovdv_bs_vol_maturities_years"],
        [mi["ovdv_bs_vols"][0]] + mi["ovdv_bs_vols"],
        "step",
        "Illustrative OVDV Black-Scholes implied vol proxy",
    )
    cds_spreads = [x / 10000.0 for x in mi["cds_par_spreads_bps"]]
    hazard = bootstrap_hazard_from_cds(
        mi["cds_maturities_years"],
        cds_spreads,
        float(mi["cds_recovery"]),
        r,
    )
    jd_vol = calibrate_jump_diffusion_vols(
        spot=float(sec["underlying_spot"]),
        maturities=mi["ovdv_bs_vol_maturities_years"],
        black_scholes_vols=mi["ovdv_bs_vols"],
        strikes=mi["jd_vol_calibration_strikes"],
        risk_free_forward=r,
        dividend_forward=q,
        hazard_rate=hazard,
        jtd_factor=float(mi["jtd_factor"]),
    )

    jd_params = OVCVParameters(
        model="jump_diffusion",
        risk_free_forward=r,
        dividend_forward=q,
        volatility=jd_vol,
        hazard_rate=hazard,
        bond_recovery=float(mi["bond_recovery"]),
        jtd_factor=float(mi["jtd_factor"]),
        borrow_cost=float(mi["borrow_cost"]),
        e2c_exponent=float(mi["e2c_exponent"]),
        e2c_reference_spot=float(sec["underlying_spot"]),
    )
    bs_params = OVCVParameters(
        model="black_scholes",
        risk_free_forward=r,
        dividend_forward=q,
        volatility=bs_vol,
        borrow_cost=float(mi["borrow_cost"]),
    )
    return bond, r, q, bs_vol, hazard, jd_vol, jd_params, bs_params, dates, maturity


def run() -> None:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sec, mi, num = cfg["security"], cfg["market_inputs"], cfg["numerics"]
    OUT.mkdir(exist_ok=True)
    n_stock, n_time = int(num["n_stock"]), int(num["n_time"])
    spot = float(sec["underlying_spot"])

    bond, r, q, bs_vol, hazard, jd_vol, jd, bs, coupon_dates, maturity = build_inputs(cfg)
    risky = price_convertible(bond, spot, jd, n_stock=n_stock, n_time=n_time)
    riskless = price_convertible(bond, spot, bs, n_stock=n_stock, n_time=n_time)

    # Bond accrued interest: contract specifies a 30/360 basis. Last coupon
    # date is 2026-07-15; settlement/as-of is 2026-11-20 in this example.
    last_coupon = date(2026, 7, 15)
    ai = accrued_interest_30_360(
        float(sec["face_per_note"]), float(sec["coupon_rate"]), last_coupon,
        parse_date(sec["valuation_date"]),
    )
    clean_risky = risky.price - ai
    clean_riskless = riskless.price - ai

    conv_price = float(sec["face_per_note"]) / float(sec["conversion_ratio_per_1000"])
    moneyness = spot / conv_price
    summary = [
        {"metric": "CUSIP", "value": sec["cusip"], "status": "contractual identifier"},
        {"metric": "valuation_date", "value": sec["valuation_date"], "status": "chosen to align with last verified share close"},
        {"metric": "underlying_spot_USD", "value": f"{spot:.4f}", "status": "public close on 2026-11-20"},
        {"metric": "maturity_years_ACT365", "value": f"{maturity:.8f}", "status": "date calculation"},
        {"metric": "conversion_price_calculated", "value": f"{conv_price:.6f}", "status": "1000 / 44.7227; indenture says approximately 22.36"},
        {"metric": "spot_to_conversion_price", "value": f"{moneyness:.4%}", "status": "below conversion price at spot reference"},
        {"metric": "accrued_interest_per_1000", "value": f"{ai:.6f}", "status": "30/360 approximation from 2026-07-15"},
        {"metric": "risky_JD_dirty_value_per_1000", "value": f"{risky.price:.6f}", "status": "ILLUSTRATIVE market inputs only"},
        {"metric": "risky_JD_clean_value_per_1000", "value": f"{clean_risky:.6f}", "status": "dirty value less accrued interest"},
        {"metric": "risky_JD_dirty_value_per_100", "value": f"{risky.price/10.0:.6f}", "status": "same value scaled to per-100 par convention"},
        {"metric": "riskless_BS_dirty_value_per_1000", "value": f"{riskless.price:.6f}", "status": "ILLUSTRATIVE market inputs only"},
        {"metric": "riskless_BS_clean_value_per_1000", "value": f"{clean_riskless:.6f}", "status": "dirty value less accrued interest"},
        {"metric": "riskless_BS_dirty_value_per_100", "value": f"{riskless.price/10.0:.6f}", "status": "same value scaled to per-100 par convention"},
        {"metric": "riskless_minus_risky_dirty", "value": f"{riskless.price - risky.price:.6f}", "status": "model-based risky/riskless gap; not an accounting CVA"},
        {"metric": "bond_recovery_cash_floor", "value": f"{jd.bond_recovery * bond.face:.6f}", "status": "per $1,000 principal using illustrative bond recovery"},
        {"metric": "JTD_adjusted_conversion_payoff_at_spot", "value": f"{bond.conversion_ratio * spot * (1.0-jd.jtd_factor):.6f}", "status": "per $1,000 principal using illustrative JTD factor"},
        {"metric": "spot_where_JTD_conversion_payoff_equals_bond_recovery", "value": f"{(jd.bond_recovery*bond.face)/(bond.conversion_ratio*(1.0-jd.jtd_factor)):.6f}", "status": "illustrative threshold at baseline recovery and JTD"},
        {"metric": "CDS_recovery", "value": mi["cds_recovery"], "status": "illustrative placeholder"},
        {"metric": "bond_recovery", "value": mi["bond_recovery"], "status": "illustrative placeholder"},
        {"metric": "JTD_factor", "value": mi["jtd_factor"], "status": "illustrative placeholder"},
        {"metric": "market_clean_price", "value": mi["market_clean_price_per_1000"], "status": "not supplied; OAS solve skipped"},
    ]
    write_csv(OUT / "summary_32076vae3.csv", summary)

    # Market input / calibration diagnostics.
    calibration_rows = []
    for i, (t, vbs, strike) in enumerate(zip(
        mi["ovdv_bs_vol_maturities_years"],
        mi["ovdv_bs_vols"],
        mi["jd_vol_calibration_strikes"],
    )):
        calibration_rows.append({
            "expiry_years": t,
            "calibration_strike_USD": strike,
            "BS_implied_vol_placeholder": vbs,
            "calibrated_JD_diffusion_vol": jd_vol.values[min(i, len(jd_vol.values)-1)],
            "hazard_rate_at_expiry_approx": hazard.at(t),
        })
    write_csv(OUT / "vol_hazard_calibration_32076vae3.csv", calibration_rows)

    # Sensitivity scenarios: one input family at a time, re-running the
    # jump-diffusion volatility calibration after any CDS/vol/rate changes.
    scenarios: list[dict[str, Any]] = []
    base_dirty = risky.price
    scenarios.append({"scenario": "BASE", "dirty_value_per_1000": base_dirty, "dirty_value_per_100": base_dirty/10.0,
                      "change_per_1000": 0.0, "change_per_100": 0.0, "pct_change": 0.0, "notes": "illustrative inputs"})

    def add_scenario(name: str, p: OVCVParameters, note: str) -> None:
        px = price_convertible(bond, spot, p, n_stock=n_stock, n_time=n_time).price
        scenarios.append({"scenario": name, "dirty_value_per_1000": px, "dirty_value_per_100": px/10.0,
                          "change_per_1000": px-base_dirty, "change_per_100": (px-base_dirty)/10.0,
                          "pct_change": (px/base_dirty-1.0)*100.0, "notes": note})

    for bump, tag in ((0.05, "+5pp"), (-0.05, "-5pp")):
        p = OVCVParameters(**{**jd.__dict__, "bond_recovery": min(1.0, max(0.0, jd.bond_recovery + bump))})
        add_scenario(f"Bond recovery {tag}", p, "CDS hazard and JD volatility held fixed")

    for bump, tag in ((0.10, "+10pp"), (-0.10, "-10pp")):
        jtd_new = min(1.0, max(0.0, jd.jtd_factor + bump))
        p = OVCVParameters(**{**jd.__dict__, "jtd_factor": jtd_new})
        add_scenario(f"JTD factor {tag}", p, "Illustrative direct JTD shock; JD volatility not recalibrated in this simple isolated shock")

    # CDS recovery shocks: market CDS par spreads fixed; re-bootstrap hazards.
    for bump, tag in ((0.10, "+10pp"), (-0.10, "-10pp")):
        cds_rec_new = min(0.95, max(0.05, float(mi["cds_recovery"]) + bump))
        h_new = bootstrap_hazard_from_cds(
            mi["cds_maturities_years"],
            [x/10000.0 for x in mi["cds_par_spreads_bps"]],
            cds_rec_new, r,
        )
        jd_vol_new = calibrate_jump_diffusion_vols(
            spot, mi["ovdv_bs_vol_maturities_years"], mi["ovdv_bs_vols"],
            mi["jd_vol_calibration_strikes"], r, q, h_new, jd.jtd_factor,
        )
        p = OVCVParameters(
            model="jump_diffusion", risk_free_forward=r, dividend_forward=q,
            volatility=jd_vol_new, hazard_rate=h_new,
            bond_recovery=jd.bond_recovery, jtd_factor=jd.jtd_factor,
            borrow_cost=jd.borrow_cost, e2c_exponent=jd.e2c_exponent,
            e2c_reference_spot=jd.e2c_reference_spot,
        )
        add_scenario(f"CDS recovery {tag}", p, "CDS par spreads fixed; hazard and JD volatility recalibrated")

    # Parallel +/− 25 bp risk-free forwards; rerun JD-vol calibration against
    # the same OVDV BS vols because the option targets depend on rates.
    for bump, tag in ((0.0025, "+25bp"), (-0.0025, "-25bp")):
        r_new = shifted_curve(r, bump, f"Risk-free forwards {tag}")
        h_new = bootstrap_hazard_from_cds(
            mi["cds_maturities_years"], [x/10000.0 for x in mi["cds_par_spreads_bps"]],
            float(mi["cds_recovery"]), r_new,
        )
        jd_vol_new = calibrate_jump_diffusion_vols(
            spot, mi["ovdv_bs_vol_maturities_years"], mi["ovdv_bs_vols"],
            mi["jd_vol_calibration_strikes"], r_new, q, h_new, jd.jtd_factor,
        )
        p = OVCVParameters(
            model="jump_diffusion", risk_free_forward=r_new, dividend_forward=q,
            volatility=jd_vol_new, hazard_rate=h_new,
            bond_recovery=jd.bond_recovery, jtd_factor=jd.jtd_factor,
            borrow_cost=jd.borrow_cost, e2c_exponent=jd.e2c_exponent,
            e2c_reference_spot=jd.e2c_reference_spot,
        )
        add_scenario(f"Risk-free curve {tag}", p, "CDS bootstrap and JD-vol calibration rerun")

    write_csv(OUT / "sensitivity_32076vae3.csv", scenarios)

    # OAS is solved only if a clean market price is entered. The price engine's
    # output is dirty, so accrued interest is added to a clean target first.
    market_clean = mi.get("market_clean_price_per_1000")
    if market_clean is not None:
        market_dirty = float(market_clean) + ai
        oas = solve_oas(bond, spot, market_dirty, bs, n_stock=n_stock, n_time=n_time)
        write_csv(OUT / "oas_solution_32076vae3.csv", [{
            "market_clean_price_per_1000": market_clean,
            "accrued_interest": ai,
            "market_dirty_price_per_1000": market_dirty,
            "solved_OAS_decimal": oas,
            "solved_OAS_bps": oas * 10000.0,
            "model_convention": "Black-Scholes mode; parallel shift to instantaneous forwards per disclosed paper",
        }])
    else:
        (OUT / "oas_solution_SKIPPED.txt").write_text(
            "OAS was not solved because market_clean_price_per_1000 is null. "
            "Enter the observed Bloomberg clean price in bond_32076vae3_config.json. "
            "The example converts it to dirty price using accrued interest and solves "
            "OAS in the paper's Black-Scholes convention.\n",
            encoding="utf-8",
        )

    coupon_rows = [
        {"coupon_date": d.isoformat(), "time_from_valuation_ACT365": year_fraction_act365(parse_date(sec["valuation_date"]), d),
         "coupon_cashflow_per_1000_USD": float(sec["face_per_note"]) * float(sec["coupon_rate"]) / 2.0}
        for d in coupon_dates
    ]
    write_csv(OUT / "coupon_schedule_32076vae3.csv", coupon_rows)

    print("Security:", sec["cusip"], "-", sec["name"])
    print("As of:", sec["valuation_date"], "Underlying spot:", spot)
    print("Conversion price:", f"${conv_price:.4f}", "Spot / conversion price:", f"{moneyness:.2%}")
    print("Accrued interest per $1,000:", f"${ai:.4f}")
    print("Risky JD dirty value (illustrative):", f"${risky.price:.4f}")
    print("Riskless BS dirty value (illustrative):", f"${riskless.price:.4f}")
    print("Outputs:", OUT)
    print("Market-data warning:", mi["market_data_status"])


if __name__ == "__main__":
    run()
