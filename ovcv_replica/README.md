# OVCV core-model research replica

This is a transparent Python research implementation of the **core one-factor Jump-Diffusion PDE** disclosed in David Frank, Bloomberg Quantitative Finance Development, *OVCV Model Description* (15 May 2014).

It is not Bloomberg software, does not contain Bloomberg source code, and should not be represented as an exact replacement for the Bloomberg Terminal function. The document discloses the model equations and broad calibration approach, but not all production numerical methods, market conventions, boundary details, calibration instruments or proprietary companion model specifications.

## Included

- Core Jump-Diffusion convertible-bond PDE on a log-stock grid.
- Black-Scholes as the zero-hazard special case.
- Bond recovery and fractional jump-to-default factor as separate controls.
- Term structures for instantaneous forward rates, dividend yields, diffusion volatility and hazard rates.
- OAS as a parallel shift to the instantaneous forward curve **in Black-Scholes mode**, following the paper.
- Discrete coupon cashflows, simple hard call/put constraints and early conversion.
- CDS spread-to-hazard bootstrap using a transparent standard approximation, with a separate CDS recovery assumption.
- Jump-Diffusion volatility calibration to user-supplied Black-Scholes implied volatilities and explicit option strikes.
- Risky vs riskless model comparison and scalar parameter sensitivity helper.
- Borrow cost included as an addition to the continuous dividend yield.

## Important limitations

1. The paper references a separate Bloomberg CDS model document, DOCS 2057273, for CDS-to-hazard calibration. That companion specification is not included here. The supplied CDS bootstrap is therefore a transparent approximation (quarterly premiums and midpoint accrued-on-default), **not** a claim to reproduce Bloomberg's CDS calibration.
2. The paper says E2C must preserve CDS-implied expected hazard rates and jointly calibrate hazard and diffusion volatility. The core PDE supports the E2C functional form `h(S,t) = h0(t) * (S0/S)^p`, but exact survival-conditioned normalization of `h0` and joint calibration are not yet implemented. For now, pass a suitably calibrated `h0` term structure and treat E2C outputs as research approximations.
3. The paper describes multiple contractual features that need extra state variables or bespoke constraints: dividend protection, N-of-M soft calls, contingent conversion, cross-currency convertibles, mandatory convertibles, exchangeables, reset convertibles, make-whole calls, dividend-forfeit clauses, CoPay clauses and averaging periods. They are **not** fully implemented in this core engine.
4. Calls and puts are represented only as simple date-based hard constraints; they are not equivalent to all Bloomberg-specific term-sheet treatments.
5. OAS solving is intentionally restricted to Black-Scholes mode. The 2014 paper explicitly describes OAS as a parallel shift to the instantaneous forward curve for Black-Scholes. It does not specify adding OAS on top of the Jump-Diffusion CDS/default mechanism.
6. The volatility calibrator requires explicit option strikes. The paper describes matching vanilla European option prices but does not prescribe a universally applicable strike-selection rule in the disclosed text.
7. The PDE uses a Crank-Nicolson-style finite-difference scheme. Bloomberg's exact grid construction, time stepping, boundary conditions, interpolation and convergence criteria are not disclosed in the paper.

## Installation

```bash
python -m pip install numpy scipy
```

## Quick start

```python
from ovcv_replica import Curve, ConvertibleBond, OVCVParameters, price_convertible

r = Curve.flat(0.04, "instantaneous SOFR forwards")
q = Curve.flat(0.01, "dividend forwards")
sigma = Curve.flat(0.30, "JD diffusion volatility")
hazard = Curve.flat(0.03, "CDS-implied hazard")

bond = ConvertibleBond(
    face=100.0,
    maturity=5.0,
    conversion_ratio=5.0,
    coupon_schedule={1.0: 2.0, 2.0: 2.0, 3.0: 2.0, 4.0: 2.0, 5.0: 2.0},
)

params = OVCVParameters(
    model="jump_diffusion",
    risk_free_forward=r,
    dividend_forward=q,
    volatility=sigma,
    hazard_rate=hazard,
    bond_recovery=0.40,
    jtd_factor=0.60,
)

result = price_convertible(bond, spot=18.0, params=params)
print(result.price)
```

## CDS recovery vs bond recovery

- `cds_recovery` is used only by `bootstrap_hazard_from_cds(...)` to translate market CDS par spreads into a hazard term structure.
- `bond_recovery` is used in the convertible default payoff `max(R_bond * Face, ConversionRatio * Stock * (1 - JTD))`.
- These controls are intentionally independent.

## Curve input convention

The model expects **instantaneous forward curves**, not par yields. If you have continuously compounded zero rates, use `Curve.from_zero_rates(times, zero_rates)` as a first conversion. Verify Bloomberg's curve representation and compounding/day-count settings before comparing outputs.

## Run the smoke test

```bash
python ovcv_replica.py
```

For model validation, increase `n_stock` and `n_time` and check price convergence. Compare only after aligning valuation date, settlement, clean/dirty price, coupon schedule, curve conventions, market volatility surface, CDS calibration and bond terms with the Terminal.

## Worked security example: First Majestic Silver CUSIP 32076VAE3

See `README_32076VAE3.md`, `bond_32076vae3_config.json` and `example_32076vae3.py` for a dated worked example using the publicly disclosed terms of First Majestic Silver's 0.125% Convertible Senior Notes due 2031. The configured share spot is the November 20, 2026 NYSE close. All rates, implied volatilities, CDS spreads, recoveries and the JTD factor in that example are explicitly marked as illustrative placeholders and must be replaced with timestamp-aligned Bloomberg inputs before any BRM comparison.

A blank market-data capture file is provided as `bloomberg_market_data_template_32076vae3.csv`. The worked example emits per-USD-1,000 and per-USD-100 values, accrued interest, an approximate CDS hazard calibration, a selected-strike JD volatility calibration, and scalar price sensitivities.

## Interactive dashboard

A Streamlit dashboard is included for entering Bloomberg term structures and calculating risky Jump-Diffusion and riskless Black-Scholes research-model prices. It includes editable OVDV volatility, risk-free interest-rate and CDS spread tables, plus independent CDS recovery, bond recovery and jump-to-default controls.

See `README_DASHBOARD.md` for setup, input conventions and limitations. On Windows, run `run_dashboard.bat`; on macOS/Linux, run `./run_dashboard.sh`.
