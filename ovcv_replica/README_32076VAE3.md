# Worked example: CUSIP 32076VAE3

## Identified instrument

**First Majestic Silver Corp. 0.125% Convertible Senior Notes due January 15, 2031**

- CUSIP: `32076VAE3`
- ISIN: `US32076VAE39`
- Currency: USD
- Aggregate initial principal: USD 350 million
- Issue/indenture date: December 8, 2025
- Coupon: 0.125% per annum, semiannual in arrears on January 15 and July 15
- Coupon day count: 360-day year of twelve 30-day months
- Maturity: January 15, 2031
- Initial conversion rate: 44.7227 common shares per USD 1,000 principal
- Initial conversion price: approximately USD 22.36 per share
- Convertible at the holder's option until the close of business on the business day immediately preceding maturity, subject to the indenture
- Senior unsecured obligations
- Optional issuer redemption: on or after January 20, 2029, if the share closing price exceeds 130% of the then-current conversion price on at least 20 trading days in a 30-consecutive-trading-day period; redemption price is 100% of principal plus accrued and unpaid interest
- Fundamental Change repurchase: 100% of principal plus accrued and unpaid interest
- Make-whole conversion feature: the indenture contains a make-whole share adjustment table and a stated make-whole cap of 63.7349 shares per USD 1,000, subject to adjustment

Sources:

1. SEC-filed indenture, dated December 8, 2025: https://www.sec.gov/Archives/edgar/data/1308648/000119312525325427/d50988dex992.htm
2. Issuer's completion announcement, December 8, 2025: https://firstmajestic.com/_resources/news/nr-20251208.pdf?v=0.2
3. First Majestic stock information page (NYSE: AG; USD close of USD 10.91, last updated November 20, 2026): https://firstmajestic.com/investors/stock-information/

## Valuation date and spot convention in the example

The example is dated **November 20, 2026** to align the valuation date to the most recent complete NYSE close retrieved for the underlying share. The close used is **USD 10.91**. The calculated initial conversion price is USD 1,000 / 44.7227 = approximately USD 22.36, so the share price is approximately 48.79% of the initial conversion price at that dated spot reference.

The script generates the remaining coupon dates from November 20, 2026 to maturity, with a coupon cash flow of USD 0.625 per USD 1,000 principal for each regular semiannual coupon date. Accrued interest is calculated separately from July 15, 2026 using 30/360. The exact Bloomberg settlement date, ex-coupon status, holiday roll and price convention must be checked before comparing against OVCV.

## Critical market-data warning

The JSON file uses **illustrative placeholders** for:

- SOFR / risk-free instantaneous forward curve
- Dividend forward curve
- OVDV Black-Scholes implied-volatility inputs
- CDS par spread term structure
- CDS recovery and bond recovery
- Jump-to-default (JTD) factor
- E2C exponent

These placeholders are deliberately visible in `bond_32076vae3_config.json`. They are not actual Bloomberg observations, not contractual terms and not claimed to be First Majestic's market calibration. The example price is therefore a **software demonstration only**, not a fair value, quote, investment recommendation or BRM validation result.

Before calibration testing, replace every placeholder with values captured from the same Bloomberg valuation timestamp and confirm curve currency, seniority, restructuring clause, day count, compounding, interpolation and settlement conventions.

## What the example runs

1. Builds the remaining coupon schedule from the security's payment terms.
2. Translates example CDS par spreads into an approximate piecewise-constant hazard curve using the project's transparent CDS bootstrap.
3. Calibrates a piecewise-constant Jump-Diffusion diffusion-volatility term structure to user-supplied Black-Scholes option prices at one explicit strike per expiry.
4. Prices the note with the core Jump-Diffusion PDE using separate bond recovery and JTD controls.
5. Prices a riskless Black-Scholes comparator using the input Black-Scholes volatility term structure.
6. Runs one-at-a-time price shocks for bond recovery, JTD, CDS recovery and risk-free forwards.
7. Calculates accrued interest and reports dirty and clean model values.
8. Skips OAS solving unless an observed Bloomberg clean market price is entered in the config. When entered, the code converts clean to dirty using accrued interest and solves OAS in the disclosed Black-Scholes parallel-forward-shift convention.

## Important model scope / limitations

This is a transparent research implementation of the core one-factor Jump-Diffusion PDE disclosed in David Frank, Bloomberg Quantitative Finance Development, *OVCV Model Description* (May 15, 2014). It is not Bloomberg software and cannot be represented as an exact replacement for Bloomberg Terminal OVCV.

For this specific note, the following contract provisions are recorded but **not fully priced** by the simple one-factor PDE:

- The issuer's 20-of-30-trading-day soft-call condition from January 20, 2029. It is path-dependent and cannot be represented by a deterministic call date without approximation.
- The holder's Fundamental Change repurchase right. This is contingent on an external event, not a scheduled put date.
- The make-whole conversion-rate table and its effective-date / share-price interpolation.
- Anti-dilution conversion-rate adjustments and potential additional amounts.

The volatility calibration in this worked example uses one selected strike for each expiry (configured as spot-ATM). It is a **term-structure slice**, not a full strike-by-expiry volatility-surface replication. The white paper's disclosed model description does not provide all of Bloomberg's production surface mapping, strike-selection and numerical conventions needed to claim exact replication.

The CDS-to-hazard routine is an explicitly disclosed approximation; the OVCV description refers to a separate Bloomberg CDS model specification for the production calibration. The E2C functional form is available in the core solver, but exact CDS-preserving hazard normalization and joint E2C/volatility calibration are not complete.

## Run

```bash
python -m pip install -r requirements.txt
python example_32076vae3.py
```

Outputs are created in `outputs_32076vae3/`:

- `summary_32076vae3.csv`
- `sensitivity_32076vae3.csv`
- `vol_hazard_calibration_32076vae3.csv`
- `coupon_schedule_32076vae3.csv`
- `oas_solution_SKIPPED.txt` until a clean market price is populated

To solve OAS, set `market_clean_price_per_1000` in the JSON to the observed Bloomberg clean price **for the same valuation and settlement conventions**. The model adds accrued interest to form a dirty target before solving.

## Run tests

```bash
python -m unittest -v
```
