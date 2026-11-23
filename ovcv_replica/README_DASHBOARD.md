# OVCV Convertible Bond Dashboard

This is a local interactive dashboard for entering Bloomberg market term structures and obtaining research-model convertible bond prices using the core one-factor Jump-Diffusion PDE described in the public 2014 Bloomberg OVCV model paper.

## Run on Windows

1. Install Python 3.10 or later.
2. Extract the project ZIP to a folder.
3. Open the folder in File Explorer.
4. Double-click `run_dashboard.bat`, or open PowerShell in the folder and run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run dashboard_app.py
```

The dashboard will open in your browser (usually at `http://localhost:8501`).

## Run on macOS / Linux

```bash
python -m pip install -r requirements.txt
python -m streamlit run dashboard_app.py
```

## Dashboard inputs

- **OVDV volatility term structure:** expiry in years, implied volatility in percent, and the calibration strike in USD for each expiry. The solver calibrates piecewise-constant Jump-Diffusion diffusion volatility to match the option prices implied by those Black-Scholes volatilities.
- **Interest-rate term structure:** either continuously compounded zero rates (which are converted to piecewise-constant instantaneous forwards) or instantaneous forward rates. Do not enter par swap rates as zero rates without bootstrapping the curve first.
- **CDS term structure:** CDS par spreads in basis points, plus the CDS recovery slider. The dashboard bootstraps an approximate piecewise-constant hazard curve.
- **Bond recovery:** recovery fraction applied to the convertible's default payoff.
- **Jump-to-default:** fraction of the stock price lost on default. 100% means the stock is wiped out by the jump.
- **Riskless comparison:** prices the bond with the Black-Scholes/no-default case alongside the risky Jump-Diffusion result.
- **Dividend yield and borrow cost:** optional equity carry assumptions.

You can edit the tables directly or paste rows copied from Excel. Tenors must be positive and strictly increasing.

## Output

The dashboard displays:

- Risky Jump-Diffusion clean and dirty value.
- Riskless Black-Scholes clean and dirty value.
- Prices per $100 of face and per bond denomination.
- Riskless-minus-risky valuation difference.
- CDS-implied hazard-rate diagnostics.
- Input OVDV volatility and calibrated Jump-Diffusion volatility by expiry.
- A downloadable CSV valuation summary.

## Important modelling boundaries

This is an independent research implementation, not Bloomberg software. In particular:

1. The volatility input is a **term-structure slice**: one volatility quote and one selected strike per expiry. It is not yet a full two-dimensional expiry-by-strike OVDV surface.
2. The strike-selection convention is user-supplied; the paper does not prescribe a unique strike convention.
3. The CDS hazard bootstrap is a transparent quarterly-premium approximation. It is not Bloomberg's proprietary companion CDS model or a guarantee of matching Bloomberg's CDS hazard curve.
4. Complex path-dependent features such as the First Majestic soft-call 20-of-30 condition, full make-whole conversion table, fundamental-change put and all notice/settlement conventions are not yet implemented in the PDE.
5. The clean-price accrued-interest adjustment uses a 30/360 US approximation for the example. Verify the exact security convention and Bloomberg settlement treatment.
6. Riskless pricing is the zero-hazard Black-Scholes comparison. It is not automatically equivalent to a Bloomberg-specific switch unless the Terminal setting is confirmed.
7. The price should not be used for a production valuation or model sign-off until the inputs, conventions, grid convergence and outputs have been independently validated against Bloomberg OVCV.

## Files

- `dashboard_app.py`: Streamlit user interface.
- `dashboard_engine.py`: dashboard calculation layer independent of Streamlit.
- `ovcv_replica.py`: PDE, CDS bootstrap and volatility calibration functions.
- `bond_32076vae3_config.json`: sample contractual terms and explicitly illustrative market inputs.
- `test_dashboard_engine.py`: dashboard-engine tests.
- `README_32076VAE3.md`: instrument details and implementation limitations.

All market data initially populated for the example are placeholders. Replace them with your own Bloomberg snapshot before using the price output.
