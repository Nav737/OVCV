from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from ovcv_replica import (
    Curve,
    ConvertibleBond,
    OVCVParameters,
    bootstrap_hazard_from_cds,
    calibrate_jump_diffusion_vols,
    price_convertible,
)

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "bond_32076vae3_config.json"

st.set_page_config(
    page_title="OVCV Convertible Bond Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_default_config() -> dict[str, Any]:
    return json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


from dashboard_engine import run_model


# ------------------------------ Header -----------------------------------
st.title("Convertible Bond · OVCV Research Dashboard")
st.caption("Manual Bloomberg market-input entry → Jump-Diffusion calibration → risky and riskless model prices")
st.warning(
    "Research implementation of the disclosed OVCV whitepaper methodology — not Bloomberg software. "
    "The curves below start with illustrative placeholders. Replace every market input with your Bloomberg snapshot before interpreting the price."
)

cfg = load_default_config()
sec = cfg["security"]
mi = cfg["market_inputs"]

with st.sidebar:
    st.header("Bond details")
    bond_name = st.text_input("Security", value=sec["name"])
    cusip = st.text_input("CUSIP", value=sec["cusip"])
    valuation_date = st.date_input("Valuation date", value=date.fromisoformat(sec["valuation_date"]))
    maturity_date = st.date_input("Maturity date", value=date.fromisoformat(sec["maturity_date"]))
    face = st.number_input("Face value per bond (USD)", min_value=1.0, value=float(sec["face_per_note"]), step=100.0)
    coupon_rate_pct = st.number_input("Annual coupon (%)", min_value=0.0, max_value=30.0, value=float(sec["coupon_rate"] * 100.0), step=0.025, format="%.4f")
    conversion_ratio = st.number_input("Conversion ratio (shares per bond)", min_value=0.0001, value=float(sec["conversion_ratio_per_1000"]), step=0.1, format="%.4f")
    spot = st.number_input("Underlying spot (USD)", min_value=0.01, value=float(sec["underlying_spot"]), step=0.10, format="%.4f")
    st.caption("The sample spot and valuation date are historical example values; update them to your Bloomberg valuation snapshot.")
    st.divider()
    st.header("Credit & default controls")
    cds_recovery_pct = st.slider("CDS recovery (%)", 0, 90, int(round(mi["cds_recovery"] * 100)), 1)
    bond_recovery_pct = st.slider("Bond recovery (%)", 0, 100, int(round(mi["bond_recovery"] * 100)), 1)
    jtd_pct = st.slider("Jump-to-default factor / stock loss (%)", 0, 100, int(round(mi["jtd_factor"] * 100)), 1)
    st.caption("JTD is entered as the fraction of the stock lost on default: 100% means the stock is reduced to zero.")
    borrow_cost_pct = st.number_input("Borrow cost (%)", min_value=-20.0, max_value=50.0, value=float(mi["borrow_cost"] * 100.0), step=0.05, format="%.3f")
    st.divider()
    st.header("Numerical resolution")
    resolution = st.selectbox("PDE grid", ["Quick", "Standard", "Higher"], index=1)
    resolution_map = {"Quick": (201, 300), "Standard": (301, 500), "Higher": (401, 700)}
    n_stock, n_time = resolution_map[resolution]

# Tabs let the user enter each Bloomberg structure separately.
tab_vol, tab_rates, tab_credit, tab_other = st.tabs([
    "① Volatility term structure", "② Interest-rate term structure", "③ CDS term structure", "④ Other inputs & run"
])

with tab_vol:
    st.subheader("OVDV implied volatility term structure")
    st.write("Enter one implied volatility quote per expiry at the selected strike or moneyness. The engine calibrates a piecewise-constant Jump-Diffusion diffusion volatility to match the corresponding Black–Scholes option prices. You can paste a copied Excel/Bloomberg range into the editable table.")
    vol_default = pd.DataFrame({
        "Tenor (years)": mi["ovdv_bs_vol_maturities_years"],
        "Implied vol (%)": [x * 100.0 for x in mi["ovdv_bs_vols"]],
        "Calibration strike (USD)": mi["jd_vol_calibration_strikes"],
    })
    vol_df = st.data_editor(vol_default, num_rows="dynamic", use_container_width=True, key="vol_editor", column_config={
        "Tenor (years)": st.column_config.NumberColumn(format="%.3f", min_value=0.001),
        "Implied vol (%)": st.column_config.NumberColumn(format="%.3f", min_value=0.001, max_value=500.0),
        "Calibration strike (USD)": st.column_config.NumberColumn(format="%.4f", min_value=0.001),
    })
    st.info("This first dashboard uses a volatility term-structure slice (one selected strike per expiry), not a full two-dimensional expiry × strike volatility surface. Enter the strike used for each volatility quote.")

with tab_rates:
    st.subheader("Risk-free interest-rate curve")
    curve_label = st.selectbox("Curve selected in Bloomberg", ["SOFR", "Fixed / Other", "Custom"], index=0)
    rate_input_type = st.radio(
        "What does your Bloomberg curve display?",
        ["Zero rates", "Instantaneous forward rates"],
        horizontal=True,
        help="Choose zero rates for continuously compounded zero-coupon rates. Choose instantaneous forwards only if your Bloomberg output is already in forward-rate form.",
    )
    if rate_input_type == "Zero rates":
        st.info("The dashboard converts the entered continuously compounded zero rates into piecewise-constant instantaneous forward rates for the PDE.")
    else:
        st.info("The first entered forward applies from valuation to the first tenor; each subsequent quote applies from the prior tenor to its own tenor. The final quote is extended beyond the last node.")
    rate_default = pd.DataFrame({
        "Tenor (years)": mi["risk_free_forward_times_years"][1:],
        "Rate (%)": [x * 100.0 for x in mi["risk_free_forward_rates"][1:]],
    })
    rate_df = st.data_editor(rate_default, num_rows="dynamic", use_container_width=True, key="rate_editor", column_config={
        "Tenor (years)": st.column_config.NumberColumn(format="%.3f", min_value=0.001),
        "Rate (%)": st.column_config.NumberColumn(format="%.4f", min_value=-10.0, max_value=50.0),
    })
    st.caption("Do not enter par swap rates as zero rates. If your Bloomberg curve is quoted as par swap rates, it needs a separate instrument-by-instrument bootstrap first.")

with tab_credit:
    st.subheader("CDS / credit spread term structure")
    st.write("Enter quoted CDS par spreads in basis points. You can paste a copied Excel/Bloomberg range into the editable table. The dashboard bootstraps a piecewise-constant hazard curve using the selected CDS recovery assumption and the entered discount curve.")
    cds_default = pd.DataFrame({
        "Tenor (years)": mi["cds_maturities_years"],
        "CDS spread (bps)": mi["cds_par_spreads_bps"],
    })
    cds_df = st.data_editor(cds_default, num_rows="dynamic", use_container_width=True, key="cds_editor", column_config={
        "Tenor (years)": st.column_config.NumberColumn(format="%.3f", min_value=0.001),
        "CDS spread (bps)": st.column_config.NumberColumn(format="%.2f", min_value=0.0, max_value=100000.0),
    })
    st.caption("The CDS bootstrap is a transparent quarterly-premium approximation. Bloomberg's companion CDS calibration conventions may differ; this is a model-replication input, not a claim of exact Bloomberg hazard bootstrapping.")

with tab_other:
    st.subheader("Other market inputs")
    st.write("Dividend and borrow assumptions affect the equity component. Leave dividend yield at zero only when that is consistent with the underlying equity and valuation setup.")
    div_default = pd.DataFrame({
        "Tenor (years)": [1.0, 3.0, 5.0],
        "Dividend yield (%)": [0.0, 0.0, 0.0],
    })
    dividend_df = st.data_editor(div_default, num_rows="dynamic", use_container_width=True, key="div_editor", column_config={
        "Tenor (years)": st.column_config.NumberColumn(format="%.3f", min_value=0.001),
        "Dividend yield (%)": st.column_config.NumberColumn(format="%.4f", min_value=-20.0, max_value=50.0),
    })
    st.markdown("#### How the dashboard prices")
    st.markdown(
        "- **Risky price:** Jump-Diffusion PDE using CDS-implied hazard, bond recovery and JTD.\n"
        "- **Riskless price:** Black–Scholes PDE using the entered implied-volatility term structure, with hazard/default risk set to zero.\n"
        "- **Clean price:** Model dirty price less accrued coupon interest (30/360 US approximation for this example)."
    )

st.divider()
run_col, note_col = st.columns([1, 3])
with run_col:
    calculate = st.button("Calculate bond price", type="primary", use_container_width=True)
with note_col:
    st.caption("Calculation uses the entered curves and selected recovery/JTD assumptions. Results are not saved to Bloomberg or BRM.")

if calculate:
    current_inputs = {
        "bond_name": bond_name,
        "cusip": cusip,
        "valuation_date": valuation_date,
        "maturity_date": maturity_date,
        "face": float(face),
        "coupon_rate_pct": float(coupon_rate_pct),
        "conversion_ratio": float(conversion_ratio),
        "spot": float(spot),
        "cds_recovery_pct": float(cds_recovery_pct),
        "bond_recovery_pct": float(bond_recovery_pct),
        "jtd_pct": float(jtd_pct),
        "borrow_cost_pct": float(borrow_cost_pct),
        "rate_input_type": rate_input_type,
        "curve_label": curve_label,
        "vol_df": vol_df.copy(),
        "rate_df": rate_df.copy(),
        "cds_df": cds_df.copy(),
        "dividend_df": dividend_df.copy(),
        "n_stock": n_stock,
        "n_time": n_time,
    }
    try:
        with st.spinner("Calibrating Jump-Diffusion volatility and solving the convertible-bond PDE…"):
            result = run_model(current_inputs)
        st.session_state["last_result"] = result
        st.session_state["last_inputs"] = current_inputs
        st.session_state["last_error"] = None
    except Exception as exc:
        st.session_state["last_error"] = str(exc)
        st.session_state.pop("last_result", None)

if st.session_state.get("last_error"):
    st.error(f"Calculation could not be completed: {st.session_state['last_error']}")

if st.session_state.get("last_result"):
    res = st.session_state["last_result"]
    inp = st.session_state["last_inputs"]
    st.divider()
    st.header("Valuation results")
    st.caption(f"{inp['bond_name']} · {inp['cusip']} · Valuation date {inp['valuation_date'].isoformat()} · {inp['curve_label']} curve · Underlying spot ${res['spot']:,.4f}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risky price · clean per $100", f"{res['risky_clean']/inp['face']*100.0:,.4f}")
    m2.metric("Risky value · clean per bond", f"${res['risky_clean']:,.2f}")
    m3.metric("Riskless price · clean per $100", f"{res['riskless_clean']/inp['face']*100.0:,.4f}")
    m4.metric("Riskless minus risky · per $100", f"{res['riskless_minus_risky']/inp['face']*100.0:,.4f}", delta_color="inverse")

    price_comparison = pd.DataFrame([
        ["Risky Jump-Diffusion", res["risky_clean"], res["risky_dirty"], res["risky_clean"] / inp["face"] * 100.0, res["risky_dirty"] / inp["face"] * 100.0],
        ["Riskless Black-Scholes", res["riskless_clean"], res["riskless_dirty"], res["riskless_clean"] / inp["face"] * 100.0, res["riskless_dirty"] / inp["face"] * 100.0],
    ], columns=["Pricing case", "Clean value / bond (USD)", "Dirty value / bond (USD)", "Clean price / $100", "Dirty price / $100"])
    st.dataframe(price_comparison.style.format({
        "Clean value / bond (USD)": "${:,.4f}",
        "Dirty value / bond (USD)": "${:,.4f}",
        "Clean price / $100": "{:,.4f}",
        "Dirty price / $100": "{:,.4f}",
    }), hide_index=True, use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Bond and conversion diagnostics")
        diag = pd.DataFrame([
            ["Conversion price", f"${res['conversion_price']:,.4f}"],
            ["Spot / conversion price", f"{res['moneyness']*100.0:,.2f}%"],
            ["Remaining maturity", f"{res['maturity_years']:,.3f} years"],
            ["Accrued interest", f"${res['accrued_interest']:,.4f} per bond"],
            ["Bond recovery", f"{inp['bond_recovery_pct']:.1f}%"],
            ["CDS recovery", f"{inp['cds_recovery_pct']:.1f}%"],
            ["JTD stock loss", f"{inp['jtd_pct']:.1f}%"],
        ], columns=["Metric", "Value"])
        st.dataframe(diag, hide_index=True, use_container_width=True)
    with right:
        st.subheader("Bootstrapped CDS hazard rates")
        st.dataframe(res["hazard_diagnostic"].style.format({"CDS spread (bps)": "{:,.2f}", "Bootstrapped hazard (%)": "{:.4f}"}), hide_index=True, use_container_width=True)

    st.subheader("OVDV volatility → calibrated Jump-Diffusion volatility")
    st.dataframe(res["vol_diagnostic"].style.format({
        "Expiry (years)": "{:.3f}",
        "Calibration strike (USD)": "${:,.4f}",
        "OVDV implied vol (%)": "{:.3f}%",
        "Calibrated JD diffusion vol (%)": "{:.3f}%",
    }), hide_index=True, use_container_width=True)
    chart_df = res["vol_diagnostic"].set_index("Expiry (years)")[["OVDV implied vol (%)", "Calibrated JD diffusion vol (%)"]]
    st.line_chart(chart_df, use_container_width=True)

    # Export a compact audit record and the curve calibration diagnostics.
    export_rows = [
        ("CUSIP", inp["cusip"]),
        ("Valuation date", inp["valuation_date"].isoformat()),
        ("Underlying spot USD", res["spot"]),
        ("Risky JD dirty price per bond", res["risky_dirty"]),
        ("Risky JD clean price per bond", res["risky_clean"]),
        ("Riskless BS dirty price per bond", res["riskless_dirty"]),
        ("Riskless BS clean price per bond", res["riskless_clean"]),
        ("Riskless minus risky per bond", res["riskless_minus_risky"]),
        ("Interest curve label", inp["curve_label"]),
        ("CDS recovery percent", inp["cds_recovery_pct"]),
        ("Bond recovery percent", inp["bond_recovery_pct"]),
        ("JTD percent", inp["jtd_pct"]),
    ]
    export_df = pd.DataFrame(export_rows, columns=["Metric", "Value"])
    st.download_button(
        "Download valuation summary (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"ovcv_{inp['cusip']}_{inp['valuation_date'].isoformat()}_summary.csv",
        mime="text/csv",
    )

    st.warning(
        "Interpretation limits: the current solver implements the paper's core one-factor PDE, coupon cashflows, early conversion, "
        "CDS-hazard approximation and user-entered volatility calibration strikes. It does not yet reproduce Bloomberg's proprietary "
        "CDS bootstrapping, full two-dimensional volatility-surface calibration, path-dependent soft-call trigger, make-whole schedule, "
        "fundamental-change put, or every terminal/notice convention. Treat the displayed values as research-model outputs, not validated Bloomberg OVCV prices."
    )

st.divider()
st.caption("OVCV Research Dashboard · Based on the publicly available 2014 Bloomberg OVCV Model Description · Independent research implementation")
