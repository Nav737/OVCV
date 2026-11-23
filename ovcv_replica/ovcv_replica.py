"""Transparent research implementation of the core model disclosed in
Bloomberg's 2014 OVCV Model Description (David Frank).

This is NOT Bloomberg software and is not expected to reproduce Bloomberg
prices exactly. The PDE is disclosed in the paper; numerical grids, event
handling and several market conventions are implementation choices here.

Units:
- time: years from valuation date
- rates, volatility, hazard, spreads: decimals (e.g. 5% = 0.05)
- bond prices and stock prices: currency units
- recovery and JTD: decimals in [0, 1]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Literal, Mapping, Optional, Sequence, Union
import math
import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import trapezoid as scipy_trapezoid
from scipy.linalg import solve_banded
from scipy.stats import norm, poisson

ModelName = Literal["jump_diffusion", "black_scholes"]
Interpolation = Literal["linear", "step"]


@dataclass(frozen=True)
class Curve:
    """Term structure of instantaneous forward rates / vols / hazard rates.

    `times` are node times in years. For step interpolation, each value applies
    from its node time up to the next node. For linear interpolation, values
    are linearly interpolated. A node at time 0 is recommended.
    """
    times: Sequence[float]
    values: Sequence[float]
    interpolation: Interpolation = "linear"
    name: str = "curve"

    def __post_init__(self):
        t = np.asarray(self.times, dtype=float)
        v = np.asarray(self.values, dtype=float)
        if t.ndim != 1 or v.ndim != 1 or len(t) != len(v) or len(t) == 0:
            raise ValueError("Curve times and values must be non-empty 1D arrays of equal length")
        if np.any(~np.isfinite(t)) or np.any(~np.isfinite(v)):
            raise ValueError("Curve contains non-finite values")
        if np.any(np.diff(t) <= 0):
            raise ValueError("Curve times must be strictly increasing")
        if t[0] < 0:
            raise ValueError("Curve times must be non-negative")
        if self.interpolation not in ("linear", "step"):
            raise ValueError("interpolation must be 'linear' or 'step'")
        object.__setattr__(self, "times", tuple(float(x) for x in t))
        object.__setattr__(self, "values", tuple(float(x) for x in v))

    def at(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x = np.asarray(t, dtype=float)
        nodes = np.asarray(self.times)
        vals = np.asarray(self.values)
        if self.interpolation == "linear":
            y = np.interp(x, nodes, vals, left=vals[0], right=vals[-1])
        else:
            idx = np.searchsorted(nodes, x, side="right") - 1
            idx = np.clip(idx, 0, len(vals) - 1)
            y = vals[idx]
        return float(y) if np.ndim(t) == 0 else y

    def integral(self, start: float, end: float, n: int = 400) -> float:
        if end < start:
            return -self.integral(end, start, n)
        if end == start:
            return 0.0
        grid = np.linspace(start, end, max(2, n + 1))
        # NumPy 2.0 introduced np.trapezoid. Use SciPy's equivalent as a
        # compatibility fallback for NumPy < 2.0.
        trapezoid = getattr(np, "trapezoid", None)
        if trapezoid is None:
            trapezoid = scipy_trapezoid
        return float(trapezoid(self.at(grid), grid))

    def discount_factor(self, t: float) -> float:
        """Discount factor from 0 to t, assuming values are instantaneous forwards."""
        return math.exp(-self.integral(0.0, float(t)))

    @classmethod
    def flat(cls, value: float, name: str = "flat", end: float = 100.0) -> "Curve":
        return cls((0.0, end), (value, value), "linear", name)

    @classmethod
    def from_zero_rates(
        cls, times: Sequence[float], zero_rates: Sequence[float], name: str = "forwards"
    ) -> "Curve":
        """Convert continuously compounded zero rates into piecewise-constant forwards."""
        t = np.asarray(times, dtype=float)
        z = np.asarray(zero_rates, dtype=float)
        if len(t) != len(z) or len(t) == 0 or np.any(t <= 0) or np.any(np.diff(t) <= 0):
            raise ValueError("Zero-rate times must be positive, strictly increasing and match rates")
        prev_t = 0.0
        prev_pv = 0.0
        forwards = []
        for ti, zi in zip(t, z):
            pv = zi * ti
            forwards.append((pv - prev_pv) / (ti - prev_t))
            prev_t, prev_pv = ti, pv
        return cls(tuple([0.0] + list(t[:-1])), tuple(forwards), "step", name)


@dataclass
class ConvertibleBond:
    face: float
    maturity: float
    conversion_ratio: float
    coupon_schedule: Mapping[float, float] = field(default_factory=dict)
    call_schedule: Mapping[float, float] = field(default_factory=dict)
    put_schedule: Mapping[float, float] = field(default_factory=dict)
    allow_early_conversion: bool = True
    name: str = "Convertible"

    def validate(self):
        if self.face <= 0 or self.maturity <= 0 or self.conversion_ratio < 0:
            raise ValueError("face and maturity must be positive; conversion_ratio must be non-negative")
        for schedule_name, schedule in (("coupon", self.coupon_schedule), ("call", self.call_schedule), ("put", self.put_schedule)):
            for t, amount in schedule.items():
                if t < 0 or t > self.maturity or amount < 0:
                    raise ValueError(f"Invalid {schedule_name} schedule item: {t} -> {amount}")


@dataclass
class OVCVParameters:
    model: ModelName
    risk_free_forward: Curve
    dividend_forward: Curve
    volatility: Curve
    hazard_rate: Optional[Curve] = None
    bond_recovery: float = 0.40
    jtd_factor: float = 1.00
    oas: float = 0.0
    borrow_cost: float = 0.0
    e2c_exponent: float = 0.0
    e2c_reference_spot: Optional[float] = None

    def validate(self):
        if self.model not in ("jump_diffusion", "black_scholes"):
            raise ValueError("model must be 'jump_diffusion' or 'black_scholes'")
        if not 0 <= self.bond_recovery <= 1:
            raise ValueError("bond_recovery must be between 0 and 1")
        if not 0 <= self.jtd_factor <= 1:
            raise ValueError("jtd_factor must be between 0 and 1")
        if self.oas < -0.50 or self.oas > 2.0:
            raise ValueError("oas is outside a reasonable validation range")
        if self.borrow_cost < -0.50 or self.borrow_cost > 2.0:
            raise ValueError("borrow_cost is outside a reasonable validation range")
        if self.e2c_exponent < 0:
            raise ValueError("e2c_exponent must be non-negative")
        if self.model == "jump_diffusion" and self.hazard_rate is None:
            raise ValueError("Jump-Diffusion requires a hazard-rate curve")
        if self.model == "jump_diffusion" and abs(self.oas) > 1e-12:
            raise ValueError("The paper documents OAS as a parallel forward-curve shift in Black-Scholes mode; do not add it to Jump-Diffusion without confirming Bloomberg's convention")


@dataclass
class PriceResult:
    price: float
    model: str
    price_type: str = "dirty"
    grid_points_stock: int = 401
    time_steps: int = 600
    notes: tuple[str, ...] = ()


def _validate_positive_curve(curve: Curve, label: str, allow_zero: bool = True):
    vals = np.asarray(curve.values)
    if np.any(vals < 0 if allow_zero else vals <= 0):
        raise ValueError(f"{label} curve contains negative or invalid values")


def _effective_curves(params: OVCVParameters):
    """Return model-specific r, q, sigma and hazard values via closures."""
    params.validate()
    r_shift = params.oas if params.model == "black_scholes" else 0.0
    if params.model == "black_scholes":
        h_curve = Curve.flat(0.0, "zero hazard")
    else:
        h_curve = params.hazard_rate
    def r_at(t): return params.risk_free_forward.at(t) + r_shift
    def q_at(t): return params.dividend_forward.at(t) + params.borrow_cost
    def sig_at(t): return params.volatility.at(t)
    def h_at(t, s=None, spot=None):
        base = h_curve.at(t)
        if params.e2c_exponent > 0 and s is not None:
            s0 = params.e2c_reference_spot or spot
            if not s0 or s0 <= 0:
                raise ValueError("E2C requires a positive e2c_reference_spot or valuation spot")
            ratio = np.maximum(np.asarray(s, dtype=float), 1e-12)
            return base * np.power(s0 / ratio, params.e2c_exponent)
        return base
    return r_at, q_at, sig_at, h_at


def _make_time_grid(maturity: float, n_steps: int, event_times: Iterable[float]) -> np.ndarray:
    grid = list(np.linspace(0.0, maturity, n_steps + 1))
    grid.extend(float(t) for t in event_times if 0 < float(t) < maturity)
    # Deduplicate floating point dates to 1e-10, preserving exact event dates.
    return np.array(sorted(set(round(x, 10) for x in grid)), dtype=float)


def price_convertible(
    bond: ConvertibleBond,
    spot: float,
    params: OVCVParameters,
    n_stock: int = 401,
    n_time: int = 600,
    log_width: Optional[float] = None,
    theta: float = 0.5,
) -> PriceResult:
    """Price a plain convertible using the paper's 1-factor PDE.

    Implemented: continuous diffusion + compensated Poisson default jump,
    bond recovery / post-default conversion payoff, continuous dividend yield,
    borrow cost as additional dividend yield, coupons, simple hard calls/puts,
    and early conversion. Complex path-dependent contractual provisions are not
    implemented in this core solver.
    """
    bond.validate()
    params.validate()
    if spot <= 0:
        raise ValueError("spot must be positive")
    if n_stock < 101 or n_stock % 2 == 0:
        raise ValueError("n_stock must be an odd integer >= 101")
    if n_time < 20:
        raise ValueError("n_time must be >= 20")
    if not 0 < theta <= 1:
        raise ValueError("theta must be in (0, 1]; 0.5 is Crank-Nicolson")

    r_at, q_at, sig_at, h_at = _effective_curves(params)
    max_sig = max(max(params.volatility.values), 0.05)
    width = log_width if log_width is not None else max(3.5, 5.0 * max_sig * math.sqrt(bond.maturity))
    x = np.linspace(math.log(spot) - width, math.log(spot) + width, n_stock)
    s_grid = np.exp(x)
    dx = x[1] - x[0]

    events = list(bond.coupon_schedule) + list(bond.call_schedule) + list(bond.put_schedule)
    times = _make_time_grid(bond.maturity, n_time, events)

    # Terminal value: redemption (including the final coupon) or conversion,
    # whichever is more valuable. The final coupon is not paid on principal
    # converted into shares; this matters for low-coupon convertible notes.
    maturity_coupon = _event_amount(bond.coupon_schedule, bond.maturity)
    v = np.maximum(bond.face + maturity_coupon, bond.conversion_ratio * s_grid)
    if not bond.allow_early_conversion:
        v = np.full_like(s_grid, bond.face + maturity_coupon)

    rf = params.bond_recovery * bond.face
    eps = 1e-12

    # Backward induction in calendar time. Each step advances the PDE in tau=T-t.
    for k in range(len(times) - 1, 0, -1):
        t0, t1 = float(times[k - 1]), float(times[k])
        dt = t1 - t0
        tm = 0.5 * (t0 + t1)
        r = float(r_at(tm))
        q = float(q_at(tm))
        sig = max(float(sig_at(tm)), 1e-8)
        h_raw = np.asarray(h_at(tm, s_grid, spot), dtype=float)
        h_nodes = np.broadcast_to(h_raw, s_grid.shape).astype(float).copy()
        if params.model == "black_scholes":
            h_nodes[:] = 0.0
        if np.any(h_nodes < 0):
            raise ValueError("Hazard rates must be non-negative")

        # PDE in x=ln(S): V_tau = a V_xx + b V_x - (r+h)V + h*D
        mu_x = r - q + params.jtd_factor * h_nodes - 0.5 * sig * sig
        aa = 0.5 * sig * sig / (dx * dx) - mu_x / (2.0 * dx)
        bb = -sig * sig / (dx * dx) - (r + h_nodes)
        cc = 0.5 * sig * sig / (dx * dx) + mu_x / (2.0 * dx)
        default_payoff = np.maximum(rf, bond.conversion_ratio * s_grid * (1.0 - params.jtd_factor))
        source = h_nodes * default_payoff

        # Boundary conditions: low-stock recovery ODE; high-stock immediate conversion.
        v_old = v.copy()
        h_low = float(h_nodes[0])
        a_low = r + h_low
        src_low = h_low * rf
        if abs(a_low) > eps:
            v_low_new = v_old[0] * math.exp(-a_low * dt) + (src_low / a_low) * (1.0 - math.exp(-a_low * dt))
        else:
            v_low_new = v_old[0] + src_low * dt
        v_high_old = bond.conversion_ratio * s_grid[-1] if bond.allow_early_conversion else bond.face
        v_high_new = v_high_old

        # Interior Crank-Nicolson / theta scheme. Coefficients are frozen at midpoint.
        old_i = v_old[1:-1]
        old_l = v_old[:-2]
        old_r = v_old[2:]
        rhs = old_i + (1.0 - theta) * dt * (aa[1:-1] * old_l + bb[1:-1] * old_i + cc[1:-1] * old_r) + dt * source[1:-1]

        # Move new-time boundary terms to RHS.
        rhs[0] += theta * dt * aa[1] * v_low_new
        rhs[-1] += theta * dt * cc[-2] * v_high_new

        # Banded matrix: (I - theta*dt*L) for interior nodes.
        n = n_stock - 2
        ab = np.zeros((3, n), dtype=float)
        ab[0, 1:] = -theta * dt * cc[1:-1][:-1]
        ab[1, :] = 1.0 - theta * dt * bb[1:-1]
        ab[2, :-1] = -theta * dt * aa[1:-1][1:]
        v_new_i = solve_banded((1, 1), ab, rhs, overwrite_ab=True, overwrite_b=False, check_finite=False)
        v = np.empty_like(v_old)
        v[0] = v_low_new
        v[-1] = v_high_new
        v[1:-1] = v_new_i

        # Apply discrete cashflows and contractual exercise at the event date.
        t_event = t0
        coupon = _event_amount(bond.coupon_schedule, t_event)
        if coupon:
            v += coupon
        put_value = _event_amount(bond.put_schedule, t_event)
        if put_value:
            v = np.maximum(v, put_value)
        call_value = _event_amount(bond.call_schedule, t_event)
        if call_value:
            if bond.allow_early_conversion:
                v = np.minimum(v, np.maximum(call_value, bond.conversion_ratio * s_grid))
            else:
                v = np.minimum(v, call_value)
        if bond.allow_early_conversion:
            v = np.maximum(v, bond.conversion_ratio * s_grid)

    notes = ["Core one-factor PDE implementation based on the published OVCV model description."]
    notes.append("Prices are dirty values; discrete coupons are added at their payment dates.")
    if params.e2c_exponent > 0:
        notes.append("E2C applies h(S,t)=h0(t)*(S0/S)^p. Exact CDS-preserving h0 and joint volatility calibration are not yet implemented.")
    if bond.call_schedule or bond.put_schedule:
        notes.append("Calls/puts are implemented as simple date-based hard constraints, not the full feature-specific OVCV logic.")
    return PriceResult(float(np.interp(math.log(spot), x, v)), params.model, "dirty", n_stock, len(times) - 1, tuple(notes))


def _event_amount(schedule: Mapping[float, float], t: float, tol: float = 1e-8) -> float:
    for event_t, amount in schedule.items():
        if abs(float(event_t) - t) <= tol:
            return float(amount)
    return 0.0


def solve_oas(
    bond: ConvertibleBond,
    spot: float,
    target_market_price: float,
    params: OVCVParameters,
    lower: float = -0.05,
    upper: float = 1.00,
    n_stock: int = 301,
    n_time: int = 400,
) -> float:
    """Solve OAS by repricing in the paper's Black-Scholes mode.

    The 2014 paper describes OAS as a parallel shift to instantaneous forward
    rates in Black-Scholes mode. It does not specify adding OAS on top of the
    Jump-Diffusion CDS/default mechanism, so this function rejects JD parameters.
    """
    if params.model != "black_scholes":
        raise ValueError("OAS solve is implemented only for Black-Scholes mode per the paper's disclosed convention")
    if target_market_price <= 0:
        raise ValueError("target_market_price must be positive")

    def objective(spread: float) -> float:
        p = OVCVParameters(
            model="black_scholes", risk_free_forward=params.risk_free_forward,
            dividend_forward=params.dividend_forward, volatility=params.volatility,
            bond_recovery=params.bond_recovery, jtd_factor=0.0, oas=spread,
            borrow_cost=params.borrow_cost,
        )
        return price_convertible(bond, spot, p, n_stock=n_stock, n_time=n_time).price - target_market_price

    f_lo, f_hi = objective(lower), objective(upper)
    if f_lo == 0:
        return lower
    if f_hi == 0:
        return upper
    if f_lo * f_hi > 0:
        raise ValueError("Could not bracket OAS solution; widen lower/upper or check market price and inputs")
    return float(brentq(objective, lower, upper, xtol=1e-7, rtol=1e-7, maxiter=100))


def _bs_call_from_integrated(
    spot: float, strike: float, total_r: float, total_q: float, total_var: float
) -> float:
    if total_var <= 1e-14:
        return max(spot * math.exp(total_r - total_q) - strike, 0.0) * math.exp(-total_r)
    std = math.sqrt(total_var)
    forward_mean = spot * math.exp(total_r - total_q)
    d1 = (math.log(forward_mean / strike) + 0.5 * total_var) / std
    d2 = d1 - std
    return math.exp(-total_r) * (forward_mean * norm.cdf(d1) - strike * norm.cdf(d2))


def price_european_call_jump_diffusion(
    spot: float,
    strike: float,
    maturity: float,
    risk_free_forward: Curve,
    dividend_forward: Curve,
    diffusion_volatility: Curve,
    hazard_rate: Curve,
    jtd_factor: float,
    n_poisson_terms: int = 100,
) -> float:
    """Price a European call under the compensated Poisson stock process.

    The disclosed stock SDE has drift r-q+eta*h and downward jumps of eta at
    Poisson arrivals. This uses the compound-Poisson/lognormal mixture for a
    time-dependent, independent hazard and piecewise time-dependent variance.
    It is used to support the paper's BS-option-price matching calibration.
    """
    if not (spot > 0 and strike > 0 and maturity > 0):
        raise ValueError("spot, strike and maturity must be positive")
    if not 0 <= jtd_factor <= 1:
        raise ValueError("jtd_factor must be in [0,1]")
    H = hazard_rate.integral(0.0, maturity)
    Rint = risk_free_forward.integral(0.0, maturity)
    Qint = dividend_forward.integral(0.0, maturity)
    V = diffusion_volatility.integral(0.0, maturity, n=800)
    if H < 0 or V < 0:
        raise ValueError("Integrated hazard and variance must be non-negative")
    if H < 1e-12 or jtd_factor < 1e-12:
        return _bs_call_from_integrated(spot, strike, Rint, Qint, V)
    max_k = max(n_poisson_terms, int(poisson.ppf(1 - 1e-12, H)) + 2)
    ks = np.arange(max_k + 1)
    probs = poisson.pmf(ks, H)
    if jtd_factor >= 1.0 - 1e-12:
        # A 100% jump wipes out the stock after the first jump.
        valid = ks == 0
        probs = probs[valid]
        ks = ks[valid]
    total = 0.0
    std = math.sqrt(max(V, 0.0))
    for k, pk in zip(ks, probs):
        if pk < 1e-16:
            continue
        jump_factor = (1.0 - jtd_factor) ** int(k)
        if jump_factor <= 0:
            continue
        # Conditional mean of S_T given N_T=k.
        forward_mean = spot * math.exp(Rint - Qint + jtd_factor * H) * jump_factor
        if std <= 1e-14:
            call = max(forward_mean - strike, 0.0) * math.exp(-Rint)
        else:
            d1 = (math.log(forward_mean / strike) + 0.5 * V) / std
            d2 = d1 - std
            call = math.exp(-Rint) * (forward_mean * norm.cdf(d1) - strike * norm.cdf(d2))
        total += float(pk) * call
    return float(total)


def calibrate_jump_diffusion_vols(
    spot: float,
    maturities: Sequence[float],
    black_scholes_vols: Sequence[float],
    strikes: Optional[Sequence[float]],
    risk_free_forward: Curve,
    dividend_forward: Curve,
    hazard_rate: Curve,
    jtd_factor: float,
    sigma_lower: float = 0.001,
    sigma_upper: float = 3.00,
) -> Curve:
    """Bootstrap piecewise-constant JD diffusion vols to match BS call prices.

    At each maturity, the previously calibrated intervals are held fixed and
    the new interval volatility is solved so the JD call price matches a BS
    call price using the supplied market-implied BS vol. This follows the
    calibration outline in section 6.2. The whitepaper does not specify the
    exact strike-selection convention, so strikes are explicit user inputs.
    """
    mats = np.asarray(maturities, dtype=float)
    vols = np.asarray(black_scholes_vols, dtype=float)
    if len(mats) == 0 or len(mats) != len(vols) or np.any(mats <= 0) or np.any(np.diff(mats) <= 0):
        raise ValueError("maturities must be positive, strictly increasing and match vols")
    if strikes is None:
        raise ValueError("Provide explicit calibration strikes; the whitepaper does not prescribe a strike convention")
    ks = np.asarray(strikes, dtype=float)
    if len(ks) != len(mats) or np.any(ks <= 0):
        raise ValueError("strikes must be positive and match maturities")
    if np.any(vols <= 0):
        raise ValueError("Black-Scholes volatilities must be positive")
    if jtd_factor >= 1:
        # Still calibratable, but for robustness the bracket may be wider.
        pass

    calibrated_nodes = [0.0]
    calibrated_vals = []
    previous_vols: list[float] = []
    previous_mats: list[float] = []

    for i, (T, vol_bs, K) in enumerate(zip(mats, vols, ks)):
        # Market call price implied by supplied Black-Scholes volatility.
        Rint = risk_free_forward.integral(0.0, float(T))
        Qint = dividend_forward.integral(0.0, float(T))
        target = _bs_call_from_integrated(spot, float(K), Rint, Qint, float(vol_bs) ** 2 * float(T))

        def candidate_price(sig_new: float) -> float:
            # Each piecewise-constant volatility is indexed by its interval
            # start. The final calibrated interval extends beyond its last
            # node, so the candidate curve has one node per interval.
            nodes = [0.0] + previous_mats
            values = previous_vols + [float(sig_new)]
            sigma_curve = Curve(tuple(nodes), tuple(values), "step", "calibrated JD volatility")
            return price_european_call_jump_diffusion(
                spot, float(K), float(T), risk_free_forward, dividend_forward,
                sigma_curve, hazard_rate, jtd_factor,
            )

        f_lo = candidate_price(sigma_lower) - target
        f_hi = candidate_price(sigma_upper) - target
        if f_lo == 0:
            sig = sigma_lower
        elif f_hi == 0:
            sig = sigma_upper
        elif f_lo * f_hi > 0:
            raise ValueError(
                f"Could not calibrate JD vol at T={T:g}: target BS call is outside the price bracket. "
                "Check strike, hazard, JTD factor and volatility bounds."
            )
        else:
            sig = float(brentq(lambda z: candidate_price(z) - target, sigma_lower, sigma_upper, xtol=1e-8))
        previous_mats.append(float(T))
        previous_vols.append(sig)
        calibrated_vals.append(sig)
        calibrated_nodes.append(float(T))

    # Each calibrated value applies from the corresponding interval start.
    return Curve(tuple(calibrated_nodes[:-1]), tuple(calibrated_vals), "step", "JD diffusion volatility")


def bootstrap_hazard_from_cds(
    maturities: Sequence[float],
    par_spreads: Sequence[float],
    cds_recovery: float,
    discount_curve: Curve,
    premium_frequency: int = 4,
) -> Curve:
    """Approximate piecewise-constant hazard bootstrap from CDS par spreads.

    Uses quarterly (configurable) premium dates, continuous discount factors,
    and a midpoint approximation for accrued premium on default. This is a
    transparent standard CDS approximation, NOT Bloomberg's referenced
    proprietary companion CDS model DOCS 2057273.
    """
    mats = np.asarray(maturities, dtype=float)
    spreads = np.asarray(par_spreads, dtype=float)
    if len(mats) == 0 or len(mats) != len(spreads) or np.any(mats <= 0) or np.any(np.diff(mats) <= 0):
        raise ValueError("CDS maturities must be positive, strictly increasing and match spreads")
    if np.any(spreads < 0) or not 0 <= cds_recovery < 1:
        raise ValueError("CDS spreads must be non-negative and CDS recovery in [0,1)")
    if premium_frequency < 1:
        raise ValueError("premium_frequency must be >= 1")

    hazards: list[float] = []
    starts = [0.0]

    def survival(t: float, candidate: Optional[float] = None, last_maturity: Optional[float] = None) -> float:
        integral = 0.0
        left = 0.0
        for j, end in enumerate(mats[:len(hazards) + (1 if candidate is not None else 0)]):
            lam = hazards[j] if j < len(hazards) else float(candidate)
            right = min(t, float(end))
            if right > left:
                integral += lam * (right - left)
            left = float(end)
            if t <= left:
                break
        if t > left:
            # At final quoted maturity, the candidate interval extends to maturity.
            lam = hazards[-1] if candidate is None and hazards else (float(candidate) if candidate is not None else 0.0)
            integral += lam * (t - left)
        return math.exp(-integral)

    for j, (T, spread) in enumerate(zip(mats, spreads)):
        prev_T = float(mats[j - 1]) if j > 0 else 0.0
        def par_spread_for(candidate_h: float) -> float:
            # Generate premium dates; include exact maturity even if not a coupon date.
            step = 1.0 / premium_frequency
            pay_dates = list(np.arange(step, T, step)) + [float(T)]
            pay_dates = sorted(set(round(float(x), 10) for x in pay_dates))
            protection = 0.0
            annuity = 0.0
            t_prev = 0.0
            q_prev = 1.0
            for t_pay in pay_dates:
                accrual = t_pay - t_prev
                q_now = survival(t_pay, candidate=candidate_h, last_maturity=float(T))
                mid = 0.5 * (t_prev + t_pay)
                df_pay = discount_curve.discount_factor(t_pay)
                df_mid = discount_curve.discount_factor(mid)
                default_prob = max(q_prev - q_now, 0.0)
                protection += (1.0 - cds_recovery) * df_mid * default_prob
                risky_annuity = accrual * df_pay * q_now + 0.5 * accrual * df_mid * default_prob
                annuity += risky_annuity
                t_prev, q_prev = t_pay, q_now
            return protection / max(annuity, 1e-16)

        # Root search for the hazard in the new maturity interval.
        lo, hi = 1e-8, 5.0
        f_lo = par_spread_for(lo) - float(spread)
        f_hi = par_spread_for(hi) - float(spread)
        if f_lo * f_hi > 0:
            raise ValueError(f"Could not bootstrap CDS hazard at maturity {T:g}")
        h = float(brentq(lambda z: par_spread_for(z) - float(spread), lo, hi, xtol=1e-10))
        hazards.append(h)
        starts.append(float(T))

    return Curve(tuple(starts[:-1]), tuple(hazards), "step", "CDS-implied hazard (approx.)")


def compare_risky_riskless(
    bond: ConvertibleBond,
    spot: float,
    risk_free_forward: Curve,
    dividend_forward: Curve,
    volatility: Curve,
    hazard_rate: Curve,
    bond_recovery: float,
    jtd_factor: float,
    n_stock: int = 401,
    n_time: int = 600,
) -> Dict[str, float]:
    """Compare JD risky value with the no-default riskless Black-Scholes value."""
    risky_params = OVCVParameters(
        model="jump_diffusion", risk_free_forward=risk_free_forward,
        dividend_forward=dividend_forward, volatility=volatility,
        hazard_rate=hazard_rate, bond_recovery=bond_recovery,
        jtd_factor=jtd_factor,
    )
    riskless_params = OVCVParameters(
        model="black_scholes", risk_free_forward=risk_free_forward,
        dividend_forward=dividend_forward, volatility=volatility,
        bond_recovery=bond_recovery, jtd_factor=0.0, oas=0.0,
    )
    risky = price_convertible(bond, spot, risky_params, n_stock=n_stock, n_time=n_time).price
    riskless = price_convertible(bond, spot, riskless_params, n_stock=n_stock, n_time=n_time).price
    return {
        "risky_price": risky,
        "riskless_price": riskless,
        "riskless_minus_risky": riskless - risky,
        "risky_discount_pct_of_riskless": (riskless - risky) / riskless * 100.0 if riskless else math.nan,
    }


def run_sensitivity(
    bond: ConvertibleBond,
    spot: float,
    base_params: OVCVParameters,
    shocks: Mapping[str, Mapping[str, float]],
    n_stock: int = 301,
    n_time: int = 400,
) -> list[dict]:
    """Manual-input sensitivity helper for scalar model parameters.

    `shocks` example: {"JTD +10pp": {"jtd_factor": 0.1}, ...}. The supplied
    value is a change, not a new absolute parameter value. Curve-node shocks
    should be applied by constructing new Curve objects outside this helper.
    """
    base_price = price_convertible(bond, spot, base_params, n_stock, n_time).price
    rows = [{"scenario": "BASE", "price": base_price, "change": 0.0, "pct_change": 0.0}]
    for name, bump in shocks.items():
        kwargs = dict(
            model=base_params.model,
            risk_free_forward=base_params.risk_free_forward,
            dividend_forward=base_params.dividend_forward,
            volatility=base_params.volatility,
            hazard_rate=base_params.hazard_rate,
            bond_recovery=base_params.bond_recovery,
            jtd_factor=base_params.jtd_factor,
            oas=base_params.oas,
            borrow_cost=base_params.borrow_cost,
            e2c_exponent=base_params.e2c_exponent,
            e2c_reference_spot=base_params.e2c_reference_spot,
        )
        for key, delta in bump.items():
            if key not in kwargs:
                raise ValueError(f"Unsupported scalar shock parameter: {key}")
            if key in ("risk_free_forward", "dividend_forward", "volatility", "hazard_rate"):
                raise ValueError(f"Curve shocks need an explicitly rebuilt Curve; cannot bump {key} as scalar")
            kwargs[key] = kwargs[key] + float(delta)
        p = OVCVParameters(**kwargs)
        price = price_convertible(bond, spot, p, n_stock, n_time).price
        rows.append({"scenario": name, "price": price, "change": price - base_price, "pct_change": (price / base_price - 1.0) * 100.0})
    return rows


if __name__ == "__main__":
    # Illustrative smoke test; these are not Bloomberg market data or expected OVCV outputs.
    r = Curve.flat(0.04, "SOFR forward")
    q = Curve.flat(0.01, "Dividend forward")
    vol = Curve.flat(0.30, "JD diffusion vol")
    hazard = Curve.flat(0.03, "hazard")
    cb = ConvertibleBond(face=100.0, maturity=5.0, conversion_ratio=5.0,
                         coupon_schedule={1.0: 2.0, 2.0: 2.0, 3.0: 2.0, 4.0: 2.0, 5.0: 2.0})
    p = OVCVParameters(model="jump_diffusion", risk_free_forward=r,
                       dividend_forward=q, volatility=vol, hazard_rate=hazard,
                       bond_recovery=0.40, jtd_factor=0.60)
    print(price_convertible(cb, spot=18.0, params=p))
