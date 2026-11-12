"""Scientific computation with NumPy and SciPy.

Statistics, regression, hypothesis tests, probability distributions, numeric
calculus, root finding, optimisation, ODE integration, numeric linear algebra,
physics helpers, unit conversion, physical constants and chemistry helpers
(molar mass, equation balancing, pH).

Like the maths engine, every operation is a pure function: dict in, JSON-safe dict out.
Atomic weights are conventional IUPAC values; radioactive elements use the mass number
of their longest-lived isotope.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np
import sympy as sp
from scipy import constants as sc
from scipy import integrate, optimize, stats

from .common import EngineError, OperationSpec, as_float, as_float_list, as_text, need
from .mathematics import parse_expression, _symbol

MAX_POINTS = 200


# --- helpers ----------------------------------------------------------------
def _fn(expression: Any, variables: list[str]):
    """Turn expression text into a fast numpy function of the given variables."""
    syms = [_symbol(v) for v in variables]
    expr = parse_expression(expression)
    unknown = expr.free_symbols - set(syms)
    if unknown:
        raise EngineError(f"The expression contains unknown symbols: {', '.join(sorted(map(str, unknown)))}.")
    return sp.lambdify(syms, expr, modules=["numpy", "scipy"])


def _bound(value: Any, name: str) -> float:
    if isinstance(value, str):
        v = value.strip().lower().replace("infinity", "oo").replace("inf", "oo")
        if v in {"oo", "+oo"}:
            return math.inf
        if v == "-oo":
            return -math.inf
        return float(parse_expression(value, name=name).evalf())
    return as_float(value, name)


def _matrix_np(value: Any, name: str, max_dim: int = 200) -> np.ndarray:
    try:
        arr = np.array(value, dtype=float)
    except (TypeError, ValueError):
        raise EngineError(f"'{name}' must be a numeric list of rows.") from None
    if arr.ndim not in (1, 2) or arr.size == 0 or max(arr.shape) > max_dim:
        raise EngineError(f"'{name}' must be a non-empty 1-D or 2-D array with sides up to {max_dim}.")
    return arr


# --- statistics -------------------------------------------------------------
def op_statistics(args: dict) -> dict:
    x = np.array(as_float_list(need(args, "data"), "data"))
    n = len(x)
    out: dict[str, Any] = {
        "n": n, "mean": float(x.mean()), "median": float(np.median(x)),
        "min": float(x.min()), "max": float(x.max()), "range": float(np.ptp(x)),
        "q1": float(np.percentile(x, 25)), "q3": float(np.percentile(x, 75)),
        "sum": float(x.sum()),
        "std_population": float(x.std(ddof=0)), "variance_population": float(x.var(ddof=0)),
    }
    out["iqr"] = out["q3"] - out["q1"]
    if n > 1:
        out.update(std_sample=float(x.std(ddof=1)), variance_sample=float(x.var(ddof=1)),
                   sem=float(stats.sem(x)))
        lo, hi = stats.t.interval(0.95, n - 1, loc=x.mean(), scale=stats.sem(x))
        out["mean_confidence_interval_95"] = [float(lo), float(hi)] if np.isfinite(lo) else None
    if n > 2:
        out.update(skewness=float(stats.skew(x)), excess_kurtosis=float(stats.kurtosis(x)))
    return out


def op_regression(args: dict) -> dict:
    x = np.array(as_float_list(need(args, "x"), "x", min_len=2))
    y = np.array(as_float_list(need(args, "y"), "y", min_len=2))
    if len(x) != len(y):
        raise EngineError("x and y must have the same length.")
    degree = int(args.get("degree", 1))
    if not 1 <= degree <= 10 or degree >= len(x):
        raise EngineError("degree must be between 1 and 10 and smaller than the number of points.")
    if degree == 1:
        r = stats.linregress(x, y)
        out: dict[str, Any] = {
            "model": "y = slope*x + intercept", "slope": float(r.slope), "intercept": float(r.intercept),
            "r": float(r.rvalue), "r_squared": float(r.rvalue**2), "p_value": float(r.pvalue),
            "slope_stderr": float(r.stderr),
        }
        predict = lambda v: r.slope * v + r.intercept  # noqa: E731
    else:
        coef = np.polyfit(x, y, degree)
        fit = np.polyval(coef, x)
        ss_res, ss_tot = float(((y - fit) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
        out = {"model": f"polynomial of degree {degree}", "coefficients_highest_first": [float(c) for c in coef],
               "r_squared": 1 - ss_res / ss_tot if ss_tot else None}
        predict = lambda v: np.polyval(coef, v)  # noqa: E731
    if args.get("predict_x") is not None:
        px = as_float_list(args["predict_x"], "predict_x", max_len=1000)
        out["predictions"] = [{"x": v, "y": float(predict(v))} for v in px]
    return out


def op_hypothesis_test(args: dict) -> dict:
    test = str(need(args, "test")).lower()
    alpha = as_float(args.get("alpha", 0.05), "alpha")
    lst = lambda k: as_float_list(need(args, k), k, min_len=2)  # noqa: E731
    if test == "t_one_sample":
        res = stats.ttest_1samp(lst("data"), as_float(need(args, "mu"), "mu"))
    elif test == "t_two_sample":
        res = stats.ttest_ind(lst("a"), lst("b"), equal_var=bool(args.get("equal_var", False)))
    elif test == "t_paired":
        a, b = lst("a"), lst("b")
        if len(a) != len(b):
            raise EngineError("Paired samples must have equal length.")
        res = stats.ttest_rel(a, b)
    elif test == "mann_whitney":
        res = stats.mannwhitneyu(lst("a"), lst("b"))
    elif test == "anova":
        groups = [as_float_list(g, "groups", min_len=2) for g in need(args, "groups")]
        if len(groups) < 2:
            raise EngineError("anova needs at least two groups.")
        res = stats.f_oneway(*groups)
    elif test == "pearson":
        a, b = lst("a"), lst("b")
        if len(a) != len(b):
            raise EngineError("a and b must have equal length.")
        res = stats.pearsonr(a, b)
    elif test == "shapiro":
        res = stats.shapiro(lst("data"))
    elif test == "chi2_gof":
        obs = as_float_list(need(args, "observed"), "observed", min_len=2)
        exp = as_float_list(args["expected"], "expected", min_len=2) if args.get("expected") else None
        res = stats.chisquare(obs, exp)
    elif test == "chi2_independence":
        table = _matrix_np(need(args, "table"), "table", 50)
        if table.ndim != 2:
            raise EngineError("'table' must be a 2-D contingency table.")
        chi2, p, dof, _ = stats.chi2_contingency(table)
        res = type("R", (), {"statistic": chi2, "pvalue": p})()
    else:
        raise EngineError("Unknown test. Use: t_one_sample, t_two_sample, t_paired, mann_whitney, anova, "
                          "pearson, shapiro, chi2_gof, chi2_independence.")
    p = float(res.pvalue)
    return {"test": test, "statistic": float(res.statistic), "p_value": p, "alpha": alpha,
            "reject_null": p < alpha,
            "interpretation": f"p = {p:.4g} is {'below' if p < alpha else 'not below'} alpha = {alpha}."}


_DISTRIBUTIONS = {
    "norm": ("loc", "scale"), "t": ("df", "loc", "scale"), "chi2": ("df", "loc", "scale"),
    "f": ("dfn", "dfd", "loc", "scale"), "binom": ("n", "p", "loc"), "poisson": ("mu", "loc"),
    "expon": ("loc", "scale"), "uniform": ("loc", "scale"), "gamma": ("a", "loc", "scale"),
    "beta": ("a", "b", "loc", "scale"), "lognorm": ("s", "loc", "scale"), "geom": ("p", "loc"),
}


def op_distribution(args: dict) -> dict:
    name = str(need(args, "name")).lower()
    if name not in _DISTRIBUTIONS:
        raise EngineError(f"Unknown distribution. Use one of: {', '.join(_DISTRIBUTIONS)}.")
    params = args.get("params") or {}
    order = _DISTRIBUTIONS[name]
    extra = set(params) - set(order)
    if extra:
        raise EngineError(f"Unknown parameter(s) {sorted(extra)}. {name} takes: {', '.join(order)}.")
    dist = getattr(stats, name)(**{k: as_float(v, k) for k, v in params.items()})
    out: dict[str, Any] = {"distribution": name, "params": params, "mean": float(dist.mean()),
                           "variance": float(dist.var()), "std": float(dist.std())}
    discrete = name in {"binom", "poisson", "geom"}
    if args.get("x") is not None:
        xs = as_float_list(args["x"] if isinstance(args["x"], (list, str)) else [args["x"]], "x", max_len=1000)
        out["at"] = [{"x": v, ("pmf" if discrete else "pdf"): float((dist.pmf if discrete else dist.pdf)(v)),
                      "cdf": float(dist.cdf(v)), "sf": float(dist.sf(v))} for v in xs]
    if args.get("q") is not None:
        qs = as_float_list(args["q"] if isinstance(args["q"], (list, str)) else [args["q"]], "q", max_len=1000)
        if any(not 0 <= q <= 1 for q in qs):
            raise EngineError("Quantile probabilities q must be between 0 and 1.")
        out["quantiles"] = [{"q": q, "value": float(dist.ppf(q))} for q in qs]
    return out


# --- numeric calculus / root finding / optimisation / ODE --------------------
def op_integrate_numeric(args: dict) -> dict:
    var = str(args.get("variable") or "x")
    f = _fn(need(args, "expression"), [var])
    a, b = _bound(need(args, "lower"), "lower"), _bound(need(args, "upper"), "upper")
    value, error = integrate.quad(lambda t: float(f(t)), a, b, limit=200)
    return {"value": float(value), "estimated_abs_error": float(error), "lower": a, "upper": b}


def op_solve_numeric(args: dict) -> dict:
    var = str(args.get("variable") or "x")
    f = _fn(need(args, "expression"), [var])
    g = lambda t: float(f(t))  # noqa: E731
    if args.get("bracket"):
        lo, hi = as_float_list(args["bracket"], "bracket", max_len=2, min_len=2)
        try:
            return {"roots": [float(optimize.brentq(g, lo, hi))], "method": "brentq"}
        except ValueError:
            raise EngineError("The function must change sign across the bracket.") from None
    if args.get("guess") is not None:
        sol = optimize.root_scalar(g, x0=as_float(args["guess"], "guess"), method="secant")
        return {"roots": [float(sol.root)], "converged": bool(sol.converged), "method": "secant"}
    lo, hi = (as_float_list(args["range"], "range", max_len=2, min_len=2) if args.get("range") else (-100.0, 100.0))
    grid = np.linspace(lo, hi, 4001)
    with np.errstate(all="ignore"):
        vals = np.array([g(t) for t in grid])
    roots: list[float] = []
    for i in range(len(grid) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
            roots.append(float(optimize.brentq(g, grid[i], grid[i + 1])))
        elif vals[i] == 0:
            roots.append(float(grid[i]))
    return {"roots": roots[:50], "searched_range": [lo, hi], "method": "sign-change scan",
            "note": "Only roots where the function changes sign inside the range are found."}


def op_optimize(args: dict) -> dict:
    var = str(args.get("variable") or "x")
    f = _fn(need(args, "expression"), [var])
    goal = str(args.get("goal", "min")).lower()
    if goal not in {"min", "max"}:
        raise EngineError("goal must be 'min' or 'max'.")
    sign = 1.0 if goal == "min" else -1.0
    g = lambda t: sign * float(f(t))  # noqa: E731
    if args.get("lower") is not None and args.get("upper") is not None:
        res = optimize.minimize_scalar(g, bounds=(as_float(args["lower"], "lower"), as_float(args["upper"], "upper")),
                                       method="bounded")
    else:
        res = optimize.minimize_scalar(g)
    return {"goal": goal, "x": float(res.x), "value": float(sign * res.fun), "converged": bool(res.success),
            "note": "Local optimum; give lower/upper bounds to search a specific interval."}


def op_ode_numeric(args: dict) -> dict:
    t_name = str(args.get("independent") or "t")
    states = args.get("states") or ["y"]
    rhs = need(args, "rhs")
    if isinstance(rhs, str):
        rhs = [rhs]
    if len(states) != len(rhs) or not 1 <= len(states) <= 12:
        raise EngineError("'states' and 'rhs' must have the same length (1 to 12).")
    fns = [_fn(r, [t_name, *states]) for r in rhs]
    y0 = as_float_list(need(args, "y0"), "y0", max_len=12)
    if len(y0) != len(states):
        raise EngineError("'y0' needs one value per state.")
    t0, t1 = as_float_list(need(args, "t_span"), "t_span", max_len=2, min_len=2)
    points = min(int(args.get("points", 50)), MAX_POINTS)
    method = str(args.get("method", "RK45"))
    if method not in {"RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"}:
        raise EngineError("method must be RK45, RK23, DOP853, Radau, BDF or LSODA.")
    sol = integrate.solve_ivp(lambda t, y: [float(f(t, *y)) for f in fns], (t0, t1), y0, method=method,
                              t_eval=np.linspace(t0, t1, points), rtol=1e-8, atol=1e-10)
    return {"success": bool(sol.success), "message": str(sol.message), "independent": t_name,
            "t": sol.t.tolist(), "states": {s: sol.y[i].tolist() for i, s in enumerate(states)},
            "final": {s: float(sol.y[i][-1]) for i, s in enumerate(states)} if sol.y.size else {}}


def op_linalg(args: dict) -> dict:
    op = str(need(args, "operation")).lower()
    A = _matrix_np(need(args, "A"), "A")
    if op in {"det", "inv", "eig"} and (A.ndim != 2 or A.shape[0] != A.shape[1]):
        raise EngineError(f"{op} needs a square matrix.")
    try:
        if op == "det":
            return {"determinant": float(np.linalg.det(A))}
        if op == "inv":
            return {"inverse": np.linalg.inv(A).tolist()}
        if op == "rank":
            return {"rank": int(np.linalg.matrix_rank(A))}
        if op == "norm":
            return {"frobenius_norm": float(np.linalg.norm(A))}
        if op == "svd":
            return {"singular_values": np.linalg.svd(A, compute_uv=False).tolist()}
        if op == "cond":
            return {"condition_number": float(np.linalg.cond(A))}
        if op == "eig":
            vals, vecs = np.linalg.eig(A)
            return {"eigenvalues": vals.tolist(), "eigenvectors_as_columns": vecs.tolist()}
        if op in {"solve", "lstsq"}:
            b = _matrix_np(need(args, "b"), "b")
            if op == "solve":
                return {"solution": np.linalg.solve(A, b).tolist()}
            x, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            return {"solution": x.tolist(), "residuals": res.tolist(), "rank": int(rank)}
    except np.linalg.LinAlgError as exc:
        raise EngineError(f"Linear algebra error: {exc}") from None
    raise EngineError("Unknown operation. Use: det, inv, rank, norm, svd, cond, eig, solve, lstsq.")


# --- physics ----------------------------------------------------------------
def op_kinematics(args: dict) -> dict:
    names = ("s", "u", "v", "a", "t")
    known = {k: args[k] for k in names if args.get(k) is not None}
    if len(known) != 3:
        raise EngineError("Give exactly three of s (displacement), u (initial velocity), v (final velocity), "
                          "a (acceleration), t (time), in SI units.")
    sy = {k: sp.Symbol(k) for k in names}
    eqs = [sp.Eq(sy["v"], sy["u"] + sy["a"] * sy["t"]),
           sp.Eq(sy["s"], sy["u"] * sy["t"] + sp.Rational(1, 2) * sy["a"] * sy["t"] ** 2),
           sp.Eq(sy["v"] ** 2, sy["u"] ** 2 + 2 * sy["a"] * sy["s"]),
           sp.Eq(sy["s"], (sy["u"] + sy["v"]) * sy["t"] / 2)]
    subs = {sy[k]: sp.Rational(repr(as_float(v, k))) for k, v in known.items()}
    unknown = [sy[k] for k in names if k not in known]
    sols = sp.solve([e.subs(subs) for e in eqs], unknown, dict=True)
    clean = []
    for s in sols:
        vals = {str(k): complex(sp.N(v)) for k, v in s.items()}
        if all(abs(v.imag) < 1e-12 for v in vals.values()):
            clean.append({k: v.real for k, v in vals.items()})
    if not clean:
        raise EngineError("No real solution for those values.")
    return {"given": {k: float(v) for k, v in known.items()}, "solutions": clean,
            "note": "Multiple solutions can occur (for example a negative time root); choose the physical one."}


def op_projectile(args: dict) -> dict:
    v0, ang = as_float(need(args, "speed"), "speed"), math.radians(as_float(need(args, "angle_deg"), "angle_deg"))
    h0, g = as_float(args.get("height", 0.0), "height"), as_float(args.get("g", sc.g), "g")
    if g <= 0:
        raise EngineError("g must be positive.")
    vy, vx = v0 * math.sin(ang), v0 * math.cos(ang)
    t_flight = (vy + math.sqrt(vy * vy + 2 * g * h0)) / g
    return {"time_of_flight_s": t_flight, "range_m": vx * t_flight, "max_height_m": h0 + vy * vy / (2 * g),
            "time_to_apex_s": max(vy, 0) / g, "impact_speed_m_s": math.hypot(vx, vy - g * t_flight),
            "assumes": "no air resistance, flat ground at y = 0"}


def op_ideal_gas(args: dict) -> dict:
    R = sc.R
    keys = ("P", "V", "n", "T")
    given = {k: as_float(args[k], k) for k in keys if args.get(k) is not None}
    if len(given) != 3:
        raise EngineError("Give exactly three of P (Pa), V (m^3), n (mol), T (K). PV = nRT.")
    if any(v <= 0 for v in given.values()):
        raise EngineError("Values must be positive.")
    missing = next(k for k in keys if k not in given)
    P, V, n, T = (given.get(k) for k in keys)
    value = {"P": lambda: n * R * T / V, "V": lambda: n * R * T / P,
             "n": lambda: P * V / (R * T), "T": lambda: P * V / (n * R)}[missing]()
    return {"solved_for": missing, "value": value, "units": {"P": "Pa", "V": "m^3", "n": "mol", "T": "K"}[missing],
            "R": R}


def op_orbit(args: dict) -> dict:
    M, r = as_float(need(args, "mass_kg"), "mass_kg"), as_float(need(args, "radius_m"), "radius_m")
    if M <= 0 or r <= 0:
        raise EngineError("mass_kg and radius_m must be positive.")
    mu = sc.G * M
    return {"circular_orbit_speed_m_s": math.sqrt(mu / r), "escape_speed_m_s": math.sqrt(2 * mu / r),
            "orbital_period_s": 2 * math.pi * math.sqrt(r**3 / mu), "surface_gravity_m_s2": mu / r**2,
            "G": sc.G, "note": "radius_m is measured from the centre of the body."}


def op_constant(args: dict) -> dict:
    query = str(need(args, "name")).strip().lower()
    table = sc.physical_constants
    exact = next((k for k in table if k.lower() == query), None)
    if exact:
        v, unit, unc = table[exact]
        return {"name": exact, "value": v, "unit": unit, "uncertainty": unc, "source": "CODATA via SciPy"}
    matches = [k for k in table if query in k.lower()][:15]
    if not matches:
        raise EngineError(f"No constant matches '{query}'.")
    return {"matches": [{"name": k, "value": table[k][0], "unit": table[k][1]} for k in matches],
            "source": "CODATA via SciPy"}


# --- unit conversion --------------------------------------------------------
_UNITS: dict[str, dict[str, float]] = {
    "length": {"m": 1, "km": 1e3, "cm": 1e-2, "mm": 1e-3, "um": 1e-6, "nm": 1e-9, "pm": 1e-12, "angstrom": 1e-10,
               "mi": sc.mile, "yd": sc.yard, "ft": sc.foot, "in": sc.inch, "nmi": sc.nautical_mile,
               "ly": sc.light_year, "au": sc.astronomical_unit, "pc": sc.parsec},
    "mass": {"kg": 1, "g": 1e-3, "mg": 1e-6, "ug": 1e-9, "t": 1e3, "lb": sc.pound, "oz": sc.ounce, "u": sc.atomic_mass},
    "time": {"s": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "min": 60, "h": 3600, "day": 86400, "week": 604800,
             "yr": sc.year},
    "energy": {"J": 1, "kJ": 1e3, "MJ": 1e6, "cal": sc.calorie, "kcal": sc.calorie * 1e3, "eV": sc.eV,
               "keV": sc.eV * 1e3, "MeV": sc.eV * 1e6, "Wh": 3600, "kWh": 3.6e6, "BTU": sc.Btu},
    "pressure": {"Pa": 1, "kPa": 1e3, "MPa": 1e6, "bar": sc.bar, "atm": sc.atm, "mmHg": sc.mmHg, "torr": sc.torr,
                 "psi": sc.psi},
    "speed": {"m/s": 1, "km/h": sc.kmh, "mph": sc.mph, "kn": sc.knot, "c": sc.c},
    "force": {"N": 1, "kN": 1e3, "lbf": sc.lbf, "dyn": sc.dyn},
    "power": {"W": 1, "kW": 1e3, "MW": 1e6, "hp": sc.hp},
    "volume": {"m3": 1, "L": sc.liter, "mL": 1e-6, "cm3": 1e-6, "gal": sc.gallon, "ft3": sc.foot**3},
    "angle": {"rad": 1, "deg": sc.degree, "arcmin": sc.arcmin, "arcsec": sc.arcsec, "turn": 2 * math.pi},
}


def _to_kelvin(v: float, u: str) -> float:
    return {"K": v, "C": v + 273.15, "F": (v - 32) * 5 / 9 + 273.15}[u]


def _from_kelvin(k: float, u: str) -> float:
    return {"K": k, "C": k - 273.15, "F": (k - 273.15) * 9 / 5 + 32}[u]


def op_convert_units(args: dict) -> dict:
    value = as_float(need(args, "value"), "value")
    src, dst = as_text(need(args, "from_unit"), "from_unit").strip(), as_text(need(args, "to_unit"), "to_unit").strip()
    if src in {"K", "C", "F"} and dst in {"K", "C", "F"}:
        k = _to_kelvin(value, src)
        if k < 0:
            raise EngineError("That temperature is below absolute zero.")
        return {"value": _from_kelvin(k, dst), "unit": dst, "category": "temperature"}
    for cat, table in _UNITS.items():
        if src in table and dst in table:
            return {"value": value * table[src] / table[dst], "unit": dst, "category": cat}
    known = sorted({u for t in _UNITS.values() for u in t} | {"K", "C", "F"})
    raise EngineError(f"Cannot convert '{src}' to '{dst}'. Units must be in the same category. Known units: {', '.join(known)}.")


# --- chemistry --------------------------------------------------------------
_ATOMIC_WEIGHT = dict(zip(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr "
    "Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir "
    "Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U".split(),
    [1.008, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998, 20.180, 22.990, 24.305, 26.982, 28.085,
     30.974, 32.06, 35.45, 39.948, 39.098, 40.078, 44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693,
     63.546, 65.38, 69.723, 72.630, 74.922, 78.971, 79.904, 83.798, 85.468, 87.62, 88.906, 91.224, 92.906, 95.95,
     98.0, 101.07, 102.91, 106.42, 107.87, 112.41, 114.82, 118.71, 121.76, 127.60, 126.90, 131.29, 132.91, 137.33,
     138.91, 140.12, 140.91, 144.24, 145.0, 150.36, 151.96, 157.25, 158.93, 162.50, 164.93, 167.26, 168.93, 173.05,
     174.97, 178.49, 180.95, 183.84, 186.21, 190.23, 192.22, 195.08, 196.97, 200.59, 204.38, 207.2, 208.98, 209.0,
     210.0, 222.0, 223.0, 226.0, 227.0, 232.04, 231.04, 238.03],
))


def _parse_group(s: str) -> Counter:
    stack: list[Counter] = [Counter()]
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in "([":
            stack.append(Counter())
            i += 1
        elif ch in ")]":
            i += 1
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            mult = int(s[i:j] or 1)
            i = j
            if len(stack) < 2:
                raise EngineError("Unbalanced brackets in the chemical formula.")
            top = stack.pop()
            for k, v in top.items():
                stack[-1][k] += v * mult
        elif ch.isupper():
            j = i + 1
            if j < len(s) and s[j].islower():
                j += 1
            element = s[i:j]
            k = j
            while k < len(s) and s[k].isdigit():
                k += 1
            count = int(s[j:k] or 1)
            i = k
            if element not in _ATOMIC_WEIGHT:
                raise EngineError(f"Unknown element symbol '{element}'.")
            stack[-1][element] += count
        else:
            raise EngineError(f"Unexpected character '{ch}' in formula '{s}'. Write charges separately.")
    if len(stack) != 1:
        raise EngineError("Unbalanced brackets in the chemical formula.")
    return stack[0]


def parse_formula(formula: str) -> Counter:
    f = re.sub(r"\s+", "", formula)
    if not f or len(f) > 200:
        raise EngineError("Give a chemical formula such as H2SO4 or CuSO4.5H2O.")
    total: Counter = Counter()
    for part in re.split(r"[·.*]", f):
        m = re.match(r"^(\d+)(?=[A-Z(\[])", part)
        mult = int(m.group(1)) if m else 1
        if m:
            part = part[m.end():]
        for k, v in _parse_group(part).items():
            total[k] += v * mult
    return total


def op_molar_mass(args: dict) -> dict:
    formula = as_text(need(args, "formula"), "formula")
    counts = parse_formula(formula)
    total = sum(_ATOMIC_WEIGHT[e] * n for e, n in counts.items())
    return {"formula": formula, "molar_mass_g_per_mol": round(total, 4),
            "composition": [{"element": e, "count": n, "mass_percent": round(100 * _ATOMIC_WEIGHT[e] * n / total, 3)}
                            for e, n in counts.items()]}


def op_balance_equation(args: dict) -> dict:
    text = as_text(need(args, "equation"), "equation")
    sides = re.split(r"\s*(?:->|=>|→|=)\s*", text.strip())
    if len(sides) != 2:
        raise EngineError("Write the equation as 'reactants -> products', e.g. 'Fe + O2 -> Fe2O3'.")
    species = [[re.sub(r"^\d+\s*", "", s.strip()) for s in side.split("+") if s.strip()] for side in sides]
    if not species[0] or not species[1]:
        raise EngineError("Both sides need at least one species.")
    names = species[0] + species[1]
    comps = [parse_formula(n) for n in names]
    elements = sorted({e for c in comps for e in c})
    n_react = len(species[0])
    M = sp.Matrix([[(1 if j < n_react else -1) * comps[j].get(e, 0) for j in range(len(names))] for e in elements])
    null = M.nullspace()
    if len(null) != 1:
        raise EngineError("The equation cannot be balanced uniquely (check the formulas).")
    v = null[0]
    lcm = sp.ilcm(*[sp.fraction(x)[1] for x in v])
    v = v * lcm
    if v[0] < 0:
        v = -v
    if any(x <= 0 for x in v):
        raise EngineError("No balanced equation with positive coefficients exists for these species.")
    g = sp.igcd(*[int(x) for x in v])
    coeffs = [int(x) // g for x in v]
    fmt = lambda cs, ns: " + ".join((f"{c} " if c != 1 else "") + n for c, n in zip(cs, ns))  # noqa: E731
    return {"balanced": f"{fmt(coeffs[:n_react], species[0])} -> {fmt(coeffs[n_react:], species[1])}",
            "coefficients": dict(zip(names, coeffs))}


def op_ph(args: dict) -> dict:
    given = {k: as_float(args[k], k) for k in ("h_conc", "oh_conc", "ph", "poh") if args.get(k) is not None}
    if len(given) != 1:
        raise EngineError("Give exactly one of h_conc (mol/L), oh_conc (mol/L), ph, poh. Assumes 25 C (pH + pOH = 14).")
    if "h_conc" in given or "oh_conc" in given:
        c = next(iter(given.values()))
        if c <= 0:
            raise EngineError("Concentration must be positive.")
        ph = -math.log10(c) if "h_conc" in given else 14 + math.log10(c)
    else:
        ph = given["ph"] if "ph" in given else 14 - given["poh"]
    return {"pH": ph, "pOH": 14 - ph, "h_conc": 10 ** (-ph), "oh_conc": 10 ** (ph - 14),
            "nature": "acidic" if ph < 7 else "basic" if ph > 7 else "neutral", "assumes": "25 C, dilute aqueous solution"}


OPERATIONS: dict[str, OperationSpec] = {
    s.name: s
    for s in [
        OperationSpec("statistics", op_statistics, "statistics(data)", "Descriptive statistics and a 95% CI for the mean."),
        OperationSpec("regression", op_regression, "regression(x, y, degree=1, predict_x=None)", "Linear or polynomial least squares fit with R^2."),
        OperationSpec("hypothesis_test", op_hypothesis_test, "hypothesis_test(test, alpha=0.05, ...)",
                      "test: t_one_sample(data,mu), t_two_sample(a,b), t_paired(a,b), mann_whitney(a,b), anova(groups), pearson(a,b), shapiro(data), chi2_gof(observed,expected?), chi2_independence(table)."),
        OperationSpec("distribution", op_distribution, "distribution(name, params, x=None, q=None)",
                      "name: norm, t, chi2, f, binom, poisson, expon, uniform, gamma, beta, lognorm, geom. params use SciPy names, e.g. {\"loc\":0,\"scale\":1} or {\"n\":10,\"p\":0.3}. x gives pdf/pmf/cdf/sf, q gives quantiles."),
        OperationSpec("integrate_numeric", op_integrate_numeric, "integrate_numeric(expression, lower, upper, variable='x')", "Numeric definite integral (quad); bounds may be 'oo'."),
        OperationSpec("solve_numeric", op_solve_numeric, "solve_numeric(expression, variable='x', bracket=None, guess=None, range=None)", "Numeric roots of expression = 0."),
        OperationSpec("optimize", op_optimize, "optimize(expression, variable='x', goal='min', lower=None, upper=None)", "Minimise or maximise a one-variable function."),
        OperationSpec("ode_numeric", op_ode_numeric, "ode_numeric(rhs, states=['y'], y0, t_span, independent='t', points=50, method='RK45')",
                      "Integrate dy/dt = rhs(t, states...). rhs is a list of expressions, one per state."),
        OperationSpec("linalg", op_linalg, "linalg(operation, A, b=None)", "Numeric linear algebra: det, inv, rank, norm, svd, cond, eig, solve, lstsq."),
        OperationSpec("kinematics", op_kinematics, "kinematics(s=, u=, v=, a=, t=)", "Constant-acceleration motion: give three of s,u,v,a,t in SI units."),
        OperationSpec("projectile", op_projectile, "projectile(speed, angle_deg, height=0, g=9.80665)", "Projectile motion without air resistance."),
        OperationSpec("ideal_gas", op_ideal_gas, "ideal_gas(P=, V=, n=, T=)", "Solve PV = nRT for the missing quantity (SI units)."),
        OperationSpec("orbit", op_orbit, "orbit(mass_kg, radius_m)", "Circular-orbit speed, escape speed, period and surface gravity."),
        OperationSpec("constant", op_constant, "constant(name)", "Look up a CODATA physical constant, e.g. 'Planck constant'."),
        OperationSpec("convert_units", op_convert_units, "convert_units(value, from_unit, to_unit)", "Convert between units of length, mass, time, energy, pressure, speed, force, power, volume, angle, temperature (K, C, F)."),
        OperationSpec("molar_mass", op_molar_mass, "molar_mass(formula)", "Molar mass and composition of a formula such as Ca(OH)2 or CuSO4.5H2O."),
        OperationSpec("balance_equation", op_balance_equation, "balance_equation(equation)", "Balance a chemical equation like 'Fe + O2 -> Fe2O3'."),
        OperationSpec("ph", op_ph, "ph(h_conc= | oh_conc= | ph= | poh=)", "Convert between pH, pOH and concentrations at 25 C."),
    ]
}
