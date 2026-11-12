"""Exact symbolic mathematics with SymPy.

Every operation is a pure function: it takes a dict of arguments and returns a
JSON-safe dict. Expressions arrive as TEXT, often written by an AI model or a
user, so they are never passed to Python `eval` unchecked:

  * only a small character set is accepted (no quotes, backslashes, braces);
  * attribute access and dunder names are rejected;
  * every identifier must be a known maths function/constant or a plain symbol
    name that is not a Python builtin or keyword;
  * parsing runs with an empty `__builtins__`.

Heavy calls additionally run inside the sandbox process pool (engines/sandbox.py)
with a hard timeout and memory cap.
"""
from __future__ import annotations

import builtins
import json
import keyword
import math
import random
import re
from typing import Any

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from .common import EngineError, OperationSpec, as_text, need

MAX_EXPR_CHARS = 2000
MAX_MATRIX_DIM = 12

# --- safe parsing -----------------------------------------------------------
_FUNCS: dict[str, Any] = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot, "sec": sp.sec, "csc": sp.csc,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "atan2": sp.atan2,
    "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "cbrt": sp.cbrt, "root": sp.root,
    "Abs": sp.Abs, "abs": sp.Abs, "sign": sp.sign, "floor": sp.floor, "ceiling": sp.ceiling, "ceil": sp.ceiling,
    "factorial": sp.factorial, "binomial": sp.binomial, "gamma": sp.gamma, "erf": sp.erf,
    "Min": sp.Min, "Max": sp.Max, "min": sp.Min, "max": sp.Max, "gcd": sp.gcd, "lcm": sp.lcm,
    "diff": sp.diff, "integrate": sp.integrate, "Integral": sp.Integral, "Derivative": sp.Derivative,
    "Sum": sp.Sum, "Product": sp.Product, "limit": sp.limit, "Matrix": sp.Matrix,
    "Rational": sp.Rational, "Piecewise": sp.Piecewise, "Heaviside": sp.Heaviside, "DiracDelta": sp.DiracDelta,
}
_CONSTS: dict[str, Any] = {
    "pi": sp.pi, "E": sp.E, "e": sp.E, "I": sp.I, "oo": sp.oo, "inf": sp.oo, "infinity": sp.oo,
}
_ALLOWED_NAMES = set(_FUNCS) | set(_CONSTS)
_BLOCKED_NAMES = set(dir(builtins)) | set(keyword.kwlist) | {"Symbol", "Function", "Integer", "Float", "Wild", "Lambda"}

_ALLOWED_CHAR = re.compile(r"[A-Za-z0-9_\s+\-*/^=<>!(),.\[\]]")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER = re.compile(r"(?<![A-Za-z_0-9])\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+")
_REPLACEMENTS = {
    "×": "*", "·": "*", "⋅": "*", "÷": "/", "−": "-", "–": "-", "π": "pi", "√": "sqrt",
    "²": "^2", "³": "^3", "∞": "oo", "≤": "<=", "≥": ">=", "≠": "!=", "\u00a0": " ",
}
TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_GLOBALS = {
    "__builtins__": {},
    "Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational,
    "Symbol": sp.Symbol, "Function": sp.Function,
}


def _normalise(text: str) -> str:
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    return text.strip()


def _check(text: str, extra: frozenset = frozenset()) -> None:
    if not text:
        raise EngineError("The expression is empty.")
    if len(text) > MAX_EXPR_CHARS:
        raise EngineError(f"The expression is longer than {MAX_EXPR_CHARS} characters.")
    bad = sorted({ch for ch in text if not _ALLOWED_CHAR.fullmatch(ch)})
    if bad:
        raise EngineError(
            f"Unsupported character(s) {bad}. Use plain text such as x^2, sqrt(x), sin(x), pi, exp(x), oo."
        )
    if "__" in text or re.search(r"\.\s*[A-Za-z_]", text):
        raise EngineError("Attribute access is not allowed in expressions.")
    for name in _IDENT.findall(_NUMBER.sub(" ", text)):
        if name in _ALLOWED_NAMES or name in extra:
            continue
        if name in _BLOCKED_NAMES or name.startswith("_") or len(name) > 24:
            raise EngineError(f"The name '{name}' is not allowed in expressions.")


def parse_expression(text: Any, extra_locals: dict | None = None, name: str = "expression") -> sp.Basic:
    """Parse text into a SymPy object using the restricted namespace."""
    text = _normalise(as_text(text, name))
    extra_locals = extra_locals or {}
    _check(text, frozenset(extra_locals))
    if re.search(r"(?<![<>=!])=(?!=)", text):
        raise EngineError("Unexpected '='. Equations are only accepted by solve, ode and verify_solution.")
    try:
        return parse_expr(
            text,
            local_dict={**_FUNCS, **_CONSTS, **extra_locals},
            global_dict=dict(_GLOBALS),
            transformations=TRANSFORMS,
            evaluate=True,
        )
    except EngineError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise EngineError(f"Could not parse '{text}': {type(exc).__name__}: {exc}") from None


def parse_relation(text: Any, extra_locals: dict | None = None) -> sp.Basic:
    """Parse an equation, inequality or bare expression (meaning `= 0`)."""
    text = _normalise(as_text(text, "equation")).replace("==", "=")
    if "!=" in text:
        raise EngineError("'!=' is not supported. Solve the equation and exclude those values instead.")
    if "<" in text or ">" in text:
        return parse_expression(text, extra_locals)
    parts = re.split(r"(?<![<>=!])=(?!=)", text)
    if len(parts) == 1:
        return parse_expression(text, extra_locals)
    if len(parts) != 2:
        raise EngineError("An equation must contain exactly one '='.")
    return sp.Eq(parse_expression(parts[0], extra_locals), parse_expression(parts[1], extra_locals))


def _symbol(name: Any) -> sp.Symbol:
    name = as_text(name, "variable").strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,15}", name) or name in _BLOCKED_NAMES or name in _ALLOWED_NAMES:
        raise EngineError(f"'{name}' is not a valid variable name.")
    return sp.Symbol(name)


def _symbols(value: Any) -> list[sp.Symbol]:
    if isinstance(value, str):
        value = [v for v in value.split(",") if v.strip()]
    return [_symbol(v) for v in value]


def _res(value: Any, **extra: Any) -> dict:
    out = {"result": str(value), "latex": sp.latex(value)}
    out.update(extra)
    return out


def _numeric(value: Any, digits: int = 15) -> str | None:
    try:
        if getattr(value, "free_symbols", None):
            return None
        return str(sp.N(value, digits))
    except Exception:  # noqa: BLE001
        return None


def _var(args: dict, default: str = "x") -> sp.Symbol:
    return _symbol(args.get("variable") or default)


# --- operations -------------------------------------------------------------
def op_evaluate(args: dict) -> dict:
    expr = parse_expression(need(args, "expression"))
    digits = int(args.get("precision", 15))
    if not 5 <= digits <= 200:
        raise EngineError("precision must be between 5 and 200.")
    exact = sp.simplify(expr)
    return _res(exact, numeric=_numeric(expr, digits))


def op_transform(args: dict) -> dict:
    methods = {
        "simplify": sp.simplify, "expand": sp.expand, "factor": sp.factor, "trigsimp": sp.trigsimp,
        "apart": sp.apart, "together": sp.together, "cancel": sp.cancel, "powsimp": sp.powsimp,
        "expand_trig": sp.expand_trig, "radsimp": sp.radsimp,
    }
    method = str(args.get("method", "simplify"))
    if method not in methods:
        raise EngineError(f"Unknown method '{method}'. Choose one of: {', '.join(sorted(methods))}.")
    return _res(methods[method](parse_expression(need(args, "expression"))), method=method)


def op_solve(args: dict) -> dict:
    eqs = need(args, "equations")
    if isinstance(eqs, str):
        eqs = [eqs]
    if not isinstance(eqs, list) or not 1 <= len(eqs) <= 12:
        raise EngineError("'equations' must be a string or a list of 1 to 12 strings.")
    rels = [parse_relation(e) for e in eqs]
    symbols = _symbols(args["variables"]) if args.get("variables") else sorted(
        set().union(*[getattr(r, "free_symbols", set()) for r in rels]), key=str
    )
    if not symbols:
        return {"result": str(sp.simplify(rels[0])), "note": "No unknowns: the statement was evaluated directly."}
    real = str(args.get("domain", "complex")).lower() == "real"
    inequality = any(getattr(r, "is_Relational", False) and not isinstance(r, sp.Eq) for r in rels)

    if len(rels) == 1 and len(symbols) == 1:
        domain = sp.S.Reals if (real or inequality) else sp.S.Complexes
        sol = sp.solveset(rels[0], symbols[0], domain=domain)
        out: dict[str, Any] = {"variable": str(symbols[0]), "domain": "real" if domain == sp.S.Reals else "complex",
                               "result": str(sol), "latex": sp.latex(sol)}
        if isinstance(sol, sp.ConditionSet):
            out["note"] = "No closed form found. Try science_compute solve_numeric for numeric roots."
        elif sol.is_FiniteSet:
            vals = sorted(sol, key=str)
            out["values"] = [str(v) for v in vals]
            out["numeric"] = [_numeric(v, 12) for v in vals]
        return out

    if inequality:
        raise EngineError("Systems of inequalities are not supported. Solve one inequality at a time.")
    solutions = sp.solve(rels, symbols, dict=True)
    return {
        "variables": [str(s) for s in symbols],
        "result": str(solutions),
        "solutions": [{str(k): str(v) for k, v in s.items()} for s in solutions],
        "latex": sp.latex(solutions),
    }


def op_differentiate(args: dict) -> dict:
    expr = parse_expression(need(args, "expression"))
    variable = args.get("variable") or "x"
    order = int(args.get("order", 1))
    if not 1 <= order <= 20:
        raise EngineError("order must be between 1 and 20.")
    if isinstance(variable, list):
        result = sp.diff(expr, *_symbols(variable))
    else:
        result = sp.diff(expr, _symbol(variable), order)
    out = _res(result)
    if args.get("point") is not None:
        sym = _symbol(variable if not isinstance(variable, list) else variable[0])
        value = result.subs(sym, parse_expression(args["point"], name="point"))
        out["at_point"] = str(sp.simplify(value))
        out["at_point_numeric"] = _numeric(value)
    return out


def op_integrate(args: dict) -> dict:
    expr = parse_expression(need(args, "expression"))
    var = _var(args)
    lower, upper = args.get("lower"), args.get("upper")
    if (lower is None) != (upper is None):
        raise EngineError("Give both 'lower' and 'upper' for a definite integral, or neither.")
    if lower is None:
        result = sp.integrate(expr, var)
        out = _res(result, kind="indefinite", note="Add an arbitrary constant C.")
    else:
        a, b = parse_expression(lower, name="lower"), parse_expression(upper, name="upper")
        result = sp.integrate(expr, (var, a, b))
        out = _res(result, kind="definite", numeric=_numeric(result))
        if result.has(sp.Integral):
            try:
                out["numeric"] = str(sp.Integral(expr, (var, a, b)).evalf(15))
            except Exception:  # noqa: BLE001
                pass
    if result.has(sp.Integral):
        out["note"] = "No closed form found by SymPy; use science_compute integrate_numeric for definite integrals."
    return out


def op_limit(args: dict) -> dict:
    expr = parse_expression(need(args, "expression"))
    var = _var(args)
    point = parse_expression(need(args, "point"), name="point")
    direction = str(args.get("direction", "+-"))
    if direction not in {"+", "-", "+-"}:
        raise EngineError("direction must be '+', '-' or '+-'.")
    return _res(sp.limit(expr, var, point, direction), direction=direction)


def op_series(args: dict) -> dict:
    expr = parse_expression(need(args, "expression"))
    var = _var(args)
    point = parse_expression(args.get("point", 0), name="point")
    order = int(args.get("order", 6))
    if not 1 <= order <= 40:
        raise EngineError("order must be between 1 and 40.")
    s = sp.series(expr, var, point, order)
    return _res(s.removeO(), with_remainder=str(s))


def _matrix(raw: Any, name: str) -> sp.Matrix:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raise EngineError(f"'{name}' must be a list of rows, e.g. [[1,2],[3,4]].") from None
    if not isinstance(raw, list) or not raw or not all(isinstance(r, list) and r for r in raw):
        raise EngineError(f"'{name}' must be a non-empty list of rows, e.g. [[1,2],[3,4]].")
    rows, cols = len(raw), len(raw[0])
    if rows > MAX_MATRIX_DIM or cols > MAX_MATRIX_DIM:
        raise EngineError(f"Matrices larger than {MAX_MATRIX_DIM}x{MAX_MATRIX_DIM} are not supported symbolically.")
    if any(len(r) != cols for r in raw):
        raise EngineError(f"All rows of '{name}' must have the same length.")
    return sp.Matrix([[parse_expression(c, name=name) for c in r] for r in raw])


def _mat_out(m: sp.Matrix) -> dict:
    return {"matrix": [[str(c) for c in m.row(i)] for i in range(m.rows)], "latex": sp.latex(m)}


def op_matrix(args: dict) -> dict:
    op = str(need(args, "operation")).lower()
    A = _matrix(need(args, "A"), "A")
    unary = {
        "determinant": lambda: _res(sp.simplify(A.det())),
        "inverse": lambda: _mat_out(A.inv()),
        "transpose": lambda: _mat_out(A.T),
        "rank": lambda: {"result": A.rank()},
        "trace": lambda: _res(A.trace()),
        "rref": lambda: {**_mat_out(A.rref()[0]), "pivots": list(A.rref()[1])},
        "nullspace": lambda: {"vectors": [[str(c) for c in v] for v in A.nullspace()]},
        "eigenvalues": lambda: {"eigenvalues": {str(k): v for k, v in A.eigenvals().items()}},
        "eigenvectors": lambda: {
            "eigenvectors": [
                {"eigenvalue": str(val), "multiplicity": mult, "vectors": [[str(c) for c in v] for v in vecs]}
                for val, mult, vecs in A.eigenvects()
            ]
        },
    }
    if op in unary:
        if op in {"determinant", "inverse", "trace", "eigenvalues", "eigenvectors"} and A.rows != A.cols:
            raise EngineError(f"{op} needs a square matrix.")
        try:
            return unary[op]()
        except sp.matrices.exceptions.NonInvertibleMatrixError:
            raise EngineError("The matrix is singular (determinant 0), so it has no inverse.") from None
    if op == "power":
        n = int(args.get("exponent", 2))
        if not -50 <= n <= 50 or A.rows != A.cols:
            raise EngineError("power needs a square matrix and an integer exponent between -50 and 50.")
        return _mat_out(A**n)
    if op in {"multiply", "add", "subtract", "solve"}:
        B = _matrix(need(args, "B"), "B")
        if op == "multiply":
            if A.cols != B.rows:
                raise EngineError(f"Cannot multiply {A.rows}x{A.cols} by {B.rows}x{B.cols}.")
            return _mat_out(A * B)
        if op == "solve":
            if A.rows != A.cols or B.rows != A.rows:
                raise EngineError("solve needs a square A and a right-hand side B with the same number of rows.")
            return _mat_out(A.LUsolve(B))
        if (A.rows, A.cols) != (B.rows, B.cols):
            raise EngineError("Matrices must have the same shape.")
        return _mat_out(A + B if op == "add" else A - B)
    raise EngineError(
        "Unknown matrix operation. Use: determinant, inverse, transpose, rank, trace, rref, nullspace, "
        "eigenvalues, eigenvectors, power, multiply, add, subtract, solve."
    )


def op_ode(args: dict) -> dict:
    fname = str(args.get("function") or "y")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,7}", fname) or fname in _BLOCKED_NAMES | _ALLOWED_NAMES:
        raise EngineError("'function' must be a short name such as y.")
    x = _var(args)
    f = sp.Function(fname)
    rel = parse_relation(need(args, "equation"), extra_locals={fname: f})
    eq = rel if isinstance(rel, sp.Eq) else sp.Eq(rel, 0)

    ics: dict = {}
    for key, value in (args.get("ics") or {}).items():  # e.g. {"y(0)": "1", "y'(0)": "0"}
        m = re.fullmatch(rf"{fname}('*)\((.+)\)", key.strip())
        if not m:
            raise EngineError(f"Initial condition '{key}' must look like {fname}(0) or {fname}'(0).")
        point = parse_expression(m.group(2), name="initial point")
        n = len(m.group(1))
        lhs = f(point) if n == 0 else sp.diff(f(x), x, n).subs(x, point)
        ics[lhs] = parse_expression(value, name="initial value")
    sol = sp.dsolve(eq, f(x), ics=ics or None)
    return _res(sol, solution=str(sol.rhs) if isinstance(sol, sp.Eq) else str(sol))


def _numeric_equal(a: sp.Basic, b: sp.Basic, samples: int = 12) -> tuple[bool | None, str | None]:
    symbols = sorted(a.free_symbols | b.free_symbols, key=str)
    rng = random.Random(12345)
    good = 0
    for _ in range(samples * 3):
        point = {s: sp.Float(rng.uniform(-3, 3)) for s in symbols}
        try:
            va, vb = complex(sp.N(a.subs(point))), complex(sp.N(b.subs(point)))
        except Exception:  # noqa: BLE001  (singularity or complex/undefined value)
            continue
        if any(math.isnan(z) or math.isinf(z) for z in (va.real, va.imag, vb.real, vb.imag)):
            continue
        if abs(va - vb) > 1e-8 * max(1.0, abs(va), abs(vb)):
            return False, ", ".join(f"{s}={float(v):.4g}" for s, v in point.items())
        good += 1
        if good >= samples:
            break
    return (True if good >= 3 else None), None


def op_verify_equal(args: dict) -> dict:
    a = parse_expression(need(args, "a"), name="a")
    b = parse_expression(need(args, "b"), name="b")
    diff = sp.simplify(a - b)
    if diff == 0:
        return {"equal": True, "method": "symbolic", "difference": "0"}
    ok, witness = _numeric_equal(a, b)
    if ok is False:
        return {"equal": False, "method": "numeric", "difference": str(diff), "counterexample": witness}
    if ok is True:
        return {"equal": True, "method": "numeric-sampling",
                "difference": str(diff), "note": "SymPy could not prove it, but the values agree at random sample points."}
    return {"equal": None, "method": "inconclusive", "difference": str(diff)}


def op_verify_solution(args: dict) -> dict:
    var = _var(args)
    rel = parse_relation(need(args, "equation"))
    value = parse_expression(need(args, "value"), name="value")
    if isinstance(rel, sp.Eq):
        residual = sp.simplify((rel.lhs - rel.rhs).subs(var, value))
    elif getattr(rel, "is_Relational", False):
        holds = rel.subs(var, value)
        return {"satisfies": bool(holds) if holds in (sp.true, sp.false) else None, "substituted": str(holds)}
    else:
        residual = sp.simplify(rel.subs(var, value))
    ok = residual == 0
    if not ok:
        num = _numeric(residual)
        ok = num is not None and abs(complex(sp.N(residual))) < 1e-9
    return {"satisfies": bool(ok), "residual": str(residual)}


def op_verify_integral(args: dict) -> dict:
    var = _var(args)
    antiderivative = parse_expression(need(args, "antiderivative"), name="antiderivative")
    integrand = parse_expression(need(args, "integrand"), name="integrand")
    d = sp.diff(antiderivative, var)
    return {**op_verify_equal({"a": str(d), "b": str(integrand)}), "derivative_of_antiderivative": str(d)}


def op_latex(args: dict) -> dict:
    return {"latex": sp.latex(parse_expression(need(args, "expression")))}


def op_number_theory(args: dict) -> dict:
    op = str(need(args, "operation")).lower()
    n = int(need(args, "n"))
    if abs(n) > 10**15:
        raise EngineError("n is too large (limit 10^15).")
    if op == "factorint":
        return {"n": n, "factors": {str(p): e for p, e in sp.factorint(n).items()}}
    if op == "isprime":
        return {"n": n, "isprime": bool(sp.isprime(n))}
    if op == "totient":
        return {"n": n, "totient": int(sp.totient(n))}
    if op == "divisors":
        if abs(n) > 10**10:
            raise EngineError("divisors is limited to |n| <= 10^10.")
        return {"n": n, "divisors": [int(d) for d in sp.divisors(abs(n))]}
    if op in {"gcd", "lcm"}:
        m = int(need(args, "m"))
        return {"result": int(sp.gcd(n, m) if op == "gcd" else sp.lcm(n, m))}
    raise EngineError("Unknown operation. Use: factorint, isprime, totient, divisors, gcd, lcm.")


OPERATIONS: dict[str, OperationSpec] = {
    s.name: s
    for s in [
        OperationSpec("evaluate", op_evaluate, "evaluate(expression, precision=15)",
                      "Simplify an expression and give an exact and numeric value."),
        OperationSpec("transform", op_transform, "transform(expression, method='simplify')",
                      "Rewrite an expression. method: simplify, expand, factor, trigsimp, apart, together, cancel, powsimp, expand_trig, radsimp."),
        OperationSpec("solve", op_solve, "solve(equations, variables=None, domain='complex')",
                      "Solve an equation, inequality or system. equations: 'x^2-4=0' or a list. domain: 'real' or 'complex'."),
        OperationSpec("differentiate", op_differentiate, "differentiate(expression, variable='x', order=1, point=None)",
                      "Derivative; variable may be a list for partial derivatives; optional evaluation point."),
        OperationSpec("integrate", op_integrate, "integrate(expression, variable='x', lower=None, upper=None)",
                      "Indefinite integral, or definite when lower and upper are given (use 'oo' for infinity)."),
        OperationSpec("limit", op_limit, "limit(expression, variable='x', point, direction='+-')",
                      "Limit as variable approaches point; direction '+', '-' or '+-'."),
        OperationSpec("series", op_series, "series(expression, variable='x', point=0, order=6)",
                      "Taylor/Laurent series expansion."),
        OperationSpec("matrix", op_matrix, "matrix(operation, A, B=None, exponent=None)",
                      "Matrices as [[1,2],[3,4]]. operation: determinant, inverse, transpose, rank, trace, rref, nullspace, eigenvalues, eigenvectors, power, multiply, add, subtract, solve (A x = B)."),
        OperationSpec("ode", op_ode, "ode(equation, function='y', variable='x', ics=None)",
                      "Solve an ordinary differential equation, e.g. equation='diff(y(x),x,2)+y(x)=0', ics={\"y(0)\":\"1\",\"y'(0)\":\"0\"}."),
        OperationSpec("verify_equal", op_verify_equal, "verify_equal(a, b)",
                      "Check whether two expressions are mathematically equal (symbolic proof, then numeric sampling)."),
        OperationSpec("verify_solution", op_verify_solution, "verify_solution(equation, value, variable='x')",
                      "Substitute a proposed solution into an equation and check it."),
        OperationSpec("verify_integral", op_verify_integral, "verify_integral(antiderivative, integrand, variable='x')",
                      "Check an antiderivative by differentiating it."),
        OperationSpec("latex", op_latex, "latex(expression)", "Convert an expression to LaTeX."),
        OperationSpec("number_theory", op_number_theory, "number_theory(operation, n, m=None)",
                      "operation: factorint, isprime, totient, divisors, gcd, lcm."),
    ]
}
