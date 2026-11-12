"""Shared helpers for the computation engines."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable


class EngineError(Exception):
    """A problem the caller can understand and fix: bad input, no solution, unsupported operation."""


@dataclass(frozen=True)
class OperationSpec:
    """One callable operation of an engine. `signature` and `summary` are shown to the AI model as tool docs."""

    name: str
    fn: Callable[[dict], dict]
    signature: str
    summary: str


def to_jsonable(obj: Any, _depth: int = 0) -> Any:
    """Convert numpy / sympy / complex / nan values into plain JSON-safe data."""
    if _depth > 12:
        return str(obj)
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, complex):
        return {"re": to_jsonable(obj.real), "im": to_jsonable(obj.imag)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v, _depth + 1) for v in obj]
    tolist = getattr(obj, "tolist", None)  # numpy scalars and arrays
    if callable(tolist):
        try:
            return to_jsonable(tolist(), _depth + 1)
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


def need(args: dict, key: str) -> Any:
    if key not in args or args[key] is None:
        raise EngineError(f"Missing required argument '{key}'.")
    return args[key]


def as_text(value: Any, name: str) -> str:
    if isinstance(value, bool):
        raise EngineError(f"'{name}' must be a string, not a boolean.")
    if isinstance(value, (int, float)):
        return repr(value)
    if not isinstance(value, str):
        raise EngineError(f"'{name}' must be a string.")
    return value


def as_float(value: Any, name: str) -> float:
    try:
        if isinstance(value, bool):
            raise ValueError
        return float(value)
    except (TypeError, ValueError):
        raise EngineError(f"'{name}' must be a number.") from None


def as_float_list(value: Any, name: str, max_len: int = 100_000, min_len: int = 1) -> list[float]:
    if isinstance(value, str):
        value = [p for p in value.replace(";", ",").replace("\n", ",").split(",") if p.strip()]
    if not isinstance(value, (list, tuple)):
        raise EngineError(f"'{name}' must be a list of numbers.")
    if not (min_len <= len(value) <= max_len):
        raise EngineError(f"'{name}' must contain between {min_len} and {max_len} numbers.")
    return [as_float(v, name) for v in value]
