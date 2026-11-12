"""Maths and science engines, including attempts to break out of the expression sandbox."""
import pytest

from engines import EngineError, EngineHub, mathematics as M, scientific as S
from tests.fakes import make_settings, run


def call(engine, op, **args):
    ops = M.OPERATIONS if engine == "math" else S.OPERATIONS
    return ops[op].fn(args)


def test_solve_and_verify():
    assert call("math", "solve", equations="x^2-5x+6=0")["values"] == ["2", "3"]
    assert call("math", "verify_solution", equation="x^2-5x+6=0", value="2")["satisfies"] is True
    assert call("math", "verify_equal", a="sin(x)^2+cos(x)^2", b="1")["equal"] is True
    assert call("math", "verify_equal", a="(x+1)^2", b="x^2+1")["equal"] is False


def test_calculus_and_matrices():
    assert call("math", "differentiate", expression="x^3", variable="x")["result"] == "3*x**2"
    assert call("math", "integrate", expression="x*exp(-x^2)", variable="x", lower="0", upper="oo")["result"] == "1/2"
    assert call("math", "limit", expression="sin(x)/x", variable="x", point="0")["result"] == "1"
    assert call("math", "matrix", operation="determinant", A=[[1, 2], [3, 4]])["result"] == "-2"
    with pytest.raises(EngineError):
        call("math", "matrix", operation="inverse", A=[[1, 2], [2, 4]])


def test_ode_with_initial_conditions():
    out = call("math", "ode", equation="diff(y(x),x,2)+y(x)=0", ics={"y(0)": "1", "y'(0)": "0"})
    assert out["solution"] == "cos(x)"


@pytest.mark.parametrize("bad", [
    "__import__('os').system('id')", "().__class__", "x.__class__", "open('/etc/passwd')", "lambda: 1",
    "exec(x)", "eval(x)", "getattr(x, x)", "[i for i in range(3)]", "a;b", "Symbol('x')", "print(1)",
])
def test_expression_sandbox_rejects_code(bad):
    with pytest.raises(EngineError):
        call("math", "evaluate", expression=bad)


def test_statistics_and_tests():
    out = call("science", "statistics", data=[2, 4, 4, 4, 5, 5, 7, 9])
    assert out["mean"] == 5.0 and out["std_population"] == 2.0
    t = call("science", "hypothesis_test", test="t_two_sample", a=[5.1, 4.9, 5.6, 5.8, 6.0], b=[4.1, 4.5, 4.9, 5.0, 4.2])
    assert t["reject_null"] is True


def test_physics_and_chemistry():
    assert abs(call("science", "projectile", speed=20, angle_deg=45)["range_m"] - 40.79) < 0.01
    assert call("science", "balance_equation", equation="Fe + O2 -> Fe2O3")["balanced"] == "4 Fe + 3 O2 -> 2 Fe2O3"
    assert abs(call("science", "molar_mass", formula="Ca(OH)2")["molar_mass_g_per_mol"] - 74.092) < 0.01
    assert abs(call("science", "convert_units", value=100, from_unit="F", to_unit="C")["value"] - 37.78) < 0.01
    with pytest.raises(EngineError):
        call("science", "convert_units", value=1, from_unit="kg", to_unit="m")


def test_numeric_methods():
    assert abs(call("science", "integrate_numeric", expression="exp(-x^2)", lower="-oo", upper="oo")["value"] - 1.7724538509) < 1e-6
    assert abs(call("science", "solve_numeric", expression="x^3-2*x-5", range=[-5, 5])["roots"][0] - 2.0945514815) < 1e-6
    ode = call("science", "ode_numeric", rhs=["-2*y"], states=["y"], y0=[1], t_span=[0, 1], points=3)
    assert abs(ode["final"]["y"] - 0.1353352832) < 1e-6


def test_hub_validates_and_caches():
    hub = EngineHub(make_settings())
    first = run(hub.run("math", "evaluate", {"expression": "2+2"}))
    assert first["result"] == "4"
    assert run(hub.run("math", "evaluate", {"expression": "2+2"})) == first
    with pytest.raises(EngineError):
        run(hub.run("math", "nope", {}))
    with pytest.raises(EngineError):
        run(hub.run("chemistry", "x", {}))


def test_sandbox_kills_runaway_computation():
    """Needs real worker processes: a runaway power tower must be stopped by the hard timeout."""
    import asyncio

    async def scenario():
        hub = EngineHub(make_settings(engine_sandbox=True, engine_workers=1, engine_timeout_seconds=3))
        await hub.start()
        try:
            with pytest.raises(EngineError):
                await hub.run("math", "evaluate", {"expression": "9^9^9^9"})
            # the worker was replaced, so the next call still works
            assert (await hub.run("math", "evaluate", {"expression": "1+1"}))["result"] == "2"
        finally:
            await hub.stop()

    asyncio.run(scenario())
