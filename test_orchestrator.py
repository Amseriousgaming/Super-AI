"""Routing, the tool loop, fallback between providers, vision requirements and reference auditing."""
import pytest

from ai.base import AllModelsFailed, NoModelAvailable, ProviderError, ImagePart, Source
from ai.orchestrator import ChatContext, HistoryEntry, Orchestrator, audit_references
from ai.router import ModelRegistry, RouteInput, Router
from ai.tools import ToolRegistry
from engines import EngineHub
from tests.fakes import ScriptedProvider, make_settings, run, spec


def build(providers, **settings_kw):
    settings = make_settings(**settings_kw)
    registry = ModelRegistry(providers, settings.provider_priority_list)
    tools = ToolRegistry(EngineHub(settings), None, None)
    return Orchestrator(settings, registry, Router(registry), tools)


def ctx(**kw):
    base = dict(question="What is the derivative of x^2?", subject="mathematics", level="secondary", mode="step-by-step",
                history=[], images=[], requested_model="auto", features={"math_verification", "scientific_calc"},
                user_id=None, conversation_id="conv12345")
    base.update(kw)
    return ChatContext(**base)


def test_model_can_use_the_math_engine_and_sees_the_result():
    p = ScriptedProvider(script=[("tool", "math_compute", {"operation": "differentiate", "args": {"expression": "x^2", "variable": "x"}}),
                                 ("text", "The derivative is $2x$.")])
    result = run(build([p]).run(ctx()))
    assert result.text == "The derivative is $2x$."
    assert [t.name for t in result.traces] == ["math_compute"] and result.traces[0].ok
    second_call_messages = p.calls[1]["messages"]
    assert second_call_messages[-1].role == "tool" and "2*x" in second_call_messages[-1].content
    assert result.usage.input_tokens == 20  # usage is summed over both model calls


def test_tool_errors_are_returned_to_the_model_not_raised():
    p = ScriptedProvider(script=[("tool", "math_compute", {"operation": "evaluate", "args": {"expression": "__import__('os')"}}),
                                 ("text", "I could not evaluate that.")])
    result = run(build([p]).run(ctx()))
    assert result.traces[0].ok is False
    assert '"ok": false' in p.calls[1]["messages"][-1].content


def test_tools_are_only_offered_when_enabled():
    p = ScriptedProvider()
    run(build([p]).run(ctx(features=set())))
    assert p.calls[0]["tools"] == []
    p2 = ScriptedProvider()
    run(build([p2]).run(ctx()))
    assert set(p2.calls[0]["tools"]) == {"math_compute", "science_compute"}


def test_falls_back_to_next_provider_on_failure():
    bad = ScriptedProvider("bad", fail_with=ProviderError("x", kind="rate_limit"))
    good = ScriptedProvider("good", script=[("text", "answer")])
    result = run(build([bad, good], provider_priority="bad,good").run(ctx()))
    assert result.text == "answer" and result.spec.provider == "good"
    assert any("bad:m1 failed" in w for w in result.warnings)


def test_all_models_failing_reports_every_failure():
    a = ScriptedProvider("a", fail_with=ProviderError("x", kind="timeout"))
    b = ScriptedProvider("b", fail_with=ProviderError("x", kind="unavailable"))
    with pytest.raises(AllModelsFailed) as exc:
        run(build([a, b], provider_priority="a,b").run(ctx()))
    assert {k for _, k in exc.value.failures} == {"timeout", "unavailable"}


def test_images_require_a_vision_model():
    image = [ImagePart("image/jpeg", "AAAA")]
    text_only = ScriptedProvider("text", specs=[spec("text", "m1")])
    seeing = ScriptedProvider("eyes", specs=[spec("eyes", "v1", vision=True)])
    with pytest.raises(NoModelAvailable):
        run(build([text_only]).run(ctx(images=image)))
    result = run(build([text_only, seeing], provider_priority="text,eyes").run(ctx(images=image)))
    assert result.spec.provider == "eyes"


def test_router_prefers_reasoning_models_for_hard_requests():
    reasoning = ScriptedProvider("deep", specs=[spec("deep", "r1", designations=("reasoning",))])
    fast = ScriptedProvider("quick", specs=[spec("quick", "f1", designations=("fast",)), spec("quick", "g1")])
    registry = ModelRegistry([fast, reasoning], ["quick", "deep"])
    router = Router(registry)
    hard = router.decide(RouteInput("Prove that sqrt(2) is irrational", "deep-reasoning", "phd", False))
    assert hard.designation == "reasoning" and hard.candidates[0].key == "deep:r1"
    easy = router.decide(RouteInput("What is 2+2?", "short", "elementary", False))
    assert easy.designation == "fast" and easy.candidates[0].key == "quick:f1"
    forced = router.decide(RouteInput("hi", "detailed", "auto", False, requested_model="deep:r1"))
    assert forced.candidates[0].key == "deep:r1"
    unknown = router.decide(RouteInput("hi", "detailed", "auto", False, requested_model="nope:x"))
    assert unknown.warnings


def test_unverified_references_are_flagged_but_retrieved_ones_are_not():
    src = [Source(1, "Real paper", "https://doi.org/10.1000/real", "s", "journal")]
    text = "See https://doi.org/10.1000/real and https://example.org/invented and 10.9999/made-up."
    flagged = audit_references(text, src, "question")
    assert "https://example.org/invented" in flagged and "10.9999/made-up" in flagged
    assert not any("real" in f for f in flagged)
    p = ScriptedProvider(script=[("text", "Read https://example.org/invented")])
    assert "were not checked" in run(build([p]).run(ctx())).text


def test_streaming_emits_deltas_and_a_final_note():
    events = []

    async def sink(event):
        events.append(event)

    p = ScriptedProvider(script=[("text", "Hello")])
    result = run(build([p]).run(ctx(), sink))
    assert result.text == "Hello" and events[0] == {"type": "delta", "text": "Hello"}


def test_history_is_trimmed_to_start_with_the_user():
    p = ScriptedProvider()
    history = [HistoryEntry("assistant", "orphan"), HistoryEntry("user", "q1"), HistoryEntry("assistant", "a1")]
    run(build([p]).run(ctx(history=history)))
    roles = [m.role for m in p.calls[0]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
