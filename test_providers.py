"""Provider adapters against a mocked HTTP transport (no network, no API keys)."""
import json

import httpx
import pytest

from ai.base import GenerationParams, Message, ProviderError, ToolSpec
from ai.providers import AnthropicProvider, OpenAICompatibleProvider
from config import ProviderConfig
from tests.fakes import make_settings, run

TOOL = ToolSpec("math_compute", "d", {"type": "object", "properties": {}})
PARAMS = GenerationParams(max_tokens=100)


def attach(provider, handler):
    provider.client = httpx.AsyncClient(base_url="https://mock.test", transport=httpx.MockTransport(handler))
    return provider


def openai_provider():
    cfg = ProviderConfig("openai", "openai_compat", "https://mock.test/v1", "sk-secret", {"general": "m1"}, vision_all=True)
    return OpenAICompatibleProvider(cfg, make_settings())


def anthropic_provider():
    cfg = ProviderConfig("anthropic", "anthropic", "https://mock.test", "sk-ant-secret", {"general": "c1"}, vision_all=True)
    return AnthropicProvider(cfg, make_settings())


def test_openai_compat_parses_text_tools_and_usage():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "<think>hidden</think>Hi", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "math_compute", "arguments": "{\"operation\": \"evaluate\"}"}}]},
            "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}})

    p = attach(openai_provider(), handler)
    out = run(p.complete(p.specs[0], [Message("user", "hi")], [TOOL], PARAMS))
    assert out.text == "Hi" and out.tool_calls[0].arguments == {"operation": "evaluate"}
    assert out.usage.input_tokens == 7 and seen["body"]["tools"][0]["function"]["name"] == "math_compute"


def test_openai_compat_streaming_assembles_text_and_tool_calls():
    sse = "\n".join([
        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t1","function":{"name":"math_compute","arguments":"{\\"a\\":"}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":"tool_calls"}]}',
        "data: [DONE]", ""])
    p = attach(openai_provider(), lambda r: httpx.Response(200, content=sse.encode()))

    async def go():
        return [e async for e in p.stream(p.specs[0], [Message("user", "hi")], [TOOL], PARAMS)]

    events = run(go())
    assert "".join(e.text for e in events if e.kind == "text") == "Hello"
    done = events[-1].completion
    assert done.text == "Hello" and done.tool_calls[0].arguments == {"a": 1}


def test_anthropic_parses_tool_use_and_sends_images_and_system():
    seen = {}

    def handler(request):
        seen["headers"], seen["body"] = request.headers, json.loads(request.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "Checking"},
                                                      {"type": "tool_use", "id": "tu1", "name": "math_compute", "input": {"operation": "latex"}}],
                                         "stop_reason": "tool_use", "usage": {"input_tokens": 11, "output_tokens": 4}})

    p = attach(anthropic_provider(), handler)
    from ai.base import ImagePart
    out = run(p.complete(p.specs[0], [Message("system", "SYS"), Message("user", "q", images=[ImagePart("image/jpeg", "AAAA")])], [TOOL], PARAMS))
    assert out.text == "Checking" and out.tool_calls[0].name == "math_compute" and out.usage.output_tokens == 4
    assert seen["body"]["system"] == "SYS" and seen["body"]["messages"][0]["content"][0]["type"] == "image"


def test_anthropic_streaming():
    sse = "\n".join([
        'data: {"type":"message_start","message":{"usage":{"input_tokens":5}}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi "}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"there"}}',
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}', ""])
    p = attach(anthropic_provider(), lambda r: httpx.Response(200, content=sse.encode()))

    async def go():
        return [e async for e in p.stream(p.specs[0], [Message("user", "hi")], [], PARAMS)]

    done = run(go())[-1].completion
    assert done.text == "Hi there" and done.usage.input_tokens == 5 and done.usage.output_tokens == 2


@pytest.mark.parametrize("status,kind", [(401, "auth"), (429, "rate_limit"), (500, "unavailable"), (529, "unavailable"), (400, "bad_request")])
def test_http_errors_are_classified_and_never_leak_keys(status, kind):
    p = attach(openai_provider(), lambda r: httpx.Response(status, json={"error": {"message": "nope sk-secret"}}, headers={"retry-after": "2"}))
    with pytest.raises(ProviderError) as exc:
        run(p.complete(p.specs[0], [Message("user", "hi")], [], PARAMS))
    assert exc.value.kind == kind and "sk-secret" not in str(exc.value)


def test_network_failure_is_unavailable():
    def boom(request):
        raise httpx.ConnectError("down")

    p = attach(openai_provider(), boom)
    with pytest.raises(ProviderError) as exc:
        run(p.complete(p.specs[0], [Message("user", "hi")], [], PARAMS))
    assert exc.value.kind == "unavailable"
