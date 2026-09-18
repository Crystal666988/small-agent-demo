"""Tests for provider request-building and response-parsing (no network)."""
from miniagent.config import Config
from miniagent.llm import LLMClient


def cfg(provider):
    return Config(
        base_url="https://api.deepseek.com" if provider == "openai" else "https://api.anthropic.com",
        auth_token="tok", model="m", max_context_tokens=8000, max_steps=8, provider=provider,
    )


def test_openai_request_shape():
    c = LLMClient(cfg("openai"))
    url, payload = c._build_request("SYS", [{"role": "user", "content": "hi"}], 100)
    assert url.endswith("/chat/completions")
    assert payload["messages"][0] == {"role": "system", "content": "SYS"}
    assert payload["messages"][1] == {"role": "user", "content": "hi"}
    assert "system" not in payload


def test_anthropic_request_shape():
    c = LLMClient(cfg("anthropic"))
    url, payload = c._build_request("SYS", [{"role": "user", "content": "hi"}], 100)
    assert url.endswith("/v1/messages")
    assert payload["system"] == "SYS"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]


def test_openai_extract_text():
    c = LLMClient(cfg("openai"))
    body = {"choices": [{"message": {"content": "hello"}}]}
    assert c._extract_text(body) == "hello"


def test_anthropic_extract_text():
    c = LLMClient(cfg("anthropic"))
    body = {"content": [{"type": "text", "text": "hello"}]}
    assert c._extract_text(body) == "hello"


def test_openai_headers_have_bearer_only():
    c = LLMClient(cfg("openai"))
    h = c._headers()
    assert h["Authorization"] == "Bearer tok"
    assert "x-api-key" not in h
