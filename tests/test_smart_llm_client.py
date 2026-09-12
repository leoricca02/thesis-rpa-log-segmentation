"""Tests for SmartLLMClient (no network: a fake transport is injected)."""

from __future__ import annotations

import json

import pytest
import requests

from smart_llm_client import SmartLLMClient, TransportError

_SCHEMA = {"type": "ARRAY", "items": {"type": "STRING"}}


def _envelope(text: str, in_t: int = 5, out_t: int = 3) -> dict:
    """Build a minimal Gemini-shaped response envelope."""
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {
            "promptTokenCount": in_t,
            "candidatesTokenCount": out_t,
            "totalTokenCount": in_t + out_t,
        },
    }


def _make_client(tmp_path, transport, **kwargs) -> SmartLLMClient:
    return SmartLLMClient(
        cache_file=str(tmp_path / "cache.json"),
        token_log_file=str(tmp_path / "tokens.csv"),
        transport=transport,
        sleep=lambda _s: None,
        **kwargs,
    )


def test_generate_content_parses_json(tmp_path):
    transport = lambda url, payload, timeout: _envelope('["a", "b"]')
    client = _make_client(tmp_path, transport)
    result = client.generate_content("m", "sys", "user", _SCHEMA)
    assert result == ["a", "b"]


def test_generate_content_strips_markdown_fences(tmp_path):
    transport = lambda u, p, t: _envelope('```json\n["x"]\n```')
    client = _make_client(tmp_path, transport)
    assert client.generate_content("m", "s", "u", _SCHEMA) == ["x"]


def test_cache_hit_skips_second_transport_call(tmp_path):
    calls = {"n": 0}

    def transport(u, p, t):
        calls["n"] += 1
        return _envelope('["cached"]')

    client = _make_client(tmp_path, transport)
    client.generate_content("m", "s", "u", _SCHEMA)
    client.generate_content("m", "s", "u", _SCHEMA)
    assert calls["n"] == 1


def test_cache_persists_across_instances(tmp_path):
    transport = lambda u, p, t: _envelope('["v"]')
    c1 = _make_client(tmp_path, transport)
    c1.generate_content("m", "s", "u", _SCHEMA)

    def fail(u, p, t):
        raise AssertionError("should have hit cache")

    c2 = _make_client(tmp_path, fail)
    assert c2.generate_content("m", "s", "u", _SCHEMA) == ["v"]


def test_schema_key_order_independence(tmp_path):
    transport = lambda u, p, t: _envelope('["v"]')
    client = _make_client(tmp_path, transport)
    k1 = client._get_cache_key("m", "s", "u", {"a": 1, "b": 2}, 0)
    k2 = client._get_cache_key("m", "s", "u", {"b": 2, "a": 1}, 0)
    assert k1 == k2


def test_different_thinking_budget_is_different_cache_key(tmp_path):
    transport = lambda u, p, t: _envelope('["v"]')
    client = _make_client(tmp_path, transport)
    k0 = client._get_cache_key("m", "s", "u", _SCHEMA, 0)
    k512 = client._get_cache_key("m", "s", "u", _SCHEMA, 512)
    assert k0 != k512


def test_retry_on_503_then_success(tmp_path):
    attempts = {"n": 0}

    def transport(u, p, t):
        attempts["n"] += 1
        if attempts["n"] < 3:
            resp = requests.Response()
            resp.status_code = 503
            raise requests.HTTPError(response=resp)
        return _envelope('["ok"]')

    client = _make_client(tmp_path, transport, max_retries=4, base_backoff=0.0)
    assert client.generate_content("m", "s", "u", _SCHEMA) == ["ok"]
    assert attempts["n"] == 3


def test_non_retryable_400_raises_immediately(tmp_path):
    attempts = {"n": 0}

    def transport(u, p, t):
        attempts["n"] += 1
        resp = requests.Response()
        resp.status_code = 400
        raise requests.HTTPError(response=resp)

    client = _make_client(tmp_path, transport, max_retries=4, base_backoff=0.0)
    with pytest.raises(TransportError):
        client.generate_content("m", "s", "u", _SCHEMA)
    assert attempts["n"] == 1


def test_exhausted_retries_raises(tmp_path):
    def transport(u, p, t):
        raise requests.ConnectionError("down")

    client = _make_client(tmp_path, transport, max_retries=2, base_backoff=0.0)
    with pytest.raises(TransportError):
        client.generate_content("m", "s", "u", _SCHEMA)


def test_blocked_response_raises_valueerror(tmp_path):
    transport = lambda u, p, t: {"promptFeedback": {"blockReason": "SAFETY"}}
    client = _make_client(tmp_path, transport)
    with pytest.raises(ValueError, match="SAFETY"):
        client.generate_content("m", "s", "u", _SCHEMA)


def test_non_json_text_raises_valueerror(tmp_path):
    transport = lambda u, p, t: _envelope("not json at all")
    client = _make_client(tmp_path, transport)
    with pytest.raises(ValueError, match="non-JSON"):
        client.generate_content("m", "s", "u", _SCHEMA)


def test_telemetry_written_with_header(tmp_path):
    transport = lambda u, p, t: _envelope('["v"]', in_t=10, out_t=4)
    log = tmp_path / "tokens.csv"
    client = SmartLLMClient(
        cache_file=str(tmp_path / "c.json"),
        token_log_file=str(log),
        transport=transport,
        sleep=lambda _s: None,
    )
    client.generate_content("mymodel", "s", "u", _SCHEMA)
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "timestamp,model,input_tokens,output_tokens,total_tokens"
    assert lines[1].split(",")[1] == "mymodel"
    assert lines[1].split(",")[2:5] == ["10", "4", "14"]


def test_corrupt_cache_file_tolerated(tmp_path):
    cache = tmp_path / "c.json"
    cache.write_text("{ this is not valid json", encoding="utf-8")
    transport = lambda u, p, t: _envelope('["v"]')
    client = SmartLLMClient(
        cache_file=str(cache),
        token_log_file=str(tmp_path / "t.csv"),
        transport=transport,
        sleep=lambda _s: None,
    )
    assert client.generate_content("m", "s", "u", _SCHEMA) == ["v"]


def test_missing_key_with_default_transport_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API Key"):
        SmartLLMClient()


def test_thinking_budget_zero_sent_in_payload(tmp_path):
    captured = {}
    def transport(url, payload, timeout):
        captured["payload"] = payload
        return _envelope('["v"]')
    client = _make_client(tmp_path, transport)
    client.generate_content("m", "s", "u", _SCHEMA, thinking_budget=0)
    cfg = captured["payload"]["generationConfig"]
    assert cfg["thinkingConfig"]["thinkingBudget"] == 0


def test_thinking_budget_512_sent_in_payload(tmp_path):
    captured = {}
    def transport(url, payload, timeout):
        captured["payload"] = payload
        return _envelope('["v"]')
    client = _make_client(tmp_path, transport)
    client.generate_content("m", "s", "u", _SCHEMA, thinking_budget=512)
    cfg = captured["payload"]["generationConfig"]
    assert cfg["thinkingConfig"]["thinkingBudget"] == 512
