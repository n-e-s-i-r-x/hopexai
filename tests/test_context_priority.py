from __future__ import annotations

import pytest

from app.config import GatewayConfig
from app.discovery import _context_window
from app.errors import PublicAPIError
from app.registry import Registry
from app.requests import parse_chat_request, parse_responses_request
from app.types import Candidate
from tests.helpers import ProviderMock, config_data, make_env


def _candidate(
    model_id: str,
    context_window: int | None,
    *,
    provider_name: str = "test-provider",
    provider_priority: int = 100,
    capabilities: dict[str, bool | None] | None = None,
) -> Candidate:
    raw = config_data(provider_name=provider_name, base_url=f"https://{provider_name}.mock")
    raw.pop("_test_model", None)
    provider = GatewayConfig.model_validate(raw).providers[0]
    base = provider.base_urls[0]
    return Candidate(
        provider=provider,
        base=base,
        model_id=model_id,
        api_format="openai_chat",
        endpoints={"chat_completions": ["/v1/chat/completions"]},
        capabilities=capabilities or {"chat": True},
        pricing="unknown",
        context_window=context_window,
        provider_priority=provider_priority,
        provider_weight=provider.selection_weight,
        model_weight=1.0,
        key_options=[("test-key", "test-fingerprint")],
    )


def _text_request():
    return parse_chat_request(
        {"model": "hopexai", "messages": [{"role": "user", "content": "hello"}]},
        max_messages=20,
        max_tools=5,
    )


def _choose_first(monkeypatch):
    monkeypatch.setattr("app.registry.secrets.choice", lambda values: values[0])
    monkeypatch.setattr(
        "app.registry.secrets.SystemRandom.choices",
        lambda self, population, weights=None, *, cum_weights=None, k=1: [population[0]],
    )


def test_verified_1m_context_preempts_provider_priority_and_small_model_affinity(monkeypatch):
    _choose_first(monkeypatch)
    smaller = _candidate("smaller", 262_144, provider_name="preferred-provider", provider_priority=1)
    larger = _candidate("large-context", 1_000_000, provider_name="fallback-provider", provider_priority=100)

    selected = Registry(None, None).choose(
        [smaller, larger],
        _text_request(),
        affinity={"provider_name": smaller.provider_name, "base_url": smaller.base_url, "model_id": smaller.model_id},
    )

    assert selected is not None
    assert selected.model_id == "large-context"


def test_random_picker_is_limited_to_1m_plus_tier(monkeypatch):
    monkeypatch.setattr("app.registry.secrets.choice", lambda values: values[0])
    seen_model_pool: list[str] = []

    def choose_last(self, population, weights=None, *, cum_weights=None, k=1):
        if len(population) > 1:
            seen_model_pool.extend(item.model_id for item in population)
        return [population[-1]]

    monkeypatch.setattr("app.registry.secrets.SystemRandom.choices", choose_last)
    candidates = [
        _candidate("large-a", 1_000_000),
        _candidate("large-b", 2_000_000),
        _candidate("small", 999_999),
    ]

    selected = Registry(None, None).choose(candidates, _text_request())

    assert selected is not None
    assert selected.model_id == "large-b"
    assert seen_model_pool == ["large-a", "large-b"]


def test_fallback_uses_largest_known_context_when_no_1m_model_is_eligible(monkeypatch):
    _choose_first(monkeypatch)
    candidates = [
        _candidate("small", 32_768),
        _candidate("largest-fallback", 524_288),
        _candidate("medium", 262_144),
    ]

    selected = Registry(None, None).choose(candidates, _text_request())

    assert selected is not None
    assert selected.model_id == "largest-fallback"


def test_unknown_contexts_remain_eligible_only_when_no_known_context_exists(monkeypatch):
    _choose_first(monkeypatch)
    known = _candidate("known", 262_144)
    unknown = _candidate("unknown", None)
    selected = Registry(None, None).choose([unknown, known], _text_request())
    assert selected is not None
    assert selected.model_id == "known"

    selected_unknown = Registry(None, None).choose([unknown], _text_request())
    assert selected_unknown is not None
    assert selected_unknown.model_id == "unknown"


def test_context_metadata_recognizes_google_and_common_provider_field_names():
    assert _context_window({"inputTokenLimit": 1_048_576}) == 1_048_576
    assert _context_window({"max_seq_len": 262_144}) == 262_144


@pytest.mark.asyncio
async def test_candidate_must_support_every_requested_feature_before_context_priority(tmp_path):
    models = [
        {
            "id": "all-features-medium-context",
            "context_window": 524_288,
            "pricing": {"prompt": 0, "completion": 0},
            "capabilities": {
                "chat": True,
                "vision": True,
                "function_calling": True,
                "structured_output": True,
            },
        },
        {
            "id": "large-context-without-tools",
            "context_window": 2_000_000,
            "pricing": {"prompt": 0, "completion": 0},
            "capabilities": {
                "chat": True,
                "vision": True,
                "function_calling": False,
                "structured_output": True,
            },
        },
        {
            "id": "large-context-without-vision",
            "context_window": 1_500_000,
            "pricing": {"prompt": 0, "completion": 0},
            "capabilities": {
                "chat": True,
                "vision": False,
                "function_calling": True,
                "structured_output": True,
            },
        },
    ]
    config = config_data()
    async with make_env(tmp_path, ProviderMock(models=models), config) as env:
        key_response = await env["client"].post(
            "/api/keys", json={}, headers={"Origin": "http://testserver"}
        )
        assert key_response.status_code == 200
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key_response.json()['api_key']}"},
            json={
                "model": "hopexai",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Describe this image and call a tool."},
                    {"type": "image_url", "image_url": {"url": "https://images.example.test/image.png"}},
                ]}],
                "tools": [{"type": "function", "function": {
                    "name": "lookup", "parameters": {"type": "object", "properties": {}}
                }}],
                "response_format": {"type": "json_object"},
            },
        )
        assert response.status_code == 200, response.text
        posted = [call for call in env["mock"].calls if call["method"] == "POST"][-1]
        assert posted["body"]["model"] == "all-features-medium-context"


@pytest.mark.parametrize("part_type", ["input_audio", "video_url", "input_file"])
def test_unimplemented_modalities_are_rejected_instead_of_silently_routed(part_type):
    with pytest.raises(PublicAPIError) as exc:
        parse_chat_request(
            {
                "model": "hopexai",
                "messages": [{"role": "user", "content": [{"type": part_type}]}],
            },
            max_messages=20,
            max_tools=5,
        )
    assert exc.value.status_code == 501
    assert exc.value.code == "unsupported_capability"


@pytest.mark.parametrize("part_type", ["input_audio", "video_url", "input_file"])
def test_unimplemented_modalities_are_rejected_on_responses_api(part_type):
    with pytest.raises(PublicAPIError) as exc:
        parse_responses_request(
            {
                "model": "hopexai",
                "input": [{"role": "user", "content": [{"type": part_type}]}],
            },
            max_messages=20,
            max_tools=5,
        )
    assert exc.value.status_code == 501
    assert exc.value.code == "unsupported_capability"
