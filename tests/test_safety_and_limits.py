from __future__ import annotations

import asyncio
import copy

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select

from app.adapters import OpenAIChatAdapter, OpenAIResponsesAdapter
from app.config import EndpointOverrides, GatewayConfig, GatewayDefaults
from app.discovery import _pricing_status
from app.models import ConversationAffinity, ResponseState
from app.requests import CanonicalRequest, parse_chat_request, parse_responses_request
from app.router import GatewayRouter
from app.types import Candidate, CanonicalDelta
from app.security import URLGuard, UnsafeUpstreamURL
from tests.helpers import ProviderMock, config_data, make_env


async def issue_key(client):
    response = await client.post("/api/keys", json={}, headers={"Origin": "http://testserver"})
    assert response.status_code == 200, response.text
    return response.json()["api_key"]


def choose_first(monkeypatch):
    monkeypatch.setattr("app.registry.secrets.choice", lambda sequence: sequence[0])
    monkeypatch.setattr(
        "app.registry.secrets.SystemRandom.choices",
        lambda self, population, weights=None, *, cum_weights=None, k=1: [population[0]],
    )


def add_second_base(config, url="https://provider-two.mock"):
    first = config["providers"][0]["base_urls"][0]
    first["url"] = "https://provider-one.mock"
    first["selection_weight"] = 100
    second = copy.deepcopy(first)
    second["url"] = url
    second["selection_weight"] = 1
    config["providers"][0]["base_urls"].append(second)


@pytest.mark.asyncio
async def test_default_output_tokens_and_configured_limit_are_enforced(tmp_path):
    config = config_data(extra_defaults={
        "default_max_output_tokens": 2048,
        "max_output_tokens_limit": 4096,
    })
    async with make_env(tmp_path, config=config) as env:
        key = await issue_key(env["client"])
        headers = {"Authorization": f"Bearer {key}"}
        valid = await env["client"].post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": "hopexai", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert valid.status_code == 200, valid.text
        call = [call for call in env["mock"].calls if call["method"] == "POST"][-1]
        assert call["body"]["max_tokens"] == 2048

        before = len([call for call in env["mock"].calls if call["method"] == "POST"])
        too_large = await env["client"].post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": "hopexai", "max_tokens": 4097, "messages": [{"role": "user", "content": "hello"}]},
        )
        assert too_large.status_code == 400
        assert too_large.json()["error"]["code"] == "max_output_tokens_exceeded"
        assert len([call for call in env["mock"].calls if call["method"] == "POST"]) == before

        responses_too_large = await env["client"].post(
            "/v1/responses",
            headers=headers,
            json={"model": "hopexai", "max_output_tokens": 4097, "input": "hello"},
        )
        assert responses_too_large.status_code == 400
        assert responses_too_large.json()["error"]["code"] == "max_output_tokens_exceeded"
        assert len([call for call in env["mock"].calls if call["method"] == "POST"]) == before


def test_tool_choice_and_json_schema_cross_protocol_translation():
    raw_config = config_data()
    raw_config.pop("_test_model", None)
    provider_config = GatewayConfig.model_validate(raw_config).providers[0]
    base = provider_config.base_urls[0]
    candidate = Candidate(
        provider=provider_config,
        base=base,
        model_id="internal-model",
        api_format="openai_chat",
        endpoints={"chat_completions": ["/v1/chat/completions"], "responses": ["/v1/responses"]},
        capabilities={"chat": True, "function_calling": True, "structured_output": True},
        pricing="unknown",
        context_window=None,
        provider_priority=provider_config.priority,
        provider_weight=provider_config.selection_weight,
        model_weight=1.0,
    )
    tool = {"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {}}}
    response_request = parse_responses_request({
        "model": "hopexai",
        "input": "hello",
        "tools": [tool],
        "tool_choice": {"type": "function", "name": "lookup"},
        "text": {"format": {"type": "json_schema", "name": "answer", "schema": {"type": "object"}, "strict": True}},
    }, 20, 5)
    chat_spec = OpenAIChatAdapter().build(candidate, response_request, "response", False)
    assert chat_spec.body["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
    assert chat_spec.body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "answer", "strict": True, "schema": {"type": "object"}},
    }

    chat_request = parse_chat_request({
        "model": "hopexai",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
    }, 20, 5)
    responses_spec = OpenAIResponsesAdapter().build(candidate, chat_request, "chat.completion", False)
    assert responses_spec.body["tool_choice"] == {"type": "function", "name": "lookup"}
    assert responses_spec.body["tools"][0]["name"] == "lookup"


def test_default_output_tokens_cannot_exceed_the_hard_limit():
    with pytest.raises(ValidationError):
        GatewayDefaults.model_validate({"default_max_output_tokens": 2048, "max_output_tokens_limit": 1024})


def test_endpoint_overrides_reject_url_paths_and_traversal():
    for path in ("//evil.example/chat", "https://evil.example/chat", "/v1/../admin", "/v1%2f..%2fadmin"):
        with pytest.raises(ValidationError):
            EndpointOverrides.model_validate({"chat_completions": path})


@pytest.mark.asyncio
async def test_private_http_requires_opt_in_and_public_http_is_still_rejected():
    guard = URLGuard(resolver=lambda host, port: ["10.0.0.2"])
    with pytest.raises(UnsafeUpstreamURL):
        await guard.validate("http://internal.example")
    await guard.validate("http://internal.example", allow_private_network=True)
    with pytest.raises(UnsafeUpstreamURL):
        await guard.validate("http://8.8.8.8", allow_private_network=True)


def test_partial_pricing_metadata_is_not_classified_as_verified_free():
    assert _pricing_status({"pricing": {"prompt": 0}}) == "unknown"
    assert _pricing_status({"free": True, "pricing": {"input": 0}}) == "unknown"
    assert _pricing_status({"pricing": {"prompt": 0, "completion": 0}}) == "free"
    assert _pricing_status({"free": True, "pricing": {"prompt": 0, "completion": 0.01}}) == "paid"


@pytest.mark.asyncio
async def test_manual_model_can_supply_verified_capability_and_free_pricing(tmp_path):
    model_id = "documented-free-vision-model"
    config = config_data(
        free_only=True,
        model_id=model_id,
        pricing={"prompt": 0.05, "completion": 0.05},
        model_capabilities={"chat": True, "streaming": True},
    )
    config["providers"][0]["base_urls"][0]["manual_models"] = [{
        "id": model_id,
        "enabled": True,
        "pricing": "free",
        "context_window": 8192,
        "capabilities": {
            "chat": True,
            "streaming": True,
            "vision": True,
        },
    }]
    async with make_env(tmp_path, config=config) as env:
        key = await issue_key(env["client"])
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "hopexai",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Describe this."},
                    {"type": "image_url", "image_url": {"url": "https://images.example.test/sample.png"}},
                ]}],
            },
        )
        assert response.status_code == 200, response.text
        post = [call for call in env["mock"].calls if call["method"] == "POST"][-1]
        assert post["body"]["model"] == model_id


@pytest.mark.asyncio
async def test_known_context_window_is_enforced_without_silent_truncation(tmp_path):
    config = config_data()
    config["_test_model"]["context_window"] = 4096
    async with make_env(tmp_path, config=config) as env:
        key = await issue_key(env["client"])
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "hopexai", "messages": [{"role": "user", "content": "x" * 12_000}]},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "context_length_exceeded"
        assert not [call for call in env["mock"].calls if call["method"] == "POST"]


@pytest.mark.asyncio
async def test_authenticated_endpoint_probe_uses_invalid_json_without_credentials(tmp_path):
    provider = ProviderMock()
    provider.protected_chat_route = True
    async with make_env(tmp_path, provider) as env:
        key = await issue_key(env["client"])
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "hopexai", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.status_code == 200, response.text
        assert provider.discovery_probe_bodies
        assert provider.discovery_probe_bodies[0]["model"] == "__hopexai_discovery_probe__"
        assert provider.discovery_probe_bodies[0]["messages"] == []
        assert provider.discovery_probe_auth == [""]


@pytest.mark.asyncio
async def test_manual_model_can_be_added_when_catalog_is_empty(tmp_path):
    config = config_data(free_only=True, model_id="manual-only-model")
    config["providers"][0]["base_urls"][0]["manual_models"] = [{
        "id": "manual-only-model",
        "pricing": "free",
        "context_window": 8192,
        "capabilities": {"chat": True, "streaming": True},
    }]
    async with make_env(tmp_path, ProviderMock(models=[]), config) as env:
        key = await issue_key(env["client"])
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "hopexai", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.status_code == 200, response.text
        post = [call for call in env["mock"].calls if call["method"] == "POST"][-1]
        assert post["body"]["model"] == "manual-only-model"


@pytest.mark.asyncio
async def test_config_can_disable_per_key_rate_limit_rotation(tmp_path, monkeypatch):
    mock = ProviderMock()
    mock.key_failures["first-upstream-key"] = (
        429,
        {"error": {"message": "API key rate limit exceeded"}},
        {},
    )
    config = config_data(keys=["first-upstream-key", "second-upstream-key"])
    config["providers"][0]["base_urls"][0]["key_rotation"] = {"rotate_on_rate_limit": False}
    async with make_env(tmp_path, mock, config) as env:
        monkeypatch.setattr("app.registry.secrets.choice", lambda sequence: sequence[0])
        key = await issue_key(env["client"])
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "hopexai", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.status_code == 503
        calls = [call for call in mock.calls if call["method"] == "POST"]
        assert [call["auth"] for call in calls] == ["Bearer first-upstream-key"]


@pytest.mark.asyncio
async def test_ambiguous_429_cools_base_instead_of_rotating_api_keys(tmp_path, monkeypatch):
    mock = ProviderMock()
    mock.key_failures["first-upstream-key"] = (
        429,
        {"error": {"message": "rate limit exceeded"}},
        {},
    )
    config = config_data(keys=["first-upstream-key", "second-upstream-key"])
    async with make_env(tmp_path, mock, config) as env:
        monkeypatch.setattr("app.registry.secrets.choice", lambda sequence: sequence[0])
        key = await issue_key(env["client"])
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "hopexai", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.status_code == 503
        upstream_calls = [call for call in mock.calls if call["method"] == "POST"]
        assert len(upstream_calls) == 1
        assert upstream_calls[0]["auth"] == "Bearer first-upstream-key"


@pytest.mark.asyncio
async def test_ambiguous_5xx_is_not_replayed_without_idempotency_opt_in(tmp_path, monkeypatch):
    config = config_data()
    add_second_base(config)
    provider = ProviderMock()
    provider.host_failures["provider-one.mock"] = (500, {"error": {"message": "temporary"}}, {})
    async with make_env(tmp_path, provider, config) as env:
        choose_first(monkeypatch)
        key = await issue_key(env["client"])
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "hopexai", "messages": [{"role": "user", "content": "chargeable request"}]},
        )
        assert response.status_code == 503
        calls = [call for call in provider.calls if call["method"] == "POST"]
        assert [call["host"] for call in calls] == ["provider-one.mock"]


@pytest.mark.asyncio
async def test_ambiguous_5xx_retries_only_with_explicit_provider_idempotency(tmp_path, monkeypatch):
    config = config_data(extra_defaults={"retry_ambiguous_failures": True})
    add_second_base(config)
    for base in config["providers"][0]["base_urls"]:
        base["supports_idempotency"] = True
    provider = ProviderMock()
    provider.host_failures["provider-one.mock"] = (500, {"error": {"message": "temporary"}}, {})
    async with make_env(tmp_path, provider, config) as env:
        choose_first(monkeypatch)
        key = await issue_key(env["client"])
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "hopexai", "messages": [{"role": "user", "content": "idempotent request"}]},
        )
        assert response.status_code == 200, response.text
        calls = [call for call in provider.calls if call["method"] == "POST"]
        assert [call["host"] for call in calls] == ["provider-one.mock", "provider-two.mock"]
        assert calls[0]["idempotency_key"]
        assert calls[0]["idempotency_key"] == calls[1]["idempotency_key"]


@pytest.mark.asyncio
async def test_reasoning_enable_and_disable_are_translated(tmp_path):
    async with make_env(tmp_path) as env:
        key = await issue_key(env["client"])
        headers = {"Authorization": f"Bearer {key}"}
        for reasoning_enabled, expected in ((True, "medium"), (False, "none")):
            response = await env["client"].post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": "hopexai",
                    "reasoning_enabled": reasoning_enabled,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert response.status_code == 200, response.text
            call = [call for call in env["mock"].calls if call["method"] == "POST"][-1]
            assert call["body"]["reasoning_effort"] == expected


@pytest.mark.asyncio
async def test_stream_usage_is_emitted_once(tmp_path):
    mock = ProviderMock()
    mock.streaming = True
    async with make_env(tmp_path, mock) as env:
        key = await issue_key(env["client"])
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "hopexai", "stream": True, "messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.status_code == 200
        assert response.text.count('"usage"') == 1
        assert '"prompt_tokens":4' in response.text
        assert '"completion_tokens":3' in response.text


@pytest.mark.asyncio
async def test_stream_does_not_fallback_after_public_output(tmp_path, monkeypatch):
    config = config_data()
    add_second_base(config)
    provider = ProviderMock()
    provider.stream_fail_after_first = True
    async with make_env(tmp_path, provider, config) as env:
        choose_first(monkeypatch)
        key = await issue_key(env["client"])
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "hopexai", "stream": True, "messages": [{"role": "user", "content": "stream"}]},
        )
        assert response.status_code == 200
        assert "first output" in response.text
        assert '"error"' in response.text
        calls = [call for call in provider.calls if call["method"] == "POST"]
        assert [call["host"] for call in calls] == ["provider-one.mock"]


@pytest.mark.asyncio
async def test_revoking_key_deletes_encrypted_response_state(tmp_path):
    async with make_env(tmp_path) as env:
        key = await issue_key(env["client"])
        headers = {"Authorization": f"Bearer {key}"}
        response = await env["client"].post(
            "/v1/responses", headers=headers,
            json={"model": "hopexai", "input": "private conversation"},
        )
        assert response.status_code == 200, response.text
        response_id = response.json()["id"]
        key_record = await env["app"].state.key_service.authenticate(key)
        async with env["app"].state.sessions() as session:
            response_count = await session.scalar(select(func.count()).select_from(ResponseState))
            affinity_count = await session.scalar(select(func.count()).select_from(ConversationAffinity))
            assert response_count == 1
            assert affinity_count == 1
        assert await env["app"].state.key_service.revoke(key_record.id)
        async with env["app"].state.sessions() as session:
            response_count = await session.scalar(select(func.count()).select_from(ResponseState))
            affinity_count = await session.scalar(select(func.count()).select_from(ConversationAffinity))
            assert response_count == 0
            assert affinity_count == 0
        late_save = await env["app"].state.runtime_state.save_response_messages(
            "resp_late", key_record.id, [], 3600, None, None, None,
        )
        assert late_save is False
        await env["app"].state.runtime_state.save_affinity(
            env["app"].state.runtime_state.affinity_digest(key_record.id, "late"),
            key_record.id, "test-provider", "https://provider.mock", "model", 3600,
        )
        async with env["app"].state.sessions() as session:
            affinity_count = await session.scalar(select(func.count()).select_from(ConversationAffinity))
            assert affinity_count == 0
        continuation = await env["client"].post(
            "/v1/responses", headers=headers,
            json={"model": "hopexai", "previous_response_id": response_id, "input": "continue"},
        )
        assert continuation.status_code == 401


@pytest.mark.asyncio
async def test_stream_iterator_closes_upstream_on_cancellation():
    raw_config = config_data()
    raw_config.pop("_test_model", None)
    config = GatewayConfig.model_validate(raw_config)
    provider = config.providers[0]
    base = provider.base_urls[0]
    candidate = Candidate(
        provider=provider,
        base=base,
        model_id="model",
        api_format="openai_chat",
        endpoints={"chat_completions": ["/v1/chat/completions"]},
        capabilities={"chat": True},
        pricing="unknown",
        context_window=None,
        provider_priority=provider.priority,
        provider_weight=provider.selection_weight,
        model_weight=1.0,
    )
    canonical: CanonicalRequest = parse_chat_request(
        {"model": "hopexai", "stream": True, "messages": [{"role": "user", "content": "cancel"}]}, 20, 5,
    )

    class FakeSession:
        closed = False

        async def deltas(self):
            yield CanonicalDelta(content="first")
            await asyncio.Event().wait()

        async def close(self):
            self.closed = True

    class FakeRegistry:
        async def candidates(self, config, request):
            return [candidate]

        def choose(self, candidates, request, **kwargs):
            return candidate

        def endpoint_kind(self, candidate, public_kind):
            return "chat_completions"

    class FakeHealth:
        async def record_success(self, *args):
            return None

        async def record_failure(self, *args, **kwargs):
            return None

    class FakeState:
        async def load_affinity(self, key):
            return None

    class FakeUpstream:
        session = FakeSession()

        async def begin_stream(self, *args):
            return self.session

    upstream = FakeUpstream()
    router = GatewayRouter(FakeRegistry(), upstream, FakeHealth(), FakeState())
    iterator = router.stream(config, canonical, "request-id", "key-id", None)
    candidate_result, delta = await anext(iterator)
    assert candidate_result is candidate
    assert delta.content == "first"
    await iterator.aclose()
    assert upstream.session.closed is True
