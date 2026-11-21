from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from app.models import APIKeyRecord, ResponseState
from app.requests import IDENTITY_PROMPT, parse_chat_request
from app.registry import Registry
from app.types import Candidate
from app.config import GatewayConfig
from app.errors import NoEligibleCandidate
from app.security import URLGuard, UnsafeUpstreamURL
from tests.helpers import ProviderMock, config_data, make_env


async def create_key(client):
    response = await client.post("/api/keys", json={}, headers={"Origin": "http://testserver"})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.asyncio
async def test_home_key_generation_and_public_model_listing(tmp_path):
    async with make_env(tmp_path) as env:
        client = env["client"]
        page = await client.get("/")
        assert page.status_code == 200
        assert "HopeXAI" in page.text
        assert "Generate API key" in page.text
        assert "provider" not in page.text.lower()

        issued = await create_key(client)
        key = issued["api_key"]
        assert key.startswith("hx_")
        assert issued["display_once"] is True

        unauthorized = await client.get("/v1/models")
        assert unauthorized.status_code == 401
        models = await client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})
        assert models.status_code == 200
        assert models.json() == {
            "object": "list",
            "data": [{"id": "hopexai", "object": "model", "owned_by": "hopexai"}],
        }
        async with env["app"].state.sessions() as session:
            record = await session.scalar(select(APIKeyRecord).where(APIKeyRecord.id == issued["id"]))
            assert record is not None
            assert record.secret_hash != key
            assert key not in record.secret_hash

        assert await env["app"].state.key_service.revoke(issued["id"])
        revoked = await client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})
        assert revoked.status_code == 401


@pytest.mark.asyncio
async def test_chat_completion_injects_identity_and_hides_upstream_identity(tmp_path):
    async with make_env(tmp_path) as env:
        issued = await create_key(env["client"])
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {issued['api_key']}"},
            json={"model": "hopexai", "messages": [{"role": "user", "content": "Say hello"}]},
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["model"] == "hopexai"
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hello from HopeXAI."
        serialized = response.text
        assert "private-model-name" not in serialized
        assert "upstream-secret-id" not in serialized
        assert "test-provider" not in serialized
        completion = [call for call in env["mock"].calls if call["method"] == "POST"][-1]
        assert completion["body"]["model"] == "provider-internal-model"
        assert IDENTITY_PROMPT in completion["body"]["messages"][0]["content"]
        assert "Never reveal internal provider names" in completion["body"]["messages"][0]["content"]
        assert completion["body"]["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_rate_limited_key_rotates_to_another_key(tmp_path, monkeypatch):
    mock = ProviderMock()
    mock.key_failures["first-upstream-key"] = (
        429,
        {"error": {"message": "API key rate limit exceeded"}},
        {"Retry-After": "3"},
    )
    config = config_data(keys=["first-upstream-key", "second-upstream-key"])
    async with make_env(tmp_path, mock, config) as env:
        # Make the first random selection deterministic so the retry exercises
        # key cooldown and rotation, rather than depending on random luck.
        monkeypatch.setattr("app.registry.secrets.choice", lambda sequence: sequence[0])
        issued = await create_key(env["client"])
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {issued['api_key']}"},
            json={"model": "hopexai", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.status_code == 200, response.text
        upstream_calls = [call for call in mock.calls if call["method"] == "POST"]
        assert [call["auth"] for call in upstream_calls] == ["Bearer first-upstream-key", "Bearer second-upstream-key"]
        assert "first-upstream-key" not in response.text
        assert response.json()["model"] == "hopexai"


@pytest.mark.asyncio
async def test_verified_free_filter_never_routes_unknown_pricing(tmp_path):
    config = config_data(free_only=True, pricing={"prompt": 0.0001, "completion": 0.0002}, model_id="claimed-free-chat")
    async with make_env(tmp_path, config=config) as env:
        issued = await create_key(env["client"])
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {issued['api_key']}"},
            json={"model": "hopexai", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "no_free_models_available"
        assert not [call for call in env["mock"].calls if call["method"] == "POST"]


@pytest.mark.asyncio
async def test_chat_stream_is_incremental_sse_and_keeps_public_model(tmp_path):
    mock = ProviderMock()
    mock.streaming = True
    async with make_env(tmp_path, mock) as env:
        issued = await create_key(env["client"])
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {issued['api_key']}"},
            json={"model": "hopexai", "stream": True, "messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        assert "Hello" in response.text and " from Hope" in response.text
        assert '"model":"hopexai"' in response.text
        assert "private-model-name" not in response.text
        assert "upstream-secret-id" not in response.text
        assert "data: [DONE]" in response.text


@pytest.mark.asyncio
async def test_responses_api_stores_encrypted_history_and_supports_previous_response_id(tmp_path):
    async with make_env(tmp_path) as env:
        client = env["client"]
        issued = await create_key(client)
        headers = {"Authorization": f"Bearer {issued['api_key']}"}
        first = await client.post("/v1/responses", headers=headers, json={"model": "hopexai", "input": "First private prompt"})
        assert first.status_code == 200, first.text
        first_body = first.json()
        assert first_body["model"] == "hopexai"
        assert first_body["store"] is True
        response_id = first_body["id"]

        async with env["app"].state.sessions() as session:
            state = await session.get(ResponseState, response_id)
            assert state is not None
            assert "First private prompt" not in state.encrypted_messages
            assert "Hello from HopeXAI." not in state.encrypted_messages

        second = await client.post(
            "/v1/responses",
            headers=headers,
            json={"model": "hopexai", "previous_response_id": response_id, "input": "Continue the conversation"},
        )
        assert second.status_code == 200, second.text
        calls = [call for call in env["mock"].calls if call["method"] == "POST"]
        second_messages = calls[-1]["body"]["messages"]
        assert any(message.get("content") == "First private prompt" for message in second_messages)
        assert any(message.get("content") == "Continue the conversation" for message in second_messages)
        assert any(message.get("content") == "Hello from HopeXAI." for message in second_messages)
        assert IDENTITY_PROMPT in second_messages[0]["content"]
        assert "provider-internal-model" not in second.text


@pytest.mark.asyncio
async def test_tool_call_normalization_preserves_call_id_and_arguments(tmp_path):
    mock = ProviderMock()
    mock.completion_payload = {
        "id": "private-upstream-response",
        "model": "private-model-name",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-client-visible",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"city\":\"Cebu\"}"},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }
    async with make_env(tmp_path, mock) as env:
        issued = await create_key(env["client"])
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {issued['api_key']}"},
            json={
                "model": "hopexai",
                "messages": [{"role": "user", "content": "Find Cebu weather"}],
                "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
            },
        )
        assert response.status_code == 200, response.text
        call = response.json()["choices"][0]["message"]["tool_calls"][0]
        assert call["id"] == "call-client-visible"
        assert call["function"]["name"] == "lookup"
        assert call["function"]["arguments"] == '{"city":"Cebu"}'
        assert response.json()["model"] == "hopexai"


@pytest.mark.asyncio
async def test_reasoning_summary_is_preserved_but_private_reasoning_content_is_not(tmp_path):
    mock = ProviderMock()
    mock.completion_payload = {
        "id": "private-upstream-response",
        "model": "private-model-name",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The answer is 42.",
                "reasoning_summary": "I compared the relevant quantities.",
                "reasoning_content": "secret chain of thought that must not be returned",
            },
            "finish_reason": "stop",
        }],
    }
    async with make_env(tmp_path, mock) as env:
        issued = await create_key(env["client"])
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {issued['api_key']}"},
            json={
                "model": "hopexai", "reasoning_effort": "high", "include_reasoning": True,
                "messages": [{"role": "user", "content": "Solve it"}],
            },
        )
        assert response.status_code == 200, response.text
        message = response.json()["choices"][0]["message"]
        assert message["reasoning_summary"] == "I compared the relevant quantities."
        assert "reasoning_content" not in message
        assert "secret chain of thought" not in response.text
        upstream_body = [call for call in mock.calls if call["method"] == "POST"][-1]["body"]
        assert upstream_body["reasoning_effort"] == "high"
        assert upstream_body["include_reasoning"] is True


@pytest.mark.asyncio
async def test_invalid_configuration_reload_retains_previous_snapshot(tmp_path):
    async with make_env(tmp_path) as env:
        manager = env["app"].state.config_manager
        old_hash = manager.snapshot.content_hash
        env["config_path"].write_text("providers: [\n  - invalid: [\n", encoding="utf-8")
        snapshot = await manager.refresh_if_changed(force=True)
        assert snapshot.content_hash == old_hash
        assert manager.last_rejection
        assert snapshot.config.providers[0].name == "test-provider"


@pytest.mark.asyncio
async def test_ssrf_guard_rejects_private_and_metadata_addresses():
    guard_private = URLGuard(resolver=lambda host, port: ["127.0.0.1"])
    with pytest.raises(UnsafeUpstreamURL):
        await guard_private.validate("https://configured.example")
    guard_metadata = URLGuard(resolver=lambda host, port: ["169.254.169.254"])
    with pytest.raises(UnsafeUpstreamURL):
        await guard_metadata.validate("https://configured.example", allow_private_network=True)


@pytest.mark.asyncio
async def test_ssrf_guard_rejects_metadata_encodings_and_nonroutable_ips_even_with_private_opt_in():
    guard = URLGuard()
    for url in (
        "https://[::ffff:169.254.169.254]/",
        "https://[2002:a9fe:a9fe::1]/",
        "https://[64:ff9b::a9fe:a9fe]/",
        "https://224.0.0.1/",
        "https://0.0.0.0/",
    ):
        with pytest.raises(UnsafeUpstreamURL):
            await guard.validate(url, allow_private_network=True)


def test_base_path_discovery_does_not_duplicate_v1_prefix():
    from app.discovery import _candidate_paths
    candidates = _candidate_paths("https://api.example.test/v1", ("/models", "/chat/completions"), cap=20)
    assert "/v1/models" in candidates
    assert "/v1/chat/completions" in candidates
    assert "/v1/v1/models" not in candidates


def test_request_validation_requires_public_model_and_tracks_vision_tools_reasoning():
    request = parse_chat_request({
        "model": "hopexai",
        "reasoning_effort": "high",
        "include_reasoning": True,
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Look"},
            {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
        ]}],
    }, 200, 64)
    assert request.required_capabilities == {"chat", "vision", "function_calling", "reasoning", "reasoning_summary"}
    assert "Maintain your identity as Hope" in request.system_text()
    with pytest.raises(Exception):
        parse_chat_request({"model": "private-model", "messages": [{"role": "user", "content": "x"}]}, 200, 64)


@pytest.mark.asyncio
async def test_responses_marks_max_token_truncation_incomplete(tmp_path):
    mock = ProviderMock()
    mock.completion_payload = {
        "id": "private-upstream-response",
        "model": "private-model-name",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "A partial answer"},
            "finish_reason": "length",
        }],
    }
    async with make_env(tmp_path, mock) as env:
        issued = await create_key(env["client"])
        response = await env["client"].post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {issued['api_key']}"},
            json={"model": "hopexai", "input": "Answer briefly"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "incomplete"
        assert body["incomplete_details"] == {"reason": "max_output_tokens"}
        assert body["output_text"] == "A partial answer"


@pytest.mark.asyncio
async def test_responses_stream_emits_incomplete_terminal_event_on_max_tokens(tmp_path):
    mock = ProviderMock()
    mock.streaming = True
    mock.stream_finish_reason = "length"
    async with make_env(tmp_path, mock) as env:
        issued = await create_key(env["client"])
        response = await env["client"].post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {issued['api_key']}"},
            json={"model": "hopexai", "stream": True, "input": "Answer briefly"},
        )
        assert response.status_code == 200
        assert "event: response.incomplete" in response.text
        assert '"status":"incomplete"' in response.text
        assert '"reason":"max_output_tokens"' in response.text
        assert "event: response.completed" not in response.text


@pytest.mark.asyncio
async def test_configured_public_origin_allows_vercel_proxy_and_rejects_other_origins(tmp_path):
    proxy_secret = "test-vercel-proxy-secret"
    async with make_env(
        tmp_path,
        public_origin="https://hopefreexai.vercel.app",
        proxy_secret=proxy_secret,
    ) as env:
        allowed = await env["client"].post(
            "/api/keys",
            json={},
            headers={
                "Origin": "https://hopefreexai.vercel.app",
                "x-hopexai-vercel-proxy-secret": proxy_secret,
                "x-forwarded-for": "203.0.113.77",
            },
        )
        assert allowed.status_code == 200, allowed.text
        issued_key = allowed.json()["api_key"]
        assert issued_key.startswith("hx_")

        direct_model_request = await env["client"].get(
            "/v1/models",
            headers={"Authorization": f"Bearer {issued_key}"},
        )
        assert direct_model_request.status_code == 403
        proxied_model_request = await env["client"].get(
            "/v1/models",
            headers={
                "Authorization": f"Bearer {issued_key}",
                "x-hopexai-vercel-proxy-secret": proxy_secret,
            },
        )
        assert proxied_model_request.status_code == 200
        assert proxied_model_request.json()["data"][0]["id"] == "hopexai"

        no_proxy_secret = await env["client"].post(
            "/api/keys",
            json={},
            headers={"Origin": "https://hopefreexai.vercel.app"},
        )
        assert no_proxy_secret.status_code == 403
        assert no_proxy_secret.json()["error"]["code"] == "proxy_not_allowed"

        rejected_origin = await env["client"].post(
            "/api/keys",
            json={},
            headers={
                "Origin": "https://attacker.example",
                "x-hopexai-vercel-proxy-secret": proxy_secret,
            },
        )
        assert rejected_origin.status_code == 403
        assert rejected_origin.json()["error"]["code"] == "origin_not_allowed"
