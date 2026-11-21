from __future__ import annotations

import json

import httpx
import pytest

from app.upstream import _extract_output_token_limit, _output_parameter_error
from tests.helpers import ProviderMock, config_data, make_env


class OutputLimitProviderMock(ProviderMock):
    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit
        self.error_message = "max_tokens is too high for this model"
        self.requested_limits: list[int] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions" and request.method == "POST":
            body = json.loads(request.content or b"{}")
            requested = body.get("max_tokens")
            if isinstance(requested, int) and not isinstance(requested, bool):
                self.requested_limits.append(requested)
                if requested > self.limit:
                    return httpx.Response(
                        400,
                        json={"error": {
                            "message": self.error_message,
                            "type": "invalid_request_error",
                            "param": "max_tokens",
                        }},
                    )
        return await super().__call__(request)


class AlternateParameterProviderMock(ProviderMock):
    def __init__(self):
        super().__init__()
        self.output_parameters: list[str] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions" and request.method == "POST":
            body = json.loads(request.content or b"{}")
            parameter = "max_completion_tokens" if "max_completion_tokens" in body else "max_tokens"
            self.output_parameters.append(parameter)
            if parameter == "max_tokens":
                return httpx.Response(400, json={"error": {
                    "message": "unsupported_parameter: max_tokens is not supported with this model; use max_completion_tokens instead",
                    "type": "invalid_request_error",
                    "param": "max_tokens",
                }})
        return await super().__call__(request)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("max_tokens must be at most 2048", 2048),
        ("requested max_completion_tokens is too large; provider maximum is 8192 tokens", 8192),
        ("max_output_tokens must not exceed 1,024", 1024),
        ("context_length_exceeded: maximum context length is 4096 tokens", None),
    ],
)
def test_provider_output_limit_extraction(message, expected):
    assert _extract_output_token_limit(message.lower()) == expected


def test_unsupported_output_parameter_selects_an_alternate_field():
    unsupported, parameter = _output_parameter_error(
        "unsupported_parameter: max_tokens is not supported with this model; use max_completion_tokens instead"
    )
    assert unsupported is True
    assert parameter == "max_completion_tokens"
    unsupported, parameter = _output_parameter_error(
        "unsupported_parameter: max_output_tokens; use max_tokens instead"
    )
    assert unsupported is True
    assert parameter == "max_tokens"


@pytest.mark.asyncio
async def test_completion_retries_at_progressively_lower_output_limits(tmp_path):
    provider = OutputLimitProviderMock(limit=1024)
    config = config_data(
        api_format="openai_chat",
        extra_defaults={"default_max_output_tokens": 4096},
    )
    async with make_env(tmp_path, provider, config) as env:
        issued = await env["client"].post("/api/keys", json={})
        assert issued.status_code == 200, issued.text
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {issued.json()['api_key']}"},
            json={"model": "hopexai", "messages": [{"role": "user", "content": "Say hello."}]},
        )

    assert response.status_code == 200, response.text
    assert provider.requested_limits == [4096, 2048, 1024]
    assert response.json()["choices"][0]["message"]["content"] == "Hello from HopeXAI."


@pytest.mark.asyncio
async def test_provider_context_rejection_retries_a_lower_output_limit(tmp_path):
    provider = OutputLimitProviderMock(limit=1024)
    provider.error_message = "context_length_exceeded: maximum context length is 4096 tokens"
    config = config_data(
        api_format="openai_chat",
        extra_defaults={"default_max_output_tokens": 4096},
    )
    async with make_env(tmp_path, provider, config) as env:
        issued = await env["client"].post("/api/keys", json={})
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {issued.json()['api_key']}"},
            json={"model": "hopexai", "messages": [{"role": "user", "content": "Say hello."}]},
        )

    assert response.status_code == 200, response.text
    assert provider.requested_limits == [4096, 2048, 1024]


@pytest.mark.asyncio
async def test_stream_setup_retries_output_limit_only_before_any_output(tmp_path):
    provider = OutputLimitProviderMock(limit=1024)
    config = config_data(
        api_format="openai_chat",
        extra_defaults={"default_max_output_tokens": 4096},
    )
    async with make_env(tmp_path, provider, config) as env:
        issued = await env["client"].post("/api/keys", json={})
        assert issued.status_code == 200, issued.text
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {issued.json()['api_key']}"},
            json={
                "model": "hopexai",
                "messages": [{"role": "user", "content": "Say hello."}],
                "stream": True,
            },
        )

    assert response.status_code == 200, response.text
    assert provider.requested_limits == [4096, 2048, 1024]
    assert "Hello" in response.text
    assert " from Hope" in response.text
    assert "data: [DONE]" in response.text


@pytest.mark.asyncio
async def test_known_small_context_window_reduces_budget_instead_of_skipping_model(tmp_path):
    provider = ProviderMock()
    config = config_data(
        api_format="openai_chat",
        extra_defaults={"default_max_output_tokens": 4096},
    )
    config["_test_model"]["context_window"] = 4096
    provider.models = [config["_test_model"]]
    async with make_env(tmp_path, provider, config) as env:
        issued = await env["client"].post("/api/keys", json={})
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {issued.json()['api_key']}"},
            json={"model": "hopexai", "messages": [{"role": "user", "content": "Say hello."}]},
        )

    assert response.status_code == 200, response.text
    upstream_requests = [call for call in provider.calls if call.get("path") == "/v1/chat/completions" and call.get("method") == "POST"]
    assert upstream_requests
    assert upstream_requests[-1]["body"]["max_tokens"] < 4096


@pytest.mark.asyncio
async def test_provider_unsupported_max_token_name_is_switched_before_retry(tmp_path):
    provider = AlternateParameterProviderMock()
    config = config_data(
        api_format="openai_chat",
        extra_defaults={"default_max_output_tokens": 4096},
    )
    async with make_env(tmp_path, provider, config) as env:
        issued = await env["client"].post("/api/keys", json={})
        assert issued.status_code == 200, issued.text
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {issued.json()['api_key']}"},
            json={"model": "hopexai", "messages": [{"role": "user", "content": "Say hello."}]},
        )

    assert response.status_code == 200, response.text
    assert provider.output_parameters == ["max_tokens", "max_completion_tokens"]
