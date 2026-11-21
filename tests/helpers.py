from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.settings import Settings


DEFAULT_MODELS = [
    {
        "id": "provider-internal-model",
        "object": "model",
        "pricing": {"prompt": 0, "completion": 0},
        "context_window": 8192,
        "capabilities": {
            "chat": True,
            "streaming": True,
            "vision": True,
            "function_calling": True,
            "structured_output": True,
            "reasoning": True,
            "reasoning_summary": True,
        },
    }
]


def config_data(
    *,
    keys: list[str] | None = None,
    free_only: bool = False,
    pricing: dict[str, Any] | None = None,
    model_capabilities: dict[str, Any] | None = None,
    model_id: str = "provider-internal-model",
    provider_name: str = "test-provider",
    base_url: str = "https://provider.mock",
    api_format: str = "auto",
    extra_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model = {
        "id": model_id,
        "object": "model",
        "pricing": pricing or {"prompt": 0, "completion": 0},
        "context_window": 8192,
        "capabilities": model_capabilities or {
            "chat": True, "streaming": True, "vision": True,
            "function_calling": True, "structured_output": True,
            "reasoning": True, "reasoning_summary": True,
        },
    }
    defaults = {"discovery_ttl_seconds": 300, "max_attempts": 5}
    if extra_defaults:
        defaults.update(extra_defaults)
    return {
        "schema_version": 1,
        "defaults": {**defaults, "free_only": free_only},
        "providers": [{
            "name": provider_name,
            "enabled": True,
            "priority": 10,
            "selection_weight": 1,
            "api_format": api_format,
            "auth": {"mode": "bearer"},
            "reasoning": {"effort_parameter": "reasoning_effort", "summary_parameter": "include_reasoning"},
            "base_urls": [{
                "url": base_url,
                "enabled": True,
                "api_keys": keys or ["good-upstream-key"],
                "model_discovery": True,
                "manual_models": [],
            }],
        }],
        "_test_model": model,
    }


def yaml_config(data: dict[str, Any]) -> str:
    copied = dict(data)
    copied.pop("_test_model", None)
    return yaml.safe_dump(copied, sort_keys=False)


class BrokenSSEStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield (
            b'data: {"id":"upstream-secret-id","object":"chat.completion.chunk",'
            b'"model":"private-model-name","choices":[{"index":0,'
            b'"delta":{"content":"first output"},"finish_reason":null}]}\n\n'
        )
        await asyncio.sleep(0)
        raise httpx.ReadTimeout("simulated stream read timeout")

    async def aclose(self) -> None:
        return None


class ProviderMock:
    def __init__(self, models: list[dict[str, Any]] | None = None):
        self.models = DEFAULT_MODELS if models is None else models
        self.calls: list[dict[str, Any]] = []
        self.key_failures: dict[str, tuple[int, dict[str, Any], dict[str, str]]] = {}
        self.host_failures: dict[str, tuple[int, dict[str, Any], dict[str, str]]] = {}
        self.completion_payload: dict[str, Any] | None = None
        self.streaming = False
        self.stream_finish_reason = "stop"
        self.stream_fail_after_first = False
        self.protected_chat_route = False
        self.discovery_probe_bodies: list[dict[str, Any]] = []
        self.discovery_probe_auth: list[str] = []
        self.return_500 = False
        self.delay = 0.0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method.upper()
        auth = request.headers.get("authorization", "")
        key = auth.removeprefix("Bearer ")
        if path == "/v1/models" and method == "GET":
            self.calls.append({"method": method, "path": path, "auth": auth})
            if not auth:
                return httpx.Response(401, json={"error": {"message": "authentication required"}})
            return httpx.Response(200, json={"object": "list", "data": self.models})
        if path == "/v1/chat/completions" and method == "OPTIONS":
            self.calls.append({"method": method, "path": path, "auth": auth})
            if self.protected_chat_route:
                return httpx.Response(401, json={"error": {"message": "authentication required"}})
            return httpx.Response(204, headers={"Allow": "POST, OPTIONS"})
        if path == "/v1/responses" and method == "OPTIONS":
            return httpx.Response(204, headers={"Allow": "POST, OPTIONS"})
        if path == "/v1/messages" and method == "OPTIONS":
            return httpx.Response(204, headers={"Allow": "POST, OPTIONS"})
        if path == "/v1/chat/completions" and method == "POST":
            try:
                body = json.loads(request.content or b"{}")
            except ValueError:
                body = {}
            if self.protected_chat_route and not auth:
                self.discovery_probe_bodies.append(body)
                self.discovery_probe_auth.append(auth)
                return httpx.Response(400, json={"error": {"message": "invalid discovery probe"}})
            self.calls.append({
                "method": method, "path": path, "host": request.url.host, "auth": auth, "body": body,
                "idempotency_key": request.headers.get("Idempotency-Key"),
            })
            if request.url.host in self.host_failures:
                status, payload, headers = self.host_failures[request.url.host]
                return httpx.Response(status, json=payload, headers=headers)
            if key in self.key_failures:
                status, payload, headers = self.key_failures[key]
                return httpx.Response(status, json=payload, headers=headers)
            if self.return_500:
                return httpx.Response(500, json={"error": {"message": "internal provider issue"}})
            if self.stream_fail_after_first and body.get("stream") is True:
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=BrokenSSEStream(),
                )
            if self.streaming or body.get("stream") is True:
                content = (
                    'data: {"id":"upstream-secret-id","object":"chat.completion.chunk","model":"private-model-name",'
                    '"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n'
                    'data: {"id":"upstream-secret-id","object":"chat.completion.chunk","model":"private-model-name",'
                    '"choices":[{"index":0,"delta":{"content":" from Hope"},"finish_reason":null}]}\n\n'
                    'data: {"id":"upstream-secret-id","object":"chat.completion.chunk","model":"private-model-name",'
                    f'"choices":[{{"index":0,"delta":{{}},"finish_reason":"{self.stream_finish_reason}"}}],"usage":{{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7}}}}\n\n'
                    'data: [DONE]\n\n'
                )
                return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content.encode())
            payload = self.completion_payload or {
                "id": "upstream-secret-id",
                "object": "chat.completion",
                "model": "private-model-name",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from HopeXAI."},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8},
            }
            return httpx.Response(200, json=payload)
        if path.endswith("openapi.json"):
            return httpx.Response(404, json={"detail": "not found"})
        if method == "POST":
            try:
                body = json.loads(request.content or b"{}")
            except ValueError:
                body = {}
            self.calls.append({"method": method, "path": path, "host": request.url.host, "auth": auth, "body": body})
        return httpx.Response(404, json={"error": {"message": "not found"}})


@asynccontextmanager
async def make_env(
    tmp_path: Path,
    provider_mock: ProviderMock | None = None,
    config: dict[str, Any] | None = None,
    *,
    public_origin: str | None = None,
    proxy_secret: str | None = None,
    vercel_runtime: bool = False,
):
    config = config or config_data()
    default_model = config.get("_test_model")
    mock = provider_mock or ProviderMock(models=[default_model] if default_model else None)
    config_path = tmp_path / "private" / "hopexai.config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml_config(config), encoding="utf-8")
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'hopexai-test.db'}",
        config_path=config_path,
        state_dir=tmp_path / "private" / "state",
        auto_create_schema=True,
        config_poll_interval=0.1,
        max_request_bytes=2_097_152,
        max_messages=200,
        max_tools=64,
        key_issue_per_hour=3,
        key_issue_per_day=10,
        api_key_requests_per_minute=1000,
        response_state_max_bytes=2_000_000,
        trusted_proxy_cidrs=(),
        log_level="WARNING",
        public_origin=public_origin,
        vercel_proxy_secret=proxy_secret,
        vercel_runtime=vercel_runtime,
    )
    transport = httpx.MockTransport(mock)
    app = create_app(
        settings,
        http_transport=transport,
        url_resolver=lambda host, port: ["1.1.1.1"],
    )
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            yield {"app": app, "client": client, "mock": mock, "settings": settings, "config_path": config_path}
