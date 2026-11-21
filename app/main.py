from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import ConfigManager, ConfigurationError
from .db import check_database, create_database, initialize_schema
from .discovery import DiscoveryService
from .errors import NoEligibleCandidate, PublicAPIError
from .health import HealthService
from .key_service import KeyService
from .presentation import (
    assistant_history_message,
    chat_completion_body,
    chat_stream_body,
    new_response_id,
    responses_body,
    responses_stream_body,
)
from .registry import Registry
from .requests import PUBLIC_MODEL, parse_chat_request, parse_responses_request
from .router import GatewayRouter
from .runtime_state import RuntimeState
from .secret_store import SecretStore
from .security import PinnedAsyncHTTPTransport, URLGuard, get_client_ip
from .settings import Settings
from .upstream import UpstreamClient

logger = logging.getLogger("hopexai.app")


def _origin_key(value: str) -> tuple[str, str, int]:
    parsed = urlsplit(value)
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid origin")
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    return parsed.scheme.lower(), parsed.hostname.rstrip(".").lower(), port


def create_app(
    settings: Settings | None = None,
    *,
    http_transport: httpx.AsyncBaseTransport | None = None,
    url_resolver=None,
) -> FastAPI:
    app_settings = settings or Settings.from_env()
    engine, sessions = create_database(app_settings)
    secret_store = SecretStore(app_settings.state_dir)
    config_manager = ConfigManager(app_settings, secret_store)
    outbound_limits = httpx.Limits(max_connections=200, max_keepalive_connections=40, keepalive_expiry=30)
    outbound_transport = http_transport if http_transport is not None else PinnedAsyncHTTPTransport(limits=outbound_limits, trust_env=False)
    client = httpx.AsyncClient(
        transport=outbound_transport,
        timeout=None,
        limits=outbound_limits,
        follow_redirects=False,
        trust_env=False,
        headers={"User-Agent": "HopeXAI-Gateway/1.0"},
    )
    guard = URLGuard(resolver=url_resolver)
    health = HealthService(sessions, secret_store)
    state = RuntimeState(sessions, secret_store, app_settings.response_state_max_bytes)
    key_service = KeyService(sessions, secret_store, app_settings)
    discovery = DiscoveryService(sessions, client, guard, secret_store, test_transport=http_transport is not None)
    registry = Registry(discovery, health)
    upstream = UpstreamClient(client, guard)
    router = GatewayRouter(registry, upstream, health, state)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        secret_store.ensure()
        if app_settings.auto_create_schema:
            await initialize_schema(engine)
        else:
            await check_database(engine)
        await config_manager.load_initial()
        logger.info("HopeXAI gateway ready; database=%s", engine.url.get_backend_name())
        try:
            yield
        finally:
            await client.aclose()
            await engine.dispose()

    application = FastAPI(
        title="HopeXAI API",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    application.state.settings = app_settings
    application.state.engine = engine
    application.state.sessions = sessions
    application.state.secret_store = secret_store
    application.state.config_manager = config_manager
    application.state.key_service = key_service
    application.state.discovery = discovery
    application.state.health = health
    application.state.runtime_state = state
    application.state.router = router

    @application.middleware("http")
    async def hardening_headers(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'self'; object-src 'none'",
        )
        response.headers.setdefault("X-Request-ID", request.state.request_id)
        if request.url.path.startswith("/v1/") or request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @application.exception_handler(PublicAPIError)
    async def handle_public_error(request: Request, exc: PublicAPIError):
        headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else {}
        return JSONResponse(exc.as_payload(), status_code=exc.status_code, headers=headers)

    @application.exception_handler(NoEligibleCandidate)
    async def handle_no_candidate(request: Request, exc: NoEligibleCandidate):
        return JSONResponse(
            {"error": {"message": exc.message, "type": "service_unavailable_error", "code": exc.code}},
            status_code=exc.status_code,
        )

    @application.exception_handler(ConfigurationError)
    async def handle_config_error(request: Request, exc: ConfigurationError):
        logger.error("configuration unavailable for request path=%s", request.url.path)
        return JSONResponse(
            {"error": {"message": "HopeXAI is temporarily unavailable.", "type": "server_error", "code": "configuration_unavailable"}},
            status_code=503,
        )

    async def authenticated_key(request: Request):
        if app_settings.vercel_proxy_secret:
            supplied_secret = request.headers.get("x-hopexai-vercel-proxy-secret", "")
            if not supplied_secret or not secrets.compare_digest(supplied_secret, app_settings.vercel_proxy_secret):
                raise PublicAPIError(403, "Requests must use the configured public gateway.", "authentication_error", "proxy_not_allowed")
        authorization = request.headers.get("authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return await key_service.authenticate(None)
        record = await key_service.authenticate(token)
        await key_service.register_request(record.id)
        return record

    async def read_json(request: Request) -> dict:
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise PublicAPIError(415, "Content-Type must be application/json.", "invalid_request_error", "unsupported_media_type")
        length = request.headers.get("content-length")
        if length:
            try:
                if int(length) > app_settings.max_request_bytes:
                    raise PublicAPIError(413, "Request body exceeds the configured size limit.", "invalid_request_error", "request_too_large")
            except ValueError:
                raise PublicAPIError(400, "Invalid Content-Length header.", "invalid_request_error", "invalid_content_length")
        chunks = bytearray()
        async for chunk in request.stream():
            if len(chunks) + len(chunk) > app_settings.max_request_bytes:
                raise PublicAPIError(413, "Request body exceeds the configured size limit.", "invalid_request_error", "request_too_large")
            chunks.extend(chunk)
        try:
            data = json.loads(chunks)
        except (ValueError, UnicodeDecodeError) as exc:
            raise PublicAPIError(400, "Request body must be valid JSON.", "invalid_request_error", "invalid_json") from exc
        if not isinstance(data, dict):
            raise PublicAPIError(400, "Request body must be a JSON object.", "invalid_request_error", "invalid_json")
        return data

    async def safe_log_request(**kwargs) -> None:
        try:
            await state.log_request(**kwargs)
        except Exception as exc:
            logger.error("request statistics write failed class=%s", type(exc).__name__)

    @application.get("/", include_in_schema=False)
    async def home():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @application.get("/healthz", include_in_schema=False)
    async def healthz():
        return {"status": "ok"}

    @application.get("/readyz", include_in_schema=False)
    async def readyz():
        try:
            await check_database(engine)
            config_manager.snapshot
            return {"status": "ready"}
        except Exception:
            return JSONResponse({"status": "not_ready"}, status_code=503)

    @application.post("/api/keys", include_in_schema=False)
    async def generate_api_key(request: Request):
        await read_json(request)
        origin = request.headers.get("origin")
        if app_settings.public_origin and not origin:
            raise PublicAPIError(403, "A same-site origin is required to generate a key.", "authentication_error", "origin_not_allowed")
        if origin:
            try:
                presented_origin = _origin_key(origin)
                if app_settings.public_origin:
                    allowed_origin = _origin_key(app_settings.public_origin)
                    allowed = presented_origin == allowed_origin
                else:
                    parsed = urlsplit(origin)
                    host = request.headers.get("host", "").lower()
                    allowed = parsed.netloc.lower() == host
                if not allowed:
                    raise PublicAPIError(403, "Cross-origin key generation is not allowed.", "authentication_error", "origin_not_allowed")
            except ValueError as exc:
                raise PublicAPIError(403, "Cross-origin key generation is not allowed.", "authentication_error", "origin_not_allowed") from exc
        if app_settings.public_origin and app_settings.vercel_proxy_secret:
            supplied_secret = request.headers.get("x-hopexai-vercel-proxy-secret", "")
            if not supplied_secret or not secrets.compare_digest(supplied_secret, app_settings.vercel_proxy_secret):
                raise PublicAPIError(403, "Key generation must use the configured public gateway.", "authentication_error", "proxy_not_allowed")
        ip = get_client_ip(
            request,
            app_settings.trusted_proxy_cidrs,
            vercel_proxy_secret=app_settings.vercel_proxy_secret,
        )
        ip_digest = secret_store.digest(ip, "key-issue-ip")
        secret, record = await key_service.issue(ip_digest)
        return JSONResponse({
            "id": record.id,
            "api_key": secret,
            "prefix": record.prefix,
            "created_at": record.created_at.isoformat(),
            "display_once": True,
        }, headers={"Cache-Control": "no-store", "Pragma": "no-cache"})

    @application.get("/v1/models")
    async def list_models(request: Request):
        await authenticated_key(request)
        return JSONResponse({
            "object": "list",
            "data": [{"id": PUBLIC_MODEL, "object": "model", "owned_by": "hopexai"}],
        }, headers={"Cache-Control": "no-store"})

    @application.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        api_key = await authenticated_key(request)
        body = await read_json(request)
        canonical = parse_chat_request(
            body,
            app_settings.max_messages,
            app_settings.max_tools,
            request.headers.get("x-hopexai-conversation-id"),
        )
        await config_manager.refresh_if_changed()
        config = config_manager.snapshot.config
        if canonical.max_output_tokens is None:
            canonical.max_output_tokens = config.defaults.default_max_output_tokens
        elif canonical.max_output_tokens > config.defaults.max_output_tokens_limit:
            raise PublicAPIError(400, "Requested output exceeds HopeXAI's configured output-token safety limit.", "invalid_request_error", "max_output_tokens_exceeded", "max_tokens")
        request_id = request.state.request_id
        affinity_key = state.affinity_digest(api_key.id, canonical.conversation_id)
        capabilities = sorted(canonical.required_capabilities)
        started = time.monotonic()

        if canonical.stream:
            chosen: dict[str, object] = {}

            async def on_candidate(candidate):
                chosen["candidate"] = candidate
                await state.save_affinity(
                    affinity_key, api_key.id, candidate.provider_name, candidate.base_url, candidate.model_id,
                    config.defaults.conversation_state_ttl_seconds,
                )

            async def generate():
                streamed_error = False
                stream_completed = False
                try:
                    async for chunk in chat_stream_body(
                        router, config, canonical, request_id, api_key.id, affinity_key,
                        disconnected=request.is_disconnected, on_candidate=on_candidate,
                    ):
                        if b'"error"' in chunk and b"[DONE]" not in chunk:
                            streamed_error = True
                        if b"data: [DONE]" in chunk:
                            stream_completed = True
                        yield chunk
                finally:
                    candidate = chosen.get("candidate")
                    await safe_log_request(
                        request_id=request_id,
                        api_key_id=api_key.id,
                        request_kind="chat.completion.stream",
                        status="error" if streamed_error else ("complete" if stream_completed else "cancelled"),
                        provider_name=getattr(candidate, "provider_name", None),
                        base_url=getattr(candidate, "base_url", None),
                        model_id=getattr(candidate, "model_id", None),
                        capabilities=capabilities,
                        latency_ms=int((time.monotonic() - started) * 1000),
                        error_code="stream_failed" if streamed_error else ("client_cancelled" if not stream_completed else None),
                    )

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive", "X-Request-ID": request_id},
            )

        try:
            result = await router.complete(config, canonical, request_id, api_key.id, affinity_key)
            await state.save_affinity(
                affinity_key, api_key.id, result.candidate.provider_name, result.candidate.base_url, result.candidate.model_id,
                config.defaults.conversation_state_ttl_seconds,
            )
            body_out = chat_completion_body(result.response, request_id)
            await safe_log_request(
                request_id=request_id,
                api_key_id=api_key.id,
                request_kind="chat.completion",
                status="complete",
                provider_name=result.candidate.provider_name,
                base_url=result.candidate.base_url,
                model_id=result.candidate.model_id,
                capabilities=capabilities,
                latency_ms=int((time.monotonic() - started) * 1000),
            )
            return JSONResponse(body_out, headers={"X-Request-ID": request_id})
        except NoEligibleCandidate as exc:
            await safe_log_request(
                request_id=request_id, api_key_id=api_key.id, request_kind="chat.completion", status="error",
                provider_name=None, base_url=None, model_id=None, capabilities=capabilities,
                latency_ms=int((time.monotonic() - started) * 1000), error_code=exc.code,
            )
            raise

    @application.post("/v1/responses")
    async def responses(request: Request):
        api_key = await authenticated_key(request)
        body = await read_json(request)
        previous_id = body.get("previous_response_id")
        previous_messages = None
        if previous_id is not None:
            if not isinstance(previous_id, str) or len(previous_id) > 80:
                raise PublicAPIError(400, "previous_response_id is invalid.", "invalid_request_error", "invalid_previous_response_id", "previous_response_id")
            previous_messages = await state.load_response_messages(previous_id, api_key.id)
        canonical = parse_responses_request(
            body,
            app_settings.max_messages,
            app_settings.max_tools,
            request.headers.get("x-hopexai-conversation-id"),
            previous_messages,
        )
        response_id = new_response_id()
        if canonical.conversation_id is None:
            canonical.conversation_id = "response-chain:" + (previous_id or response_id)
        await config_manager.refresh_if_changed()
        config = config_manager.snapshot.config
        if canonical.max_output_tokens is None:
            canonical.max_output_tokens = config.defaults.default_max_output_tokens
        elif canonical.max_output_tokens > config.defaults.max_output_tokens_limit:
            raise PublicAPIError(400, "Requested output exceeds HopeXAI's configured output-token safety limit.", "invalid_request_error", "max_output_tokens_exceeded", "max_output_tokens")
        request_id = request.state.request_id
        affinity_key = state.affinity_digest(api_key.id, canonical.conversation_id)
        capabilities = sorted(canonical.required_capabilities)
        started = time.monotonic()

        if canonical.stream:
            chosen: dict[str, object] = {}

            async def on_candidate(candidate):
                chosen["candidate"] = candidate
                await state.save_affinity(
                    affinity_key, api_key.id, candidate.provider_name, candidate.base_url, candidate.model_id,
                    config.defaults.conversation_state_ttl_seconds,
                )

            async def on_complete(canonical_response, response_payload):
                candidate = chosen.get("candidate")
                messages = canonical.messages + [assistant_history_message(canonical_response)]
                stored = False
                if canonical.store:
                    stored = await state.save_response_messages(
                        response_id, api_key.id, messages,
                        config.defaults.conversation_state_ttl_seconds,
                        getattr(candidate, "provider_name", None),
                        getattr(candidate, "base_url", None),
                        getattr(candidate, "model_id", None),
                    )
                response_payload["store"] = stored
                await safe_log_request(
                    request_id=request_id, api_key_id=api_key.id, request_kind="response.stream", status="complete",
                    provider_name=getattr(candidate, "provider_name", None),
                    base_url=getattr(candidate, "base_url", None), model_id=getattr(candidate, "model_id", None),
                    capabilities=capabilities, latency_ms=int((time.monotonic() - started) * 1000),
                )

            async def generate():
                streamed_error = False
                stream_completed = False
                try:
                    async for chunk in responses_stream_body(
                        router, config, canonical, response_id, request_id, api_key.id, affinity_key,
                        disconnected=request.is_disconnected, on_candidate=on_candidate, on_complete=on_complete,
                    ):
                        if b'"type":"error"' in chunk or b'"error":{' in chunk:
                            streamed_error = True
                        if b'"type":"response.completed"' in chunk or b'"type":"response.incomplete"' in chunk:
                            stream_completed = True
                        yield chunk
                finally:
                    if streamed_error or not stream_completed:
                        candidate = chosen.get("candidate")
                        await safe_log_request(
                            request_id=request_id, api_key_id=api_key.id, request_kind="response.stream",
                            status="error" if streamed_error else "cancelled",
                            provider_name=getattr(candidate, "provider_name", None), base_url=getattr(candidate, "base_url", None),
                            model_id=getattr(candidate, "model_id", None), capabilities=capabilities,
                            latency_ms=int((time.monotonic() - started) * 1000),
                            error_code="stream_failed" if streamed_error else "client_cancelled",
                        )

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive", "X-Request-ID": request_id},
            )

        try:
            result = await router.complete(config, canonical, request_id, api_key.id, affinity_key)
            await state.save_affinity(
                affinity_key, api_key.id, result.candidate.provider_name, result.candidate.base_url, result.candidate.model_id,
                config.defaults.conversation_state_ttl_seconds,
            )
            conversation_messages = canonical.messages + [assistant_history_message(result.response)]
            stored = False
            if canonical.store:
                stored = await state.save_response_messages(
                    response_id, api_key.id, conversation_messages,
                    config.defaults.conversation_state_ttl_seconds,
                    result.candidate.provider_name, result.candidate.base_url, result.candidate.model_id,
                )
            body_out = responses_body(result.response, response_id, stored=stored)
            await safe_log_request(
                request_id=request_id, api_key_id=api_key.id, request_kind="response", status="complete",
                provider_name=result.candidate.provider_name, base_url=result.candidate.base_url,
                model_id=result.candidate.model_id, capabilities=capabilities,
                latency_ms=int((time.monotonic() - started) * 1000),
            )
            return JSONResponse(body_out, headers={"X-Request-ID": request_id})
        except NoEligibleCandidate as exc:
            await safe_log_request(
                request_id=request_id, api_key_id=api_key.id, request_kind="response", status="error",
                provider_name=None, base_url=None, model_id=None, capabilities=capabilities,
                latency_ms=int((time.monotonic() - started) * 1000), error_code=exc.code,
            )
            raise

    application.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    return application


app = create_app()
