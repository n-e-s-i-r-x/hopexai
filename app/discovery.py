from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit

import httpx
from sqlalchemy import or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .config import BaseURLConfig, GatewayDefaults, ProviderConfig
from .models import DiscoveryCache, DiscoveryLease, utcnow
from .security import URLGuard

logger = logging.getLogger("hopexai.discovery")
DISCOVERABLE_KINDS = {
    "models": ("/models",),
    "chat_completions": ("/chat/completions",),
    "responses": ("/responses",),
    "messages": ("/messages",),
    "embeddings": ("/embeddings",),
    "image_generation": ("/images/generations",),
    "audio_transcriptions": ("/audio/transcriptions",),
    "audio_speech": ("/audio/speech",),
    "moderations": ("/moderations",),
    "rerank": ("/rerank", "/reranking"),
    "files": ("/files",),
}


@dataclass(frozen=True, slots=True)
class DiscoveryData:
    endpoints: dict[str, list[str]]
    api_format: str
    models: list[dict[str, Any]]
    provider_capabilities: dict[str, bool | None]
    last_error_code: str | None = None


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _insert_for(session: AsyncSession, model):
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        return pg_insert(model)
    if dialect == "sqlite":
        return sqlite_insert(model)
    raise RuntimeError(f"unsupported database dialect: {dialect}")


def _config_fingerprint(provider: ProviderConfig, base: BaseURLConfig, secret_store=None) -> str:
    auth = base.auth or provider.auth
    value = {
        "provider": provider.name,
        "provider_format": provider.api_format,
        "base_url": base.url,
        "base_format": base.api_format,
        "auth": {"mode": auth.mode, "header_name": auth.header_name, "prefix": auth.prefix, "query_parameter": auth.query_parameter},
        "key_fingerprints": [secret_store.digest(key.get_secret_value(), "discovery-key") for key in base.api_keys] if secret_store else len(base.api_keys),
        "model_discovery": base.model_discovery,
        "overrides": base.endpoint_overrides.as_paths(),
        "default_capabilities": base.default_capabilities.as_dict(),
        "include_models": base.include_models,
        "exclude_models": base.exclude_models,
        "manual_models": [model.model_dump(mode="json") for model in base.manual_models],
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _origin(base_url: str) -> str:
    parsed = urlsplit(base_url)
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{host}{port}"


def _route_url(base_url: str, path: str) -> str:
    parsed = urlsplit(base_url)
    if path.startswith("/"):
        return _origin(base_url) + path
    prefix = parsed.path.rstrip("/")
    return _origin(base_url) + (prefix + "/" if prefix else "/") + path.lstrip("/")


def _base_prefixes(base_url: str) -> list[str]:
    parsed = urlsplit(base_url)
    path = parsed.path.rstrip("/")
    values = [path, "", "/v1", "/api/v1", "/api"]
    if path and not path.endswith(("/v1", "/api/v1", "/api", "/v1beta", "/v1beta1")):
        values.append(path + "/v1")
        values.append(path + "/api/v1")
    deduped: list[str] = []
    for value in values:
        normalized = "/" + "/".join(part for part in value.split("/") if part) if value else ""
        if normalized not in deduped:
            deduped.append(normalized)
    return deduped[:8]


def _candidate_paths(base_url: str, suffixes: tuple[str, ...], cap: int = 8) -> list[str]:
    results: list[str] = []
    for prefix in _base_prefixes(base_url):
        for suffix in suffixes:
            path = (prefix.rstrip("/") + suffix) or "/"
            if path not in results:
                results.append(path)
            if len(results) >= cap:
                return results
    return results


def _auth_url_headers(base_url: str, auth, api_key: str | None) -> tuple[str, dict[str, str]]:
    headers = {"Accept": "application/json"}
    if auth.mode == "none" or api_key is None:
        return base_url, headers
    if auth.mode == "bearer":
        headers[auth.header_name] = auth.prefix + api_key
    elif auth.mode == "header":
        headers[auth.header_name] = auth.prefix + api_key
    elif auth.mode == "query":
        parsed = urlsplit(base_url)
        query = urlencode({auth.query_parameter: api_key})
        base_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, ""))
    return base_url, headers


def _parse_catalog(data: Any) -> list[dict[str, Any]] | None:
    records = None
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        for name in ("data", "models", "model_list", "items", "results"):
            if isinstance(data.get(name), list):
                records = data[name]
                break
    if records is None:
        return None
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records[:20_000]:
        if isinstance(record, str):
            model_id = record
            metadata: dict[str, Any] = {}
        elif isinstance(record, dict):
            model_id = record.get("id", record.get("name", record.get("model")))
            metadata = record
        else:
            continue
        if not isinstance(model_id, str):
            continue
        model_id = model_id.strip()
        if not model_id or len(model_id) > 256 or any(ord(char) < 32 for char in model_id) or model_id in seen:
            continue
        seen.add(model_id)
        capabilities = _model_capabilities(metadata)
        pricing = _pricing_status(metadata)
        context_window = _context_window(metadata)
        output.append({
            "id": model_id,
            "capabilities": capabilities,
            "pricing": pricing,
            "context_window": context_window,
            "selection_weight": 1.0,
        })
        if len(output) >= 10_000:
            break
    return output


def _context_window(metadata: dict[str, Any]) -> int | None:
    candidates = [
        metadata.get("context_window"), metadata.get("context_length"), metadata.get("max_context_length"),
        metadata.get("max_model_len"), metadata.get("context_length_limit"), metadata.get("max_input_tokens"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            candidate = candidate.get("value")
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool) and 0 < candidate <= 10_000_000:
            return int(candidate)
    return None


def _pricing_status(metadata: dict[str, Any]) -> str:
    pricing = metadata.get("pricing")
    if not isinstance(pricing, dict):
        pricing = metadata.get("cost") if isinstance(metadata.get("cost"), dict) else None
    if isinstance(pricing, dict) and pricing:
        values: dict[str, Decimal] = {}
        for key in ("prompt", "completion", "input", "output", "input_token", "output_token", "request"):
            if key not in pricing:
                continue
            value = pricing[key]
            if isinstance(value, dict):
                value = value.get("value", value.get("price"))
            try:
                number = Decimal(str(value).replace("$", "").strip())
            except (InvalidOperation, ValueError):
                return "unknown"
            if number < 0:
                return "unknown"
            values[key] = number
        if any(value > 0 for value in values.values()):
            return "paid"
        has_input_price = bool({"prompt", "input", "input_token"} & values.keys())
        has_output_price = bool({"completion", "output", "output_token"} & values.keys())
        if values and has_input_price and has_output_price:
            return "free"
        # A missing input or output price is not evidence that the omitted side
        # is free, even when the published price fields that are present are 0.
        return "unknown"
    # An explicit provider-provided free flag is useful only when it does not
    # conflict with a partial/paid price schedule.
    return "free" if metadata.get("free") is True else "unknown"


def _bool_from_metadata(metadata: dict[str, Any], names: tuple[str, ...]) -> bool | None:
    for name in names:
        value = metadata.get(name)
        if isinstance(value, bool):
            return value
    return None


def _model_capabilities(metadata: dict[str, Any]) -> dict[str, bool | None]:
    result: dict[str, bool | None] = {
        "chat": None,
        "responses": None,
        "streaming": None,
        "vision": None,
        "image_generation": None,
        "audio_transcription": None,
        "audio_generation": None,
        "function_calling": None,
        "structured_output": None,
        "moderation": None,
        "reranking": None,
        "file_operations": None,
        "embeddings": None,
        "reasoning": None,
        "reasoning_summary": None,
    }
    raw = metadata.get("capabilities", metadata.get("supported_capabilities", {}))
    if isinstance(raw, dict):
        aliases = {
            "tools": "function_calling", "tool_calling": "function_calling", "functions": "function_calling",
            "json": "structured_output", "json_mode": "structured_output", "vision": "vision",
            "image_understanding": "vision", "reasoning_summaries": "reasoning_summary",
            "stream": "streaming", "audio_in": "audio_transcription", "audio_out": "audio_generation",
        }
        for key, value in raw.items():
            target = aliases.get(key, key)
            if target in result and isinstance(value, bool):
                result[target] = value
    fields = {
        "chat": ("supports_chat", "chat_completions", "chat"),
        "vision": ("supports_vision", "vision"),
        "function_calling": ("supports_tools", "tool_calling", "function_calling"),
        "streaming": ("supports_streaming", "streaming"),
        "structured_output": ("supports_json_mode", "structured_output"),
        "reasoning": ("reasoning", "supports_reasoning"),
        "reasoning_summary": ("reasoning_summary", "supports_reasoning_summary"),
        "embeddings": ("supports_embeddings", "embeddings"),
        "image_generation": ("supports_image_generation", "image_generation"),
        "audio_transcription": ("supports_audio_transcription", "audio_transcription"),
        "audio_generation": ("supports_audio_generation", "audio_generation"),
        "moderation": ("supports_moderation", "moderation"),
        "reranking": ("supports_reranking", "reranking"),
        "file_operations": ("supports_files", "file_operations"),
    }
    for target, names in fields.items():
        if result[target] is None:
            result[target] = _bool_from_metadata(metadata, names)
    modalities: list[str] = []
    for key in ("input_modalities", "modalities", "input_types"):
        value = metadata.get(key)
        if isinstance(value, list):
            modalities.extend(str(item).lower() for item in value)
        elif isinstance(value, dict):
            modalities.extend(str(item).lower() for item, supported in value.items() if supported is True)
    architecture = metadata.get("architecture")
    if isinstance(architecture, dict):
        value = architecture.get("input_modalities", architecture.get("modality"))
        if isinstance(value, list):
            modalities.extend(str(item).lower() for item in value)
        elif isinstance(value, str):
            modalities.append(value.lower())
        task = str(architecture.get("task", "")).lower()
    else:
        task = ""
    if result["vision"] is None and any("image" in item or "vision" in item for item in modalities):
        result["vision"] = True
    if result["chat"] is None:
        task_value = str(metadata.get("task", metadata.get("type", task))).lower()
        if any(token in task_value for token in ("text-generation", "text_generation", "chat", "language-model")):
            result["chat"] = True
        supported = metadata.get("supported_parameters")
        if isinstance(supported, list) and any(item in {"messages", "tools", "tool_choice"} for item in supported):
            result["chat"] = True
    supported = metadata.get("supported_parameters")
    if isinstance(supported, list):
        normalized = {str(item).lower() for item in supported}
        if result["function_calling"] is None and {"tools", "tool_choice"} & normalized:
            result["function_calling"] = True
        if result["streaming"] is None and "stream" in normalized:
            result["streaming"] = True
        if result["reasoning"] is None and {"reasoning_effort", "thinking", "reasoning"} & normalized:
            result["reasoning"] = True
        if result["structured_output"] is None and {"response_format", "json_schema"} & normalized:
            result["structured_output"] = True
    return result


def _merge_capabilities(*values: dict[str, Any]) -> dict[str, bool | None]:
    result: dict[str, bool | None] = {}
    for value in values:
        for name, state in value.items():
            if state is not None:
                result[name] = state
            elif name not in result:
                result[name] = None
    return result


def _openapi_endpoint_map(spec: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, bool | None], bool]:
    endpoints: dict[str, list[str]] = {kind: [] for kind in DISCOVERABLE_KINDS}
    caps: dict[str, bool | None] = {}
    info = spec.get("info", {})
    title = str(info.get("title", "")).lower() if isinstance(info, dict) else ""
    paths = spec.get("paths", {})
    if not isinstance(paths, dict):
        return endpoints, caps, "anthropic" in title
    route_rules: tuple[tuple[str, re.Pattern[str], str], ...] = (
        ("models", re.compile(r"(?:^|/)models/?$", re.I), "get"),
        ("chat_completions", re.compile(r"(?:^|/)chat/completions/?$", re.I), "post"),
        ("responses", re.compile(r"(?:^|/)responses/?$", re.I), "post"),
        ("messages", re.compile(r"(?:^|/)messages/?$", re.I), "post"),
        ("embeddings", re.compile(r"(?:^|/)embeddings/?$", re.I), "post"),
        ("image_generation", re.compile(r"(?:^|/)images/generations/?$", re.I), "post"),
        ("audio_transcriptions", re.compile(r"(?:^|/)audio/transcriptions/?$", re.I), "post"),
        ("audio_speech", re.compile(r"(?:^|/)audio/speech/?$", re.I), "post"),
        ("moderations", re.compile(r"(?:^|/)moderations/?$", re.I), "post"),
        ("rerank", re.compile(r"(?:^|/)rerank(?:ing)?/?$", re.I), "post"),
        ("files", re.compile(r"(?:^|/)files/?$", re.I), "post"),
    )
    for path, path_item in paths.items():
        if not isinstance(path, str) or not path.startswith("/") or ".." in path.split("/") or not isinstance(path_item, dict):
            continue
        lower = path.lower()
        for kind, matcher, method in route_rules:
            if matcher.search(lower) and isinstance(path_item.get(method), dict):
                if path not in endpoints[kind]:
                    endpoints[kind].append(path)
                operation = path_item[method]
                text = json.dumps(operation, ensure_ascii=True).lower()
                if '"stream"' in text or "streaming" in text:
                    caps["streaming"] = True
                if '"tools"' in text or "tool_choice" in text or "tool calling" in text:
                    caps["function_calling"] = True
                if "response_format" in text or "json_schema" in text:
                    caps["structured_output"] = True
        # API descriptions may list an explicit base-level capability.
    is_anthropic = "anthropic" in title and bool(endpoints.get("messages"))
    if endpoints.get("chat_completions") or endpoints.get("responses") or endpoints.get("messages"):
        caps["chat"] = True  # API operation evidence; per-model support remains unknown unless metadata says otherwise.
    if endpoints.get("responses"):
        caps["responses"] = True
    return endpoints, caps, is_anthropic


class DiscoveryService:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        client: httpx.AsyncClient,
        guard: URLGuard,
        secret_store,
        *,
        test_transport: bool = False,
    ):
        self.sessions = sessions
        self.client = client
        self.guard = guard
        self.secret_store = secret_store
        self.test_transport = test_transport
        self.owner = str(uuid.uuid4())
        self._local_locks: dict[str, asyncio.Lock] = {}

    async def get(self, provider: ProviderConfig, base: BaseURLConfig, defaults: GatewayDefaults, *, force: bool = False) -> DiscoveryData:
        key = hashlib.sha256((provider.name + "\0" + base.url).encode("utf-8")).hexdigest()
        lock = self._local_locks.setdefault(key, asyncio.Lock())
        async with lock:
            fingerprint = _config_fingerprint(provider, base, self.secret_store)
            cached = await self._load(provider.name, base.url)
            now = utcnow()
            if not force and cached and cached.config_hash == fingerprint and _as_utc(cached.expires_at) and _as_utc(cached.expires_at) > now:
                return self._to_data(cached)
            acquired = await self._claim_lease(key, defaults.discovery_lease_seconds)
            if not acquired:
                # Another process is refreshing the same base. Wait briefly for
                # its committed snapshot, avoiding duplicate model-list traffic.
                for _ in range(15):
                    await asyncio.sleep(0.2)
                    fresh = await self._load(provider.name, base.url)
                    if fresh and fresh.config_hash == fingerprint and _as_utc(fresh.expires_at) and _as_utc(fresh.expires_at) > utcnow():
                        return self._to_data(fresh)
                if cached:
                    return self._to_data(cached)
                return self._manual_fallback(provider, base)
            try:
                discovered = await self._discover(provider, base, defaults)
                expires = utcnow() + timedelta(seconds=defaults.discovery_ttl_seconds)
                await self._save(provider.name, base.url, fingerprint, discovered, expires)
                return discovered
            except Exception as exc:
                # Stale successful discoveries remain usable during a provider
                # outage or malformed docs response.
                logger.warning("endpoint/model discovery failed provider=%s base=%s class=%s", provider.name, base.url, type(exc).__name__)
                if cached:
                    return self._to_data(cached)
                fallback = self._manual_fallback(provider, base)
                await self._save(provider.name, base.url, fingerprint, fallback, utcnow() + timedelta(seconds=min(60, defaults.discovery_ttl_seconds)))
                return fallback
            finally:
                await self._release_lease(key)

    async def _claim_lease(self, key: str, lease_seconds: int) -> bool:
        now = utcnow()
        async with self.sessions() as session:
            async with session.begin():
                stmt = _insert_for(session, DiscoveryLease).values(
                    cache_key=key, owner=self.owner, lease_until=now + timedelta(seconds=lease_seconds), updated_at=now,
                ).on_conflict_do_nothing(index_elements=[DiscoveryLease.cache_key])
                await session.execute(stmt)
                result = await session.execute(
                    update(DiscoveryLease)
                    .where(
                        DiscoveryLease.cache_key == key,
                        or_(DiscoveryLease.owner == self.owner, DiscoveryLease.lease_until <= now),
                    )
                    .values(owner=self.owner, lease_until=now + timedelta(seconds=lease_seconds), updated_at=now)
                )
                return result.rowcount == 1

    async def _release_lease(self, key: str) -> None:
        now = utcnow()
        async with self.sessions() as session:
            async with session.begin():
                await session.execute(
                    update(DiscoveryLease)
                    .where(DiscoveryLease.cache_key == key, DiscoveryLease.owner == self.owner)
                    .values(owner=None, lease_until=now, updated_at=now)
                )

    async def _load(self, provider_name: str, base_url: str) -> DiscoveryCache | None:
        async with self.sessions() as session:
            return await session.get(DiscoveryCache, (provider_name, base_url))

    async def _save(self, provider_name: str, base_url: str, config_hash: str, data: DiscoveryData, expires: datetime) -> None:
        now = utcnow()
        values = {
            "provider_name": provider_name,
            "base_url": base_url,
            "config_hash": config_hash,
            "endpoints": data.endpoints,
            "api_format": data.api_format,
            "models": data.models,
            "provider_capabilities": data.provider_capabilities,
            "discovered_at": now,
            "expires_at": expires,
            "last_error_code": data.last_error_code,
        }
        async with self.sessions() as session:
            async with session.begin():
                stmt = _insert_for(session, DiscoveryCache).values(**values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[DiscoveryCache.provider_name, DiscoveryCache.base_url],
                    set_={key: value for key, value in values.items() if key not in {"provider_name", "base_url"}},
                )
                await session.execute(stmt)

    def _to_data(self, record: DiscoveryCache) -> DiscoveryData:
        endpoints = record.endpoints if isinstance(record.endpoints, dict) else {}
        normalized = {key: value if isinstance(value, list) else ([value] if isinstance(value, str) else []) for key, value in endpoints.items()}
        return DiscoveryData(
            endpoints=normalized,
            api_format=record.api_format,
            models=record.models if isinstance(record.models, list) else [],
            provider_capabilities=record.provider_capabilities if isinstance(record.provider_capabilities, dict) else {},
            last_error_code=record.last_error_code,
        )

    def _manual_fallback(self, provider: ProviderConfig, base: BaseURLConfig) -> DiscoveryData:
        endpoints = {key: [path] for key, path in base.endpoint_overrides.as_paths().items()}
        models = []
        for manual in base.manual_models:
            if not manual.enabled:
                continue
            models.append({
                "id": manual.id,
                "capabilities": manual.capabilities.as_dict(),
                "pricing": manual.pricing or "unknown",
                "context_window": manual.context_window,
                "selection_weight": manual.selection_weight,
            })
        configured_format = base.api_format or provider.api_format
        api_format = configured_format if configured_format != "auto" else "auto"
        return DiscoveryData(endpoints, api_format, models, {}, "discovery_unavailable")

    async def _discover(self, provider: ProviderConfig, base: BaseURLConfig, defaults: GatewayDefaults) -> DiscoveryData:
        endpoints: dict[str, list[str]] = {kind: [] for kind in DISCOVERABLE_KINDS}
        for kind, path in base.endpoint_overrides.as_paths().items():
            endpoints.setdefault(kind, []).append(path)
        provider_caps: dict[str, bool | None] = {}
        models: list[dict[str, Any]] = []
        format_from_spec: str | None = None

        await self.guard.validate(base.url, allow_private_network=base.allow_private_network)

        # Inspect the configured base, its common version roots, and response
        # metadata without credentials. Redirects are followed only when same-
        # origin and publicly resolved; cross-origin locations are never used.
        for prefix in _base_prefixes(base.url)[:5]:
            root_path = prefix or "/"
            root_url = _route_url(base.url, root_path)
            for method in ("GET", "HEAD", "OPTIONS"):
                try:
                    response = await self.guard.safe_stream(
                        self.client, method, root_url,
                        allow_private_network=base.allow_private_network,
                        timeout=defaults.discovery_timeout_seconds,
                        headers={"Accept": "application/json, application/yaml, text/html"},
                    )
                    try:
                        # Touch the relevant headers for status/content/version
                        # inspection; values are not returned publicly or logged.
                        _ = (
                            response.status_code,
                            response.headers.get("content-type"),
                            response.headers.get("api-version"),
                            response.headers.get("x-api-version"),
                            response.headers.get("allow"),
                            response.headers.get("location"),
                        )
                    finally:
                        await response.aclose()
                except Exception:
                    continue

        # Read OpenAPI metadata without credentials. Only recognized API paths
        # on the configured origin are retained; arbitrary documented URLs are
        # never fetched.
        docs_paths = _candidate_paths(base.url, ("/openapi.json", "/swagger.json", "/api-docs", "/docs/openapi.json"), cap=16)
        for path in docs_paths:
            try:
                status, headers, data = await self._fetch_json(
                    _route_url(base.url, path), base.allow_private_network,
                    defaults.discovery_timeout_seconds, headers={"Accept": "application/json"},
                )
            except Exception:
                continue
            if status != 200 or not isinstance(data, dict) or not (data.get("openapi") or data.get("swagger")):
                continue
            found, caps, anthropic = _openapi_endpoint_map(data)
            self._merge_endpoints(endpoints, found)
            provider_caps = _merge_capabilities(provider_caps, caps)
            if anthropic:
                format_from_spec = "anthropic"
            elif found.get("chat_completions"):
                format_from_spec = "openai_chat"
            elif found.get("responses"):
                format_from_spec = "openai_responses"
            break

        # Direct, bounded probes are limited to established API route candidates.
        # They do not include credentials and do not follow cross-origin redirects.
        model_candidates = _candidate_paths(base.url, ("/models",), cap=12)
        if base.model_discovery:
            for path in model_candidates:
                if not any(existing == path for existing in endpoints.get("models", [])):
                    try:
                        status, headers, data = await self._fetch_json(
                            _route_url(base.url, path), base.allow_private_network,
                            defaults.discovery_timeout_seconds, headers={"Accept": "application/json"},
                        )
                    except Exception:
                        continue
                    if status == 200:
                        catalog = _parse_catalog(data)
                        if catalog is not None:
                            endpoints["models"].append(path)
                            models = catalog
                            break
                    elif status in {401, 403}:
                        # Authentication-gated list route: this is a candidate,
                        # but the actual catalog read is performed below only
                        # against the explicitly configured origin.
                        endpoints["models"].append(path)
                        break

        # If docs did not expose a model path, inspect the bounded, standard
        # candidates for a recognizable model-list shape or auth-required route.
        if base.model_discovery and not endpoints.get("models"):
            for path in model_candidates:
                try:
                    status, _, data = await self._fetch_json(
                        _route_url(base.url, path), base.allow_private_network,
                        defaults.discovery_timeout_seconds, headers={"Accept": "application/json"},
                    )
                except Exception:
                    continue
                if status == 200 and _parse_catalog(data) is not None:
                    endpoints["models"].append(path)
                    models = _parse_catalog(data) or []
                    break
                if status in {401, 403}:
                    endpoints["models"].append(path)
                    break

        # Verify completion routes using OPTIONS/HEAD/GET and a final malformed,
        # unauthenticated compatibility check where needed. The sentinel request
        # has no valid model or messages and cannot generate billable output.
        for kind, suffixes in (("chat_completions", ("/chat/completions",)), ("responses", ("/responses",)), ("messages", ("/messages",))):
            for path in _candidate_paths(base.url, suffixes, cap=10):
                if path in endpoints.get(kind, []):
                    continue
                verified, evidence = await self._probe_post_route(base.url, path, kind, base.allow_private_network, defaults.discovery_timeout_seconds)
                if verified:
                    endpoints[kind].append(path)
                    break

        # Capability routes besides chat are learned from specs and metadata,
        # not by sending arbitrary POST requests to undocumented paths.
        for kind, suffixes in DISCOVERABLE_KINDS.items():
            if kind in {"models", "chat_completions", "responses", "messages"}:
                continue
            # Use known, bounded paths only when OPTIONS clearly advertises POST.
            for path in _candidate_paths(base.url, suffixes, cap=4):
                if path in endpoints.get(kind, []):
                    continue
                try:
                    response = await self.guard.safe_stream(
                        self.client, "OPTIONS", _route_url(base.url, path),
                        allow_private_network=base.allow_private_network,
                        timeout=defaults.discovery_timeout_seconds,
                        headers={"Accept": "application/json"},
                    )
                    try:
                        allow = (response.headers.get("allow", "") + "," + response.headers.get("access-control-allow-methods", "")).upper()
                        if response.status_code in {200, 204} and "POST" in allow:
                            endpoints[kind].append(path)
                            break
                    finally:
                        await response.aclose()
                except Exception:
                    continue

        if base.model_discovery and (not models) and endpoints.get("models"):
            models = await self._fetch_authenticated_catalog(provider, base, endpoints["models"], defaults.discovery_timeout_seconds)

        if format_from_spec is None:
            if endpoints.get("chat_completions"):
                format_from_spec = "openai_chat"
            elif endpoints.get("responses"):
                format_from_spec = "openai_responses"
            elif endpoints.get("messages"):
                format_from_spec = "anthropic" if "anthropic" in provider.name.lower() else "auto"

        configured_format = base.api_format or provider.api_format
        api_format = configured_format if configured_format != "auto" else (format_from_spec or "auto")
        # Provider-level API capability evidence is not copied into per-model
        # capability claims except for features that the OpenAPI operation proves.
        if endpoints.get("chat_completions") or endpoints.get("responses") or endpoints.get("messages"):
            provider_caps["chat"] = True
        if endpoints.get("responses"):
            provider_caps["responses"] = True
        if any("stream" in name for name in provider_caps):
            provider_caps["streaming"] = True

        # Ensure model IDs are restricted to configured inclusions/exclusions.
        if base.include_models is not None:
            allowed = set(base.include_models)
            models = [model for model in models if model.get("id") in allowed]
        excluded = set(base.exclude_models)
        if excluded:
            models = [model for model in models if model.get("id") not in excluded]

        # Merge manual metadata for matching IDs; manual entries are the
        # authoritative per-model override and can also add undiscoverable IDs.
        by_id = {model["id"]: model for model in models if isinstance(model.get("id"), str)}
        for manual in base.manual_models:
            if not manual.enabled or manual.id in excluded:
                continue
            current = by_id.get(manual.id, {
                "id": manual.id,
                "capabilities": {},
                "pricing": "unknown",
                "context_window": None,
                "selection_weight": 1.0,
            })
            current["capabilities"] = _merge_capabilities(
                current.get("capabilities", {}), manual.capabilities.as_dict()
            )
            if manual.pricing is not None:
                current["pricing"] = manual.pricing
            current["context_window"] = manual.context_window or current.get("context_window")
            current["selection_weight"] = manual.selection_weight
            by_id[manual.id] = current
        models = list(by_id.values())

        return DiscoveryData(endpoints, api_format, models, provider_caps, None)

    @staticmethod
    def _merge_endpoints(target: dict[str, list[str]], source: dict[str, list[str]]) -> None:
        for kind, paths in source.items():
            current = target.setdefault(kind, [])
            for path in paths:
                if path not in current:
                    current.append(path)

    async def _probe_post_route(
        self,
        base_url: str,
        path: str,
        kind: str,
        allow_private: bool,
        timeout: float,
    ) -> tuple[bool, str]:
        url = _route_url(base_url, path)
        allow_post = False
        saw_protected = False
        for method in ("OPTIONS", "HEAD", "GET"):
            try:
                response = await self.guard.safe_stream(
                    self.client, method, url, allow_private_network=allow_private,
                    timeout=timeout, headers={"Accept": "application/json"},
                )
                try:
                    allow = (response.headers.get("allow", "") + "," + response.headers.get("access-control-allow-methods", "")).upper()
                    if response.status_code in {200, 204, 405} and "POST" in allow:
                        return True, f"{method}_allow_post"
                    if response.status_code in {401, 403}:
                        saw_protected = True
                    if response.status_code == 405:
                        allow_post = "POST" in allow
                finally:
                    await response.aclose()
            except Exception:
                continue
        if allow_post:
            return True, "allow_header"
        # If an exact candidate is protected by auth, use an invalid, credential-
        # free payload only to confirm the route. No successful result is accepted.
        if saw_protected:
            probe_body = {"model": "__hopexai_discovery_probe__", "max_tokens": 0, "messages": []}
            if kind == "responses":
                probe_body = {"model": "__hopexai_discovery_probe__", "max_output_tokens": 0, "input": []}
            elif kind == "messages":
                probe_body = {"model": "__hopexai_discovery_probe__", "max_tokens": 0, "messages": []}
            try:
                response = await self.guard.safe_stream(
                    self.client, "POST", url, allow_private_network=allow_private,
                    timeout=timeout, headers={"Accept": "application/json", "Content-Type": "application/json"},
                    json=probe_body,
                )
                try:
                    # A 2xx is intentionally not considered proof; the sentinel
                    # could be accepted by a permissive test service.
                    if response.status_code in {400, 401, 403, 404, 405, 415, 422}:
                        if response.status_code != 404:
                            return True, "controlled_invalid_request"
                finally:
                    await response.aclose()
            except Exception:
                pass
        return False, "not_verified"

    async def _fetch_authenticated_catalog(
        self,
        provider: ProviderConfig,
        base: BaseURLConfig,
        model_paths: list[str],
        timeout: float,
    ) -> list[dict[str, Any]]:
        auth = base.auth or provider.auth
        keys = [item.get_secret_value() for item in base.api_keys]
        if auth.mode != "none" and not keys:
            return []
        attempts = keys[: min(3, len(keys))] if auth.mode != "none" else [None]
        for path in model_paths[:3]:
            for key in attempts:
                url, headers = _auth_url_headers(_route_url(base.url, path), auth, key)
                try:
                    status, _, data = await self._fetch_json(url, base.allow_private_network, timeout, headers=headers)
                except Exception:
                    continue
                if status == 200:
                    catalog = _parse_catalog(data)
                    if catalog is not None:
                        return catalog
                if status in {403, 429}:
                    # A forbidden catalog read or a rate limit does not identify
                    # this credential as invalid. Do not rotate keys to probe
                    # around policy or provider-wide quota restrictions.
                    return []
        return []

    async def _fetch_json(
        self,
        url: str,
        allow_private: bool,
        timeout: float,
        *,
        headers: dict[str, str] | None = None,
        method: str = "GET",
        json_body: dict[str, Any] | None = None,
    ) -> tuple[int, httpx.Headers, Any]:
        if self.test_transport:
            # Mock transports intentionally bypass DNS, but retain scheme/host
            # validation. This flag is only set by tests and never from env.
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("invalid test URL")
        content_headers = dict(headers or {})
        kwargs: dict[str, Any] = {"headers": content_headers}
        if json_body is not None:
            kwargs["json"] = json_body
        response = await self.guard.safe_stream(
            self.client, method, url,
            allow_private_network=allow_private,
            timeout=httpx.Timeout(timeout, connect=min(timeout, 5.0)),
            **kwargs,
        )
        try:
            content_length = response.headers.get("content-length")
            if content_length and content_length.isdigit() and int(content_length) > 2_000_000:
                return response.status_code, response.headers, None
            data_bytes = bytearray()
            async for chunk in response.aiter_bytes():
                if len(data_bytes) + len(chunk) > 2_000_000:
                    return response.status_code, response.headers, None
                data_bytes.extend(chunk)
            body = bytes(data_bytes)
            if not body:
                return response.status_code, response.headers, None
            media_type = response.headers.get("content-type", "").lower()
            if "json" not in media_type and body[:1] not in {b"{", b"["}:
                return response.status_code, response.headers, None
            try:
                return response.status_code, response.headers, json.loads(body)
            except (ValueError, UnicodeDecodeError):
                return response.status_code, response.headers, None
        finally:
            await response.aclose()
