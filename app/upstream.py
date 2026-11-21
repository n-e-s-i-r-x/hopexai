from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, AsyncIterator
from urllib.parse import urlencode, urlsplit, urlunsplit

import httpx

from .adapters import AdapterError, AdapterRegistry, BaseAdapter
from .config import GatewayDefaults
from .errors import UpstreamFailure
from .requests import CanonicalRequest
from .security import URLGuard, UnsafeUpstreamURL
from .types import Candidate, CanonicalDelta, CanonicalResponse

logger = logging.getLogger("hopexai.upstream")
MAX_UPSTREAM_JSON_BYTES = 8_000_000
MAX_UPSTREAM_ERROR_BYTES = 64_000


def endpoint_url(base_url: str, endpoint_path: str) -> str:
    parsed = urlsplit(base_url)
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port else ""
    origin = f"{parsed.scheme}://{host}{port}"
    if endpoint_path.startswith("/"):
        return origin + endpoint_path
    base_path = parsed.path.rstrip("/")
    path = base_path + "/" + endpoint_path.lstrip("/") if base_path else "/" + endpoint_path.lstrip("/")
    return origin + path


def _auth_target(candidate: Candidate, url: str, extra_headers: dict[str, str]) -> tuple[str, dict[str, str]]:
    auth = candidate.base.auth or candidate.provider.auth
    headers = {"Accept": "application/json", **extra_headers}
    if auth.mode == "none" or candidate.api_key is None:
        return url, headers
    if auth.mode in {"bearer", "header"}:
        headers[auth.header_name] = auth.prefix + candidate.api_key
    elif auth.mode == "query":
        parsed = urlsplit(url)
        query = urlencode({auth.query_parameter: candidate.api_key})
        url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, ""))
    return url, headers


def _retry_after(headers: httpx.Headers) -> int | None:
    value = headers.get("retry-after")
    if not value:
        return None
    try:
        return max(1, min(int(float(value)), 86_400))
    except (ValueError, OverflowError):
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(1, min(int((when - datetime.now(timezone.utc)).total_seconds()), 86_400))
        except Exception:
            return None


def _classify_http_failure(status: int, body: bytes, headers: httpx.Headers, candidate: Candidate, defaults: GatewayDefaults) -> UpstreamFailure:
    text = body[:MAX_UPSTREAM_ERROR_BYTES].decode("utf-8", "ignore").lower()
    retry_after = _retry_after(headers)
    idempotent_retry = candidate.base.supports_idempotency and defaults.retry_ambiguous_failures
    if status == 401:
        return UpstreamFailure("invalid_credentials", status, True, "key", retry_after, "upstream rejected credentials")
    if status == 403:
        invalid = any(token in text for token in ("invalid api key", "invalid_api_key", "incorrect api key", "revoked key"))
        if invalid:
            return UpstreamFailure("invalid_credentials", status, True, "key", retry_after, "upstream rejected credentials")
        return UpstreamFailure("permission_denied", status, True, "model", retry_after, "upstream denied this operation")
    if status == 429:
        if any(token in text for token in (
            "quota", "billing", "insufficient", "credits exhausted", "payment required",
            "organization limit", "organisation limit", "project limit",
        )):
            return UpstreamFailure("account_quota_exhausted", status, True, "base", retry_after, "upstream account quota unavailable")
        if any(token in text for token in (
            "api key rate limit", "api-key rate limit", "key rate limit",
            "rate limit for this key", "per-key rate limit", "per key rate limit",
        )):
            return UpstreamFailure("rate_limited", status, True, "key", retry_after, "upstream key rate limited")
        # A generic 429 does not identify whether the limit is key-, project-,
        # or account-wide. Cool down the configured base instead of rotating
        # credentials and potentially bypassing a shared provider quota.
        return UpstreamFailure("rate_limited", status, True, "base", retry_after, "upstream rate limit scope was ambiguous")
    if status == 404:
        model_missing = any(token in text for token in ("model_not_found", "model not found", "unknown model", "model does not exist", "no such model"))
        if model_missing:
            return UpstreamFailure("model_unavailable", status, True, "model", retry_after, "upstream model was not found")
        return UpstreamFailure("endpoint_not_found", status, True, "endpoint", retry_after, "upstream endpoint was not found")
    if status in {400, 405, 415, 422}:
        if any(token in text for token in ("context_length_exceeded", "maximum context length", "too many tokens", "context window")):
            return UpstreamFailure("model_context_limit", status, True, "model", retry_after, "upstream model context limit exceeded")
        return UpstreamFailure("upstream_invalid_request", status, True, "endpoint", retry_after, "upstream rejected translated request")
    if status == 408:
        return UpstreamFailure("upstream_timeout", status, idempotent_retry, "base", retry_after, "upstream timed out")
    if status >= 500:
        return UpstreamFailure("provider_error", status, idempotent_retry, "base", retry_after, "upstream server error")
    return UpstreamFailure("upstream_rejected", status, False, "base", retry_after, "upstream rejected request")


async def _read_limited(response: httpx.Response, limit: int) -> bytes:
    content_length = response.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > limit:
        raise UpstreamFailure("upstream_response_too_large", response.status_code, False, "model", detail_for_logs="upstream response exceeded configured limit")
    content = bytearray()
    async for chunk in response.aiter_bytes():
        if len(content) + len(chunk) > limit:
            raise UpstreamFailure("upstream_response_too_large", response.status_code, False, "model", detail_for_logs="upstream response exceeded configured limit")
        content.extend(chunk)
    return bytes(content)


@dataclass(slots=True)
class UpstreamStream:
    response: httpx.Response
    adapter: BaseAdapter
    request: CanonicalRequest
    single_response: CanonicalResponse | None = None
    closed: bool = False

    async def deltas(self) -> AsyncIterator[CanonicalDelta]:
        if self.single_response is not None:
            item = self.single_response
            if item.content is not None:
                yield CanonicalDelta(content=item.content)
            if item.reasoning_summary:
                yield CanonicalDelta(reasoning_summary=item.reasoning_summary)
            if item.tool_calls:
                yield CanonicalDelta(tool_calls=[
                    {"index": index, "id": call.get("id"), "type": "function", "function": call.get("function", {})}
                    for index, call in enumerate(item.tool_calls)
                ])
            yield CanonicalDelta(finish_reason=item.finish_reason, usage=item.usage, refusal=item.refusal)
            return
        try:
            async for delta in self.adapter.stream_deltas(self.response, self.request):
                yield delta
        except UpstreamFailure:
            raise
        except httpx.TimeoutException as exc:
            raise UpstreamFailure("upstream_read_timeout", None, False, "base", detail_for_logs="upstream stream read timeout") from exc
        except httpx.HTTPError as exc:
            raise UpstreamFailure("upstream_connection_error", None, False, "base", detail_for_logs="upstream stream transport error") from exc
        except AdapterError as exc:
            raise UpstreamFailure("incompatible_stream", self.response.status_code, False, "endpoint", detail_for_logs="upstream stream format was incompatible") from exc

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            await self.response.aclose()


class UpstreamClient:
    def __init__(self, client: httpx.AsyncClient, guard: URLGuard, adapters: AdapterRegistry | None = None):
        self.client = client
        self.guard = guard
        self.adapters = adapters or AdapterRegistry()

    async def complete(
        self,
        candidate: Candidate,
        request: CanonicalRequest,
        public_kind: str,
        defaults: GatewayDefaults,
        request_id: str,
    ) -> CanonicalResponse:
        adapter = self.adapters.for_candidate(candidate)
        try:
            spec = adapter.build(candidate, request, public_kind, False)
        except AdapterError as exc:
            raise UpstreamFailure("unsupported_upstream_format", None, True, "endpoint", detail_for_logs="adapter could not translate request") from exc
        endpoint = candidate.endpoint_path or self._first_endpoint(candidate, spec.endpoint_kind)
        url = endpoint_url(candidate.base.url, endpoint)
        url, headers = _auth_target(candidate, url, spec.headers)
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"
        if candidate.base.supports_idempotency:
            headers[candidate.base.idempotency_header] = request_id
        timeout = httpx.Timeout(defaults.request_timeout_seconds, connect=defaults.connect_timeout_seconds)
        try:
            response = await self.guard.safe_stream(
                self.client,
                "POST",
                url,
                allow_private_network=candidate.base.allow_private_network,
                timeout=timeout,
                headers=headers,
                json=spec.body,
            )
        except UnsafeUpstreamURL as exc:
            raise UpstreamFailure("unsafe_upstream_url", None, False, "base", detail_for_logs="configured upstream URL failed security validation") from exc
        except httpx.ConnectTimeout as exc:
            raise UpstreamFailure("upstream_connect_timeout", None, True, "base", detail_for_logs="upstream connection timed out") from exc
        except httpx.ConnectError as exc:
            raise UpstreamFailure("upstream_connect_error", None, True, "base", detail_for_logs="upstream connection failed") from exc
        except httpx.PoolTimeout as exc:
            raise UpstreamFailure("gateway_pool_timeout", None, True, "base", detail_for_logs="outbound connection pool is full") from exc
        except httpx.TimeoutException as exc:
            raise UpstreamFailure("upstream_timeout", None, candidate.base.supports_idempotency and defaults.retry_ambiguous_failures, "base", detail_for_logs="upstream request timed out") from exc
        except httpx.HTTPError as exc:
            raise UpstreamFailure("upstream_connection_error", None, False, "base", detail_for_logs="upstream transport failed") from exc
        try:
            if response.status_code < 200 or response.status_code >= 300:
                body = await _read_limited(response, MAX_UPSTREAM_ERROR_BYTES)
                raise _classify_http_failure(response.status_code, body, response.headers, candidate, defaults)
            body = await _read_limited(response, MAX_UPSTREAM_JSON_BYTES)
            if not body:
                raise UpstreamFailure("incompatible_response", response.status_code, candidate.base.supports_idempotency and defaults.retry_ambiguous_failures, "model", detail_for_logs="upstream returned an empty response")
            try:
                data = json.loads(body)
            except (ValueError, UnicodeDecodeError) as exc:
                raise UpstreamFailure("incompatible_response", response.status_code, candidate.base.supports_idempotency and defaults.retry_ambiguous_failures, "model", detail_for_logs="upstream response was not valid JSON") from exc
            try:
                return adapter.parse_response(data, request)
            except AdapterError as exc:
                raise UpstreamFailure("incompatible_response", response.status_code, candidate.base.supports_idempotency and defaults.retry_ambiguous_failures, "model", detail_for_logs="upstream response did not match adapter") from exc
        except UpstreamFailure:
            raise
        except httpx.TimeoutException as exc:
            raise UpstreamFailure("upstream_read_timeout", None, candidate.base.supports_idempotency and defaults.retry_ambiguous_failures, "base", detail_for_logs="upstream response read timed out") from exc
        except httpx.HTTPError as exc:
            raise UpstreamFailure("upstream_read_error", None, candidate.base.supports_idempotency and defaults.retry_ambiguous_failures, "base", detail_for_logs="upstream response read failed") from exc
        finally:
            await response.aclose()

    async def begin_stream(
        self,
        candidate: Candidate,
        request: CanonicalRequest,
        public_kind: str,
        defaults: GatewayDefaults,
        request_id: str,
    ) -> UpstreamStream:
        adapter = self.adapters.for_candidate(candidate)
        try:
            spec = adapter.build(candidate, request, public_kind, True)
        except AdapterError as exc:
            raise UpstreamFailure("unsupported_upstream_format", None, True, "endpoint", detail_for_logs="adapter could not translate streaming request") from exc
        endpoint = candidate.endpoint_path or self._first_endpoint(candidate, spec.endpoint_kind)
        url = endpoint_url(candidate.base.url, endpoint)
        url, headers = _auth_target(candidate, url, spec.headers)
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream, application/json"
        headers["Cache-Control"] = "no-cache"
        if candidate.base.supports_idempotency:
            headers[candidate.base.idempotency_header] = request_id
        timeout = httpx.Timeout(defaults.request_timeout_seconds, connect=defaults.connect_timeout_seconds)
        try:
            response = await self.guard.safe_stream(
                self.client,
                "POST",
                url,
                allow_private_network=candidate.base.allow_private_network,
                timeout=timeout,
                headers=headers,
                json=spec.body,
            )
        except UnsafeUpstreamURL as exc:
            raise UpstreamFailure("unsafe_upstream_url", None, False, "base", detail_for_logs="configured upstream URL failed security validation") from exc
        except httpx.ConnectTimeout as exc:
            raise UpstreamFailure("upstream_connect_timeout", None, True, "base", detail_for_logs="upstream connection timed out") from exc
        except httpx.ConnectError as exc:
            raise UpstreamFailure("upstream_connect_error", None, True, "base", detail_for_logs="upstream connection failed") from exc
        except httpx.TimeoutException as exc:
            raise UpstreamFailure("upstream_timeout", None, candidate.base.supports_idempotency and defaults.retry_ambiguous_failures, "base", detail_for_logs="upstream connection timed out") from exc
        except httpx.HTTPError as exc:
            raise UpstreamFailure("upstream_connection_error", None, False, "base", detail_for_logs="upstream transport failed") from exc
        if response.status_code < 200 or response.status_code >= 300:
            try:
                body = await _read_limited(response, MAX_UPSTREAM_ERROR_BYTES)
                raise _classify_http_failure(response.status_code, body, response.headers, candidate, defaults)
            except UpstreamFailure:
                raise
            except httpx.TimeoutException as exc:
                raise UpstreamFailure("upstream_read_timeout", response.status_code, candidate.base.supports_idempotency and defaults.retry_ambiguous_failures, "base", detail_for_logs="upstream error response read timed out") from exc
            finally:
                await response.aclose()
        content_type = response.headers.get("content-type", "").lower()
        if "text/event-stream" in content_type:
            return UpstreamStream(response, adapter, request)
        if "json" in content_type:
            try:
                body = await _read_limited(response, MAX_UPSTREAM_JSON_BYTES)
                data = json.loads(body)
                normalized = adapter.parse_response(data, request)
                return UpstreamStream(response, adapter, request, single_response=normalized)
            except UpstreamFailure:
                await response.aclose()
                raise
            except httpx.TimeoutException as exc:
                await response.aclose()
                raise UpstreamFailure("upstream_read_timeout", response.status_code, candidate.base.supports_idempotency and defaults.retry_ambiguous_failures, "base", detail_for_logs="provider JSON stream read timed out") from exc
            except Exception as exc:
                await response.aclose()
                raise UpstreamFailure("incompatible_response", response.status_code, False, "model", detail_for_logs="provider ignored streaming or returned an incompatible JSON response") from exc
        await response.aclose()
        raise UpstreamFailure("incompatible_stream", response.status_code, False, "endpoint", detail_for_logs="upstream did not return a recognized stream")

    @staticmethod
    def _first_endpoint(candidate: Candidate, kind: str) -> str:
        paths = candidate.endpoints.get(kind, [])
        if not paths:
            raise UpstreamFailure("endpoint_not_found", None, True, "endpoint", detail_for_logs="no discovered endpoint path")
        return paths[0]
