from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import AsyncIterator

from .config import GatewayConfig
from .errors import NoEligibleCandidate, UpstreamFailure
from .health import HealthService
from .registry import Registry
from .requests import CanonicalRequest
from .runtime_state import RuntimeState
from .types import Candidate, CanonicalDelta, CanonicalResponse
from .upstream import UpstreamClient

logger = logging.getLogger("hopexai.router")


@dataclass(slots=True)
class CompletionResult:
    response: CanonicalResponse
    candidate: Candidate


class GatewayRouter:
    def __init__(self, registry: Registry, upstream: UpstreamClient, health: HealthService, state: RuntimeState):
        self.registry = registry
        self.upstream = upstream
        self.health = health
        self.state = state

    async def complete(
        self,
        config: GatewayConfig,
        request: CanonicalRequest,
        request_id: str,
        api_key_id: str,
        affinity_key: str | None,
    ) -> CompletionResult:
        candidates = await self.registry.candidates(config, request)
        affinity = await self.state.load_affinity(affinity_key)
        failed_models: set[tuple[str, str, str]] = set()
        failed_bases: set[tuple[str, str]] = set()
        failed_providers: set[str] = set()
        tried_keys: set[tuple[str, str, str]] = set()
        excluded_endpoints: dict[tuple[str, str, str], set[str]] = {}
        last_failure: UpstreamFailure | None = None
        defaults = config.defaults

        for attempt in range(defaults.max_attempts):
            candidate = self.registry.choose(
                candidates,
                request,
                affinity=affinity,
                failed_models=failed_models,
                failed_bases=failed_bases,
                failed_providers=failed_providers,
                tried_keys=tried_keys,
                excluded_endpoints=excluded_endpoints,
            )
            if candidate is None:
                break
            try:
                response = await self.upstream.complete(candidate, request, request.kind, defaults, request_id)
                await self.health.record_success(candidate.provider_name, candidate.base_url, candidate.model_id, candidate.api_key)
                return CompletionResult(response, candidate)
            except UpstreamFailure as failure:
                last_failure = failure
                await self._record_failure(candidate, failure, defaults)
                self._exclude_failure(
                    candidate, request, failure, failed_models, failed_bases,
                    failed_providers, tried_keys, excluded_endpoints,
                )
                logger.warning(
                    "upstream attempt failed request_id=%s provider=%s base=%s model=%s category=%s attempt=%d",
                    request_id, candidate.provider_name, candidate.base_url, candidate.model_id, failure.error_code, attempt + 1,
                )
                if not failure.retry_safe:
                    break
        if last_failure is not None:
            raise self._public_fallback_error(last_failure)
        raise NoEligibleCandidate(
            "upstream_unavailable",
            "No compatible HopeXAI upstream is currently available.",
            503,
        )

    async def stream(
        self,
        config: GatewayConfig,
        request: CanonicalRequest,
        request_id: str,
        api_key_id: str,
        affinity_key: str | None,
    ) -> AsyncIterator[tuple[Candidate, CanonicalDelta]]:
        candidates = await self.registry.candidates(config, request)
        affinity = await self.state.load_affinity(affinity_key)
        failed_models: set[tuple[str, str, str]] = set()
        failed_bases: set[tuple[str, str]] = set()
        failed_providers: set[str] = set()
        tried_keys: set[tuple[str, str, str]] = set()
        excluded_endpoints: dict[tuple[str, str, str], set[str]] = {}
        last_failure: UpstreamFailure | None = None
        defaults = config.defaults

        for attempt in range(defaults.max_attempts):
            candidate = self.registry.choose(
                candidates,
                request,
                affinity=affinity,
                failed_models=failed_models,
                failed_bases=failed_bases,
                failed_providers=failed_providers,
                tried_keys=tried_keys,
                excluded_endpoints=excluded_endpoints,
            )
            if candidate is None:
                break
            session = None
            emitted = False
            finished = False
            pending_usage: dict[str, int] = {}
            last_yielded_usage: dict[str, int] = {}
            try:
                session = await self.upstream.begin_stream(candidate, request, request.kind, defaults, request_id)
                async for delta in session.deltas():
                    if delta.usage:
                        pending_usage.update(delta.usage)
                        if "prompt_tokens" in pending_usage or "completion_tokens" in pending_usage:
                            pending_usage["total_tokens"] = pending_usage.get("prompt_tokens", 0) + pending_usage.get("completion_tokens", 0)
                    if not delta.has_output:
                        continue
                    if not emitted:
                        emitted = True
                        await self.health.record_success(candidate.provider_name, candidate.base_url, candidate.model_id, candidate.api_key)
                    if delta.usage:
                        delta.usage = dict(pending_usage)
                        last_yielded_usage = dict(pending_usage)
                    if delta.finish_reason:
                        finished = True
                    yield candidate, delta
                if not emitted:
                    # Preserve provider usage/termination metadata but do not
                    # send speculative role chunks before a stream has output.
                    emitted = True
                    await self.health.record_success(candidate.provider_name, candidate.base_url, candidate.model_id, candidate.api_key)
                    yield candidate, CanonicalDelta(finish_reason="stop", usage=pending_usage)
                elif not finished:
                    # A provider may close the SSE stream without an explicit
                    # finish event; terminate the public protocol cleanly. Do
                    # not duplicate usage already delivered on an earlier chunk.
                    final_usage = pending_usage if pending_usage != last_yielded_usage else {}
                    yield candidate, CanonicalDelta(finish_reason="stop", usage=final_usage)
                elif pending_usage and pending_usage != last_yielded_usage:
                    # Usage can arrive in a separate, post-finish SSE event.
                    yield candidate, CanonicalDelta(usage=pending_usage)
                return
            except UpstreamFailure as failure:
                last_failure = failure
                # A read timeout can be replayed only when an operator explicitly
                # enabled idempotent retries and the provider is configured to
                # honor the same idempotency key. It is never switched mid-output.
                if (
                    failure.error_code in {"upstream_read_timeout", "incompatible_stream", "incompatible_response"}
                    and candidate.base.supports_idempotency
                    and defaults.retry_ambiguous_failures
                ):
                    failure.retry_safe = True
                if emitted:
                    await self._record_failure(candidate, failure, defaults)
                    logger.warning(
                        "stream failed after output request_id=%s provider=%s base=%s model=%s category=%s",
                        request_id, candidate.provider_name, candidate.base_url, candidate.model_id, failure.error_code,
                    )
                    raise self._public_fallback_error(failure)
                await self._record_failure(candidate, failure, defaults)
                self._exclude_failure(
                    candidate, request, failure, failed_models, failed_bases,
                    failed_providers, tried_keys, excluded_endpoints,
                )
                logger.warning(
                    "stream setup failed request_id=%s provider=%s base=%s model=%s category=%s attempt=%d",
                    request_id, candidate.provider_name, candidate.base_url, candidate.model_id, failure.error_code, attempt + 1,
                )
                if not failure.retry_safe:
                    break
            finally:
                if session is not None:
                    await session.close()
        if last_failure is not None:
            raise self._public_fallback_error(last_failure)
        raise NoEligibleCandidate(
            "upstream_unavailable",
            "No compatible HopeXAI upstream is currently available.",
            503,
        )

    async def _record_failure(self, candidate: Candidate, failure: UpstreamFailure, defaults) -> None:
        if failure.scope == "endpoint":
            return
        await self.health.record_failure(
            candidate.provider_name,
            candidate.base_url,
            candidate.model_id,
            candidate.api_key,
            failure,
            provider_cooldown_seconds=defaults.provider_failure_cooldown_seconds,
            model_cooldown_seconds=defaults.model_failure_cooldown_seconds,
            key_cooldown_seconds=defaults.key_rate_limit_cooldown_seconds,
            account_quota_cooldown_seconds=defaults.account_quota_cooldown_seconds,
            invalid_key_cooldown_seconds=candidate.base.key_rotation.invalid_key_cooldown_seconds,
        )

    def _exclude_failure(
        self,
        candidate: Candidate,
        request: CanonicalRequest,
        failure: UpstreamFailure,
        failed_models: set[tuple[str, str, str]],
        failed_bases: set[tuple[str, str]],
        failed_providers: set[str],
        tried_keys: set[tuple[str, str, str]],
        excluded_endpoints: dict[tuple[str, str, str], set[str]],
    ) -> None:
        if failure.scope == "key":
            if failure.error_code == "rate_limited" and not candidate.base.key_rotation.rotate_on_rate_limit:
                failed_bases.add((candidate.provider_name, candidate.base_url))
            else:
                tried_keys.add((candidate.provider_name, candidate.base_url, candidate.key_fingerprint))
        elif failure.scope == "model":
            failed_models.add((candidate.provider_name, candidate.base_url, candidate.model_id))
        elif failure.scope == "base":
            failed_bases.add((candidate.provider_name, candidate.base_url))
        elif failure.scope == "provider":
            failed_providers.add(candidate.provider_name)
        elif failure.scope == "endpoint":
            kind = self.registry.endpoint_kind(candidate, request.kind)
            key = (candidate.provider_name, candidate.base_url, kind)
            excluded_endpoints.setdefault(key, set()).add(candidate.endpoint_path or "")

    @staticmethod
    def _public_fallback_error(failure: UpstreamFailure) -> NoEligibleCandidate:
        if failure.error_code == "model_context_limit":
            return NoEligibleCandidate(
                "context_length_exceeded",
                "The conversation exceeds the context limit of the available compatible models.",
                400,
            )
        if failure.error_code == "upstream_invalid_request":
            return NoEligibleCandidate(
                "upstream_request_rejected",
                "The request could not be processed by an available compatible model. Check the request format and supported options.",
                400,
            )
        if failure.error_code == "upstream_response_too_large":
            return NoEligibleCandidate(
                "upstream_response_too_large",
                "The upstream response exceeded HopeXAI's safety limit.",
                502,
            )
        return NoEligibleCandidate(
            "upstream_unavailable",
            "No compatible HopeXAI upstream could complete the request.",
            503,
        )
