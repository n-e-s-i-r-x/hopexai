from __future__ import annotations

import asyncio
import logging
import secrets
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from .adapters import AdapterError, AdapterRegistry
from .config import BaseURLConfig, GatewayConfig, ProviderConfig
from .discovery import DiscoveryData, DiscoveryService
from .errors import NoEligibleCandidate
from .health import HealthService, HealthSnapshot, as_utc
from .requests import CanonicalRequest
from .types import Candidate

logger = logging.getLogger("hopexai.registry")
PREFERRED_CONTEXT_WINDOW_TOKENS = 1_000_000


class Registry:
    def __init__(self, discovery: DiscoveryService, health: HealthService, adapters: AdapterRegistry | None = None):
        self.discovery = discovery
        self.health = health
        self.adapters = adapters or AdapterRegistry()

    async def candidates(self, config: GatewayConfig, request: CanonicalRequest) -> list[Candidate]:
        now = datetime.now(timezone.utc)
        health_snapshot = await self.health.snapshot()
        providers = [provider for provider in config.providers if provider.enabled]
        discovered = await self._discover_all(providers, config)
        result: list[Candidate] = []
        saw_configured_models = False
        saw_free_candidate = False
        saw_pricing_filtered = False
        saw_capability_miss = False
        saw_context_miss = False

        for provider, base, data in discovered:
            if not base.enabled:
                continue
            provider_state = health_snapshot.providers.get((provider.name, base.url))
            cooldown_until = as_utc(provider_state.cooldown_until) if provider_state else None
            if cooldown_until and cooldown_until > now:
                continue
            effective_format = base.api_format or provider.api_format
            if effective_format == "auto":
                effective_format = data.api_format
            if effective_format == "auto":
                if data.endpoints.get("chat_completions"):
                    effective_format = "openai_chat"
                elif data.endpoints.get("responses"):
                    effective_format = "openai_responses"
                elif data.endpoints.get("messages") and "anthropic" in provider.name.lower():
                    effective_format = "anthropic"
            manual_models = {manual.id: manual for manual in base.manual_models}
            disabled_models = {manual.id for manual in base.manual_models if not manual.enabled}
            model_records = {
                model["id"]: dict(model)
                for model in data.models
                if isinstance(model, dict) and isinstance(model.get("id"), str)
            }
            for manual in base.manual_models:
                if manual.enabled:
                    model_records.setdefault(manual.id, {
                        "id": manual.id,
                        "capabilities": {},
                        "pricing": "unknown",
                        "context_window": None,
                        "selection_weight": manual.selection_weight,
                    })
            for model in model_records.values():
                if not isinstance(model, dict) or not isinstance(model.get("id"), str):
                    continue
                model_id = model["id"]
                manual = manual_models.get(model_id)
                if model_id in base.exclude_models or model_id in disabled_models:
                    continue
                if not base.model_discovery and manual is None:
                    continue
                if base.include_models is not None and model_id not in base.include_models and manual is None:
                    continue
                saw_configured_models = True
                model_health = health_snapshot.models.get((provider.name, base.url, model_id))
                unavailable_until = as_utc(model_health.unavailable_until) if model_health else None
                if unavailable_until and unavailable_until > now:
                    continue

                model_caps = model.get("capabilities") if isinstance(model.get("capabilities"), dict) else {}
                default_caps = base.default_capabilities.as_dict()
                manual_caps = manual.capabilities.as_dict() if manual is not None else {}
                capabilities: dict[str, bool | None] = {}
                for capability_name in set(default_caps) | set(model_caps) | set(manual_caps):
                    model_value = model_caps.get(capability_name)
                    default_value = default_caps.get(capability_name)
                    manual_value = manual_caps.get(capability_name)
                    if manual_value is not None:
                        capabilities[capability_name] = manual_value
                    else:
                        capabilities[capability_name] = model_value if model_value is not None else default_value

                pricing = model.get("pricing") if model.get("pricing") in {"free", "paid", "unknown"} else "unknown"
                if manual is not None and manual.pricing is not None:
                    pricing = manual.pricing
                if config.defaults.free_only or base.free_only:
                    if pricing != "free":
                        saw_pricing_filtered = True
                        continue
                    saw_free_candidate = True
                else:
                    saw_free_candidate = saw_free_candidate or pricing == "free"

                # Chat and response requests share the same normalized gateway
                # implementation. Every required feature must be explicitly true.
                missing = [cap for cap in request.required_capabilities if capabilities.get(cap) is not True]
                if missing:
                    saw_capability_miss = True
                    continue

                context_window = manual.context_window if manual is not None and manual.context_window is not None else model.get("context_window")
                if not isinstance(context_window, int) or isinstance(context_window, bool) or context_window <= 0:
                    context_window = None
                required_tokens = request.estimated_input_tokens() + (request.max_output_tokens or 1024)
                if context_window is not None and required_tokens > context_window:
                    saw_context_miss = True
                    continue

                endpoints = {
                    key: value if isinstance(value, list) else ([value] if isinstance(value, str) else [])
                    for key, value in data.endpoints.items()
                }
                for kind, path in base.endpoint_overrides.as_paths().items():
                    endpoints[kind] = [path] + [item for item in endpoints.get(kind, []) if item != path]
                candidate = Candidate(
                    provider=provider,
                    base=base,
                    model_id=model_id,
                    api_format=effective_format,
                    endpoints=endpoints,
                    capabilities=capabilities,
                    pricing=pricing,
                    context_window=context_window,
                    provider_priority=provider.priority,
                    provider_weight=provider.selection_weight,
                    model_weight=(manual.selection_weight if manual is not None else (float(model.get("selection_weight", 1.0)) if isinstance(model.get("selection_weight", 1.0), (int, float)) else 1.0)),
                )
                try:
                    adapter = self.adapters.for_candidate(candidate)
                    endpoint_kind = adapter.endpoint_kind(candidate, request.kind)
                except AdapterError:
                    saw_capability_miss = True
                    continue
                endpoint_options = endpoints.get(endpoint_kind, [])
                if not endpoint_options:
                    saw_capability_miss = True
                    continue

                auth = base.auth or provider.auth
                if auth.mode == "none":
                    key_values: list[str | None] = [None]
                else:
                    key_values = [item.get_secret_value() for item in base.api_keys]
                eligible_keys: list[tuple[str | None, str]] = []
                for key_value in key_values:
                    fingerprint = self.health.key_fingerprint(key_value)
                    key_state = health_snapshot.keys.get((provider.name, base.url, fingerprint))
                    if key_state:
                        key_cooldown = as_utc(key_state.cooldown_until)
                        if key_state.invalid_at is not None:
                            if base.key_rotation.invalid_key_cooldown_seconds == 0:
                                continue
                            invalid_until = as_utc(key_state.invalid_at)
                            if invalid_until and invalid_until.timestamp() + base.key_rotation.invalid_key_cooldown_seconds > now.timestamp():
                                continue
                        if key_cooldown and key_cooldown > now:
                            continue
                    eligible_keys.append((key_value, fingerprint))
                if not eligible_keys:
                    continue
                candidate.key_options = eligible_keys
                result.append(candidate)

        if result:
            return result
        enabled_bases = [base for provider in config.providers if provider.enabled for base in provider.base_urls if base.enabled]
        free_only_covers_all = config.defaults.free_only or bool(enabled_bases) and all(base.free_only for base in enabled_bases)
        if free_only_covers_all and saw_configured_models and saw_pricing_filtered and not saw_free_candidate:
            raise NoEligibleCandidate(
                "no_free_models_available",
                "No verified-free models are currently available. Paid and unknown-priced models were not selected.",
                503,
            )
        if saw_context_miss:
            raise NoEligibleCandidate(
                "context_length_exceeded",
                "The conversation is too large for the available model context windows. Reduce the input or choose a supported context-management strategy.",
                400,
            )
        if saw_capability_miss:
            caps = ", ".join(sorted(request.required_capabilities))
            raise NoEligibleCandidate(
                "capability_unavailable",
                f"No currently available model is verified to support this request ({caps}).",
                503,
            )
        raise NoEligibleCandidate(
            "upstream_unavailable",
            "No compatible HopeXAI upstream is currently available.",
            503,
        )

    async def _discover_all(self, providers: list[ProviderConfig], config: GatewayConfig):
        jobs = []
        for provider in providers:
            for base in provider.base_urls:
                if base.enabled:
                    jobs.append((provider, base, self.discovery.get(provider, base, config.defaults)))
        if not jobs:
            return []
        results = await asyncio.gather(*(job[2] for job in jobs), return_exceptions=True)
        output = []
        for (provider, base, _), result in zip(jobs, results):
            if isinstance(result, Exception):
                logger.warning("discovery result unavailable provider=%s base=%s error=%s", provider.name, base.url, type(result).__name__)
                result = self.discovery._manual_fallback(provider, base)
            output.append((provider, base, result))
        return output

    def choose(
        self,
        candidates: list[Candidate],
        request: CanonicalRequest,
        *,
        affinity: dict[str, str] | None = None,
        failed_models: set[tuple[str, str, str]] | None = None,
        failed_bases: set[tuple[str, str]] | None = None,
        failed_providers: set[str] | None = None,
        tried_keys: set[tuple[str, str, str]] | None = None,
        excluded_endpoints: dict[tuple[str, str, str], set[str]] | None = None,
    ) -> Candidate | None:
        failed_models = failed_models or set()
        failed_bases = failed_bases or set()
        failed_providers = failed_providers or set()
        tried_keys = tried_keys or set()
        excluded_endpoints = excluded_endpoints or {}
        viable: list[Candidate] = []
        for candidate in candidates:
            provider_key = candidate.provider_name
            base_key = (candidate.provider_name, candidate.base_url)
            model_key = (candidate.provider_name, candidate.base_url, candidate.model_id)
            if provider_key in failed_providers or base_key in failed_bases or model_key in failed_models:
                continue
            adapter = self.adapters.for_candidate(candidate)
            endpoint_kind = adapter.endpoint_kind(candidate, request.kind)
            endpoint_key = (candidate.provider_name, candidate.base_url, endpoint_kind)
            excluded = excluded_endpoints.get(endpoint_key, set())
            available_paths = [path for path in candidate.endpoints.get(endpoint_kind, []) if path not in excluded]
            if not available_paths:
                continue
            eligible_keys = [item for item in candidate.key_options if (candidate.provider_name, candidate.base_url, item[1]) not in tried_keys]
            if not eligible_keys:
                continue
            clone = replace(candidate)
            clone.key_options = eligible_keys
            clone.endpoint_path = secrets.choice(available_paths)
            viable.append(clone)
        if not viable:
            return None

        # Context preference outranks provider priority and affinity: when any
        # compatible, healthy 1M+ model exists, select only within that tier.
        large_context = [
            item for item in viable
            if item.context_window is not None
            and item.context_window >= PREFERRED_CONTEXT_WINDOW_TOKENS
        ]
        if large_context:
            viable = large_context
        else:
            known_context = [item for item in viable if item.context_window is not None]
            if known_context:
                largest_context = max(item.context_window for item in known_context)
                viable = [item for item in known_context if item.context_window == largest_context]

        if affinity:
            same_provider = [item for item in viable if item.provider_name == affinity.get("provider_name")]
            if same_provider:
                viable = same_provider
                same_base = [item for item in viable if item.base_url == affinity.get("base_url")]
                if same_base:
                    viable = same_base
                    same_model = [item for item in viable if item.model_id == affinity.get("model_id")]
                    if same_model:
                        viable = same_model

        best_priority = min(item.provider_priority for item in viable)
        tier = [item for item in viable if item.provider_priority == best_priority]
        provider_groups: dict[str, list[Candidate]] = {}
        for item in tier:
            provider_groups.setdefault(item.provider_name, []).append(item)
        provider_names = list(provider_groups)
        provider_weights = [provider_groups[name][0].provider_weight for name in provider_names]
        selected_provider = secrets.SystemRandom().choices(provider_names, weights=provider_weights, k=1)[0]
        provider_items = provider_groups[selected_provider]

        base_groups: dict[str, list[Candidate]] = {}
        for item in provider_items:
            base_groups.setdefault(item.base_url, []).append(item)
        base_urls = list(base_groups)
        base_weights = [base_groups[url][0].base.selection_weight for url in base_urls]
        selected_base = secrets.SystemRandom().choices(base_urls, weights=base_weights, k=1)[0]
        base_items = base_groups[selected_base]
        model = secrets.SystemRandom().choices(base_items, weights=[item.model_weight for item in base_items], k=1)[0]
        key, fingerprint = secrets.choice(model.key_options)
        model.api_key = key
        model.key_fingerprint = fingerprint
        return model

    def endpoint_kind(self, candidate: Candidate, public_kind: str) -> str:
        return self.adapters.for_candidate(candidate).endpoint_kind(candidate, public_kind)
