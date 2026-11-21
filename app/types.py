from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import BaseURLConfig, ProviderConfig


@dataclass(slots=True)
class Candidate:
    provider: ProviderConfig
    base: BaseURLConfig
    model_id: str
    api_format: str
    endpoints: dict[str, list[str]]
    capabilities: dict[str, bool | None]
    pricing: str
    context_window: int | None
    provider_priority: int
    provider_weight: float
    model_weight: float
    api_key: str | None = field(default=None, repr=False)
    key_fingerprint: str = "none"
    key_options: list[tuple[str | None, str]] = field(default_factory=list, repr=False)
    endpoint_path: str | None = None

    @property
    def provider_name(self) -> str:
        return self.provider.name

    @property
    def base_url(self) -> str:
        return self.base.url


@dataclass(slots=True)
class CanonicalResponse:
    content: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning_summary: str | None = None
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    refusal: str | None = None


@dataclass(slots=True)
class CanonicalDelta:
    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning_summary: str | None = None
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    refusal: str | None = None

    @property
    def has_output(self) -> bool:
        return bool(self.content or self.reasoning_summary or self.tool_calls or self.finish_reason or self.refusal)
