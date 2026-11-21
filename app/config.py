from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import os
import re
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator, model_validator

from .secret_store import SecretStore
from .settings import Settings

logger = logging.getLogger("hopexai.config")
ApiFormat = Literal["auto", "openai_chat", "openai_responses", "anthropic"]
PricingStatus = Literal["free", "paid", "unknown"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class CapabilitySet(StrictModel):
    chat: bool | None = None
    responses: bool | None = None
    streaming: bool | None = None
    vision: bool | None = None
    image_generation: bool | None = None
    audio_transcription: bool | None = None
    audio_generation: bool | None = None
    function_calling: bool | None = None
    structured_output: bool | None = None
    moderation: bool | None = None
    reranking: bool | None = None
    file_operations: bool | None = None
    embeddings: bool | None = None
    reasoning: bool | None = None
    reasoning_summary: bool | None = None

    def as_dict(self) -> dict[str, bool | None]:
        return self.model_dump()


class AuthConfig(StrictModel):
    mode: Literal["bearer", "header", "query", "none"] = "bearer"
    header_name: str = "Authorization"
    prefix: str = "Bearer "
    query_parameter: str = "key"

    @field_validator("header_name")
    @classmethod
    def valid_header(cls, value: str) -> str:
        if not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", value):
            raise ValueError("header_name is not a valid HTTP header name")
        return value

    @field_validator("query_parameter")
    @classmethod
    def valid_query_parameter(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value):
            raise ValueError("query_parameter must be a simple parameter name")
        return value

    @field_validator("prefix")
    @classmethod
    def valid_prefix(cls, value: str) -> str:
        if "\r" in value or "\n" in value or len(value) > 80:
            raise ValueError("prefix contains invalid header characters")
        return value


class ReasoningMapping(StrictModel):
    enabled_parameter: str | None = None
    effort_parameter: str | None = "reasoning_effort"
    summary_parameter: str | None = None
    disabled_effort_value: str | None = "none"
    effort_map: dict[str, str] = Field(default_factory=lambda: {
        "minimal": "minimal", "low": "low", "medium": "medium",
        "high": "high", "extra_high": "xhigh", "maximum": "max",
    })

    @field_validator("enabled_parameter", "effort_parameter", "summary_parameter")
    @classmethod
    def valid_parameter(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", value):
            raise ValueError("reasoning parameter must be a simple field name")
        return value

    @field_validator("disabled_effort_value")
    @classmethod
    def valid_disabled_value(cls, value: str | None) -> str | None:
        if value is not None and (not value or len(value) > 40 or any(ord(char) < 32 for char in value)):
            raise ValueError("disabled_effort_value must be a short single-line string")
        return value

    @field_validator("effort_map")
    @classmethod
    def valid_effort_map(cls, value: dict[str, str]) -> dict[str, str]:
        allowed = {"minimal", "low", "medium", "high", "extra_high", "maximum"}
        if any(key not in allowed or not isinstance(mapped, str) or not mapped or len(mapped) > 40 for key, mapped in value.items()):
            raise ValueError("effort_map must map supported effort names to short strings")
        return value


class EndpointOverrides(StrictModel):
    models: str | None = None
    chat_completions: str | None = None
    responses: str | None = None
    messages: str | None = None
    embeddings: str | None = None
    image_generation: str | None = None
    audio_transcriptions: str | None = None
    audio_speech: str | None = None
    moderations: str | None = None
    rerank: str | None = None
    files: str | None = None

    @model_validator(mode="after")
    def validate_paths(self) -> "EndpointOverrides":
        for field_name, value in self.model_dump().items():
            if value is None:
                continue
            parsed = urlsplit(value)
            if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
                raise ValueError(f"endpoint_overrides.{field_name} must be a path, not a URL or query")
            path = value.strip()
            if (
                not path
                or path.startswith("//")
                or not re.fullmatch(r"/?[A-Za-z0-9._~/-]+", path)
                or any(part in {".", ".."} for part in path.split("/"))
            ):
                raise ValueError(f"endpoint_overrides.{field_name} is not a safe relative path")
        return self

    def as_paths(self) -> dict[str, str]:
        return {k: v for k, v in self.model_dump().items() if v}


class ManualModel(StrictModel):
    id: str = Field(min_length=1, max_length=256)
    enabled: bool = True
    selection_weight: float = Field(default=1.0, gt=0, le=1000)
    pricing: PricingStatus | None = None
    context_window: int | None = Field(default=None, gt=0, le=10_000_000)
    capabilities: CapabilitySet = Field(default_factory=CapabilitySet)

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if any(ord(char) < 32 for char in value) or value.strip() != value:
            raise ValueError("model ID contains invalid whitespace or control characters")
        return value


class KeyRotation(StrictModel):
    rotate_on_rate_limit: bool = True
    invalid_key_cooldown_seconds: int = Field(default=0, ge=0, le=31_536_000)


class BaseURLConfig(StrictModel):
    allow_private_network: bool = False
    url: str = Field(min_length=8, max_length=2048)
    enabled: bool = True
    selection_weight: float = Field(default=1.0, gt=0, le=1000)
    api_keys: list[SecretStr] = Field(default_factory=list, max_length=100)
    auth: AuthConfig | None = None
    api_format: ApiFormat | None = None
    model_discovery: bool = True
    free_only: bool = False
    include_models: list[str] | None = None
    exclude_models: list[str] = Field(default_factory=list)
    default_capabilities: CapabilitySet = Field(default_factory=CapabilitySet)
    manual_models: list[ManualModel] = Field(default_factory=list, max_length=1000)
    endpoint_overrides: EndpointOverrides = Field(default_factory=EndpointOverrides)
    key_rotation: KeyRotation = Field(default_factory=KeyRotation)
    supports_idempotency: bool = False
    idempotency_header: str = "Idempotency-Key"

    @field_validator("url")
    @classmethod
    def validate_base_url(cls, value: str, info: Any) -> str:
        parsed = urlsplit(value)
        allow_private = bool(info.data.get("allow_private_network", False))
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            raise ValueError("url must be an absolute http(s) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("credentials must not be embedded in url")
        try:
            if parsed.port is not None and not 1 <= parsed.port <= 65535:
                raise ValueError("url port is outside the valid range")
        except ValueError as exc:
            if "outside the valid range" in str(exc):
                raise
            raise ValueError("url contains an invalid port") from exc
        if parsed.query or parsed.fragment:
            raise ValueError("url must not contain query parameters or a fragment")
        if parsed.scheme == "http" and not allow_private:
            raise ValueError("plain HTTP is allowed only with allow_private_network: true")
        host = parsed.hostname.rstrip(".").lower()
        if host in {"metadata.google.internal", "metadata"}:
            raise ValueError("cloud metadata hosts are forbidden")
        if host == "localhost" and not allow_private:
            raise ValueError("localhost requires an explicit private-network opt-in")
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            ip = None
        if ip is not None:
            if ip in {ipaddress.ip_address("169.254.169.254"), ipaddress.ip_address("169.254.170.2")}:
                raise ValueError("cloud metadata service addresses are forbidden")
            if not allow_private and not ip.is_global:
                raise ValueError("non-public IP literals require allow_private_network: true")
        return value.rstrip("/")

    @field_validator("include_models", "exclude_models")
    @classmethod
    def valid_model_lists(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if len(value) > 10_000 or any(not item or len(item) > 256 or any(ord(c) < 32 for c in item) for item in value):
            raise ValueError("model lists contain an invalid ID or are too large")
        return value

    @field_validator("api_keys")
    @classmethod
    def unique_keys(cls, value: list[SecretStr]) -> list[SecretStr]:
        raw = [item.get_secret_value() for item in value]
        if any(not item.strip() or "\r" in item or "\n" in item for item in raw):
            raise ValueError("api_keys must contain non-empty single-line values")
        if len(set(raw)) != len(raw):
            raise ValueError("duplicate API keys are not allowed in one base URL")
        return value

    @field_validator("idempotency_header")
    @classmethod
    def valid_idempotency_header(cls, value: str) -> str:
        if not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", value):
            raise ValueError("idempotency_header is not a valid HTTP header name")
        return value

    @model_validator(mode="after")
    def auth_has_credentials(self) -> "BaseURLConfig":
        # The provider-level auth mode may be overridden later; credential checks
        # are completed by ProviderConfig once provider auth is known.
        ids = [model.id for model in self.manual_models]
        if len(ids) != len(set(ids)):
            raise ValueError("manual_models contains duplicate IDs")
        if len(self.exclude_models) != len(set(self.exclude_models)):
            raise ValueError("exclude_models contains duplicate IDs")
        return self


class ProviderConfig(StrictModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    enabled: bool = True
    priority: int = Field(default=100, ge=0, le=100_000)
    selection_weight: float = Field(default=1.0, gt=0, le=1000)
    api_format: ApiFormat = "auto"
    auth: AuthConfig = Field(default_factory=AuthConfig)
    reasoning: ReasoningMapping = Field(default_factory=ReasoningMapping)
    base_urls: list[BaseURLConfig] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_base_urls(self) -> "ProviderConfig":
        urls = [item.url.lower().rstrip("/") for item in self.base_urls]
        if len(urls) != len(set(urls)):
            raise ValueError("base_urls contains duplicate URLs")
        for base in self.base_urls:
            auth = base.auth or self.auth
            if auth.mode != "none" and base.enabled and not base.api_keys:
                raise ValueError(f"enabled base URL {base.url} requires at least one api_key")
        return self


class GatewayDefaults(StrictModel):
    default_max_output_tokens: int = Field(default=1024, ge=1, le=32768)
    max_output_tokens_limit: int = Field(default=32768, ge=1, le=1_000_000)
    request_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    connect_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    discovery_timeout_seconds: float = Field(default=8.0, gt=0, le=60)
    discovery_ttl_seconds: int = Field(default=1800, ge=30, le=604800)
    discovery_lease_seconds: int = Field(default=45, ge=10, le=300)
    max_attempts: int = Field(default=5, ge=1, le=20)
    provider_failure_cooldown_seconds: int = Field(default=30, ge=1, le=3600)
    model_failure_cooldown_seconds: int = Field(default=60, ge=1, le=3600)
    key_rate_limit_cooldown_seconds: int = Field(default=45, ge=1, le=3600)
    account_quota_cooldown_seconds: int = Field(default=300, ge=1, le=86400)
    conversation_state_ttl_seconds: int = Field(default=86400, ge=60, le=2_592_000)
    free_only: bool = False
    retry_ambiguous_failures: bool = False

    @model_validator(mode="after")
    def output_default_within_limit(self) -> "GatewayDefaults":
        if self.default_max_output_tokens > self.max_output_tokens_limit:
            raise ValueError("default_max_output_tokens must not exceed max_output_tokens_limit")
        return self


class GatewayConfig(StrictModel):
    schema_version: Literal[1] = 1
    defaults: GatewayDefaults = Field(default_factory=GatewayDefaults)
    providers: list[ProviderConfig] = Field(default_factory=list, max_length=1000)

    @model_validator(mode="after")
    def unique_provider_names(self) -> "GatewayConfig":
        names = [provider.name for provider in self.providers]
        if len(names) != len(set(names)):
            raise ValueError("provider names must be unique")
        return self


@dataclass(frozen=True, slots=True)
class ConfigSnapshot:
    config: GatewayConfig
    content_hash: str
    loaded_at: float


class ConfigurationError(RuntimeError):
    def __init__(self, message: str, details: tuple[str, ...] = ()):
        super().__init__(message)
        self.details = details


class ConfigManager:
    def __init__(self, settings: Settings, secret_store: SecretStore):
        self.settings = settings
        self.secret_store = secret_store
        self._snapshot: ConfigSnapshot | None = None
        self._last_check = 0.0
        self._lock = asyncio.Lock()
        self.last_rejection: tuple[str, ...] = ()

    @property
    def snapshot(self) -> ConfigSnapshot:
        if self._snapshot is None:
            raise ConfigurationError("configuration has not been loaded")
        return self._snapshot

    async def load_initial(self) -> ConfigSnapshot:
        self.secret_store.ensure()
        try:
            raw = await asyncio.to_thread(self._read_config_bytes)
        except ConfigurationError:
            raise
        except OSError as exc:
            raise ConfigurationError("configuration file is missing or unreadable") from exc
        parsed = self._parse_and_validate(raw)
        self._snapshot = ConfigSnapshot(parsed, hashlib.sha256(raw).hexdigest(), time.monotonic())
        self._last_check = time.monotonic()
        logger.info("configuration loaded: providers=%d", len(parsed.providers))
        return self._snapshot

    async def refresh_if_changed(self, force: bool = False) -> ConfigSnapshot:
        now = time.monotonic()
        if not force and now - self._last_check < self.settings.config_poll_interval:
            return self.snapshot
        async with self._lock:
            now = time.monotonic()
            if not force and now - self._last_check < self.settings.config_poll_interval:
                return self.snapshot
            self._last_check = now
            try:
                raw = await asyncio.to_thread(self._read_config_bytes)
            except (OSError, ConfigurationError):
                if self._snapshot is None:
                    raise ConfigurationError("configuration file is missing or unreadable")
                logger.error("configuration reload rejected: file is missing, unreadable, or has unsafe permissions; retaining last known-good configuration")
                self.last_rejection = ("configuration file is missing, unreadable, or has unsafe permissions",)
                return self._snapshot
            digest = hashlib.sha256(raw).hexdigest()
            if self._snapshot is not None and digest == self._snapshot.content_hash:
                return self._snapshot
            try:
                parsed = self._parse_and_validate(raw)
            except ConfigurationError as exc:
                self.last_rejection = exc.details or (str(exc),)
                logger.error("configuration reload rejected; retaining last known-good configuration (%s)", "; ".join(self.last_rejection))
                return self.snapshot
            self._snapshot = ConfigSnapshot(parsed, digest, time.monotonic())
            self.last_rejection = ()
            logger.info("configuration reloaded: providers=%d", len(parsed.providers))
            return self._snapshot

    def _read_config_bytes(self) -> bytes:
        path = self.settings.config_path
        if path.is_symlink():
            raise ConfigurationError("configuration file must not be a symbolic link", ("configuration file must not be a symbolic link",))
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ConfigurationError("configuration path must be a regular file", ("configuration path must be a regular file",))
        if os.name == "posix" and info.st_mode & 0o077:
            try:
                os.chmod(path, 0o600)
            except OSError as exc:
                raise ConfigurationError("configuration file permissions are too broad", ("configuration file permissions must be owner-only",)) from exc
            info = path.stat()
            if info.st_mode & 0o077:
                raise ConfigurationError("configuration file permissions are too broad", ("configuration file permissions must be owner-only",))
        return path.read_bytes()

    def _parse_and_validate(self, raw: bytes) -> GatewayConfig:
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConfigurationError("configuration file must be UTF-8", ("configuration file must be UTF-8",)) from exc
        try:
            data = yaml.safe_load(decoded)
        except yaml.YAMLError as exc:
            raise ConfigurationError("configuration YAML could not be parsed", ("configuration YAML could not be parsed",)) from exc
        if not isinstance(data, dict):
            raise ConfigurationError("configuration root must be a YAML mapping", ("configuration root must be a YAML mapping",))
        self._decrypt_provider_keys(data)
        try:
            return GatewayConfig.model_validate(data)
        except ValidationError as exc:
            safe_errors = tuple(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors(include_input=False)
            )
            raise ConfigurationError("configuration validation failed", safe_errors) from exc
        except Exception as exc:
            raise ConfigurationError("configuration validation failed", ("configuration validation failed",)) from exc

    def _decrypt_provider_keys(self, data: dict[str, Any]) -> None:
        providers = data.get("providers", [])
        if not isinstance(providers, list):
            return
        for provider in providers:
            if not isinstance(provider, dict):
                continue
            bases = provider.get("base_urls", [])
            if not isinstance(bases, list):
                continue
            for base in bases:
                if not isinstance(base, dict):
                    continue
                keys = base.get("api_keys", [])
                if isinstance(keys, list):
                    try:
                        base["api_keys"] = [self.secret_store.decrypt_config_secret(key) if isinstance(key, str) else key for key in keys]
                    except Exception as exc:
                        # Keep the reason deliberately value-free; encrypted material
                        # must never be copied into logs or validation responses.
                        raise ConfigurationError("encrypted provider credential is invalid", ("encrypted provider credential could not be decrypted",)) from exc

    def redacted_dump(self) -> dict[str, Any]:
        data = self.snapshot.config.model_dump(mode="json")
        for provider in data.get("providers", []):
            for base in provider.get("base_urls", []):
                if base.get("api_keys"):
                    base["api_keys"] = ["[REDACTED]" for _ in base["api_keys"]]
        return data
