from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class APIKeyRecord(Base):
    __tablename__ = "api_keys"
    __table_args__ = (
        Index("ix_api_keys_prefix", "prefix"),
        Index("ix_api_keys_revoked", "revoked_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    secret_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    prefix: Mapped[str] = mapped_column(String(24), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    requests_total: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rate_window_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rate_window_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class KeyIssuanceEvent(Base):
    __tablename__ = "key_issuance_events"
    __table_args__ = (Index("ix_key_issuance_ip_created", "ip_digest", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ip_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class KeyIssueThrottle(Base):
    __tablename__ = "key_issue_throttles"

    ip_digest: Mapped[str] = mapped_column(String(64), primary_key=True)
    hour_started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    hour_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    day_started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    day_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class ProviderHealth(Base):
    __tablename__ = "provider_health"
    __table_args__ = (Index("ix_provider_health_cooldown", "cooldown_until"),)

    provider_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    base_url: Mapped[str] = mapped_column(String(2048), primary_key=True)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cooldown_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class ModelHealth(Base):
    __tablename__ = "model_health"
    __table_args__ = (Index("ix_model_health_unavailable", "unavailable_until"),)

    provider_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    base_url: Mapped[str] = mapped_column(String(2048), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    unavailable_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class KeyHealth(Base):
    __tablename__ = "upstream_key_health"
    __table_args__ = (Index("ix_upstream_key_health_cooldown", "cooldown_until"),)

    provider_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    base_url: Mapped[str] = mapped_column(String(2048), primary_key=True)
    key_fingerprint: Mapped[str] = mapped_column(String(64), primary_key=True)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cooldown_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    invalid_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class DiscoveryCache(Base):
    __tablename__ = "discovery_cache"
    __table_args__ = (Index("ix_discovery_expires", "expires_at"),)

    provider_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    base_url: Mapped[str] = mapped_column(String(2048), primary_key=True)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    endpoints: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    api_format: Mapped[str] = mapped_column(String(32), default="auto", nullable=False)
    models: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    provider_capabilities: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    discovered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)


class DiscoveryLease(Base):
    __tablename__ = "discovery_leases"

    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner: Mapped[str | None] = mapped_column(String(36), nullable=True)
    lease_until: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class ConversationAffinity(Base):
    __tablename__ = "conversation_affinity"
    __table_args__ = (
        Index("ix_affinity_expires", "expires_at"),
        Index("ix_affinity_key_owner", "api_key_id"),
    )

    affinity_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    api_key_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    provider_name: Mapped[str] = mapped_column(String(64), nullable=False)
    base_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    model_id: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ResponseState(Base):
    __tablename__ = "response_states"
    __table_args__ = (
        Index("ix_response_states_key_expires", "api_key_id", "expires_at"),
        UniqueConstraint("id", "api_key_id", name="uq_response_state_key_owner"),
    )

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    api_key_id: Mapped[str] = mapped_column(ForeignKey("api_keys.id", ondelete="CASCADE"), nullable=False)
    encrypted_messages: Mapped[str] = mapped_column(Text, nullable=False)
    provider_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    base_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    model_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class RequestStat(Base):
    __tablename__ = "request_stats"
    __table_args__ = (
        Index("ix_request_stats_created", "created_at"),
        Index("ix_request_stats_key_created", "api_key_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    request_id: Mapped[str] = mapped_column(String(36), nullable=False)
    api_key_id: Mapped[str | None] = mapped_column(ForeignKey("api_keys.id", ondelete="SET NULL"), nullable=True)
    request_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    provider_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    base_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    model_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    capability_set: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
