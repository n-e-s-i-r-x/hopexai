from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .errors import PublicAPIError
from .models import (
    APIKeyRecord,
    ConversationAffinity,
    DiscoveryCache,
    DiscoveryLease,
    KeyHealth,
    ModelHealth,
    ProviderHealth,
    ResponseState,
    RequestStat,
    utcnow,
)
from .secret_store import SecretStore


def _insert_for(session: AsyncSession, model):
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        return pg_insert(model)
    if dialect == "sqlite":
        return sqlite_insert(model)
    raise RuntimeError(f"unsupported database dialect: {dialect}")


class RuntimeState:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        secret_store: SecretStore,
        state_max_bytes: int,
    ):
        self.sessions = sessions
        self.secret_store = secret_store
        self.state_max_bytes = state_max_bytes

    def affinity_digest(self, api_key_id: str, conversation_id: str | None) -> str | None:
        if not conversation_id:
            return None
        return self.secret_store.digest(api_key_id + "\0" + conversation_id, "conversation-affinity")

    async def load_affinity(self, affinity_key: str | None) -> dict[str, str] | None:
        if not affinity_key:
            return None
        now = utcnow()
        async with self.sessions() as session:
            record = await session.get(ConversationAffinity, affinity_key)
            if record is None or _as_utc(record.expires_at) <= now:
                return None
            return {"provider_name": record.provider_name, "base_url": record.base_url, "model_id": record.model_id}

    async def save_affinity(
        self,
        affinity_key: str | None,
        api_key_id: str,
        provider_name: str,
        base_url: str,
        model_id: str,
        ttl_seconds: int,
    ) -> None:
        if not affinity_key:
            return
        now = utcnow()
        values = {
            "affinity_key": affinity_key,
            "api_key_id": api_key_id,
            "provider_name": provider_name,
            "base_url": base_url,
            "model_id": model_id,
            "updated_at": now,
            "expires_at": now + timedelta(seconds=ttl_seconds),
        }
        async with self.sessions() as session:
            async with session.begin():
                owner = await session.scalar(
                    select(APIKeyRecord).where(APIKeyRecord.id == api_key_id).with_for_update()
                )
                if owner is None or owner.revoked_at is not None:
                    return
                stmt = _insert_for(session, ConversationAffinity).values(**values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[ConversationAffinity.affinity_key],
                    set_={key: value for key, value in values.items() if key != "affinity_key"},
                )
                await session.execute(stmt)

    async def load_response_messages(self, response_id: str, api_key_id: str) -> list[dict[str, Any]]:
        now = utcnow()
        async with self.sessions() as session:
            record = await session.scalar(
                select(ResponseState).where(
                    ResponseState.id == response_id,
                    ResponseState.api_key_id == api_key_id,
                    ResponseState.expires_at > now,
                )
            )
        if record is None:
            raise PublicAPIError(404, "The referenced HopeXAI response was not found or has expired.", "invalid_request_error", "response_not_found", "previous_response_id")
        try:
            raw = self.secret_store.decrypt_blob(record.encrypted_messages)
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError("invalid stored messages")
            return parsed
        except Exception as exc:
            raise PublicAPIError(500, "Stored conversation state is unavailable.", "server_error", "response_state_unavailable") from exc

    async def save_response_messages(
        self,
        response_id: str,
        api_key_id: str,
        messages: list[dict[str, Any]],
        ttl_seconds: int,
        provider_name: str | None,
        base_url: str | None,
        model_id: str | None,
    ) -> bool:
        serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(serialized) > self.state_max_bytes:
            return False
        now = utcnow()
        state = ResponseState(
            id=response_id,
            api_key_id=api_key_id,
            encrypted_messages=self.secret_store.encrypt_blob(serialized),
            provider_name=provider_name,
            base_url=base_url,
            model_id=model_id,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
        )
        async with self.sessions() as session:
            async with session.begin():
                owner = await session.scalar(
                    select(APIKeyRecord).where(APIKeyRecord.id == api_key_id).with_for_update()
                )
                if owner is None or owner.revoked_at is not None:
                    return False
                session.add(state)
        return True

    async def log_request(
        self,
        request_id: str,
        api_key_id: str | None,
        request_kind: str,
        status: str,
        provider_name: str | None,
        base_url: str | None,
        model_id: str | None,
        capabilities: list[str],
        latency_ms: int,
        error_code: str | None = None,
    ) -> None:
        async with self.sessions() as session:
            async with session.begin():
                session.add(RequestStat(
                    request_id=request_id,
                    api_key_id=api_key_id,
                    request_kind=request_kind,
                    status=status,
                    provider_name=provider_name,
                    base_url=base_url,
                    model_id=model_id,
                    capability_set=capabilities,
                    latency_ms=max(0, latency_ms),
                    error_code=error_code,
                    created_at=utcnow(),
                ))

    async def cleanup(self) -> None:
        now = utcnow()
        async with self.sessions() as session:
            async with session.begin():
                await session.execute(ResponseState.__table__.delete().where(ResponseState.expires_at < now))
                await session.execute(ConversationAffinity.__table__.delete().where(ConversationAffinity.expires_at < now))
                await session.execute(DiscoveryCache.__table__.delete().where(DiscoveryCache.expires_at < now))
                await session.execute(DiscoveryLease.__table__.delete().where(DiscoveryLease.lease_until < now))
                health_cutoff = now - timedelta(days=90)
                await session.execute(ProviderHealth.__table__.delete().where(ProviderHealth.updated_at < health_cutoff))
                await session.execute(ModelHealth.__table__.delete().where(ModelHealth.updated_at < health_cutoff))
                await session.execute(KeyHealth.__table__.delete().where(KeyHealth.updated_at < health_cutoff))
                # Keep a bounded operational history; no prompts or completions
                # are stored in this metrics table.
                await session.execute(RequestStat.__table__.delete().where(RequestStat.created_at < now - timedelta(days=30)))


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
