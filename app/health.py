from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .errors import UpstreamFailure
from .models import KeyHealth, ModelHealth, ProviderHealth, utcnow


def as_utc(value: datetime | None) -> datetime | None:
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
    raise RuntimeError(f"unsupported database dialect for concurrency-safe upsert: {dialect}")


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    providers: dict[tuple[str, str], ProviderHealth]
    models: dict[tuple[str, str, str], ModelHealth]
    keys: dict[tuple[str, str, str], KeyHealth]


class HealthService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession], secret_store):
        self.sessions = sessions
        self.secret_store = secret_store

    async def snapshot(self) -> HealthSnapshot:
        async with self.sessions() as session:
            providers = list(await session.scalars(select(ProviderHealth)))
            models = list(await session.scalars(select(ModelHealth)))
            keys = list(await session.scalars(select(KeyHealth)))
        return HealthSnapshot(
            {(x.provider_name, x.base_url): x for x in providers},
            {(x.provider_name, x.base_url, x.model_id): x for x in models},
            {(x.provider_name, x.base_url, x.key_fingerprint): x for x in keys},
        )

    def key_fingerprint(self, key: str | None) -> str:
        return self.secret_store.digest(key or "", "upstream-key-fingerprint")

    async def record_failure(
        self,
        provider_name: str,
        base_url: str,
        model_id: str,
        key: str | None,
        failure: UpstreamFailure,
        *,
        provider_cooldown_seconds: int,
        model_cooldown_seconds: int,
        key_cooldown_seconds: int,
        account_quota_cooldown_seconds: int,
        invalid_key_cooldown_seconds: int = 0,
    ) -> None:
        now = utcnow()
        code = failure.error_code[:64]
        async with self.sessions() as session:
            async with session.begin():
                if failure.scope == "key":
                    fingerprint = self.key_fingerprint(key)
                    is_invalid = failure.error_code == "invalid_credentials"
                    cooldown = invalid_key_cooldown_seconds if is_invalid else (
                        failure.retry_after_seconds or key_cooldown_seconds
                    )
                    until = None if is_invalid and cooldown == 0 else now + timedelta(seconds=max(1, cooldown))
                    values: dict[str, Any] = {
                        "provider_name": provider_name,
                        "base_url": base_url,
                        "key_fingerprint": fingerprint,
                        "consecutive_failures": 1,
                        "cooldown_until": until,
                        "invalid_at": now if is_invalid else None,
                        "last_error_code": code,
                        "updated_at": now,
                    }
                    stmt = _insert_for(session, KeyHealth).values(**values)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=[KeyHealth.provider_name, KeyHealth.base_url, KeyHealth.key_fingerprint],
                        set_={
                            "consecutive_failures": KeyHealth.consecutive_failures + 1,
                            "cooldown_until": until,
                            "invalid_at": now if is_invalid else KeyHealth.invalid_at,
                            "last_error_code": code,
                            "updated_at": now,
                        },
                    )
                    await session.execute(stmt)
                    return
                if failure.scope == "model":
                    until = now + timedelta(seconds=model_cooldown_seconds)
                    values = {
                        "provider_name": provider_name,
                        "base_url": base_url,
                        "model_id": model_id,
                        "consecutive_failures": 1,
                        "unavailable_until": until,
                        "last_error_code": code,
                        "updated_at": now,
                    }
                    stmt = _insert_for(session, ModelHealth).values(**values)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=[ModelHealth.provider_name, ModelHealth.base_url, ModelHealth.model_id],
                        set_={
                            "consecutive_failures": ModelHealth.consecutive_failures + 1,
                            "unavailable_until": until,
                            "last_error_code": code,
                            "updated_at": now,
                        },
                    )
                    await session.execute(stmt)
                    return
                if failure.error_code == "account_quota_exhausted":
                    cooldown = account_quota_cooldown_seconds
                else:
                    cooldown = failure.retry_after_seconds or provider_cooldown_seconds
                until = now + timedelta(seconds=max(1, cooldown))
                values = {
                    "provider_name": provider_name,
                    "base_url": base_url,
                    "consecutive_failures": 1,
                    "cooldown_until": until,
                    "last_error_code": code,
                    "updated_at": now,
                }
                stmt = _insert_for(session, ProviderHealth).values(**values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[ProviderHealth.provider_name, ProviderHealth.base_url],
                    set_={
                        "consecutive_failures": ProviderHealth.consecutive_failures + 1,
                        "cooldown_until": until,
                        "last_error_code": code,
                        "updated_at": now,
                    },
                )
                await session.execute(stmt)

    async def record_success(self, provider_name: str, base_url: str, model_id: str, key: str | None) -> None:
        now = utcnow()
        fingerprint = self.key_fingerprint(key)
        async with self.sessions() as session:
            async with session.begin():
                await session.execute(
                    update(ProviderHealth)
                    .where(ProviderHealth.provider_name == provider_name, ProviderHealth.base_url == base_url)
                    .values(consecutive_failures=0, cooldown_until=None, last_error_code=None, updated_at=now)
                )
                await session.execute(
                    update(ModelHealth)
                    .where(
                        ModelHealth.provider_name == provider_name,
                        ModelHealth.base_url == base_url,
                        ModelHealth.model_id == model_id,
                    )
                    .values(consecutive_failures=0, unavailable_until=None, last_error_code=None, updated_at=now)
                )
                await session.execute(
                    update(KeyHealth)
                    .where(
                        KeyHealth.provider_name == provider_name,
                        KeyHealth.base_url == base_url,
                        KeyHealth.key_fingerprint == fingerprint,
                    )
                    .values(consecutive_failures=0, cooldown_until=None, invalid_at=None, last_error_code=None, updated_at=now)
                )

    async def reset(self) -> None:
        async with self.sessions() as session:
            async with session.begin():
                await session.execute(ProviderHealth.__table__.delete())
                await session.execute(ModelHealth.__table__.delete())
                await session.execute(KeyHealth.__table__.delete())
