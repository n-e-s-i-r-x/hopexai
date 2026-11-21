from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import re
import secrets
import time
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .errors import PublicAPIError
from .models import APIKeyRecord, ConversationAffinity, KeyIssueThrottle, KeyIssuanceEvent, ResponseState, utcnow
from .secret_store import SecretStore
from .settings import Settings


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class KeyService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession], secrets_store: SecretStore, settings: Settings):
        self.sessions = sessions
        self.secrets_store = secrets_store
        self.settings = settings
        self._serverless_lock = asyncio.Lock()
        self._issue_events: dict[str, list[float]] = {}
        self._request_windows: dict[str, tuple[float, int]] = {}

    def fingerprint(self, value: str) -> str:
        return self.secrets_store.digest(value, "api-key-hash")

    @staticmethod
    def _b64url(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    @staticmethod
    def _unb64url(value: str) -> bytes:
        padded = value + "=" * (-len(value) % 4)
        return base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)

    def _serverless_key_id(self, secret: str) -> str:
        return str(uuid.UUID(hex=self.secrets_store.digest(secret, "api-key-id")[:32]))

    def _serverless_record(self, secret: str, created_at: datetime) -> APIKeyRecord:
        return APIKeyRecord(
            id=self._serverless_key_id(secret),
            secret_hash=self.fingerprint(secret),
            prefix=secret[:12],
            created_at=created_at,
        )

    def _issue_serverless_key(self, now: datetime) -> tuple[str, APIKeyRecord]:
        issued_at = int(now.timestamp())
        nonce = self._b64url(secrets.token_bytes(24))
        payload = f"v1.{issued_at}.{nonce}"
        signature = hmac.new(
            self.secrets_store.key,
            b"hopexai:serverless-api-key:" + payload.encode("ascii"),
            hashlib.sha256,
        ).digest()
        secret = f"hx_{payload}.{self._b64url(signature)}"
        return secret, self._serverless_record(secret, now)

    def _authenticate_serverless_key(self, secret: str) -> APIKeyRecord | None:
        match = re.fullmatch(r"hx_v1\.(\d{1,12})\.([A-Za-z0-9_-]{32})\.([A-Za-z0-9_-]{43})", secret)
        if match is None:
            return None
        version = "v1"
        issued_at, nonce, signature_text = match.groups()
        try:
            if int(issued_at) > int(time.time()) + 300:
                return None
            if len(self._unb64url(nonce)) != 24:
                return None
            signature = self._unb64url(signature_text)
            if len(signature) != hashlib.sha256().digest_size:
                return None
            payload = f"{version}.{issued_at}.{nonce}"
            expected = hmac.new(
                self.secrets_store.key,
                b"hopexai:serverless-api-key:" + payload.encode("ascii"),
                hashlib.sha256,
            ).digest()
        except (ValueError, UnicodeEncodeError):
            return None
        if not hmac.compare_digest(signature, expected):
            return None
        created_at = datetime.fromtimestamp(int(issued_at), timezone.utc)
        return self._serverless_record(secret, created_at)

    async def _issue_serverless(self, ip_digest: str) -> tuple[str, APIKeyRecord]:
        now = utcnow()
        timestamp = now.timestamp()
        async with self._serverless_lock:
            events = [item for item in self._issue_events.get(ip_digest, []) if timestamp - item < 86_400]
            hourly = sum(timestamp - item < 3_600 for item in events)
            if hourly >= self.settings.key_issue_per_hour or len(events) >= self.settings.key_issue_per_day:
                self._issue_events[ip_digest] = events
                raise PublicAPIError(429, "Key generation is temporarily rate limited. Try again later.", "rate_limit_error", "key_generation_rate_limited")
            events.append(timestamp)
            self._issue_events[ip_digest] = events
            if len(self._issue_events) > 4096:
                self._issue_events = {
                    key: values for key, values in self._issue_events.items()
                    if values and timestamp - values[-1] < 86_400
                }
        return self._issue_serverless_key(now)

    async def issue(self, ip_digest: str) -> tuple[str, APIKeyRecord]:
        if self.settings.vercel_runtime:
            return await self._issue_serverless(ip_digest)
        now = utcnow()
        async with self.sessions() as session:
            async with session.begin():
                await self._ensure_issue_bucket(session, ip_digest, now)
                bucket = await session.scalar(
                    select(KeyIssueThrottle).where(KeyIssueThrottle.ip_digest == ip_digest).with_for_update()
                )
                if bucket is None:
                    raise RuntimeError("could not initialize key issuance limiter")
                hour_start = _as_utc(bucket.hour_started_at) or now
                day_start = _as_utc(bucket.day_started_at) or now
                if now - hour_start >= timedelta(hours=1):
                    bucket.hour_started_at = now
                    bucket.hour_count = 0
                if now - day_start >= timedelta(days=1):
                    bucket.day_started_at = now
                    bucket.day_count = 0
                if bucket.hour_count >= self.settings.key_issue_per_hour or bucket.day_count >= self.settings.key_issue_per_day:
                    raise PublicAPIError(429, "Key generation is temporarily rate limited. Try again later.", "rate_limit_error", "key_generation_rate_limited")
                bucket.hour_count += 1
                bucket.day_count += 1
                bucket.updated_at = now
                secret = "hx_" + secrets.token_urlsafe(32)
                record = APIKeyRecord(
                    id=str(uuid.uuid4()),
                    secret_hash=self.fingerprint(secret),
                    prefix=secret[:12],
                    created_at=now,
                )
                session.add(record)
                session.add(KeyIssuanceEvent(ip_digest=ip_digest, created_at=now))
            # `record` is detached only after commit; the returned fields are safe
            # metadata and do not contain the key hash.
            return secret, record

    async def _ensure_issue_bucket(self, session: AsyncSession, ip_digest: str, now: datetime) -> None:
        existing = await session.get(KeyIssueThrottle, ip_digest)
        if existing is not None:
            return
        try:
            async with session.begin_nested():
                session.add(KeyIssueThrottle(
                    ip_digest=ip_digest,
                    hour_started_at=now,
                    hour_count=0,
                    day_started_at=now,
                    day_count=0,
                    updated_at=now,
                ))
                await session.flush()
        except Exception:
            # A concurrent process may have inserted the same digest first.
            # The transaction remains usable because the insert was a savepoint.
            pass

    async def authenticate(self, bearer_value: str | None) -> APIKeyRecord:
        if not bearer_value:
            raise PublicAPIError(401, "A valid HopeXAI API key is required.", "authentication_error", "invalid_api_key")
        secret = bearer_value.strip()
        if not secret.startswith("hx_") or len(secret) < 30 or len(secret) > 128:
            raise PublicAPIError(401, "A valid HopeXAI API key is required.", "authentication_error", "invalid_api_key")
        if self.settings.vercel_runtime:
            record = self._authenticate_serverless_key(secret)
            if record is None:
                raise PublicAPIError(401, "A valid HopeXAI API key is required.", "authentication_error", "invalid_api_key")
            return record
        key_hash = self.fingerprint(secret)
        async with self.sessions() as session:
            record = await session.scalar(select(APIKeyRecord).where(APIKeyRecord.secret_hash == key_hash))
            if record is None or record.revoked_at is not None:
                raise PublicAPIError(401, "A valid HopeXAI API key is required.", "authentication_error", "invalid_api_key")
            # Return a detached, non-secret record; callers only use its opaque ID.
            return record

    async def register_request(self, key_id: str) -> None:
        if self.settings.vercel_runtime:
            now = time.monotonic()
            async with self._serverless_lock:
                started, count = self._request_windows.get(key_id, (now, 0))
                if now - started >= 60:
                    started, count = now, 0
                if count >= self.settings.api_key_requests_per_minute:
                    raise PublicAPIError(429, "Request rate limit exceeded. Please retry shortly.", "rate_limit_error", "api_key_rate_limited")
                self._request_windows[key_id] = (started, count + 1)
                if len(self._request_windows) > 8192:
                    self._request_windows = {
                        key: value for key, value in self._request_windows.items()
                        if now - value[0] < 60
                    }
            return
        now = utcnow()
        async with self.sessions() as session:
            async with session.begin():
                record = await session.scalar(select(APIKeyRecord).where(APIKeyRecord.id == key_id).with_for_update())
                if record is None or record.revoked_at is not None:
                    raise PublicAPIError(401, "A valid HopeXAI API key is required.", "authentication_error", "invalid_api_key")
                started = _as_utc(record.rate_window_started_at)
                if started is None or now - started >= timedelta(minutes=1):
                    record.rate_window_started_at = now
                    record.rate_window_count = 1
                elif record.rate_window_count >= self.settings.api_key_requests_per_minute:
                    raise PublicAPIError(429, "Request rate limit exceeded. Please retry shortly.", "rate_limit_error", "api_key_rate_limited")
                else:
                    record.rate_window_count += 1
                record.requests_total += 1
                record.last_used_at = now

    async def revoke(self, key_id: str) -> bool:
        async with self.sessions() as session:
            async with session.begin():
                record = await session.get(APIKeyRecord, key_id, with_for_update=True)
                if record is None:
                    return False
                if record.revoked_at is None:
                    record.revoked_at = utcnow()
                await session.execute(delete(ResponseState).where(ResponseState.api_key_id == key_id))
                await session.execute(delete(ConversationAffinity).where(ConversationAffinity.api_key_id == key_id))
                return True

    async def list_keys(self) -> list[APIKeyRecord]:
        async with self.sessions() as session:
            result = await session.scalars(select(APIKeyRecord).order_by(APIKeyRecord.created_at.desc()))
            return list(result)

    async def cleanup_old_issuance_events(self) -> None:
        cutoff = utcnow() - timedelta(days=8)
        async with self.sessions() as session:
            async with session.begin():
                await session.execute(
                    KeyIssuanceEvent.__table__.delete().where(KeyIssuanceEvent.created_at < cutoff)
                )
                await session.execute(
                    KeyIssueThrottle.__table__.delete().where(KeyIssueThrottle.updated_at < cutoff)
                )
