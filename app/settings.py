from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int, minimum: int = 0, maximum: int = 2**31 - 1) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if not minimum <= parsed <= maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _float(name: str, default: float, minimum: float = 0.0, maximum: float = 3600.0) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc
    if not minimum <= parsed <= maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
    return parsed


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str
    config_path: Path
    state_dir: Path
    auto_create_schema: bool = False
    config_poll_interval: float = 2.0
    max_request_bytes: int = 2_097_152
    max_messages: int = 200
    max_tools: int = 64
    key_issue_per_hour: int = 3
    key_issue_per_day: int = 10
    api_key_requests_per_minute: int = 120
    response_state_max_bytes: int = 2_000_000
    trusted_proxy_cidrs: tuple[str, ...] = ()
    log_level: str = "INFO"
    public_origin: str | None = None
    vercel_proxy_secret: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        root = Path(os.getenv("HOPEXAI_HOME", Path.cwd())).expanduser().resolve()
        db_url = os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{root / 'var' / 'hopexai.db'}")
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        config_path = Path(os.getenv("HOPEXAI_CONFIG", str(root / "private" / "hopexai.config.yaml"))).expanduser()
        state_dir = Path(os.getenv("HOPEXAI_STATE_DIR", str(root / "private" / "state"))).expanduser()
        if not config_path.is_absolute():
            config_path = root / config_path
        if not state_dir.is_absolute():
            state_dir = root / state_dir
        config_path = config_path.absolute()
        state_dir = state_dir.absolute()
        proxies = tuple(x.strip() for x in os.getenv("HOPEXAI_TRUSTED_PROXY_CIDRS", "").split(",") if x.strip())
        return cls(
            database_url=db_url,
            config_path=config_path,
            state_dir=state_dir,
            auto_create_schema=_bool("HOPEXAI_AUTO_CREATE_SCHEMA", False),
            config_poll_interval=_float("HOPEXAI_CONFIG_POLL_INTERVAL", 2.0, 0.1, 60.0),
            max_request_bytes=_int("HOPEXAI_MAX_REQUEST_BYTES", 2_097_152, 16_384, 50_000_000),
            max_messages=_int("HOPEXAI_MAX_MESSAGES", 200, 1, 10_000),
            max_tools=_int("HOPEXAI_MAX_TOOLS", 64, 0, 1000),
            key_issue_per_hour=_int("HOPEXAI_KEY_ISSUE_PER_HOUR", 3, 1, 1000),
            key_issue_per_day=_int("HOPEXAI_KEY_ISSUE_PER_DAY", 10, 1, 10_000),
            api_key_requests_per_minute=_int("HOPEXAI_API_KEY_RPM", 120, 1, 100_000),
            response_state_max_bytes=_int("HOPEXAI_RESPONSE_STATE_MAX_BYTES", 2_000_000, 10_000, 50_000_000),
            trusted_proxy_cidrs=proxies,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            public_origin=os.getenv("HOPEXAI_PUBLIC_ORIGIN") or None,
            vercel_proxy_secret=os.getenv("HOPEXAI_VERCEL_PROXY_SECRET") or None,
        )
