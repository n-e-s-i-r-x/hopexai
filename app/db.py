from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from .models import Base
from .settings import Settings


def _prepare_sqlite_path(url: str) -> None:
    prefix = "sqlite+aiosqlite:///"
    if not url.startswith(prefix) or url.endswith(":memory:"):
        return
    raw_path = url[len(prefix):]
    if raw_path.startswith("/"):
        path = Path("/" + raw_path.lstrip("/"))
    else:
        path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)


def create_database(settings: Settings) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    _prepare_sqlite_path(settings.database_url)
    kwargs: dict[str, object] = {"pool_pre_ping": True}
    if settings.database_url.startswith("postgresql+"):
        kwargs.update({"pool_size": 20, "max_overflow": 20, "pool_recycle": 1800})
    engine = create_async_engine(settings.database_url, **kwargs)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    return engine, session_factory


async def initialize_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def check_database(engine: AsyncEngine) -> None:
    from sqlalchemy import text

    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))
        await connection.execute(text("SELECT id FROM api_keys LIMIT 1"))


async def get_session(session_factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        yield session
