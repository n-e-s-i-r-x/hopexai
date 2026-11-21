from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import timezone
from pathlib import Path
from typing import Any

import httpx
import yaml
from sqlalchemy import select

from .config import ConfigManager, ConfigurationError
from .db import check_database, create_database
from .discovery import DiscoveryService
from .health import HealthService, as_utc
from .key_service import KeyService
from .models import KeyHealth, ProviderHealth, ModelHealth
from .runtime_state import RuntimeState
from .secret_store import SecretStore
from .security import PinnedAsyncHTTPTransport, URLGuard
from .settings import Settings


def _settings() -> Settings:
    settings = Settings.from_env()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return settings


def _config_path(settings: Settings) -> Path:
    return settings.config_path


def _ensure_setup(settings: Settings) -> None:
    settings.config_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    settings.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(settings.config_path.parent, 0o700)
        os.chmod(settings.state_dir, 0o700)
    except OSError:
        pass
    if not settings.config_path.exists():
        example = Path(__file__).resolve().parents[1] / "config" / "hopexai.config.example.yaml"
        if not example.exists():
            raise SystemExit("Configuration example not found; copy config/hopexai.config.example.yaml manually.")
        shutil.copyfile(example, settings.config_path)
        os.chmod(settings.config_path, 0o600)
        print(f"Created private configuration: {settings.config_path}")
    SecretStore(settings.state_dir).ensure()
    print(f"Persistent master key ready: {settings.state_dir / 'master.key'}")
    print("Next: edit the private YAML file, then run `python -m app.cli config validate` and `python -m app.cli db migrate`.")


async def _validate_config(settings: Settings) -> None:
    manager = ConfigManager(settings, SecretStore(settings.state_dir))
    try:
        snapshot = await manager.load_initial()
    except ConfigurationError as exc:
        print("Configuration is invalid:", file=sys.stderr)
        for detail in exc.details or (str(exc),):
            print(f"  - {detail}", file=sys.stderr)
        raise SystemExit(2)
    enabled = sum(1 for provider in snapshot.config.providers if provider.enabled)
    bases = sum(1 for provider in snapshot.config.providers for base in provider.base_urls if base.enabled)
    key_count = sum(len(base.api_keys) for provider in snapshot.config.providers for base in provider.base_urls)
    print(f"Configuration valid: {enabled} enabled providers, {bases} enabled base URLs, {key_count} configured upstream keys.")
    print(f"SHA-256: {snapshot.content_hash}")
    print("Credential values were not displayed.")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=".hopexai-config-", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
        os.chmod(path, 0o600)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


async def _encrypt_config(settings: Settings) -> None:
    secret_store = SecretStore(settings.state_dir)
    manager = ConfigManager(settings, secret_store)
    await manager.load_initial()
    path = _config_path(settings)
    try:
        raw = yaml.safe_load(path.read_text("utf-8"))
    except Exception:
        raise SystemExit("Could not parse configuration file; no changes were made.")
    count = 0
    for provider in raw.get("providers", []) if isinstance(raw, dict) else []:
        for base in provider.get("base_urls", []) if isinstance(provider, dict) else []:
            if not isinstance(base, dict) or not isinstance(base.get("api_keys"), list):
                continue
            encrypted = []
            for item in base["api_keys"]:
                if isinstance(item, str) and not item.startswith("ENC[v1:"):
                    encrypted.append(secret_store.encrypt_config_secret(item))
                    count += 1
                else:
                    encrypted.append(item)
            base["api_keys"] = encrypted
    serialized = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True).encode("utf-8")
    try:
        manager._parse_and_validate(serialized)
    except ConfigurationError as exc:
        print("Encrypted configuration failed validation; the original file was retained.", file=sys.stderr)
        for detail in exc.details:
            print(f"  - {detail}", file=sys.stderr)
        raise SystemExit(2)
    _atomic_write(path, serialized)
    print(f"Encrypted {count} plaintext upstream key(s) in {path}. No credentials were printed.")


async def _keys_list(settings: Settings) -> None:
    engine, sessions = create_database(settings)
    try:
        await check_database(engine)
        service = KeyService(sessions, SecretStore(settings.state_dir), settings)
        records = await service.list_keys()
        if not records:
            print("No HopeXAI API keys have been generated.")
        for record in records:
            status = "revoked" if record.revoked_at else "active"
            created = record.created_at.astimezone(timezone.utc).isoformat()
            print(f"{record.id}  {record.prefix}…  {status:<7} requests={record.requests_total} created={created}")
    finally:
        await engine.dispose()


async def _keys_revoke(settings: Settings, key_id: str) -> None:
    engine, sessions = create_database(settings)
    try:
        service = KeyService(sessions, SecretStore(settings.state_dir), settings)
        if await service.revoke(key_id):
            print(f"Revoked HopeXAI API key {key_id}.")
        else:
            print("No key with that ID exists.", file=sys.stderr)
            raise SystemExit(1)
    finally:
        await engine.dispose()


async def _diagnose(settings: Settings, force: bool) -> None:
    secret_store = SecretStore(settings.state_dir)
    secret_store.ensure()
    manager = ConfigManager(settings, secret_store)
    snapshot = await manager.load_initial()
    engine, sessions = create_database(settings)
    client_limits = httpx.Limits(max_connections=40, max_keepalive_connections=10)
    client = httpx.AsyncClient(
        transport=PinnedAsyncHTTPTransport(limits=client_limits, trust_env=False),
        timeout=None,
        limits=client_limits,
        follow_redirects=False,
        trust_env=False,
        headers={"User-Agent": "HopeXAI-Gateway-Diagnostics/1.0"},
    )
    guard = URLGuard()
    discovery = DiscoveryService(sessions, client, guard, secret_store)
    health = HealthService(sessions, secret_store)
    try:
        await check_database(engine)
        state = await health.snapshot()
        if not snapshot.config.providers:
            print("No upstream providers are configured.")
        for provider in snapshot.config.providers:
            for base in provider.base_urls:
                if not provider.enabled or not base.enabled:
                    print(f"{provider.name} {base.url}: disabled")
                    continue
                print(f"\n{provider.name}  {base.url}")
                try:
                    data = await discovery.get(provider, base, snapshot.config.defaults, force=force)
                except Exception as exc:
                    print(f"  discovery: failed ({type(exc).__name__})")
                    continue
                print(f"  API format: {data.api_format}")
                provider_health = state.providers.get((provider.name, base.url))
                if provider_health is None:
                    print("  base health: no recorded failures")
                else:
                    until = as_utc(provider_health.cooldown_until)
                    cooldown = f"; cooldown until {until.isoformat()}" if until and until > datetime_now() else ""
                    print(f"  base health: failures={provider_health.consecutive_failures}{cooldown}")
                for kind, paths in sorted(data.endpoints.items()):
                    if paths:
                        print(f"  {kind}: {', '.join(paths)}")
                print(f"  models discovered: {len(data.models)}")
                for model in data.models[:100]:
                    caps = model.get("capabilities", {})
                    enabled = ", ".join(key for key, value in caps.items() if value is True) or "none verified"
                    model_health = state.models.get((provider.name, base.url, model.get("id", "")))
                    health_label = "healthy/untried"
                    if model_health is not None:
                        until = as_utc(model_health.unavailable_until)
                        health_label = f"failures={model_health.consecutive_failures}"
                        if until and until > datetime_now():
                            health_label += f"; unavailable until {until.isoformat()}"
                    print(f"    - {model.get('id')} | pricing={model.get('pricing', 'unknown')} | context={model.get('context_window') or 'unknown'} | capabilities={enabled} | health={health_label}")
                for item in base.api_keys:
                    key = item.get_secret_value()
                    fingerprint = health.key_fingerprint(key)
                    status = state.keys.get((provider.name, base.url, fingerprint))
                    if status and status.invalid_at:
                        label = "invalid"
                    elif status and as_utc(status.cooldown_until) and as_utc(status.cooldown_until) > datetime_now():
                        label = "cooldown"
                    else:
                        label = "configured (secret hidden)"
                    print(f"  upstream key: {label}")
    finally:
        await client.aclose()
        await engine.dispose()


def datetime_now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


async def _reset_health(settings: Settings) -> None:
    engine, sessions = create_database(settings)
    try:
        service = HealthService(sessions, SecretStore(settings.state_dir))
        await service.reset()
        print("Provider, model, and upstream-key health/cooldown state was reset.")
    finally:
        await engine.dispose()


async def _cleanup(settings: Settings) -> None:
    engine, sessions = create_database(settings)
    try:
        state = RuntimeState(sessions, SecretStore(settings.state_dir), settings.response_state_max_bytes)
        keys = KeyService(sessions, SecretStore(settings.state_dir), settings)
        await state.cleanup()
        await keys.cleanup_old_issuance_events()
        print("Expired response state, affinities, discovery cache/leases, stale health, request statistics, and key-issuance records were removed.")
    finally:
        await engine.dispose()


def _migrate() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=project_root, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hopexai", description="Private HopeXAI gateway administration CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup", help="create the private config and persistent encryption key")
    config = sub.add_parser("config", help="validate or protect the private YAML configuration")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    config_sub.add_parser("validate", help="validate without displaying credentials")
    config_sub.add_parser("encrypt-keys", help="encrypt plaintext API keys in the private YAML file")

    db = sub.add_parser("db", help="database maintenance")
    db_sub = db.add_subparsers(dest="db_command", required=True)
    db_sub.add_parser("migrate", help="apply Alembic migrations")
    db_sub.add_parser("cleanup", help="remove expired operational records")

    keys = sub.add_parser("keys", help="private HopeXAI API key operations")
    keys_sub = keys.add_subparsers(dest="keys_command", required=True)
    keys_sub.add_parser("list", help="list IDs and status only (never key values)")
    revoke = keys_sub.add_parser("revoke", help="revoke a HopeXAI key by its private CLI ID")
    revoke.add_argument("key_id")

    providers = sub.add_parser("providers", help="private upstream diagnostics")
    providers_sub = providers.add_subparsers(dest="providers_command", required=True)
    diagnose = providers_sub.add_parser("diagnose", help="discover endpoints/models and report health")
    diagnose.add_argument("--force", action="store_true", help="refresh even when a valid cache exists")

    health = sub.add_parser("health", help="private health-state operations")
    health_sub = health.add_subparsers(dest="health_command", required=True)
    health_sub.add_parser("reset", help="clear provider, model, and key cooldown state")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = _settings()
    if args.command == "setup":
        _ensure_setup(settings)
    elif args.command == "config" and args.config_command == "validate":
        asyncio.run(_validate_config(settings))
    elif args.command == "config" and args.config_command == "encrypt-keys":
        asyncio.run(_encrypt_config(settings))
    elif args.command == "db" and args.db_command == "migrate":
        _migrate()
    elif args.command == "db" and args.db_command == "cleanup":
        asyncio.run(_cleanup(settings))
    elif args.command == "keys" and args.keys_command == "list":
        asyncio.run(_keys_list(settings))
    elif args.command == "keys" and args.keys_command == "revoke":
        asyncio.run(_keys_revoke(settings, args.key_id))
    elif args.command == "providers" and args.providers_command == "diagnose":
        asyncio.run(_diagnose(settings, args.force))
    elif args.command == "health" and args.health_command == "reset":
        asyncio.run(_reset_health(settings))
    else:
        parser.error("unknown command")


if __name__ == "__main__":
    main()
