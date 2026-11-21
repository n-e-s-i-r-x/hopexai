from __future__ import annotations

import base64
import json
import tomllib
from pathlib import Path

import httpx
import pytest

from app.config import ConfigManager
from app.main import create_app
from app.secret_store import SecretStore, SecretStoreError
from app.settings import Settings


def test_vercel_copies_the_active_project_config_into_ephemeral_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv("HOPEXAI_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("HOPEXAI_CONFIG", str(tmp_path / "ignored.yaml"))
    monkeypatch.setenv("HOPEXAI_CONFIG_YAML", "ignored old environment config")
    monkeypatch.setenv("DATABASE_URL", "postgresql://ignored.example/ignored")
    monkeypatch.setenv("HOPEXAI_VERCEL_PROXY_SECRET", "legacy-proxy-secret")
    project_config = Path(__file__).resolve().parents[1] / "config" / "hopexai.config.yaml"

    settings = Settings.from_env()

    assert settings.vercel_runtime is True
    assert settings.auto_create_schema is True
    assert settings.database_url.startswith("sqlite+aiosqlite:////")
    assert str(tmp_path / "runtime" / "var" / "hopexai.db") in settings.database_url
    assert settings.config_path.read_text("utf-8") == project_config.read_text("utf-8")
    assert settings.public_origin is None
    assert settings.vercel_proxy_secret is None


@pytest.mark.asyncio
async def test_vercel_key_generation_works_with_project_config_and_stable_master_key(tmp_path, monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv("HOPEXAI_MASTER_KEY", base64.urlsafe_b64encode(b"v" * 32).decode("ascii").rstrip("="))
    monkeypatch.setenv("HOPEXAI_RUNTIME_DIR", str(tmp_path / "runtime"))
    app = create_app(Settings.from_env())

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="https://hopefreexai.vercel.app",
        ) as client:
            page = await client.get("/")
            issued = await client.post(
                "/api/keys",
                json={},
                headers={"Origin": "https://hopefreexai.vercel.app"},
            )
            assert page.status_code == 200
            assert issued.status_code == 200, issued.text
            api_key = issued.json()["api_key"]
            assert api_key.startswith("hx_v1.")
            models = await client.get("/v1/models", headers={"Authorization": f"Bearer {api_key}"})
            assert models.status_code == 200, models.text


def test_provider_keys_can_reference_server_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("HOPEXAI_PROVIDER_API_KEY", "sk-provider-secret-from-env")
    active = Path(__file__).resolve().parents[1] / "config" / "hopexai.config.yaml"
    raw = active.read_text("utf-8").replace("REPLACE_WITH_PROVIDER_API_KEY", "${HOPEXAI_PROVIDER_API_KEY}")
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        config_path=tmp_path / "hopexai.config.yaml",
        state_dir=tmp_path / "state",
    )
    config = ConfigManager(settings, SecretStore(settings.state_dir))._parse_and_validate(raw.encode("utf-8"))

    api_key = config.providers[0].base_urls[0].api_keys[0].get_secret_value()
    assert api_key == "sk-provider-secret-from-env"


def test_config_folder_has_one_active_editable_config_not_example_files():
    config_dir = Path(__file__).resolve().parents[1] / "config"
    assert (config_dir / "hopexai.config.yaml").is_file()
    assert not list(config_dir.glob("*.example.yaml"))


def test_vercel_requires_a_stable_master_key_instead_of_a_function_local_random_key(tmp_path, monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.delenv("HOPEXAI_MASTER_KEY", raising=False)
    state_dir = tmp_path / "state"

    with pytest.raises(SecretStoreError, match="HOPEXAI_MASTER_KEY is required on Vercel"):
        SecretStore(state_dir).ensure()

    assert not state_dir.exists()


def test_vercel_entrypoint_and_function_settings_are_single_backend():
    root = Path(__file__).resolve().parents[1]
    vercel = json.loads((root / "vercel.json").read_text("utf-8"))
    project = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))

    assert project["tool"]["vercel"]["entrypoint"] == "app.main:app"
    assert "app/main.py" in vercel["functions"]
    assert vercel["functions"]["app/main.py"]["maxDuration"] == 300
    assert "config/hopexai.config.yaml" in vercel["functions"]["app/main.py"]["includeFiles"]
    assert vercel.get("rewrites") is None
