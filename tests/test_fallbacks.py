from __future__ import annotations

import copy

import pytest

from tests.helpers import ProviderMock, config_data, make_env


async def issue_key(client):
    response = await client.post("/api/keys", json={}, headers={"Origin": "http://testserver"})
    assert response.status_code == 200
    return response.json()["api_key"]


def deterministic_first(monkeypatch):
    monkeypatch.setattr("app.registry.secrets.choice", lambda sequence: sequence[0])
    monkeypatch.setattr(
        "app.registry.secrets.SystemRandom.choices",
        lambda self, population, weights=None, *, cum_weights=None, k=1: [population[0]],
    )


@pytest.mark.asyncio
async def test_account_quota_failure_falls_back_to_another_base_url(tmp_path, monkeypatch):
    config = config_data(keys=["primary-key"])
    first_base = config["providers"][0]["base_urls"][0]
    first_base["url"] = "https://provider-one.mock"
    first_base["selection_weight"] = 100
    second_base = copy.deepcopy(first_base)
    second_base["url"] = "https://provider-two.mock"
    second_base["api_keys"] = ["secondary-key"]
    second_base["selection_weight"] = 1
    config["providers"][0]["base_urls"].append(second_base)

    provider = ProviderMock(models=[config["_test_model"]])
    provider.host_failures["provider-one.mock"] = (
        429,
        {"error": {"code": "insufficient_quota", "message": "account quota exhausted"}},
        {},
    )
    async with make_env(tmp_path, provider, config) as env:
        deterministic_first(monkeypatch)
        key = await issue_key(env["client"])
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "hopexai", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.status_code == 200, response.text
        calls = [call for call in provider.calls if call["method"] == "POST"]
        assert [call["host"] for call in calls] == ["provider-one.mock", "provider-two.mock"]
        assert response.json()["model"] == "hopexai"
        assert "provider-one" not in response.text and "provider-two" not in response.text


@pytest.mark.asyncio
async def test_invalid_manual_endpoint_tries_another_discovered_endpoint(tmp_path, monkeypatch):
    config = config_data()
    config["providers"][0]["base_urls"][0]["endpoint_overrides"] = {
        "chat_completions": "/wrong/chat-endpoint",
    }
    provider = ProviderMock(models=[config["_test_model"]])
    async with make_env(tmp_path, provider, config) as env:
        # The manual override is attempted first, then the independently
        # discovered /v1/chat/completions path is used after a bounded 404.
        monkeypatch.setattr("app.registry.secrets.choice", lambda sequence: sequence[0])
        key = await issue_key(env["client"])
        response = await env["client"].post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "hopexai", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.status_code == 200, response.text
        calls = [call for call in provider.calls if call["method"] == "POST"]
        assert [call["path"] for call in calls] == ["/wrong/chat-endpoint", "/v1/chat/completions"]
        assert response.json()["model"] == "hopexai"
