from __future__ import annotations

import asyncio

import httpx
import pytest

from app.security import PinnedAsyncHTTPTransport, URLGuard


@pytest.mark.asyncio
async def test_guard_pins_validated_dns_address_preserves_host_and_disables_reuse():
    seen: list[tuple[str, str]] = []
    connections = 0

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        nonlocal connections
        connections += 1
        try:
            while True:
                request_line = await reader.readline()
                if not request_line:
                    break
                host = ""
                while True:
                    line = await reader.readline()
                    if line in {b"\r\n", b"\n", b""}:
                        break
                    name, _, value = line.decode("latin-1").partition(":")
                    if name.lower() == "host":
                        host = value.strip()
                seen.append((request_line.decode("ascii").strip(), host))
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nok")
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    guard = URLGuard(resolver=lambda host, requested_port: ["127.0.0.1"])
    transport = PinnedAsyncHTTPTransport(limits=httpx.Limits(max_connections=2), trust_env=False)
    try:
        async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
            for _ in range(2):
                response = await guard.safe_request(
                    client,
                    "GET",
                    f"http://provider.example:{port}/health",
                    allow_private_network=True,
                    timeout=2.0,
                )
                assert response.status_code == 200
                assert response.content == b"ok"
                await response.aclose()
        assert seen == [
            ("GET /health HTTP/1.1", f"provider.example:{port}"),
            ("GET /health HTTP/1.1", f"provider.example:{port}"),
        ]
        # A validated connection must not be reused for another request: that
        # request may resolve to a different address or use a stricter policy.
        assert connections == 2
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_pinned_transport_refuses_requests_without_urlguard_authorization():
    transport = PinnedAsyncHTTPTransport(limits=httpx.Limits(max_connections=1), trust_env=False)
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        with pytest.raises(httpx.ConnectError):
            await client.get("http://127.0.0.1:1/should-not-connect", timeout=1.0)
