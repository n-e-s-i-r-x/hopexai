from __future__ import annotations

import asyncio
import ipaddress
import secrets
import socket
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urljoin, urlsplit

import httpcore
import httpx
from httpcore._backends.auto import AutoBackend
from fastapi import Request


class UnsafeUpstreamURL(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _PinnedDestination:
    host: str
    port: int
    addresses: tuple[str, ...]


_PINNED_DESTINATION: ContextVar[_PinnedDestination | None] = ContextVar("hopexai_pinned_destination", default=None)


def _normalize_host(host: str) -> str:
    try:
        return ipaddress.ip_address(host).compressed.lower()
    except ValueError:
        return host.encode("idna").decode("ascii").rstrip(".").lower()


class _PinnedNetworkBackend:
    """Connect to the exact public/private IPs validated for this request."""

    def __init__(self):
        self.backend = AutoBackend()

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        pin = _PINNED_DESTINATION.get()
        if pin is None or _normalize_host(host) != pin.host or port != pin.port:
            raise httpcore.ConnectError("outbound destination was not authorized")
        last_error = None
        for address in pin.addresses[:8]:
            try:
                return await self.backend.connect_tcp(
                    host=address,
                    port=port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise httpcore.ConnectError("outbound destination has no validated address")

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        return await self.backend.connect_unix_socket(path=path, timeout=timeout, socket_options=socket_options)

    async def sleep(self, seconds):
        await self.backend.sleep(seconds)


class PinnedAsyncHTTPTransport(httpx.AsyncHTTPTransport):
    """HTTPX transport whose TCP connections use URLGuard's validated IP pin."""

    def __init__(self, *, limits: httpx.Limits | None = None, trust_env: bool = False, verify=True, cert=None):
        super().__init__(limits=limits, trust_env=trust_env, verify=verify, cert=cert, proxy=None)
        pool = self._pool
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=pool._ssl_context,
            max_connections=pool._max_connections,
            # Reuse could carry a connection authorized under a different
            # request's DNS results or private-network policy. Re-resolve and
            # pin each outbound request instead.
            max_keepalive_connections=0,
            keepalive_expiry=pool._keepalive_expiry,
            http1=pool._http1,
            http2=pool._http2,
            retries=pool._retries,
            local_address=pool._local_address,
            uds=pool._uds,
            network_backend=_PinnedNetworkBackend(),
            socket_options=pool._socket_options,
        )


class URLGuard:
    """Reject SSRF destinations before every outbound request.

    Private networks are denied by default. A provider base URL may opt in
    explicitly for a trusted internal service; redirects to a different origin
    are never followed, and redirect destinations are always checked as public.
    """

    def __init__(self, resolver: Callable[[str, int], list[str]] | None = None):
        self.resolver = resolver or self._resolve

    @staticmethod
    def _resolve(host: str, port: int) -> list[str]:
        values = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        return list({record[4][0] for record in values})

    async def validate(self, url: str | httpx.URL, allow_private_network: bool = False) -> list[str]:
        parsed = urlsplit(str(url))
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            raise UnsafeUpstreamURL("upstream URL must use http(s) and have a host")
        if parsed.scheme == "http" and not allow_private_network:
            raise UnsafeUpstreamURL("plain HTTP requires an explicit private-network opt-in")
        if parsed.username is not None or parsed.password is not None:
            raise UnsafeUpstreamURL("userinfo is forbidden in upstream URLs")
        if parsed.fragment:
            raise UnsafeUpstreamURL("fragments are forbidden in upstream URLs")
        host = parsed.hostname.rstrip(".").lower()
        if host in {"metadata", "metadata.google.internal"}:
            raise UnsafeUpstreamURL("cloud metadata hosts are forbidden")
        if host == "localhost" and not allow_private_network:
            raise UnsafeUpstreamURL("localhost is forbidden")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            addresses = [ipaddress.ip_address(host)]
        except ValueError:
            try:
                resolved = await asyncio.to_thread(self.resolver, host, port)
            except (OSError, ValueError, socket.gaierror) as exc:
                raise UnsafeUpstreamURL("upstream hostname did not resolve") from exc
            if not resolved:
                raise UnsafeUpstreamURL("upstream hostname did not resolve")
            try:
                addresses = [ipaddress.ip_address(item) for item in resolved]
            except ValueError as exc:
                raise UnsafeUpstreamURL("upstream hostname returned an invalid address") from exc
        for address in addresses:
            effective_address = address.ipv4_mapped if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped else address
            if self._is_metadata(address) or self._is_metadata(effective_address):
                raise UnsafeUpstreamURL("cloud metadata addresses are forbidden")
            if (
                address.is_multicast
                or address.is_unspecified
                or address.is_reserved
                or (isinstance(address, ipaddress.IPv6Address) and (address.sixtofour is not None or address.teredo is not None))
            ):
                raise UnsafeUpstreamURL("upstream resolved to a reserved or non-routable address")
            if not allow_private_network and not effective_address.is_global:
                raise UnsafeUpstreamURL("upstream resolved to a non-public network address")
            if parsed.scheme == "http" and effective_address.is_global:
                raise UnsafeUpstreamURL("plain HTTP is restricted to explicitly allowed private addresses")
        return [address.compressed for address in addresses]

    @staticmethod
    def _is_metadata(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        metadata = [
            ipaddress.ip_network("169.254.169.254/32"),
            ipaddress.ip_network("169.254.170.2/32"),
            ipaddress.ip_network("100.100.100.200/32"),
        ]
        if isinstance(address, ipaddress.IPv6Address):
            metadata.append(ipaddress.ip_network("fd00:ec2::254/128"))
        return any(address in network for network in metadata)

    async def safe_stream(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        allow_private_network: bool,
        timeout: httpx.Timeout | float,
        max_redirects: int = 2,
        **kwargs,
    ) -> httpx.Response:
        current = url
        request_method = method.upper()
        request_allow_private = allow_private_network
        for redirect_count in range(max_redirects + 1):
            addresses = await self.validate(current, allow_private_network=request_allow_private)
            parsed_current = urlsplit(current)
            host = parsed_current.hostname or ""
            port = parsed_current.port or (443 if parsed_current.scheme.lower() == "https" else 80)
            pin = _PinnedDestination(_normalize_host(host), port, tuple(addresses))
            pin_token = _PINNED_DESTINATION.set(pin)
            try:
                request = client.build_request(request_method, current, timeout=timeout, **kwargs)
                response = await client.send(request, stream=True, follow_redirects=False)
            finally:
                _PINNED_DESTINATION.reset(pin_token)
            if response.status_code not in {301, 302, 303, 307, 308}:
                return response
            location = response.headers.get("location")
            if not location or redirect_count >= max_redirects or request_method not in {"GET", "HEAD", "OPTIONS"}:
                return response
            target = urljoin(current, location)
            old_parts, new_parts = urlsplit(current), urlsplit(target)
            try:
                old_origin = (old_parts.scheme.lower(), old_parts.hostname.lower() if old_parts.hostname else None, old_parts.port)
                new_origin = (new_parts.scheme.lower(), new_parts.hostname.lower() if new_parts.hostname else None, new_parts.port)
            except ValueError:
                await response.aclose()
                return response
            if old_origin != new_origin:
                # Never forward credentials across origins, even for a redirect
                # that otherwise appears safe.
                return response
            try:
                await self.validate(target, allow_private_network=False)
            except UnsafeUpstreamURL:
                return response
            await response.aclose()
            current = target
            request_allow_private = False
        raise UnsafeUpstreamURL("too many upstream redirects")

    async def safe_request(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        allow_private_network: bool,
        timeout: httpx.Timeout | float,
        max_redirects: int = 2,
        **kwargs,
    ) -> httpx.Response:
        response = await self.safe_stream(
            client,
            method,
            url,
            allow_private_network=allow_private_network,
            timeout=timeout,
            max_redirects=max_redirects,
            **kwargs,
        )
        await response.aread()
        return response


def get_client_ip(
    request: Request,
    trusted_proxy_cidrs: tuple[str, ...] = (),
    *,
    vercel_proxy_secret: str | None = None,
) -> str:
    peer = request.client.host if request.client else "0.0.0.0"
    try:
        peer_ip = ipaddress.ip_address(peer)
    except ValueError:
        return "0.0.0.0"

    # Vercel rewrites identify the visitor in X-Forwarded-For. Trust that value
    # only when Vercel has overwritten a private marker header with the shared
    # deployment secret; do not trust a browser-supplied forwarding header.
    if vercel_proxy_secret:
        supplied_secret = request.headers.get("x-hopexai-vercel-proxy-secret", "")
        if supplied_secret and secrets.compare_digest(supplied_secret, vercel_proxy_secret):
            forwarded = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
            try:
                return str(ipaddress.ip_address(forwarded))
            except ValueError:
                return str(peer_ip)

    trusted_networks = []
    for cidr in trusted_proxy_cidrs:
        try:
            trusted_networks.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            continue

    def is_trusted(value: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        return any(value.version == network.version and value in network for network in trusted_networks)

    if not is_trusted(peer_ip):
        return str(peer_ip)
    forwarded = request.headers.get("x-forwarded-for", "")
    chain: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for raw in forwarded.split(","):
        try:
            chain.append(ipaddress.ip_address(raw.strip()))
        except ValueError:
            continue
    current = peer_ip
    for candidate in reversed(chain):
        if not is_trusted(current):
            break
        current = candidate
    return str(current)
