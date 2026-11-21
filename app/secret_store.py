from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
import re
import stat
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class SecretStoreError(RuntimeError):
    pass


class SecretStore:
    """Stable key material for config encryption and keyed digests.

    Traditional deployments keep the key in a permission-restricted file. Vercel
    deployments supply the same 32-byte key through a server-only environment
    variable because function filesystems are ephemeral. The key is never
    returned through the HTTP API.
    """

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.key_path = state_dir / "master.key"
        self._key: bytes | None = None

    def ensure(self) -> bytes:
        if self._key is not None:
            return self._key
        configured_key = os.getenv("HOPEXAI_MASTER_KEY")
        if configured_key:
            try:
                if re.fullmatch(r"[0-9a-fA-F]{64}", configured_key):
                    material = bytes.fromhex(configured_key)
                else:
                    padded = configured_key + "=" * (-len(configured_key) % 4)
                    material = base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
            except (binascii.Error, ValueError, UnicodeEncodeError) as exc:
                raise SecretStoreError("HOPEXAI_MASTER_KEY must be a 32-byte base64url or 64-character hex value") from exc
            if len(material) != 32:
                raise SecretStoreError("HOPEXAI_MASTER_KEY must decode to exactly 32 bytes")
            self._key = material
            return material
        if os.getenv("VERCEL", "").strip().lower() in {"1", "true", "yes"}:
            raise SecretStoreError("HOPEXAI_MASTER_KEY is required on Vercel and must remain stable across deployments")
        if self.state_dir.is_symlink():
            raise SecretStoreError("state directory must not be a symbolic link")
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.chmod(self.state_dir, 0o700)
        except OSError:
            pass
        if self.key_path.is_symlink():
            raise SecretStoreError("master key path must not be a symbolic link")
        try:
            fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            try:
                material = os.urandom(32)
                os.write(fd, material)
                os.fsync(fd)
            finally:
                os.close(fd)
        try:
            info = self.key_path.stat()
            if not stat.S_ISREG(info.st_mode):
                raise SecretStoreError("master key path is not a regular file")
            if info.st_mode & 0o077:
                os.chmod(self.key_path, 0o600)
            material = self.key_path.read_bytes()
        except OSError as exc:
            raise SecretStoreError("unable to read the persistent master key") from exc
        if len(material) != 32:
            raise SecretStoreError("master key file must contain exactly 32 bytes")
        self._key = material
        return material

    @property
    def key(self) -> bytes:
        if self._key is None:
            return self.ensure()
        return self._key

    def digest(self, value: str | bytes, purpose: str) -> str:
        raw = value.encode("utf-8") if isinstance(value, str) else value
        purpose_key = hmac.new(self.key, ("hopexai:" + purpose).encode("utf-8"), hashlib.sha256).digest()
        return hmac.new(purpose_key, raw, hashlib.sha256).hexdigest()

    def encrypt_config_secret(self, plaintext: str) -> str:
        nonce = os.urandom(12)
        encrypted = AESGCM(self.key).encrypt(nonce, plaintext.encode("utf-8"), b"hopexai/config-secret/v1")
        token = base64.urlsafe_b64encode(nonce + encrypted).decode("ascii").rstrip("=")
        return f"ENC[v1:{token}]"

    def decrypt_config_secret(self, value: str) -> str:
        if not (value.startswith("ENC[v1:") and value.endswith("]")):
            return value
        token = value[7:-1]
        try:
            padded = token + "=" * (-len(token) % 4)
            raw = base64.urlsafe_b64decode(padded.encode("ascii"))
            nonce, ciphertext = raw[:12], raw[12:]
            if len(nonce) != 12 or not ciphertext:
                raise ValueError("invalid encrypted value")
            return AESGCM(self.key).decrypt(nonce, ciphertext, b"hopexai/config-secret/v1").decode("utf-8")
        except Exception as exc:
            raise SecretStoreError("encrypted configuration value could not be decrypted") from exc

    def encrypt_blob(self, plaintext: bytes) -> str:
        nonce = os.urandom(12)
        ciphertext = AESGCM(self.key).encrypt(nonce, plaintext, b"hopexai/response-state/v1")
        return "v1:" + base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii").rstrip("=")

    def decrypt_blob(self, token: str) -> bytes:
        if not token.startswith("v1:"):
            raise SecretStoreError("unsupported encrypted state version")
        try:
            value = token[3:]
            raw = base64.urlsafe_b64decode((value + "=" * (-len(value) % 4)).encode("ascii"))
            return AESGCM(self.key).decrypt(raw[:12], raw[12:], b"hopexai/response-state/v1")
        except Exception as exc:
            raise SecretStoreError("encrypted response state could not be decrypted") from exc
