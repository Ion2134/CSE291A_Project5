"""Encrypted API key handling.

The API key is never stored in plaintext. ``.env`` holds:

- ``MCP_HARDENING_MASTER_KEY``  - a Fernet key (the key-encryption key)
- ``MCP_HARDENING_API_KEY_ENC`` - the API key, Fernet-encrypted under the master key

The key is decrypted once when a ``Keyring`` is built. Independent HMAC subkeys are derived from it with
HKDF-SHA256: request and response (per-call verification is one HMAC each way), and mandate (the
user's side signs mandates with it; see ``mandate.py``).
"""

from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import dataclass, field

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

MASTER_KEY_ENV = "MCP_HARDENING_MASTER_KEY"
API_KEY_ENC_ENV = "MCP_HARDENING_API_KEY_ENC"

_HKDF_SALT = b"mcp-hardening/v1"


def _derive(api_key: bytes, info: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=_HKDF_SALT, info=info).derive(api_key)


@dataclass(frozen=True)
class Keyring:
    kid: str
    request_key: bytes = field(repr=False)
    response_key: bytes = field(repr=False)
    ephemeral: bool = False
    mandate_key: bytes = field(default=b"", repr=False)

    @classmethod
    def from_api_key(cls, api_key: bytes, *, ephemeral: bool = False) -> Keyring:
        if not api_key:
            raise ValueError("API key is empty")
        return cls(
            kid=hashlib.sha256(api_key).hexdigest()[:8],
            request_key=_derive(api_key, b"request"),
            response_key=_derive(api_key, b"response"),
            ephemeral=ephemeral,
            mandate_key=_derive(api_key, b"mandate"),
        )

    @classmethod
    def from_encrypted(cls, master_key: str | bytes, api_key_enc: str | bytes) -> Keyring:
        try:
            api_key = Fernet(master_key).decrypt(api_key_enc)
        except (InvalidToken, ValueError) as exc:
            raise ValueError("could not decrypt MCP hardening API key (wrong master key or corrupt token)") from exc
        return cls.from_api_key(api_key)

    @classmethod
    def from_env(cls) -> Keyring | None:
        """Keyring from the two env vars, or None if they aren't set."""
        master = os.environ.get(MASTER_KEY_ENV)
        enc = os.environ.get(API_KEY_ENC_ENV)
        if not master or not enc:
            return None
        return cls.from_encrypted(master, enc)

    @classmethod
    def ephemeral_key(cls) -> Keyring:
        """Random in-memory key. Fine for in-process simulation where client and server share it."""
        return cls.from_api_key(secrets.token_bytes(32), ephemeral=True)


def generate_encrypted_api_key(api_key: bytes | None = None) -> tuple[str, str]:
    """Return ``(master_key, api_key_enc)`` for a new (or given) API key."""
    master = Fernet.generate_key()
    key = api_key or secrets.token_urlsafe(32).encode()
    return master.decode(), Fernet(master).encrypt(key).decode()
