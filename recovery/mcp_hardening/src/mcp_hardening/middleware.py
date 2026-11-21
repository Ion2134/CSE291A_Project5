"""Two-way per-call signing middleware for MCP-shaped tool calls.

Request  (client → server): ``params["_meta"]["mcp_hardening"] = {v, kid, nonce, ts, sig?}``
    sig = HMAC-SHA256(k_req, "v1|req|kid|tool|nonce|ts|canonical(arguments)")
Response (server → client): ``result["_meta"]["mcp_hardening"] = {v, kid, nonce, sig}``
    sig = HMAC-SHA256(k_resp, "v1|resp|kid|tool|nonce|canonical(structuredContent)|isError")

The response is bound to the request nonce, so a response can't be replayed onto another call.
Mandate  (user → server, carried by the agent): ``params["_meta"]["mcp_hardening_mandate"]``; for a
    protected tool the server checks the call against the user's signed rules (``mandate.py``).

Each check runs only when its toggle in ``HardeningConfig`` is on; any combination is valid.
All checks are constant-time HMAC compares plus a dict lookup - no I/O, no asymmetric crypto.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Callable, NoReturn, Protocol

from opentelemetry import trace

from mcp_hardening import errors
from mcp_hardening.canonical import request_mac_input, response_mac_input
from mcp_hardening.config import HardeningConfig
from mcp_hardening.errors import HardeningError
from mcp_hardening.keys import Keyring
from mcp_hardening.mandate import MANDATE_KEY, MandateUses, Resolvers, sign_mandate, verify_mandate
from mcp_hardening.replay import NonceCache
from mcp_hardening.telemetry import semconv

log = logging.getLogger(__name__)

ENVELOPE_KEY = "mcp_hardening"
ENVELOPE_VERSION = 1


def _mac(key: bytes, message: bytes) -> str:
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def _span_set(**attrs: Any) -> None:
    """Record attributes on the active OTel span (no-op when there isn't one)."""
    span = trace.get_current_span()
    if not span.is_recording():
        return
    for key, value in attrs.items():
        if value is not None:
            span.set_attribute(key, value)


def envelope_from(meta: Any) -> dict[str, Any] | None:
    """Pull the hardening envelope out of an MCP ``_meta`` (or A2A metadata) dict."""
    if not isinstance(meta, dict):
        return None
    env = meta.get(ENVELOPE_KEY)
    return env if isinstance(env, dict) else None


def failure_reason_from_message(message: str | None) -> str | None:
    """Recover the reason from a stringified ``HardeningError`` (e.g. a failed A2A task over HTTP)."""
    if not message or "mcp_hardening:" not in message:
        return None
    tail = message.split("mcp_hardening:", 1)[1]
    reason = tail.split(" ", 1)[0].strip()
    return reason if reason in errors.FAILURE_REASONS else None


class ToolServer(Protocol):
    def tools_list(self) -> dict[str, Any]: ...

    def tools_call(self, params: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class ClientMiddleware:
    """Runs on the caller side: signs outgoing calls, verifies incoming results."""

    config: HardeningConfig
    keyring: Keyring | None
    clock: Callable[[], float] = time.time

    def sign(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Envelope to attach to the call, or None when every security mechanism is off."""
        if not self.config.security_enabled:
            return None
        assert self.keyring is not None
        env: dict[str, Any] = {
            "v": ENVELOPE_VERSION,
            "kid": self.keyring.kid,
            "nonce": secrets.token_hex(16),
            "ts": int(self.clock()),
        }
        if self.config.request_auth.enabled:
            env["sig"] = _mac(
                self.keyring.request_key,
                request_mac_input(env["kid"], tool, env["nonce"], env["ts"], arguments),
            )
        _span_set(**{semconv.HARDENING_REQUEST_SIGNED: "sig" in env, semconv.HARDENING_KID: env["kid"]})
        return env

    def attach(self, params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Sign an MCP ``tools/call`` params dict. Returns (params_with_meta, envelope)."""
        env = self.sign(str(params.get("name", "")), dict(params.get("arguments") or {}))
        if env is None:
            return params, None
        meta = dict(params.get("_meta") or {})
        meta[ENVELOPE_KEY] = env
        return {**params, "_meta": meta}, env

    def verify_response(
        self,
        tool: str,
        request_envelope: dict[str, Any] | None,
        result: dict[str, Any],
    ) -> None:
        """Check the server's response signature over the content actually received.

        ``result`` is MCP-shaped: ``{"structuredContent", "isError", "_meta"}``. Raises HardeningError.
        """
        if not self.config.response_auth.enabled:
            return
        assert self.keyring is not None
        env = envelope_from(result.get("_meta"))
        if env is None or not isinstance(env.get("sig"), str):
            self._fail(errors.MISSING_RESPONSE_SIG)
        assert env is not None
        nonce = env.get("nonce")
        if request_envelope is not None and nonce != request_envelope.get("nonce"):
            self._fail(errors.BAD_RESPONSE_SIG, "response not bound to this request's nonce")
        if env.get("kid") != self.keyring.kid:
            self._fail(errors.UNKNOWN_KID, "response signed with an unknown key")
        expected = _mac(
            self.keyring.response_key,
            response_mac_input(
                self.keyring.kid,
                tool,
                str(nonce),
                result.get("structuredContent"),
                bool(result.get("isError", False)),
            ),
        )
        if not hmac.compare_digest(expected, env["sig"]):
            self._fail(errors.BAD_RESPONSE_SIG)
        _span_set(**{semconv.HARDENING_RESPONSE_VERIFIED: True})

    @staticmethod
    def _fail(reason: str, detail: str = "") -> NoReturn:
        _span_set(
            **{
                semconv.HARDENING_RESPONSE_VERIFIED: False,
                semconv.HARDENING_FAILURE_REASON: reason,
                semconv.HARDENING_FAILURE_STAGE: "response",
            }
        )
        raise HardeningError(reason, detail)


@dataclass
class ServerMiddleware:
    """Runs in front of the real tool server: verifies calls before the handler, signs results."""

    config: HardeningConfig
    keyring: Keyring | None
    nonces: NonceCache | None
    clock: Callable[[], float] = time.time
    mandate_uses: MandateUses | None = None  # set when mandates are single-use

    def verify_request(self, params: dict[str, Any]) -> dict[str, Any] | None:
        """Run every enabled request check. Returns the envelope (if any). Raises HardeningError."""
        cfg = self.config
        env = envelope_from(params.get("_meta"))
        if not (cfg.request_auth.enabled or cfg.freshness_check.enabled or cfg.replay_protection.enabled):
            return env
        if env is None:
            self._fail(errors.MISSING_ENVELOPE)
        assert env is not None
        tool = str(params.get("name", ""))
        nonce, ts = env.get("nonce"), env.get("ts")

        if cfg.request_auth.enabled:
            assert self.keyring is not None
            if env.get("kid") != self.keyring.kid:
                self._fail(errors.UNKNOWN_KID, f"kid={env.get('kid')!r}")
            sig = env.get("sig")
            if not isinstance(sig, str) or not isinstance(nonce, str) or not isinstance(ts, int):
                self._fail(errors.BAD_REQUEST_SIG, "malformed envelope")
            expected = _mac(
                self.keyring.request_key,
                request_mac_input(self.keyring.kid, tool, nonce, ts, dict(params.get("arguments") or {})),
            )
            if not hmac.compare_digest(expected, sig):
                self._fail(errors.BAD_REQUEST_SIG)

        if cfg.freshness_check.enabled:
            if not isinstance(ts, int) or isinstance(ts, bool):
                self._fail(errors.STALE_TIMESTAMP, "missing timestamp")
            skew = abs(self.clock() - ts)
            if skew > cfg.freshness_check.window_s:
                self._fail(errors.STALE_TIMESTAMP, f"skew {skew:.0f}s > {cfg.freshness_check.window_s:.0f}s")

        # Replay last, so a forged or stale request can't burn a legitimate nonce.
        if cfg.replay_protection.enabled:
            assert self.nonces is not None
            if not isinstance(nonce, str) or not nonce:
                self._fail(errors.REPLAYED_NONCE, "missing nonce")
            if not self.nonces.check_and_add(nonce):
                self._fail(errors.REPLAYED_NONCE)

        _span_set(**{semconv.HARDENING_REQUEST_VERIFIED: True})
        return env

    def verify_mandate(self, params: dict[str, Any], resolvers: Resolvers) -> bool:
        """Check a protected tool call against the user's signed mandate. Returns whether a check ran.

        Raises HardeningError (stage ``request``) when the mandate is missing, forged, stale, or does
        not allow this call. Unprotected tools pass through untouched.
        """
        cfg = self.config.mandate_binding
        tool = str(params.get("name", ""))
        if not cfg.enabled or tool not in cfg.protected_tools:
            return False
        assert self.keyring is not None
        meta = params.get("_meta")
        token = meta.get(MANDATE_KEY) if isinstance(meta, dict) else None
        try:
            verify_mandate(token, tool, dict(params.get("arguments") or {}), key=self.keyring.mandate_key,
                           kid=self.keyring.kid, resolvers=resolvers, max_age_s=cfg.max_age_s, clock=self.clock,
                           uses=self.mandate_uses)
        except HardeningError as exc:
            self._fail(exc.reason, exc.detail)
        _span_set(**{semconv.HARDENING_MANDATE_VERIFIED: True})
        return True

    def record_mandate_use(self, params: dict[str, Any]) -> None:
        """Count one use of the call's mandate. Call only after the protected tool actually ran."""
        if self.mandate_uses is None:
            return
        token = (params.get("_meta") or {}).get(MANDATE_KEY) or {}
        self.mandate_uses.record(str(token.get("id")), str(params.get("name", "")), int(token.get("iat", 0)))

    def sign_response(
        self, tool: str, request_envelope: dict[str, Any] | None, result: dict[str, Any]
    ) -> dict[str, Any]:
        if not self.config.response_auth.enabled:
            return result
        assert self.keyring is not None
        nonce = (request_envelope or {}).get("nonce")
        if not isinstance(nonce, str) or not nonce:
            nonce = secrets.token_hex(16)
        sig = _mac(
            self.keyring.response_key,
            response_mac_input(
                self.keyring.kid,
                tool,
                nonce,
                result.get("structuredContent"),
                bool(result.get("isError", False)),
            ),
        )
        meta = dict(result.get("_meta") or {})
        meta[ENVELOPE_KEY] = {"v": ENVELOPE_VERSION, "kid": self.keyring.kid, "nonce": nonce, "sig": sig}
        return {**result, "_meta": meta}

    @staticmethod
    def _fail(reason: str, detail: str = "") -> NoReturn:
        _span_set(
            **{
                semconv.HARDENING_REQUEST_VERIFIED: False,
                semconv.HARDENING_FAILURE_REASON: reason,
                semconv.HARDENING_FAILURE_STAGE: "request",
            }
        )
        raise HardeningError(reason, detail)


@dataclass
class HardeningRuntime:
    """Everything one run needs, built once and shared by all episodes and servers."""

    config: HardeningConfig
    keyring: Keyring | None
    nonces: NonceCache | None
    client: ClientMiddleware = field(init=False)
    server: ServerMiddleware = field(init=False)
    clock: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        self.client = ClientMiddleware(self.config, self.keyring, self.clock)
        mandates = self.config.mandate_binding
        uses = MandateUses(mandates.max_age_s, self.clock) if mandates.enabled and mandates.single_use else None
        self.server = ServerMiddleware(self.config, self.keyring, self.nonces, self.clock, uses)

    @classmethod
    def build(
        cls,
        config: HardeningConfig,
        keyring: Keyring | None = None,
        *,
        allow_ephemeral: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> HardeningRuntime:
        """Build a runtime. The key comes from ``keyring``, else the encrypted env vars, else (if
        ``allow_ephemeral``) a random in-memory key. Raises if a key is needed and none is available."""
        if config.security_enabled and keyring is None:
            keyring = Keyring.from_env()
            if keyring is None:
                if not allow_ephemeral:
                    raise ValueError(
                        "MCP hardening is enabled but MCP_HARDENING_MASTER_KEY / MCP_HARDENING_API_KEY_ENC "
                        "are not set (run `mcp-hardening keygen`)"
                    )
                log.warning("MCP hardening: no encrypted API key in env; using an ephemeral in-memory key")
                keyring = Keyring.ephemeral_key()
        nonces = None
        if config.replay_protection.enabled:
            nonces = NonceCache(
                ttl_s=config.replay_protection.nonce_ttl_s,
                max_entries=config.replay_protection.max_nonces,
                clock=clock,
            )
        return cls(config=config, keyring=keyring, nonces=nonces, clock=clock)

    @property
    def enabled(self) -> bool:
        return self.config.security_enabled

    def wrap(self, server: ToolServer, resolvers: Resolvers | None = None) -> ToolServer:
        """Put the server-side middleware in front of ``server`` (identity when security is off).

        ``resolvers`` are the lookups (name -> function) a user mandate's rules may call; the tool
        server owns them, so the hardening layer holds no domain rules of its own.
        """
        return HardenedToolServer(server, self, resolvers) if self.enabled else server

    def sign_mandate(self, mandate: dict[str, Any] | None) -> dict[str, Any] | None:
        """The user's side: sign ``mandate`` for the agent to carry (None when mandate binding is off)."""
        if mandate is None or not self.config.mandate_binding.enabled:
            return None
        assert self.keyring is not None
        return sign_mandate(self.keyring.mandate_key, self.keyring.kid, mandate, clock=self.clock)


class HardenedToolServer:
    """Duck-typed MCP tool server wrapper: verify request → inner handler → sign response."""

    def __init__(self, inner: ToolServer, runtime: HardeningRuntime, resolvers: Resolvers | None = None) -> None:
        self._inner = inner
        self._runtime = runtime
        self._resolvers = dict(resolvers or {})

    @property
    def inner(self) -> ToolServer:
        return self._inner

    def tools_list(self) -> dict[str, Any]:
        return self._inner.tools_list()

    def tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        env = self._runtime.server.verify_request(params)
        checked = self._runtime.server.verify_mandate(params, self._resolvers)
        result = self._inner.tools_call(params)  # a failed tool raises, so it never counts as a use
        if checked:
            self._runtime.server.record_mandate_use(params)
        return self._runtime.server.sign_response(str(params.get("name", "")), env, result)
