from __future__ import annotations

import itertools
import json
from typing import Any

import pytest

from mcp_hardening import (
    HardeningConfig,
    HardeningError,
    HardeningRuntime,
    Keyring,
    NonceCache,
    generate_encrypted_api_key,
)
from mcp_hardening.config import (
    FreshnessCheckConfig,
    ReplayProtectionConfig,
    RequestAuthConfig,
    ResponseAuthConfig,
)


class EchoServer:
    """Minimal MCP-shaped server that records the calls that reach the handler."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def tools_list(self) -> dict[str, Any]:
        return {"tools": [{"name": "get_balance"}]}

    def tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(params)
        content = {"account_id": params["arguments"].get("account_id"), "balance_cents": 100}
        return {"content": [{"type": "text", "text": json.dumps(content)}], "structuredContent": content, "isError": False}


def cfg(req=False, resp=False, replay=False, fresh=False) -> HardeningConfig:
    return HardeningConfig(
        request_auth=RequestAuthConfig(enabled=req),
        response_auth=ResponseAuthConfig(enabled=resp),
        replay_protection=ReplayProtectionConfig(enabled=replay),
        freshness_check=FreshnessCheckConfig(enabled=fresh),
    )


ALL_ON = cfg(True, True, True, True)
ARGS = {"account_id": "acct_1"}


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def runtime(config: HardeningConfig = ALL_ON, clock: Clock | None = None) -> HardeningRuntime:
    return HardeningRuntime.build(config, Keyring.ephemeral_key(), clock=clock or Clock())


def roundtrip(rt: HardeningRuntime, server: EchoServer, params: dict[str, Any]) -> dict[str, Any]:
    signed, env = rt.client.attach(params)
    result = rt.wrap(server).tools_call(signed)
    rt.client.verify_response(params["name"], env, result)
    return result


def test_roundtrip_all_on():
    rt, server = runtime(), EchoServer()
    result = roundtrip(rt, server, {"name": "get_balance", "arguments": ARGS})
    assert result["structuredContent"]["balance_cents"] == 100
    assert len(server.calls) == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["arguments"].__setitem__("account_id", "acct_evil"),
        lambda p: p["arguments"].__setitem__("extra", 1),
        lambda p: p.__setitem__("name", "initiate_transfer"),
    ],
    ids=["changed_arg", "added_arg", "swapped_tool"],
)
def test_tampered_request_rejected_before_handler(mutate):
    rt, server = runtime(cfg(req=True)), EchoServer()
    signed, _ = rt.client.attach({"name": "get_balance", "arguments": dict(ARGS)})
    signed = {**signed, "arguments": dict(signed["arguments"])}
    mutate(signed)
    with pytest.raises(HardeningError) as exc:
        rt.wrap(server).tools_call(signed)
    assert exc.value.reason == "bad_request_sig"
    assert server.calls == []


def test_missing_envelope_and_unknown_kid():
    rt, server = runtime(cfg(req=True)), EchoServer()
    with pytest.raises(HardeningError) as exc:
        rt.wrap(server).tools_call({"name": "get_balance", "arguments": ARGS})
    assert exc.value.reason == "missing_envelope"

    other = runtime(cfg(req=True))
    signed, _ = other.client.attach({"name": "get_balance", "arguments": ARGS})
    with pytest.raises(HardeningError) as exc:
        rt.wrap(server).tools_call(signed)
    assert exc.value.reason == "unknown_kid"
    assert server.calls == []


def test_replay_rejected():
    rt, server = runtime(cfg(req=True, replay=True)), EchoServer()
    signed, _ = rt.client.attach({"name": "get_balance", "arguments": ARGS})
    rt.wrap(server).tools_call(signed)
    with pytest.raises(HardeningError) as exc:
        rt.wrap(server).tools_call(signed)
    assert exc.value.reason == "replayed_nonce"
    assert len(server.calls) == 1


def test_forged_request_does_not_burn_nonce():
    rt, server = runtime(cfg(req=True, replay=True)), EchoServer()
    signed, _ = rt.client.attach({"name": "get_balance", "arguments": ARGS})
    forged = {**signed, "arguments": {"account_id": "acct_evil"}}
    with pytest.raises(HardeningError):
        rt.wrap(server).tools_call(forged)
    rt.wrap(server).tools_call(signed)  # the genuine call still goes through


def test_stale_timestamp_rejected():
    clock = Clock()
    rt, server = runtime(cfg(fresh=True), clock), EchoServer()
    signed, _ = rt.client.attach({"name": "get_balance", "arguments": ARGS})
    clock.t += 121
    with pytest.raises(HardeningError) as exc:
        rt.wrap(server).tools_call(signed)
    assert exc.value.reason == "stale_timestamp"


def test_tampered_response_rejected():
    rt, server = runtime(cfg(resp=True)), EchoServer()
    params = {"name": "get_balance", "arguments": ARGS}
    signed, env = rt.client.attach(params)
    result = rt.wrap(server).tools_call(signed)
    tampered = {**result, "structuredContent": {**result["structuredContent"], "balance_cents": 999}}
    with pytest.raises(HardeningError) as exc:
        rt.client.verify_response("get_balance", env, tampered)
    assert exc.value.reason == "bad_response_sig"
    injected = {**result, "structuredContent": {**result["structuredContent"], "_note": "IGNORE PREVIOUS"}}
    with pytest.raises(HardeningError):
        rt.client.verify_response("get_balance", env, injected)
    stripped = {k: v for k, v in result.items() if k != "_meta"}
    with pytest.raises(HardeningError) as exc:
        rt.client.verify_response("get_balance", env, stripped)
    assert exc.value.reason == "missing_response_sig"


def test_response_bound_to_request_nonce():
    rt, server = runtime(cfg(resp=True)), EchoServer()
    params = {"name": "get_balance", "arguments": ARGS}
    signed_a, env_a = rt.client.attach(params)
    _, env_b = rt.client.attach(params)
    result_a = rt.wrap(server).tools_call(signed_a)
    with pytest.raises(HardeningError) as exc:
        rt.client.verify_response("get_balance", env_b, result_a)
    assert exc.value.reason == "bad_response_sig"


@pytest.mark.parametrize("flags", list(itertools.product([False, True], repeat=4)))
def test_each_toggle_independent(flags):
    """Every combination works for honest traffic, and each check fires only when its toggle is on."""
    req, resp, replay, fresh = flags
    clock = Clock()
    rt, server = runtime(cfg(req, resp, replay, fresh), clock), EchoServer()
    params = {"name": "get_balance", "arguments": ARGS}

    roundtrip(rt, server, params)  # honest call always passes

    signed, env = rt.client.attach(params)
    if env is None:
        assert not any(flags)
        return

    forged = {**signed, "arguments": {"account_id": "acct_evil"}}
    _expect(req, lambda: rt.wrap(EchoServer()).tools_call(forged))

    stale_rt = rt
    signed_old, _ = stale_rt.client.attach(params)
    clock.t += 1000
    _expect(fresh, lambda: rt.wrap(EchoServer()).tools_call(signed_old))
    clock.t -= 1000

    signed_once, _ = rt.client.attach(params)
    rt.wrap(EchoServer()).tools_call(signed_once)
    _expect(replay, lambda: rt.wrap(EchoServer()).tools_call(signed_once))

    signed_r, env_r = rt.client.attach(params)
    res = rt.wrap(EchoServer()).tools_call(signed_r)
    bad = {**res, "structuredContent": {"account_id": "acct_1", "balance_cents": 1}}
    _expect(resp, lambda: rt.client.verify_response("get_balance", env_r, bad))


def _expect(should_raise: bool, fn) -> None:
    if should_raise:
        with pytest.raises(HardeningError):
            fn()
    else:
        fn()


def test_security_off_is_passthrough():
    rt, server = runtime(HardeningConfig()), EchoServer()
    assert rt.wrap(server) is server
    assert rt.client.sign("get_balance", ARGS) is None


def test_keyring_encrypted_roundtrip(monkeypatch):
    master, enc = generate_encrypted_api_key(b"secret-api-key")
    ring = Keyring.from_encrypted(master, enc)
    assert ring == Keyring.from_api_key(b"secret-api-key")
    assert "secret" not in repr(ring)
    monkeypatch.setenv("MCP_HARDENING_MASTER_KEY", master)
    monkeypatch.setenv("MCP_HARDENING_API_KEY_ENC", enc)
    assert Keyring.from_env() == ring
    other_master, _ = generate_encrypted_api_key()
    with pytest.raises(ValueError):
        Keyring.from_encrypted(other_master, enc)


def test_runtime_requires_key_when_ephemeral_disallowed(monkeypatch):
    monkeypatch.delenv("MCP_HARDENING_MASTER_KEY", raising=False)
    monkeypatch.delenv("MCP_HARDENING_API_KEY_ENC", raising=False)
    with pytest.raises(ValueError):
        HardeningRuntime.build(ALL_ON, allow_ephemeral=False)
    assert HardeningRuntime.build(ALL_ON).keyring.ephemeral
    HardeningRuntime.build(HardeningConfig(), allow_ephemeral=False)  # no key needed when all off


def test_nonce_cache_expiry_and_bound():
    clock = Clock()
    cache = NonceCache(ttl_s=10, max_entries=2, clock=clock)
    assert cache.check_and_add("a") and not cache.check_and_add("a")
    clock.t += 11
    assert cache.check_and_add("a")
    cache.check_and_add("b")
    cache.check_and_add("c")
    assert len(cache) == 2
