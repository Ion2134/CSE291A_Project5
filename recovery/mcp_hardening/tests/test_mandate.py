"""Signed user mandates: the rules are data, the facts come from the server's resolvers."""

import pytest

from mcp_hardening import MANDATE_KEY, HardeningConfig, HardeningError, HardeningRuntime, Keyring, load_config
from mcp_hardening.config import MandateBindingConfig
from mcp_hardening.mandate import MandateEvalError, evaluate

BALANCES = {"acct_a": 80_000, "acct_b": 120_000}
RESOLVERS = {"balance": lambda account: BALANCES[account], "accounts": lambda: list(BALANCES)}


def arg(name):
    return {"arg": name}


def op(name, *args, **extra):
    return {"op": name, "args": list(args), **extra}


def fn(name, *args):
    return {"fn": name, "args": list(args)}


HALF_OF_B = {"tools": {"pay": {"require": [
    op("eq", arg("from"), "acct_b"),
    op("eq", arg("amount"), op("//", fn("balance", "acct_b"), 2)),
    op("le", arg("amount"), 100_000),
]}}}


class Server:
    """A tool server that records what ran."""

    def __init__(self):
        self.ran = []

    def tools_list(self):
        return {"tools": []}

    def tools_call(self, params):
        self.ran.append((params["name"], params.get("arguments")))
        return {"structuredContent": {"ok": True}, "isError": False}


def runtime(config=None, clock=lambda: 1_000_000.0):
    config = config or HardeningConfig(mandate_binding=MandateBindingConfig(enabled=True, protected_tools=("pay",)))
    return HardeningRuntime.build(config, Keyring.ephemeral_key(), clock=clock)


def call(server, rt, tool, arguments, token):
    params = {"name": tool, "arguments": arguments, "_meta": {MANDATE_KEY: token} if token is not None else {}}
    return server.tools_call(params)


def test_evaluates_comparisons_arithmetic_and_ranking():
    assert evaluate(op("//", fn("balance", "acct_b"), 2), {}, RESOLVERS) == 60_000
    assert evaluate(op("argmax", fn("accounts"), key="balance"), {}, RESOLVERS) == "acct_b"
    assert evaluate(op("and", op("gt", arg("x"), 1), op("not", op("eq", arg("x"), 5))), {"x": 3}, RESOLVERS)
    with pytest.raises(MandateEvalError, match="unknown resolver"):
        evaluate(fn("secret_answer"), {}, RESOLVERS)
    with pytest.raises(MandateEvalError, match="unknown op"):
        evaluate(op("exec", "rm -rf /"), {}, RESOLVERS)


def test_allows_the_call_the_user_meant_and_refuses_others():
    rt = runtime()
    inner = Server()
    server = rt.wrap(inner, resolvers=RESOLVERS)
    token = rt.sign_mandate(HALF_OF_B)

    call(server, rt, "pay", {"from": "acct_b", "amount": 60_000}, token)
    assert inner.ran == [("pay", {"from": "acct_b", "amount": 60_000})]

    with pytest.raises(HardeningError) as wrong_amount:
        call(server, rt, "pay", {"from": "acct_b", "amount": 50_000}, token)
    assert wrong_amount.value.reason == "mandate_violation"
    assert "rule 1" in str(wrong_amount.value) and "50000 vs 60000" in str(wrong_amount.value)

    with pytest.raises(HardeningError, match="rule 0"):
        call(server, rt, "pay", {"from": "acct_a", "amount": 60_000}, token)
    assert len(inner.ran) == 1  # neither refused call reached the tool


def test_rules_use_live_facts_at_execution_time():
    rt = runtime()
    server = rt.wrap(Server(), resolvers=RESOLVERS)
    token = rt.sign_mandate(HALF_OF_B)
    BALANCES["acct_b"] = 100_000
    try:
        call(server, rt, "pay", {"from": "acct_b", "amount": 50_000}, token)  # half of the *current* balance
    finally:
        BALANCES["acct_b"] = 120_000


def test_missing_forged_or_stale_mandates_are_refused():
    now = [1_000_000.0]
    rt = runtime(clock=lambda: now[0])
    server = rt.wrap(Server(), resolvers=RESOLVERS)
    good = {"from": "acct_b", "amount": 60_000}

    with pytest.raises(HardeningError) as missing:
        call(server, rt, "pay", good, None)
    assert missing.value.reason == "missing_mandate"

    token = rt.sign_mandate(HALF_OF_B)
    loosened = {**token, "mandate": {"tools": {"pay": {"require": []}}}}  # the agent rewrites the rules
    with pytest.raises(HardeningError) as forged:
        call(server, rt, "pay", good, loosened)
    assert forged.value.reason == "bad_mandate_sig"

    other_key = runtime().sign_mandate(HALF_OF_B)
    with pytest.raises(HardeningError) as unknown:
        call(server, rt, "pay", good, other_key)
    assert unknown.value.reason in ("unknown_kid", "bad_mandate_sig")

    now[0] += 3601
    with pytest.raises(HardeningError) as stale:
        call(server, rt, "pay", good, token)
    assert stale.value.reason == "mandate_expired"


def test_protected_tool_not_in_the_mandate_is_refused_and_others_pass():
    rt = runtime()
    inner = Server()
    server = rt.wrap(inner, resolvers=RESOLVERS)
    read_only = rt.sign_mandate({"tools": {}})
    with pytest.raises(HardeningError, match="does not allow pay"):
        call(server, rt, "pay", {"from": "acct_b", "amount": 1}, read_only)
    call(server, rt, "get_balance", {"account_id": "acct_a"}, None)  # unprotected: no mandate needed
    assert inner.ran == [("get_balance", {"account_id": "acct_a"})]


def test_resolver_errors_fail_closed():
    rt = runtime()
    server = rt.wrap(Server(), resolvers=RESOLVERS)
    token = rt.sign_mandate({"tools": {"pay": {"require": [op("eq", arg("amount"), fn("balance", "acct_zzz"))]}}})
    with pytest.raises(HardeningError, match="could not be checked"):
        call(server, rt, "pay", {"amount": 1}, token)


def test_toggle_and_presets():
    assert not load_config("all_on").mandate_binding.enabled  # existing presets are unchanged
    cfg = load_config("all_on_mandate")
    assert cfg.mandate_binding.enabled and cfg.request_auth.enabled
    assert cfg.mandate_binding.protected_tools == ("initiate_transfer",)
    only = load_config("mandate_only")
    assert only.mandate_binding.enabled and only.security_enabled and not only.request_auth.enabled
    assert HardeningConfig.from_dict(cfg.to_dict()) == cfg

    off = runtime(load_config("all_off"))
    assert off.sign_mandate(HALF_OF_B) is None
    inner = Server()
    assert off.wrap(inner) is inner  # off means no middleware at all
    with pytest.raises(ValueError, match="list of strings"):
        HardeningConfig.from_dict({"mandate_binding": {"protected_tools": "initiate_transfer"}})


def single_use_runtime(clock=lambda: 1_000_000.0):
    return runtime(HardeningConfig(mandate_binding=MandateBindingConfig(
        enabled=True, protected_tools=("pay",), single_use=True)), clock=clock)


GOOD = {"from": "acct_b", "amount": 60_000}


def test_single_use_refuses_a_second_payment_under_the_same_mandate():
    rt = single_use_runtime()
    inner = Server()
    server = rt.wrap(inner, resolvers=RESOLVERS)
    token = rt.sign_mandate(HALF_OF_B)
    call(server, rt, "pay", GOOD, token)
    with pytest.raises(HardeningError) as again:
        call(server, rt, "pay", GOOD, token)  # a retry, or a replayed token: new nonce, same mandate
    assert again.value.reason == "mandate_used"
    assert len(inner.ran) == 1

    call(server, rt, "pay", GOOD, rt.sign_mandate(HALF_OF_B))  # a new request gets a new mandate
    assert len(inner.ran) == 2


def test_max_uses_allows_a_request_that_needs_several_calls():
    rt = single_use_runtime()
    server = rt.wrap(Server(), resolvers=RESOLVERS)
    two_bills = rt.sign_mandate({"tools": {"pay": {"require": [op("le", arg("amount"), 5_000)], "max_uses": 2}}})
    call(server, rt, "pay", {"amount": 1_000}, two_bills)
    call(server, rt, "pay", {"amount": 2_000}, two_bills)
    with pytest.raises(HardeningError, match="allows pay 2 time"):
        call(server, rt, "pay", {"amount": 3_000}, two_bills)
    with pytest.raises(ValueError, match="max_uses"):
        rt.sign_mandate({"tools": {"pay": {"require": [], "max_uses": 0}}})


def test_only_calls_that_ran_use_up_a_mandate():
    rt = single_use_runtime()

    class Failing(Server):
        def tools_call(self, params):
            raise ValueError("insufficient funds")

    token = rt.sign_mandate(HALF_OF_B)
    with pytest.raises(ValueError, match="insufficient funds"):
        call(rt.wrap(Failing(), resolvers=RESOLVERS), rt, "pay", GOOD, token)
    with pytest.raises(HardeningError, match="rule 1"):  # refused by the rules: not a use either
        call(rt.wrap(Server(), resolvers=RESOLVERS), rt, "pay", {**GOOD, "amount": 1}, token)
    call(rt.wrap(Server(), resolvers=RESOLVERS), rt, "pay", GOOD, token)  # still unused, so it runs


def test_single_use_off_allows_repeats_and_old_uses_are_forgotten():
    rt = runtime()  # single_use defaults to off
    server = rt.wrap(Server(), resolvers=RESOLVERS)
    token = rt.sign_mandate(HALF_OF_B)
    call(server, rt, "pay", GOOD, token)
    call(server, rt, "pay", GOOD, token)

    now = [1_000_000.0]
    rt = single_use_runtime(clock=lambda: now[0])
    server = rt.wrap(Server(), resolvers=RESOLVERS)
    call(server, rt, "pay", GOOD, rt.sign_mandate(HALF_OF_B))
    now[0] += 3601
    assert rt.server.mandate_uses.count("anything", "pay") == 0 and not rt.server.mandate_uses._uses
    assert load_config("all_on_mandate").mandate_binding.single_use
    assert not HardeningConfig().mandate_binding.single_use
