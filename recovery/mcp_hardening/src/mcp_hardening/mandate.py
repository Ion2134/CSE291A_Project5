"""Signed user mandates: bind a protected tool call to what the user actually asked for.

Request and response signing prove a call came from the agent and was not changed on the way. They
cannot tell whether the call is what the *user* wanted: an agent that gets the amount wrong signs the
wrong amount honestly. A mandate closes that gap. The user's side (the app taking the request) signs
the rules a call must satisfy; the agent carries the token unchanged; the tool server checks the call
against those rules right before running it.

This module knows nothing about any domain. The rules are data inside the signed mandate, and the
only facts they can use are the lookup functions the tool server registers (its ``resolvers``)::

    {"v": 1,
     "tools": {
       "initiate_transfer": {"require": [
         {"op": "eq", "args": [{"arg": "payee_id"}, "payee_7"]},
         {"op": "eq", "args": [{"arg": "amount_cents"},
                               {"op": "//", "args": [{"fn": "balance", "args": ["acct_42"]}, 2]}]},
         {"op": "le", "args": [{"arg": "amount_cents"}, 100000]}]}}}

Expressions (JSON, never ``eval``):

- a literal: number, string, bool, null, or a list of expressions
- ``{"arg": name}``: the call's argument ``name``
- ``{"fn": name, "args": [...]}``: a resolver registered by the tool server
- ``{"op": name, "args": [...]}``: one of ``OPS`` (comparisons, ``and``/``or``/``not``, arithmetic,
  ``min``/``max``, and ``argmax``/``argmin`` over a list with ``"key"`` naming a resolver)

A tool listed under ``tools`` is allowed when every ``require`` entry is true. A protected tool that is
not listed is not allowed at all. With ``single_use`` on, a mandate allows each tool at most its
``max_uses`` (default 1) successful calls, so a retry or a replayed token can't repeat a payment; a new
user request gets a new mandate (a new ``id``), so later requests are unaffected. Anything that can't be evaluated (unknown resolver or op, a resolver
error) fails closed.
"""

from __future__ import annotations

import hashlib
import hmac
import operator
import secrets
import time
from collections.abc import Callable, Mapping
from typing import Any

from mcp_hardening import errors
from mcp_hardening.canonical import PROTOCOL_VERSION, canonical_json
from mcp_hardening.errors import HardeningError

MANDATE_KEY = "mcp_hardening_mandate"
MANDATE_VERSION = 1

Resolvers = Mapping[str, Callable[..., Any]]


class MandateUses:
    """How many times each mandate has been used for each protected tool (single-use mandates).

    Entries are dropped once their mandate is past ``max_age_s``: an expired mandate is refused anyway.
    """

    def __init__(self, max_age_s: float, clock: Callable[[], float] = time.time) -> None:
        self.max_age_s = max_age_s
        self.clock = clock
        self._uses: dict[tuple[str, str], list[int]] = {}  # (mandate id, tool) -> [count, iat]

    def count(self, mandate_id: str, tool: str) -> int:
        self._prune()
        return self._uses.get((mandate_id, tool), [0, 0])[0]

    def record(self, mandate_id: str, tool: str, iat: int) -> None:
        entry = self._uses.setdefault((mandate_id, tool), [0, iat])
        entry[0] += 1

    def _prune(self) -> None:
        cutoff = self.clock() - self.max_age_s
        for key in [key for key, (_, iat) in self._uses.items() if iat < cutoff]:
            del self._uses[key]


def max_uses(token: dict[str, Any], tool: str) -> int:
    """How many successful calls of ``tool`` the mandate in ``token`` allows (default 1)."""
    rules = (token.get("mandate", {}).get("tools") or {}).get(tool) or {}
    return int(rules.get("max_uses", 1))


class MandateEvalError(ValueError):
    """A rule could not be evaluated (bad expression, unknown resolver, resolver failure)."""


def _arith(fn: Callable[[Any, Any], Any]) -> Callable[..., Any]:
    def apply(*values: Any) -> Any:
        if len(values) != 2:
            raise MandateEvalError("arithmetic ops take exactly two arguments")
        return fn(*values)
    return apply


def _compare(fn: Callable[[Any, Any], bool]) -> Callable[..., bool]:
    def apply(*values: Any) -> bool:
        if len(values) != 2:
            raise MandateEvalError("comparisons take exactly two arguments")
        try:
            return bool(fn(*values))
        except TypeError as exc:
            raise MandateEvalError(f"cannot compare {values[0]!r} and {values[1]!r}") from exc
    return apply


# Eager ops: every argument is evaluated first, then the function is applied.
OPS: dict[str, Callable[..., Any]] = {
    "eq": _compare(operator.eq), "ne": _compare(operator.ne),
    "lt": _compare(operator.lt), "le": _compare(operator.le),
    "gt": _compare(operator.gt), "ge": _compare(operator.ge),
    "in": _compare(lambda value, options: value in options),
    "+": _arith(operator.add), "-": _arith(operator.sub), "*": _arith(operator.mul), "//": _arith(operator.floordiv),
    "min": lambda *values: min(values), "max": lambda *values: max(values),
}
LOGIC_OPS = ("and", "or", "not")  # short-circuit
RANK_OPS = ("argmax", "argmin")   # need a "key" resolver


def evaluate(expr: Any, arguments: Mapping[str, Any], resolvers: Resolvers) -> Any:
    """Evaluate one expression against the call's ``arguments`` and the server's ``resolvers``."""
    if isinstance(expr, list):
        return [evaluate(item, arguments, resolvers) for item in expr]
    if not isinstance(expr, dict):
        return expr  # literal
    if "arg" in expr:
        return arguments.get(expr["arg"])
    if "fn" in expr:
        name = expr["fn"]
        if name not in resolvers:
            raise MandateEvalError(f"unknown resolver {name!r}")
        values = [evaluate(item, arguments, resolvers) for item in expr.get("args", [])]
        try:
            return resolvers[name](*values)
        except Exception as exc:  # a lookup that fails can't vouch for anything
            raise MandateEvalError(f"resolver {name!r} failed: {exc}") from exc
    if "op" in expr:
        name, raw_args = expr["op"], expr.get("args", [])
        if name == "and":
            return all(evaluate(item, arguments, resolvers) for item in raw_args)
        if name == "or":
            return any(evaluate(item, arguments, resolvers) for item in raw_args)
        if name == "not":
            if len(raw_args) != 1:
                raise MandateEvalError("'not' takes one argument")
            return not evaluate(raw_args[0], arguments, resolvers)
        if name in RANK_OPS:
            key = expr.get("key")
            if key not in resolvers:
                raise MandateEvalError(f"{name} needs a known 'key' resolver, got {key!r}")
            if len(raw_args) != 1:
                raise MandateEvalError(f"{name} takes one argument (a list)")
            items = evaluate(raw_args[0], arguments, resolvers)
            if not isinstance(items, list) or not items:
                raise MandateEvalError(f"{name} needs a non-empty list")
            pick = max if name == "argmax" else min
            return pick(items, key=lambda item: resolvers[key](item))
        if name not in OPS:
            raise MandateEvalError(f"unknown op {name!r}")
        return OPS[name](*(evaluate(item, arguments, resolvers) for item in raw_args))
    raise MandateEvalError(f"not an expression: {expr!r}")


def describe(expr: Any) -> str:
    """Short human-readable form of an expression, for error details."""
    if isinstance(expr, list):
        return "[" + ", ".join(describe(item) for item in expr) + "]"
    if not isinstance(expr, dict):
        return repr(expr)
    if "arg" in expr:
        return str(expr["arg"])
    if "fn" in expr:
        return f"{expr['fn']}({', '.join(describe(a) for a in expr.get('args', []))})"
    args = [describe(a) for a in expr.get("args", [])]
    symbols = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "in": "in",
               "+": "+", "-": "-", "*": "*", "//": "//"}
    if expr.get("op") in symbols and len(args) == 2:
        return f"({args[0]} {symbols[expr['op']]} {args[1]})"
    key = f", key={expr['key']}" if "key" in expr else ""
    return f"{expr.get('op')}({', '.join(args)}{key})"


def _mac_input(kid: str, mandate_id: str, iat: int, mandate: Any) -> bytes:
    return "|".join([PROTOCOL_VERSION, "mandate", kid, mandate_id, str(iat), canonical_json(mandate)]).encode()


def sign_mandate(key: bytes, kid: str, mandate: dict[str, Any], *, clock: Callable[[], float] = time.time) -> dict:
    """The token the user's side hands to the agent: the mandate plus its signature."""
    _check_shape(mandate)
    token = {"v": MANDATE_VERSION, "kid": kid, "id": secrets.token_hex(8), "iat": int(clock()), "mandate": mandate}
    token["sig"] = hmac.new(key, _mac_input(kid, token["id"], token["iat"], mandate), hashlib.sha256).hexdigest()
    return token


def verify_mandate(
    token: Any,
    tool: str,
    arguments: Mapping[str, Any],
    *,
    key: bytes,
    kid: str,
    resolvers: Resolvers,
    max_age_s: float,
    clock: Callable[[], float] = time.time,
    uses: MandateUses | None = None,
) -> None:
    """Raise ``HardeningError`` unless ``token`` is genuine, fresh, and allows this call.

    With ``uses`` (single-use mandates), also refuse a call past the tool's ``max_uses``. Recording a
    use is the caller's job, once the call has actually run (``MandateUses.record``).
    """
    if not isinstance(token, dict) or not isinstance(token.get("mandate"), dict):
        raise HardeningError(errors.MISSING_MANDATE, f"{tool} needs a signed user mandate")
    if token.get("kid") != kid:
        raise HardeningError(errors.UNKNOWN_KID, "mandate signed with an unknown key")
    sig, mandate_id, iat = token.get("sig"), token.get("id"), token.get("iat")
    if not isinstance(sig, str) or not isinstance(mandate_id, str) or not isinstance(iat, int):
        raise HardeningError(errors.BAD_MANDATE_SIG, "malformed mandate token")
    expected = hmac.new(key, _mac_input(kid, mandate_id, iat, token["mandate"]), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        raise HardeningError(errors.BAD_MANDATE_SIG)
    age = clock() - iat
    if age > max_age_s or age < -max_age_s:
        raise HardeningError(errors.MANDATE_EXPIRED, f"mandate is {age:.0f}s old (max {max_age_s:.0f}s)")

    rules = (token["mandate"].get("tools") or {}).get(tool)
    if not isinstance(rules, dict):
        raise HardeningError(errors.MANDATE_VIOLATION, f"the user's mandate does not allow {tool}")
    if uses is not None:
        allowed, used = max_uses(token, tool), uses.count(mandate_id, tool)
        if used >= allowed:
            raise HardeningError(errors.MANDATE_USED,
                                 f"this mandate allows {tool} {allowed} time(s) and was already used {used}")
    for index, requirement in enumerate(rules.get("require", [])):
        try:
            ok = evaluate(requirement, arguments, resolvers)
        except MandateEvalError as exc:
            raise HardeningError(errors.MANDATE_VIOLATION, f"rule {index} could not be checked: {exc}") from exc
        if ok is not True:
            raise HardeningError(errors.MANDATE_VIOLATION, _violation(index, requirement, arguments, resolvers))


def _violation(index: int, requirement: Any, arguments: Mapping[str, Any], resolvers: Resolvers) -> str:
    """Which rule failed, with the values it compared, so the agent can correct the call."""
    detail = f"rule {index} failed: {describe(requirement)}"
    if isinstance(requirement, dict) and requirement.get("op") in OPS and len(requirement.get("args", [])) == 2:
        try:
            left, right = (evaluate(a, arguments, resolvers) for a in requirement["args"])
            detail += f" with {left!r} vs {right!r}"
        except MandateEvalError:
            pass
    return detail


def _check_shape(mandate: Any) -> None:
    if not isinstance(mandate, dict) or not isinstance(mandate.get("tools", {}), dict):
        raise ValueError("a mandate is {'tools': {tool_name: {'require': [expression, ...]}}}")
    for tool, rules in mandate.get("tools", {}).items():
        if not isinstance(rules, dict) or not isinstance(rules.get("require", []), list):
            raise ValueError(f"mandate rules for {tool!r} must be {{'require': [expression, ...]}}")
        limit = rules.get("max_uses", 1)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError(f"max_uses for {tool!r} must be a whole number >= 1, got {limit!r}")
