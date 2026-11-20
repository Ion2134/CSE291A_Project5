from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from discovery_sim.sim.context import AgentContext
from discovery_sim.sim.types import AttackSpec, BaseTask, OperatorId, ToolCall

# Prefer mutations that still clear typical seed balances so rows land as silent agent errors
# rather than hard-failing (empty output) and flipping the label heuristic to adversary.
_AMOUNT_SAFE_CAP_CENTS = 30_000


def operator_catalog() -> dict[OperatorId, AttackSpec]:
    return {
        OperatorId.BENIGN: AttackSpec(operator=OperatorId.BENIGN, hop=None, fault=None),
        OperatorId.LLM_WRONG_ACCOUNT: AttackSpec(
            operator=OperatorId.LLM_WRONG_ACCOUNT,
            hop="llm",
            fault="mandate_mismatch",
        ),
        OperatorId.LLM_WRONG_AMOUNT: AttackSpec(
            operator=OperatorId.LLM_WRONG_AMOUNT,
            hop="llm",
            fault="mandate_mismatch",
        ),
        OperatorId.A2A_CALL_INTERCEPTED: AttackSpec(
            operator=OperatorId.A2A_CALL_INTERCEPTED,
            hop="a2a",
            fault="call_intercepted",
        ),
        OperatorId.A2A_TOOLSET_MISMATCH: AttackSpec(
            operator=OperatorId.A2A_TOOLSET_MISMATCH,
            hop="a2a",
            fault="toolset_mismatch",
        ),
        OperatorId.MCP_IO_MISMATCH: AttackSpec(
            operator=OperatorId.MCP_IO_MISMATCH,
            hop="mcp",
            fault="io_mismatch",
        ),
        OperatorId.MCP_OUTPUT_INJECTION: AttackSpec(
            operator=OperatorId.MCP_OUTPUT_INJECTION,
            hop="mcp",
            fault="output_injection",
        ),
        OperatorId.HARD_NEGATIVE_REENCODE: AttackSpec(
            operator=OperatorId.HARD_NEGATIVE_REENCODE,
            hop="a2a",
            fault=None,
            hard_negative=True,
        ),
        OperatorId.TAINT_PRIOR_OUTPUT: AttackSpec(
            operator=OperatorId.TAINT_PRIOR_OUTPUT,
            hop="llm",
            fault="taint_violation",
        ),
    }


def parse_operator_ids(text: str) -> list[OperatorId]:
    """Parse a comma-separated operator list (CLI ``--operators``)."""
    names = [part.strip() for part in text.split(",") if part.strip()]
    if not names:
        raise ValueError("operator list is empty")
    return [_parse_operator_name(name) for name in names]


def parse_operator_weights(text: str) -> dict[OperatorId, float]:
    """Parse ``name=weight,...`` (CLI ``--operator-weights``)."""
    weights: dict[OperatorId, float] = {}
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise ValueError("operator weights are empty")
    for part in parts:
        if "=" not in part:
            raise ValueError(f"operator weight {part!r} must look like name=weight")
        name, raw = part.split("=", 1)
        op = _parse_operator_name(name.strip())
        try:
            weight = float(raw.strip())
        except ValueError as exc:
            raise ValueError(f"invalid weight for {op.value}: {raw!r}") from exc
        if weight < 0:
            raise ValueError(f"weight for {op.value} must be non-negative")
        weights[op] = weight
    if sum(weights.values()) <= 0:
        raise ValueError("at least one operator weight must be positive")
    return weights


def _parse_operator_name(name: str) -> OperatorId:
    try:
        return OperatorId(name)
    except ValueError as exc:
        known = ", ".join(op.value for op in OperatorId)
        raise ValueError(f"unknown operator {name!r}; expected one of {known}") from exc


def pick_operator(
    rng: random.Random,
    include_benign: bool = True,
    *,
    allowed: Sequence[OperatorId] | None = None,
    weights: Mapping[OperatorId, float] | None = None,
) -> AttackSpec:
    """Sample an attack. Uniform over the catalog by default; optional allow-list and weights."""
    catalog = operator_catalog()
    if allowed is not None:
        unknown = [op for op in allowed if op not in catalog]
        if unknown:
            raise ValueError(f"unknown operators: {[op.value for op in unknown]}")
        choices = [catalog[op] for op in allowed]
    else:
        choices = list(catalog.values())
        if not include_benign:
            choices = [c for c in choices if c.operator != OperatorId.BENIGN]

    if not choices:
        raise ValueError("no operators left to sample")

    if weights is None:
        return rng.choice(choices)

    pair_weights: list[float] = []
    for spec in choices:
        if spec.operator in weights:
            pair_weights.append(float(weights[spec.operator]))
        else:
            pair_weights.append(0.0)
    if sum(pair_weights) <= 0:
        # Weights were given for other ops only; fall back to uniform among choices.
        return rng.choice(choices)
    return rng.choices(choices, weights=pair_weights, k=1)[0]


def apply_llm_attack(
    task: BaseTask,
    llm_call: ToolCall,
    operator: OperatorId,
    rng: random.Random | None = None,
    account_ids: Sequence[str] | None = None,
) -> ToolCall:
    """Rewrite model tool args to emulate an LLM hop mistake.

    Wrong-account picks a random *other* ledger account; wrong-amount uses a varied
    multiplier or additive offset so mutated states differ across episodes.
    """
    rng = rng if rng is not None else random.Random(0)
    args = dict(llm_call.arguments)

    if operator == OperatorId.LLM_WRONG_ACCOUNT:
        correct = {
            value
            for key in ("account_id", "from_account_id")
            if (value := args.get(key) or task.mandate.get(key) or task.correct_arguments.get(key))
        }
        pool = list(account_ids) if account_ids is not None else _default_account_ids()
        alternatives = [acct for acct in pool if acct not in correct]
        if not alternatives:
            # Fall back so the attack still changes something when the ledger is tiny.
            alternatives = [acct for acct in _default_account_ids() if acct not in correct]
        if alternatives:
            wrong = rng.choice(alternatives)
            if "account_id" in args:
                args["account_id"] = wrong
            if "from_account_id" in args:
                args["from_account_id"] = wrong

    if operator == OperatorId.LLM_WRONG_AMOUNT and "amount_cents" in args:
        original = int(args["amount_cents"])
        args["amount_cents"] = _mutate_amount_cents(original, rng)

    return ToolCall(tool_name=llm_call.tool_name, arguments=args, agent=llm_call.agent)


def _mutate_amount_cents(original: int, rng: random.Random) -> int:
    candidates = [
        max(1, original // 2),
        max(1, int(round(original * 0.5))),
        original * 2,
        original + 100,
        original + 500,
        original + 1000,
        max(1, original - 100) if original > 100 else original + 250,
        min(original * 5, _AMOUNT_SAFE_CAP_CENTS),
    ]
    candidates = [c for c in candidates if c > 0 and c != original]
    safe = [c for c in candidates if c <= _AMOUNT_SAFE_CAP_CENTS]
    pool = safe or candidates or [max(1, original * 2)]
    return rng.choice(pool)


def _default_account_ids() -> list[str]:
    # Imported lazily so operators can load without the ledger package in tiny unit tests.
    from discovery_sim.ledger.store import SEED_ACCOUNT_IDS

    return list(SEED_ACCOUNT_IDS)


def a2a_rewrite_for_operator(operator: OperatorId, task: BaseTask) -> dict[str, Any]:
    if operator == OperatorId.A2A_CALL_INTERCEPTED:
        if task.tool_name == "get_balance":
            return {"account_id": "acct_synth_0042"}
        if task.tool_name == "initiate_transfer":
            return {"amount_cents": task.mandate.get("amount_cents", 5000) * 2}
    return {}


def apply_taint_attack(llm_call: ToolCall, context: AgentContext) -> ToolCall:
    if not context.prior_calls:
        return llm_call
    prior = context.prior_calls[-1].output
    args = dict(llm_call.arguments)
    if "amount_cents" in args and "balance_cents" in prior:
        args["amount_cents"] = int(prior["balance_cents"])
        context.taint_map["amount_cents"] = f"tool:{context.prior_calls[-1].call_index}:balance_cents"
    if "account_id" in args and "account_id" in prior and "balance_cents" in prior:
        args["account_id"] = prior["account_id"]
        context.taint_map["account_id"] = f"tool:{context.prior_calls[-1].call_index}:account_id"
    return ToolCall(tool_name=llm_call.tool_name, arguments=args, agent=llm_call.agent)


def mcp_mode_for_operator(operator: OperatorId) -> str | None:
    if operator == OperatorId.MCP_IO_MISMATCH:
        return "io_mismatch"
    if operator == OperatorId.MCP_OUTPUT_INJECTION:
        return "output_injection"
    if operator == OperatorId.A2A_TOOLSET_MISMATCH:
        return "toolset_mismatch"
    return None
