from __future__ import annotations

import random
from typing import Any

from discovery_sim.sim.context import AgentContext
from discovery_sim.sim.types import AttackSpec, BaseTask, OperatorId, ToolCall


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


def pick_operator(rng: random.Random, include_benign: bool = True) -> AttackSpec:
    catalog = operator_catalog()
    choices = list(catalog.values())
    if not include_benign:
        choices = [c for c in choices if c.operator != OperatorId.BENIGN]
    return rng.choice(choices)


def apply_llm_attack(task: BaseTask, llm_call: ToolCall, operator: OperatorId) -> ToolCall:
    args = dict(llm_call.arguments)
    if operator == OperatorId.LLM_WRONG_ACCOUNT and "account_id" in args:
        args["account_id"] = "acct_synth_0042"
    if operator == OperatorId.LLM_WRONG_AMOUNT and "amount_cents" in args:
        args["amount_cents"] = int(args["amount_cents"]) * 10
    if operator == OperatorId.LLM_WRONG_ACCOUNT and "from_account_id" in args:
        args["from_account_id"] = "acct_synth_0042"
    return ToolCall(tool_name=llm_call.tool_name, arguments=args, agent=llm_call.agent)


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
