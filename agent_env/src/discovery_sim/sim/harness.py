from __future__ import annotations

import random
from typing import Any

from discovery_sim.a2a.types import Message, SendMessageRequest, data_part
from discovery_sim.agents.bank import BankAgent
from discovery_sim.agents.base import McpBackedAgent
from discovery_sim.agents.mitm import MitmAgent
from discovery_sim.agents.payment import PaymentAgent
from discovery_sim.ledger.store import SyntheticLedger
from discovery_sim.mcp.hacked import HackedMcpToolServer
from discovery_sim.mcp.server import bank_mcp_server, payment_mcp_server
from discovery_sim.sim.operators import (
    a2a_rewrite_for_operator,
    apply_llm_attack,
    apply_taint_attack,
    mcp_mode_for_operator,
)
from discovery_sim.sim.context import AgentContext
from discovery_sim.sim.labels import llm_call_matches_task
from discovery_sim.sim.types import AttackSpec, BaseTask, CallRecord, OperatorId, ToolCall


class AgentPool:
    def __init__(self, ledger: SyntheticLedger | None = None) -> None:
        self.ledger = ledger or SyntheticLedger()
        self._bank: BankAgent | None = None
        self._payment: PaymentAgent | None = None

    def bank(self, mcp_mode: str | None = None) -> BankAgent:
        mcp = bank_mcp_server(self.ledger)
        if mcp_mode:
            mcp = HackedMcpToolServer(mcp, mcp_mode, ledger_getter=lambda: self.ledger)
        return BankAgent("http://127.0.0.1:8002", ledger=self.ledger, mcp=mcp)

    def payment(self, mcp_mode: str | None = None) -> PaymentAgent:
        mcp = payment_mcp_server(self.ledger)
        if mcp_mode:
            mcp = HackedMcpToolServer(mcp, mcp_mode, ledger_getter=lambda: self.ledger)
        return PaymentAgent("http://127.0.0.1:8003", ledger=self.ledger, mcp=mcp)


async def execute_tool_call(
    pool: AgentPool,
    task: BaseTask,
    llm_call: ToolCall,
    attack: AttackSpec,
    call_index: int = 0,
    context: AgentContext | None = None,
    rng: random.Random | None = None,
) -> CallRecord:
    llm_effective = llm_call
    if attack.hop == "llm" and attack.operator in (
        OperatorId.LLM_WRONG_ACCOUNT,
        OperatorId.LLM_WRONG_AMOUNT,
    ):
        llm_effective = apply_llm_attack(
            task,
            llm_call,
            attack.operator,
            rng=rng,
            account_ids=list(pool.ledger.accounts),
        )
    if attack.operator == OperatorId.TAINT_PRIOR_OUTPUT and context is not None:
        llm_effective = apply_taint_attack(llm_effective, context)

    a2a_args = dict(llm_effective.arguments)
    mcp_args = dict(llm_effective.arguments)
    hidden_adversary = False

    mcp_mode = None
    if attack.hop == "mcp":
        mcp_mode = mcp_mode_for_operator(attack.operator)

    upstream: McpBackedAgent
    if llm_effective.agent == "bank":
        upstream = pool.bank(mcp_mode=mcp_mode)
    else:
        upstream = pool.payment(mcp_mode=mcp_mode)

    agent: McpBackedAgent | MitmAgent = upstream
    a2a_tool = llm_effective.tool_name

    if attack.hop == "a2a" and attack.operator == OperatorId.A2A_CALL_INTERCEPTED:
        rewrite = a2a_rewrite_for_operator(attack.operator, task)
        agent = MitmAgent("http://127.0.0.1:8010", upstream, rewrite_arguments=rewrite)
        mcp_args = {**a2a_args, **rewrite}
        a2a_args = dict(mcp_args)
        hidden_adversary = llm_call_matches_task(task, llm_effective.arguments)

    if attack.hop == "a2a" and attack.operator == OperatorId.A2A_TOOLSET_MISMATCH:
        hacked = HackedMcpToolServer(
            upstream.mcp_server(),  # type: ignore[arg-type]
            "toolset_mismatch",
            ledger_getter=lambda: pool.ledger,
        )
        if llm_effective.agent == "bank":
            upstream = BankAgent("http://127.0.0.1:8002", ledger=pool.ledger, mcp=hacked)
        else:
            upstream = PaymentAgent("http://127.0.0.1:8003", ledger=pool.ledger, mcp=hacked)
        agent = MitmAgent("http://127.0.0.1:8010", upstream)
        a2a_tool = "fetch_balance_alias" if llm_effective.tool_name == "get_balance" else llm_effective.tool_name

    if attack.operator == OperatorId.HARD_NEGATIVE_REENCODE:
        agent = upstream

    request = SendMessageRequest(
        message=Message(
            role="user",
            parts=[data_part(a2a_args)],
            metadata={"skillId": a2a_tool},
        )
    )

    error = None
    output: dict[str, Any] = {}
    try:
        response = await agent.send_message(request)
        if response.task and response.task.artifacts:
            part = response.task.artifacts[0]["parts"][0]
            output = dict(part.get("data") or {})
        elif response.task and response.task.status.get("state") == "failed":
            error = str(response.task.status.get("message", "failed"))
    except Exception as exc:
        error = str(exc)

    if attack.hop == "mcp" and attack.operator == OperatorId.MCP_IO_MISMATCH:
        hidden_adversary = True

    return CallRecord(
        base_task=task,
        call_index=call_index,
        llm_call=llm_effective,
        a2a_tool=a2a_tool,
        a2a_arguments=a2a_args,
        mcp_arguments=mcp_args,
        output=output,
        attack=attack,
        hidden_adversary=hidden_adversary,
        error=error,
    )
