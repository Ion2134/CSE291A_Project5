from __future__ import annotations

import random
from typing import Any

from mcp_hardening import ENVELOPE_KEY, MANDATE_KEY, HardeningError, HardeningRuntime

from discovery_sim.a2a.types import Message, SendMessageRequest, data_part
from discovery_sim.agents.bank import BankAgent
from discovery_sim.agents.base import McpBackedAgent
from discovery_sim.agents.mitm import MitmAgent
from discovery_sim.agents.payment import PaymentAgent
from discovery_sim.ledger.store import SyntheticLedger
from discovery_sim.mcp.hacked import IO_MISMATCH_TOOLS, HackedMcpToolServer
from discovery_sim.mcp.server import bank_mcp_server, payment_mcp_server
from discovery_sim.sim.operators import (
    a2a_rewrite_for_operator,
    apply_llm_attack,
    apply_taint_attack,
    mcp_mode_for_operator,
)
from discovery_sim.sim.context import AgentContext
from discovery_sim.sim.intent import ledger_resolvers
from discovery_sim.sim.labels import llm_call_matches_task
from discovery_sim.sim.types import AttackSpec, BaseTask, CallRecord, OperatorId, ToolCall


class AgentPool:
    def __init__(
        self,
        ledger: SyntheticLedger | None = None,
        runtime: HardeningRuntime | None = None,
    ) -> None:
        self.ledger = ledger or SyntheticLedger()
        # Shared MCP hardening middleware (None or all-off = no hardening).
        self.runtime = runtime
        # The user's signed mandate for this episode (``authorize``); None when mandate binding is off.
        self.mandate_token: dict[str, Any] | None = None
        self._bank: BankAgent | None = None
        self._payment: PaymentAgent | None = None

    def authorize(self, intent: dict[str, Any] | None) -> None:
        """The user's side signs what they asked for; the agent carries it on every call."""
        self.mandate_token = self.runtime.sign_mandate(intent) if self.runtime is not None else None

    def _server(self, mcp: Any, mcp_mode: str | None) -> Any:
        # Hardening wraps the *real* server first; an attack wraps on top of that. The genuine server
        # signs its output, so anything the hacked layer changes breaks the response signature.
        if self.runtime is not None:
            # The bank side's lookups a user mandate's rules may call, read from the live ledger.
            mcp = self.runtime.wrap(mcp, resolvers=ledger_resolvers(self.ledger))
        if mcp_mode:
            mcp = HackedMcpToolServer(mcp, mcp_mode, ledger_getter=lambda: self.ledger)
        return mcp

    def bank(self, mcp_mode: str | None = None) -> BankAgent:
        mcp = self._server(bank_mcp_server(self.ledger), mcp_mode)
        return BankAgent("http://127.0.0.1:8002", ledger=self.ledger, mcp=mcp)

    def payment(self, mcp_mode: str | None = None) -> PaymentAgent:
        mcp = self._server(payment_mcp_server(self.ledger), mcp_mode)
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

    # Client-side middleware signs what the agent intends to call: after its own (llm-hop) mistakes,
    # before anything on the wire can touch it.
    runtime = pool.runtime if pool.runtime is not None and pool.runtime.enabled else None
    envelope = runtime.client.sign(llm_effective.tool_name, llm_effective.arguments) if runtime else None
    metadata: dict[str, Any] = {"skillId": a2a_tool}
    if envelope is not None:
        metadata[ENVELOPE_KEY] = envelope
    if pool.mandate_token is not None:
        metadata[MANDATE_KEY] = pool.mandate_token  # passed along unchanged; the agent can't re-sign it
    hardening = _hardening_record(runtime, envelope)

    request = SendMessageRequest(
        message=Message(
            role="user",
            parts=[data_part(a2a_args)],
            metadata=metadata,
        )
    )

    error = None
    output: dict[str, Any] = {}
    artifact: dict[str, Any] | None = None
    try:
        response = await agent.send_message(request)
        if response.task and response.task.artifacts:
            artifact = response.task.artifacts[0]
            output = dict(artifact["parts"][0].get("data") or {})
        elif response.task and response.task.status.get("state") == "failed":
            error = str(response.task.status.get("message", "failed"))
    except HardeningError as exc:
        # Server-side middleware refused the call before the handler ran: the call fails.
        error = _hardening_failed(hardening, exc, "request")
    except Exception as exc:
        error = str(exc)

    if runtime is not None and artifact is not None:
        try:
            runtime.client.verify_response(
                llm_effective.tool_name,
                envelope,
                {"structuredContent": output, "isError": False, "_meta": artifact.get("metadata")},
            )
            if runtime.config.response_auth.enabled:
                hardening["response_verified"] = True
        except HardeningError as exc:
            # The result can't be trusted, so the agent never sees it: the call fails.
            error = _hardening_failed(hardening, exc, "response")
            output = {}

    _record_mandate(runtime, llm_effective.tool_name, hardening)

    if (attack.hop == "mcp" and attack.operator == OperatorId.MCP_IO_MISMATCH
            and llm_effective.tool_name in IO_MISMATCH_TOOLS):
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
        hardening=hardening,
    )


def _hardening_failed(hardening: dict[str, Any], exc: HardeningError, stage: str) -> str:
    hardening["failure_reason"] = exc.reason
    hardening["failure_stage"] = stage
    hardening[f"{stage}_verified"] = False
    return str(exc)


# The rejections that come from the mandate check itself (not from the request signature before it).
MANDATE_FAILURES = ("missing_mandate", "bad_mandate_sig", "mandate_expired", "mandate_violation")


def _record_mandate(runtime: HardeningRuntime | None, tool: str, hardening: dict[str, Any]) -> None:
    """``mandate_verified``: True if a protected call passed the user's mandate and ran, False if the
    mandate refused it, None if no mandate check ran (off, an unprotected tool, or refused before it)."""
    if runtime is None:
        return
    cfg = runtime.config.mandate_binding
    verified = None
    if cfg.enabled and tool in cfg.protected_tools:
        reason, stage = hardening.get("failure_reason"), hardening.get("failure_stage")
        if reason in MANDATE_FAILURES:
            verified = False
        elif stage != "request":  # it got past every request check, including the mandate
            verified = True
    hardening["mandate_verified"] = verified
    # Whether that mandate is single-use, i.e. the call can't be repeated under it (recovery reads this).
    hardening["mandate_single_use"] = cfg.single_use if verified is not None else None
    # The mandate refused the call exactly as the agent signed it: the request signature was checked
    # (and passed, since the mandate is checked after it), so nothing changed it on the way. That makes
    # it the agent's own mistake, not tampering. Without request_auth there's no way to tell, so no.
    hardening["own_call_refused"] = (hardening.get("failure_reason") == "mandate_violation"
                                     and runtime.config.request_auth.enabled) if verified is not None else None


def _hardening_record(runtime: HardeningRuntime | None, envelope: dict[str, Any] | None) -> dict[str, Any]:
    """Per-call hardening outcome. ``*_verified`` is None when that check is off (or never ran)."""
    if runtime is None:
        return {}
    cfg = runtime.config
    request_checks = cfg.request_auth.enabled or cfg.freshness_check.enabled or cfg.replay_protection.enabled
    return {
        # Whether the user's mandate was carried (it's checked only for protected tools).
        "mandate_attached": True if cfg.mandate_binding.enabled else None,
        "kid": envelope.get("kid") if envelope else None,
        "request_signed": bool(envelope and "sig" in envelope),
        # Optimistic: flipped to False if the server rejects the call.
        "request_verified": True if request_checks else None,
        "response_verified": None,
        "failure_reason": None,
        "failure_stage": None,
    }


def _llm_matches_task(task: BaseTask, args: dict[str, Any]) -> bool:
    return all(args.get(k) == v for k, v in task.correct_arguments.items())
