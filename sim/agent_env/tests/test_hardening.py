from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from mcp_hardening import HardeningError, HardeningRuntime, GenAITelemetry, Keyring, load_config, read_telemetry, semconv

from discovery_sim.a2a.client import A2AClient
from discovery_sim.agents.bank import BankAgent
from discovery_sim.ledger.store import SyntheticLedger
from discovery_sim.llm.chat import ChatTurn, ScriptChatClient, ToolCallRequest
from discovery_sim.mcp.server import bank_mcp_server
from discovery_sim.sim.agent_tasks import agent_task_index
from discovery_sim.sim.episode import EpisodeRunner, run_episodes
from discovery_sim.sim.harness import AgentPool, execute_tool_call
from discovery_sim.sim.operators import operator_catalog
from discovery_sim.sim.run_config import RunConfig
from discovery_sim.sim.task_defs import all_task_defs
from discovery_sim.sim.types import BaseTask, OperatorId, ToolCall


def scripted(preset: str, operator: OperatorId):
    runner = EpisodeRunner(seed=42, config=RunConfig(mode="scripted", multi_step=False, hardening=load_config(preset)))
    return asyncio.run(runner.run_one(operator_override=operator, scenario_index=0))


@pytest.mark.parametrize(
    "preset,operator,reason",
    [
        ("all_on", OperatorId.A2A_CALL_INTERCEPTED, "bad_request_sig"),
        ("all_on", OperatorId.MCP_IO_MISMATCH, "bad_response_sig"),
        ("all_on", OperatorId.MCP_OUTPUT_INJECTION, "bad_response_sig"),
        # Each mechanism on its own only catches what it is for.
        ("request_only", OperatorId.A2A_CALL_INTERCEPTED, "bad_request_sig"),
        ("request_only", OperatorId.MCP_IO_MISMATCH, None),
        ("response_only", OperatorId.MCP_OUTPUT_INJECTION, "bad_response_sig"),
        ("response_only", OperatorId.A2A_CALL_INTERCEPTED, None),
        # Not tampering in transit: honestly signed, so it goes through (Laya's job).
        ("all_on", OperatorId.LLM_WRONG_ACCOUNT, None),
        ("all_on", OperatorId.BENIGN, None),
    ],
)
def test_hardening_blocks_tampering(preset, operator, reason):
    ep = scripted(preset, operator)
    rec, ex = ep.records[0], ep.examples[0]
    assert rec.hardening["failure_reason"] == reason
    assert ex.meta["hardening"] == rec.hardening
    if reason:
        assert rec.error.startswith(f"mcp_hardening:{reason}")
        assert rec.output == {}
        assert ep.hard_fail and not ep.silent_wrong
        # Labels still describe what happened: the attacker caused the bad step.
        assert ex.labels.output_correct is False
        assert ex.labels.cause == "adversary"
    else:
        assert rec.error is None


def test_benign_all_on_marks_verified():
    ep = scripted("all_on", OperatorId.BENIGN)
    h = ep.records[0].hardening
    assert ep.task_correct
    assert h["request_signed"] and h["request_verified"] is True and h["response_verified"] is True


def test_default_config_is_baseline():
    ep = scripted("default", OperatorId.MCP_IO_MISMATCH)
    assert ep.records[0].hardening == {}
    assert ep.silent_wrong  # no defenses: the attack goes through silently, as before


def test_intercepted_transfer_never_touches_ledger():
    tdef = next(t for t in all_task_defs() if t.task_id == "agent_transfer")
    task = BaseTask(
        task_id="t",
        template=tdef.template,
        user_message=tdef.user_message,
        mandate=dict(tdef.mandate),
        target_agent="payment",
        tool_name="initiate_transfer",
        correct_arguments=dict(tdef.correct_arguments),
    )
    runtime = HardeningRuntime.build(load_config("request_only"), Keyring.ephemeral_key())
    pool = AgentPool(SyntheticLedger(), runtime=runtime)
    balance_before = pool.ledger.get_balance("acct_synth_0031")["balance_cents"]
    attack = operator_catalog()[OperatorId.A2A_CALL_INTERCEPTED]
    rec = asyncio.run(
        execute_tool_call(pool, task, ToolCall("initiate_transfer", dict(tdef.correct_arguments), "payment"), attack)
    )
    assert rec.hardening["failure_reason"] == "bad_request_sig"
    assert pool.ledger.payments == {}
    assert pool.ledger.get_balance("acct_synth_0031")["balance_cents"] == balance_before


def _transfer_chat():
    return ScriptChatClient(
        [
            ChatTurn(content=None, tool_calls=[ToolCallRequest("c0", "bank__get_balance", {"acct": "x"})]),  # malformed
            ChatTurn(content=None, tool_calls=[ToolCallRequest("c1", "bank__get_balance", {"account_id": "acct_synth_0031"})]),
            ChatTurn(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        "c2",
                        "payment__initiate_transfer",
                        {"from_account_id": "acct_synth_0031", "payee_id": "payee_synth_0007", "amount_cents": 5000},
                    )
                ],
            ),
            ChatTurn(content="Paid.", tool_calls=[]),
        ]
    )


def test_agent_mode_telemetry_turns(tmp_path: Path):
    cfg = load_config("all_on")
    telemetry = GenAITelemetry(cfg.telemetry, tmp_path)
    runner = EpisodeRunner(
        seed=1,
        config=RunConfig(mode="agent", hardening=cfg),
        chat_client=_transfer_chat(),
        telemetry=telemetry,
    )
    ep = asyncio.run(
        runner.run_one(operator_override=OperatorId.BENIGN, scenario_index=agent_task_index("agent_balance_then_pay"))
    )
    telemetry.shutdown()
    assert ep.task_correct

    rows = read_telemetry(tmp_path / "telemetry").to_pylist()
    ops = [r[semconv.GEN_AI_OPERATION_NAME] for r in rows]
    assert ops.count("invoke_agent") == 1
    assert ops.count("chat") == 4
    assert ops.count("execute_tool") == 3  # 1 malformed + 2 real

    agent = next(r for r in rows if r[semconv.GEN_AI_OPERATION_NAME] == "invoke_agent")
    assert agent[semconv.GEN_AI_CONVERSATION_ID] == ep.episode_id
    assert agent["status_code"] == "OK"
    assert all(r["trace_id"] == agent["trace_id"] for r in rows)
    assert all(r["parent_span_id"] == agent["span_id"] for r in rows if r is not agent)

    tools = {r[semconv.GEN_AI_TOOL_CALL_ID]: r for r in rows if r[semconv.GEN_AI_OPERATION_NAME] == "execute_tool"}
    assert tools["c0"][semconv.SIM_MALFORMED] is True and tools["c0"][semconv.ERROR_TYPE] == "malformed_tool_call"
    assert tools["c0"]["status_code"] == "ERROR"
    for cid in ("c1", "c2"):
        assert tools[cid]["status_code"] == "OK"
        assert tools[cid][semconv.HARDENING_REQUEST_VERIFIED] is True
        assert tools[cid][semconv.HARDENING_RESPONSE_VERIFIED] is True
        assert tools[cid][semconv.SIM_OUTPUT_CORRECT] is True
    chat = next(r for r in rows if r[semconv.GEN_AI_OPERATION_NAME] == "chat")
    assert chat[semconv.GEN_AI_PROVIDER_NAME] == "script"
    assert "gen_ai.input.messages" in json.loads(chat["attributes_json"])


def test_run_episodes_writes_telemetry_and_manifest(tmp_path: Path):
    cfg = load_config("all_on")
    asyncio.run(run_episodes(RunConfig(episodes=12, seed=3, mode="scripted", hardening=cfg), tmp_path))
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["hardening"] == cfg.to_dict()
    assert manifest["hardening_key"]["kid"]
    baseline = json.loads((tmp_path / "baseline.json").read_text())
    assert baseline["hardening"] == cfg.to_dict()
    assert set(baseline["hardening_blocked_calls"]) == {"total", "by_reason", "by_operator"}

    table = read_telemetry(tmp_path / "telemetry")
    ops = table.column(semconv.GEN_AI_OPERATION_NAME).to_pylist()
    episodes = sum(1 for _ in open(tmp_path / "episodes.jsonl"))
    assert ops.count("invoke_agent") == episodes
    assert ops.count("execute_tool") == sum(1 for _ in open(tmp_path / "examples.jsonl"))
    rows = table.to_pylist()
    blocked = [r for r in rows if r[semconv.HARDENING_FAILURE_REASON]]
    assert len(blocked) == baseline["hardening_blocked_calls"]["total"]
    # Episode spans carry an explicit outcome: ERROR exactly for episodes that didn't complete.
    eps = {json.loads(line)["episode_id"]: json.loads(line) for line in open(tmp_path / "episodes.jsonl")}
    for r in rows:
        if r[semconv.GEN_AI_OPERATION_NAME] == "invoke_agent":
            ep = eps[r[semconv.GEN_AI_CONVERSATION_ID]]
            assert r["status_code"] == ("OK" if ep["task_completed"] else "ERROR")


def test_telemetry_off(tmp_path: Path):
    asyncio.run(run_episodes(RunConfig(episodes=3, seed=1, mode="scripted", hardening=load_config("all_off")), tmp_path))
    assert not (tmp_path / "telemetry").exists()


def test_http_mcp_and_a2a_paths():
    runtime = HardeningRuntime.build(load_config("all_on"), Keyring.ephemeral_key())
    ledger = SyntheticLedger()
    app = BankAgent("http://bank", ledger=ledger, mcp=runtime.wrap(bank_mcp_server(ledger))).starlette_app()
    transport = httpx.ASGITransport(app=app)

    async def _run():
        signed = A2AClient("http://bank", runtime=runtime, transport=transport)
        result = await signed.call_tool("get_balance", {"account_id": "acct_synth_0031"})
        assert result["structuredContent"]["balance_cents"] == 120_000

        msg = {"role": "user", "parts": [{"data": {"account_id": "acct_synth_0042"}}], "metadata": {"skillId": "get_balance"}}
        a2a = await signed.send_message(msg)
        assert a2a["task"]["artifacts"][0]["parts"][0]["data"]["balance_cents"] == 45_000

        unsigned = A2AClient("http://bank", transport=transport)
        with pytest.raises(HardeningError) as exc:
            await unsigned.call_tool("get_balance", {"account_id": "acct_synth_0031"})
        assert exc.value.reason == "missing_envelope"

        other_key = HardeningRuntime.build(load_config("all_on"), Keyring.ephemeral_key())
        with pytest.raises(HardeningError) as exc:
            await A2AClient("http://bank", runtime=other_key, transport=transport).send_message(msg)
        assert exc.value.reason == "unknown_kid"

    asyncio.run(_run())
