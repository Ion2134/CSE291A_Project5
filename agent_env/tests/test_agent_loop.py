from __future__ import annotations

import asyncio
import json

from discovery_sim.llm.chat import ChatTurn, ScriptChatClient, ToolCallRequest
from discovery_sim.sim.episode import EpisodeRunner
from discovery_sim.sim.run_config import RunConfig
from discovery_sim.sim.types import OperatorId


def test_agent_loop_balance_then_transfer():
    chat = ScriptChatClient(
        [
            ChatTurn(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="c1",
                        name="bank__get_balance",
                        arguments={"account_id": "acct_synth_0031"},
                    )
                ],
            ),
            ChatTurn(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="c2",
                        name="payment__initiate_transfer",
                        arguments={
                            "from_account_id": "acct_synth_0031",
                            "payee_id": "payee_synth_0007",
                            "amount_cents": 5000,
                        },
                    )
                ],
            ),
            ChatTurn(content="Paid $50 after checking balance.", tool_calls=[]),
        ]
    )
    runner = EpisodeRunner(
        seed=1,
        config=RunConfig(mode="agent", multi_step=False),
        chat_client=chat,
    )
    ep = asyncio.run(runner.run_one(operator_override=OperatorId.BENIGN, scenario_index=2))
    assert ep is not None
    assert len(ep.records) == 2
    assert ep.task_completed
    assert ep.task_correct
    assert ep.examples[0].meta["mode"] == "agent"
    assert ep.examples[0].meta["live_llm"] is False
    assert "conversation" in ep.examples[1].input_state


def test_agent_loop_mcp_attack_on_model_tool_call():
    chat = ScriptChatClient(
        [
            ChatTurn(
                content=None,
                tool_calls=[
                    ToolCallRequest("c1", "bank__get_balance", {"account_id": "acct_synth_0031"})
                ],
            ),
            ChatTurn(content="Balance retrieved.", tool_calls=[]),
        ]
    )
    runner = EpisodeRunner(
        seed=1,
        config=RunConfig(mode="agent"),
        chat_client=chat,
    )
    ep = asyncio.run(
        runner.run_one(
            operator_override=OperatorId.MCP_IO_MISMATCH,
            scenario_index=0,
            fault_call_index_override=0,
        )
    )
    assert ep is not None
    assert ep.attack_applied
    assert ep.examples[0].labels.output_correct is False
    assert ep.examples[0].meta["operator"] == OperatorId.MCP_IO_MISMATCH.value
