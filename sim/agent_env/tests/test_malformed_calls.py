from __future__ import annotations

import asyncio
import json

from discovery_sim.llm.chat import ChatTurn, ScriptChatClient, ToolCallRequest, _parse_chat_completion
from discovery_sim.mcp.openai_tools import validate_tool_call
from discovery_sim.sim.agent_tasks import agent_task_index
from discovery_sim.sim.episode import EpisodeRunner
from discovery_sim.sim.run_config import RunConfig
from discovery_sim.sim.types import OperatorId


def _run(turns: list[ChatTurn], task_id: str = "agent_balance"):
    runner = EpisodeRunner(seed=1, config=RunConfig(mode="agent"), chat_client=ScriptChatClient(turns))
    return asyncio.run(
        runner.run_one(operator_override=OperatorId.BENIGN, scenario_index=agent_task_index(task_id))
    )


def _call(name: str, arguments: dict, **kw) -> ChatTurn:
    return ChatTurn(content=None, tool_calls=[ToolCallRequest("c1", name, arguments, **kw)])


def test_validate_tool_call():
    assert validate_tool_call("bank__get_balance", {"account_id": "acct_synth_0031"}) is None
    assert "unknown tool" in validate_tool_call("bank__get_balanse", {})
    assert "missing" in validate_tool_call("bank__get_balance", {})
    assert "unexpected" in validate_tool_call("bank__get_balance", {"account_id": "a", "acct": "a"})
    assert "must be integer" in validate_tool_call(
        "payment__initiate_transfer",
        {"from_account_id": "a", "payee_id": "p", "amount_cents": "5000"},
    )
    assert ">= 1" in validate_tool_call(
        "payment__initiate_transfer",
        {"from_account_id": "a", "payee_id": "p", "amount_cents": 0},
    )


def test_parse_keeps_raw_arguments_and_json_errors():
    body = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "bank__get_balance", "arguments": '{"account_id": '}}
                    ]
                }
            }
        ]
    }
    tc = _parse_chat_completion(body).tool_calls[0]
    assert tc.raw_arguments == '{"account_id": '
    assert tc.parse_error and "not valid JSON" in tc.parse_error
    assert tc.arguments == {}


def test_malformed_call_is_reported_to_model_and_retry_recovers():
    ep = _run(
        [
            _call("bank__get_balance", {"acct": "acct_synth_0031"}),
            _call("bank__get_balance", {"account_id": "acct_synth_0031"}),
            ChatTurn(content="Balance retrieved.", tool_calls=[]),
        ]
    )
    assert ep is not None
    assert len(ep.malformed_calls) == 1
    bad = ep.malformed_calls[0]
    assert "missing required argument(s): account_id" in bad["error"]
    assert bad["task_id"] == "agent_balance"
    assert bad["conversation"][-1]["tool_calls"][0]["function"]["name"] == "bank__get_balance"
    # The retry ran as tool round 0, so the attack schedule isn't shifted by the bad call.
    assert len(ep.records) == 1 and ep.records[0].call_index == 0
    assert ep.task_correct
    assert ep.model_failure is False


def test_model_that_never_recovers_is_a_model_failure_not_dropped():
    ep = _run(
        [
            _call("bank__get_balance", {}, raw_arguments="{not json", parse_error="arguments are not valid JSON"),
            ChatTurn(content="Sorry, I couldn't do that.", tool_calls=[]),
        ]
    )
    assert ep is not None
    assert ep.records == []
    assert ep.model_failure is True
    assert ep.malformed_calls[0]["raw_arguments"] == "{not json"


def test_model_sees_its_raw_arguments_and_the_error():
    chat = ScriptChatClient(
        [
            _call("bank__nope", {"account_id": "acct_synth_0031"}),
            ChatTurn(content="Done.", tool_calls=[]),
        ]
    )
    seen: list[list[dict]] = []
    original = chat.complete

    async def spy(messages, tools):
        seen.append([dict(m) for m in messages])
        return await original(messages, tools)

    chat.complete = spy  # type: ignore[method-assign]
    runner = EpisodeRunner(seed=1, config=RunConfig(mode="agent"), chat_client=chat)
    asyncio.run(
        runner.run_one(
            operator_override=OperatorId.BENIGN,
            scenario_index=agent_task_index("agent_balance"),
        )
    )
    tool_reply = seen[1][-1]
    assert tool_reply["role"] == "tool"
    assert "unknown tool" in json.loads(tool_reply["content"])["error"]
