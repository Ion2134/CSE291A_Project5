"""io_mismatch only changes some tools' output; on any other tool the call is genuine and labelled as such."""

import asyncio

from discovery_sim.ledger.store import SyntheticLedger
from discovery_sim.mcp.hacked import IO_MISMATCH_TOOLS
from discovery_sim.sim.harness import AgentPool, execute_tool_call
from discovery_sim.sim.labels import label_call
from discovery_sim.sim.operators import operator_catalog
from discovery_sim.sim.types import BaseTask, OperatorId, ToolCall


def labelled(tool: str, arguments: dict):
    task = BaseTask(task_id="t", template="t", user_message="m", mandate={}, target_agent="bank",
                    tool_name=tool, correct_arguments=dict(arguments))
    pool = AgentPool(SyntheticLedger())
    attack = operator_catalog()[OperatorId.MCP_IO_MISMATCH]
    record = asyncio.run(execute_tool_call(pool, task, ToolCall(tool, dict(arguments), "bank"), attack))
    return record, label_call(task, record, pool.ledger)


def test_unaffected_tool_is_a_correct_call():
    assert "list_accounts" not in IO_MISMATCH_TOOLS
    record, labels = labelled("list_accounts", {})
    assert record.output == {"accounts": SyntheticLedger().list_accounts()}  # genuine output
    assert not record.hidden_adversary
    assert (labels.cause, labels.output_correct) == ("none", True)


def test_affected_tool_is_still_an_attack():
    record, labels = labelled("get_balance", {"account_id": "acct_synth_0031"})
    assert record.hidden_adversary
    assert (labels.cause, labels.output_correct) == ("adversary", False)
