from __future__ import annotations

from discovery_sim.sim.labels import label_call
from discovery_sim.sim.operators import operator_catalog
from discovery_sim.sim.types import BaseTask, CallRecord, OperatorId, ToolCall

CATALOG = operator_catalog()
MANDATE = {"from_account_id": "acct_synth_0031", "payee_id": "payee_synth_0007", "amount_cents": 5000,
           "cap_cents": 100_000}
TASK = BaseTask(task_id="t", template="transfer_under_cap", user_message="Pay payee_synth_0007 $50.",
                mandate=MANDATE, target_agent="payment", tool_name="initiate_transfer",
                correct_arguments={k: MANDATE[k] for k in ("from_account_id", "payee_id", "amount_cents")})


def _record(call_args, received_args, output, operator, error=None):
    call = ToolCall(tool_name="initiate_transfer", arguments=call_args, agent="payment")
    return CallRecord(base_task=TASK, call_index=0, llm_call=call, a2a_tool="initiate_transfer",
                      a2a_arguments=dict(received_args), mcp_arguments=dict(received_args), output=output,
                      attack=CATALOG[operator], error=error)


def test_refused_off_mandate_call_is_an_agent_error():
    """The bank refused exactly what the agent asked for (more than the account holds): agent's fault."""
    too_much = {**TASK.correct_arguments, "amount_cents": 900_000}
    labels = label_call(TASK, _record(too_much, too_much, {}, OperatorId.LLM_WRONG_AMOUNT, "insufficient funds"))
    assert (labels.output_correct, labels.cause) == (False, "agent")


def test_refusal_of_a_tampered_call_is_still_an_adversary_error():
    """The agent asked correctly, but the call was rewritten on the way and then refused."""
    rewritten = {**TASK.correct_arguments, "amount_cents": 900_000}
    labels = label_call(TASK, _record(TASK.correct_arguments, rewritten, {}, OperatorId.A2A_CALL_INTERCEPTED,
                                      "insufficient funds"))
    assert labels.cause == "adversary"


def test_on_mandate_call_that_fails_is_unchanged():
    """A correct call that comes back empty (e.g. a hijacked tool) is not blamed on the agent."""
    labels = label_call(TASK, _record(TASK.correct_arguments, TASK.correct_arguments, {},
                                      OperatorId.A2A_TOOLSET_MISMATCH, "failed"))
    assert labels.cause == "adversary"
