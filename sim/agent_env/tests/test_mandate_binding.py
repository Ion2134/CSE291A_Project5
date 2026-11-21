"""Signed user mandates in the sim: every task's intent allows its correct transfer and nothing else."""

import asyncio
import json
import random

import pytest
from mcp_hardening import HardeningError, HardeningRuntime, Keyring, load_config, verify_mandate

from discovery_sim.ledger.store import SyntheticLedger
from discovery_sim.sim.episode import run_episodes
from discovery_sim.sim.hard_tasks import HARD_FAMILIES
from discovery_sim.sim.intent import READ_ONLY_INTENT, ledger_resolvers
from discovery_sim.sim.run_config import RunConfig
from discovery_sim.sim.task_defs import agent_tasks

RUNTIME = HardeningRuntime.build(load_config("all_on_mandate"), Keyring.ephemeral_key())
CFG = RUNTIME.config.mandate_binding


def check(task, arguments, ledger=None):
    """Raise HardeningError unless the task's signed intent allows ``initiate_transfer(arguments)``."""
    token = RUNTIME.sign_mandate(task.user_intent)
    verify_mandate(token, "initiate_transfer", arguments, key=RUNTIME.keyring.mandate_key, kid=RUNTIME.keyring.kid,
                   resolvers=ledger_resolvers(ledger or SyntheticLedger()), max_age_s=CFG.max_age_s)


def planned_transfer(task):
    return next((args for tool, args in task.plan or () if tool == "payment__initiate_transfer"), None)


def hard_tasks(family, count=60):
    rng = random.Random(family)
    return [HARD_FAMILIES[family](rng, SyntheticLedger()) for _ in range(count)]


@pytest.mark.parametrize("family", sorted(HARD_FAMILIES))
def test_each_hard_family_allows_its_correct_transfer_and_refuses_changes(family):
    for task in hard_tasks(family):
        correct = planned_transfer(task)
        if correct is None:  # conditional task whose condition is false: paying at all is the mistake
            assert task.mandate.get("no_transfer")
            instruction = task.instruction
            with pytest.raises(HardeningError, match="rule 0"):  # the balance condition
                check(task, {"from_account_id": instruction["account_id"], "payee_id": instruction["payee_id"],
                             "amount_cents": instruction["amount_cents"]})
            continue
        check(task, correct)
        for change in ({"amount_cents": correct["amount_cents"] + 1}, {"payee_id": "payee_synth_9999"},
                       {"from_account_id": next(a for a in SyntheticLedger().accounts
                                                if a != correct["from_account_id"])}):
            with pytest.raises(HardeningError) as refused:
                check(task, {**correct, **change})
            assert refused.value.reason == "mandate_violation"


def test_intents_are_rules_not_the_hidden_answer():
    """A derived amount is stated as a rule over the live balance, so it follows the ledger."""
    task = next(t for t in hard_tasks("derived_transfer") if "balance" in json.dumps(t.user_intent))
    account = task.mandate["from_account_id"]
    ledger = SyntheticLedger()
    ledger.accounts[account].balance_cents += 10_000
    with pytest.raises(HardeningError):
        check(task, planned_transfer(task), ledger)  # the old amount no longer matches the rule


def test_core_tasks_use_literal_intents():
    for task in agent_tasks("core"):
        if task.mandate.get("payee_id"):
            correct = {key: task.mandate[key] for key in ("from_account_id", "payee_id", "amount_cents")}
            check(task, correct)
            with pytest.raises(HardeningError):
                check(task, {**correct, "amount_cents": correct["amount_cents"] * 2})
        else:
            assert task.user_intent == READ_ONLY_INTENT
            with pytest.raises(HardeningError, match="does not allow"):
                check(task, {"from_account_id": "acct_synth_0031", "payee_id": "payee_synth_0007", "amount_cents": 1})


def _run(tmp_path, preset, **overrides):
    config = RunConfig(episodes=40, seed=4, mode="agent", llm_backend="oracle", task_set="hard",
                       hardening=load_config(preset), **overrides)
    asyncio.run(run_episodes(config, tmp_path))
    return [json.loads(line) for line in (tmp_path / "examples.jsonl").read_text().splitlines()]


def test_benign_episodes_are_never_refused(tmp_path):
    examples = _run(tmp_path, "all_on_mandate", operators=("benign",))
    assert not [x for x in examples if x["meta"]["hardening"].get("failure_reason")]
    assert all(x["meta"]["hardening"]["mandate_attached"] for x in examples)


def test_agent_amount_mistakes_are_refused_before_the_transfer_runs(tmp_path):
    examples = _run(tmp_path, "all_on_mandate", operators=("llm_wrong_amount",))
    transfers = [x for x in examples if x["meta"]["fault"] and x["meta"]["hop"] == "llm"
                 and "initiate_transfer" in x["input"]["state_text"].split("\ntool: ", 1)[1].split("\n", 1)[0]]
    assert transfers and all(x["meta"]["hardening"]["failure_reason"] == "mandate_violation" for x in transfers)
    assert all(x["meta"]["hardening"]["failure_stage"] == "request" for x in transfers)


def test_turned_off_it_changes_nothing(tmp_path):
    examples = _run(tmp_path, "all_on", operators=("llm_wrong_amount",))
    assert not [x for x in examples if x["meta"]["hardening"].get("failure_reason") == "mandate_violation"]
    assert all("mandate_attached" in x["meta"]["hardening"] and x["meta"]["hardening"]["mandate_attached"] is None
               for x in examples)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["hardening"]["mandate_binding"]["enabled"] is False


def test_verified_transfers_skip_the_classifiers_and_cannot_be_rolled_back(tmp_path):
    """A mandate is a hard rule: a transfer it allowed is never put to the classifier vote."""
    asyncio.run(run_episodes(RunConfig(episodes=40, seed=4, mode="agent", llm_backend="oracle", task_set="hard",
                                       operators=("benign",), state_db=True, recovery=True,
                                       recovery_classifiers=("oracle", "rules"),
                                       hardening=load_config("all_on_mandate")), tmp_path))
    examples = [json.loads(line) for line in (tmp_path / "examples.jsonl").read_text().splitlines()]
    transfers = [x for x in examples
                 if x["input"]["state_text"].split("\ntool: ", 1)[1].split("\n", 1)[0] == "initiate_transfer"]
    reads = [x for x in examples if x not in transfers]
    assert transfers and all(x["meta"]["hardening"]["mandate_verified"] is True for x in transfers)
    assert all(x["meta"]["classifier_outputs"] is None for x in transfers)
    assert all(x["meta"]["recovery"]["reason"] == "mandate_verified" for x in transfers)
    assert not any(x["meta"]["recovery"]["rolled_back"] for x in transfers)
    # Unprotected reads are still classified as before.
    assert reads and all(x["meta"]["classifier_outputs"] is not None for x in reads)
    assert all(x["meta"]["hardening"]["mandate_verified"] is None for x in reads)


def test_refused_transfers_record_the_mandate_failure(tmp_path):
    examples = _run(tmp_path, "all_on_mandate", operators=("llm_wrong_amount",))
    refused = [x for x in examples if x["meta"]["hardening"].get("failure_reason") == "mandate_violation"]
    assert refused and all(x["meta"]["hardening"]["mandate_verified"] is False for x in refused)


def test_single_use_keeps_a_verified_transfer_whose_reply_was_tampered_and_refuses_repeats(tmp_path):
    """mcp_output_injection on a transfer: the payment is right, its reply isn't. Keep it, withhold the reply."""
    asyncio.run(run_episodes(RunConfig(episodes=40, seed=4, mode="agent", llm_backend="oracle", task_set="hard",
                                       operators=("mcp_output_injection",), state_db=True, recovery=True,
                                       hardening=load_config("all_on_mandate")), tmp_path))
    examples = [json.loads(line) for line in (tmp_path / "examples.jsonl").read_text().splitlines()]
    withheld = [x for x in examples if x["meta"]["recovery"]["reason"] == "verified_reply_withheld"]
    assert withheld
    for x in withheld:
        assert x["meta"]["hardening"]["mandate_verified"] and x["meta"]["hardening"]["failure_stage"] == "response"
        assert not x["meta"]["recovery"]["rolled_back"] and x["meta"]["recovery"]["reply_withheld"]
    episodes = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert sum(e["task_correct"] for e in episodes) >= len(withheld)  # each kept transfer completes its task


def test_without_single_use_a_tampered_reply_still_rolls_the_transfer_back(tmp_path):
    from mcp_hardening import HardeningConfig

    cfg = load_config("all_on_mandate").to_dict()
    cfg["mandate_binding"]["single_use"] = False
    asyncio.run(run_episodes(RunConfig(episodes=40, seed=4, mode="agent", llm_backend="oracle", task_set="hard",
                                       operators=("mcp_output_injection",), state_db=True, recovery=True,
                                       hardening=HardeningConfig.from_dict(cfg)), tmp_path))
    examples = [json.loads(line) for line in (tmp_path / "examples.jsonl").read_text().splitlines()]
    assert not [x for x in examples if x["meta"]["recovery"]["reason"] == "verified_reply_withheld"]
    assert [x for x in examples if x["meta"]["recovery"]["reason"] == "response_sig_failed"
            and x["meta"]["hardening"]["mandate_verified"]]


def _scripted_episode(tmp_path, turns, preset="all_on_mandate", retry=False,
                      operator="benign"):
    from discovery_sim.llm.chat import ScriptChatClient
    from discovery_sim.sim.episode import EpisodeRunner
    from discovery_sim.sim.task_defs import agent_task_index
    from discovery_sim.sim.types import OperatorId

    hardening = load_config(preset) if isinstance(preset, str) else preset
    config = RunConfig(mode="agent", hardening=hardening, state_db=retry, recovery=retry, retry_refused=retry)
    runner = EpisodeRunner(seed=1, config=config, chat_client=ScriptChatClient(turns))
    if retry:
        from state_db import StateStore
        runner.store = StateStore(tmp_path / "state.db")
    try:
        return asyncio.run(runner.run_one(operator_override=OperatorId(operator),
                                          scenario_index=agent_task_index("agent_transfer")))
    finally:
        if runner.store is not None:
            runner.store.close()


def _transfer(call_id, amount):
    from discovery_sim.llm.chat import ChatTurn, ToolCallRequest

    return ChatTurn(content=None, tool_calls=[ToolCallRequest(call_id, "payment__initiate_transfer", {
        "from_account_id": "acct_synth_0031", "payee_id": "payee_synth_0007", "amount_cents": amount})])


def test_an_agent_mistake_is_refused_and_ends_the_episode(tmp_path):
    """Today any failed call ends the episode, so the agent can't send its corrected transfer. (The
    refused call doesn't use up the mandate: see test_only_calls_that_ran_use_up_a_mandate.)"""
    from discovery_sim.llm.chat import ChatTurn

    ep = _scripted_episode(tmp_path, [_transfer("c1", 50_000),           # its own mistake: $500, not $50
                                      _transfer("c2", 5_000),            # the correction it never gets to send
                                      ChatTurn(content="Paid.", tool_calls=[])])
    (only,) = ep.records
    assert only.hardening["failure_reason"] == "mandate_violation" and only.output == {}
    assert not ep.task_correct


def test_a_second_payment_under_the_same_mandate_is_refused(tmp_path):
    from discovery_sim.llm.chat import ChatTurn

    ep = _scripted_episode(tmp_path, [_transfer("c1", 5_000), _transfer("c2", 5_000),
                                      ChatTurn(content="Paid twice?", tool_calls=[])])
    assert ep.records[0].output.get("status") == "completed"
    assert ep.records[1].hardening["failure_reason"] == "mandate_used"


def test_with_retries_the_agent_corrects_its_own_refused_transfer(tmp_path):
    from discovery_sim.llm.chat import ChatTurn

    ep = _scripted_episode(tmp_path, [_transfer("c1", 50_000),           # its own mistake: $500, not $50
                                      _transfer("c2", 5_000),            # corrected after the refusal
                                      ChatTurn(content="Paid.", tool_calls=[])], retry=True)
    first, second = ep.records
    assert first.hardening["failure_reason"] == "mandate_violation" and first.hardening["own_call_refused"]
    assert ep.examples[0].meta["recovery"]["reason"] == "retry_allowed"
    assert second.hardening["mandate_verified"] is True and second.output.get("status") == "completed"
    assert ep.task_correct and not ep.hard_fail  # judged on the call that ran; the ledger has the one payment


def test_retries_stop_at_the_cap(tmp_path):
    from discovery_sim.llm.chat import ChatTurn

    wrong = [_transfer(f"c{i}", 50_000) for i in range(5)]
    ep = _scripted_episode(tmp_path, [*wrong, ChatTurn(content="?", tool_calls=[])], retry=True)
    reasons = [ex.meta["recovery"]["reason"] for ex in ep.examples]
    assert reasons == ["retry_allowed", "retry_allowed", "cap_reached"]  # max_recoveries=2, then it ends
    assert not ep.task_correct


def test_tampered_calls_are_never_retried(tmp_path):
    """An intercepted transfer fails its request signature: that's an attack, so no retry."""
    examples = _run(tmp_path, "all_on_mandate", operators=("a2a_call_intercepted",), state_db=True, recovery=True,
                    retry_refused=True)
    blocked = [x for x in examples if x["meta"]["hardening"].get("failure_reason") == "bad_request_sig"]
    assert blocked and all(x["meta"]["recovery"]["reason"] == "blocked_before_tool" for x in blocked)
    assert not [x for x in examples if x["meta"]["recovery"]["reason"] == "retry_allowed"
                and x["meta"]["hardening"]["failure_reason"] != "mandate_violation"]


def test_without_request_signing_a_refusal_is_not_retried(tmp_path):
    """mandate_only can't tell the agent's own call from a tampered one, so it doesn't retry."""
    from discovery_sim.llm.chat import ChatTurn

    ep = _scripted_episode(tmp_path, [_transfer("c1", 50_000), _transfer("c2", 5_000),
                                      ChatTurn(content="Paid.", tool_calls=[])], preset="mandate_only", retry=True)
    (only,) = ep.records
    assert only.hardening["failure_reason"] == "mandate_violation" and only.hardening["own_call_refused"] is False
    assert ep.examples[0].meta["recovery"]["reason"] == "blocked_before_tool"


def test_retry_needs_recovery():
    from discovery_sim.llm.chat import ScriptChatClient
    from discovery_sim.sim.orchestrator import AgentOrchestrator

    with pytest.raises(ValueError, match="retry_refused needs recovery"):
        AgentOrchestrator(ScriptChatClient([]), retry_refused=True)
