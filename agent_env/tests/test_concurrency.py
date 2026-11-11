from __future__ import annotations

import asyncio
import json

from discovery_sim.llm.chat import ChatTurn, ToolCallRequest
from discovery_sim.sim import episode, orchestrator
from discovery_sim.sim.episode import ERRORS_FILE, run_episodes
from discovery_sim.sim.run_config import RunConfig


def test_concurrent_scripted_run(tmp_path):
    out = tmp_path / "conc"
    summary = asyncio.run(
        run_episodes(
            RunConfig(episodes=12, seed=99, mode="scripted", concurrency=4),
            out,
        )
    )
    assert summary["episodes"] == 12
    assert summary["concurrency"] == 4
    assert (out / "episodes.jsonl").read_text().count("\n") == 12


class _AlwaysCheckBalanceChat:
    """Calls get_balance first on every task."""

    model_id = "fake"

    def __init__(self) -> None:
        self._turn = 0

    async def complete(self, messages, tools):
        self._turn += 1
        if self._turn == 1:
            return ChatTurn(
                content=None,
                tool_calls=[ToolCallRequest("c1", "bank__get_balance", {"account_id": "acct_synth_0031"})],
            )
        return ChatTurn(content="Done.", tool_calls=[])


def _buggy_expected_arguments(tool_name, mandate, call_args=None):
    # Reintroduces the old mandate.py bug (fixed since) to stand in for any simulator crash.
    return {"account_id": mandate["account_id"]}


def test_crashing_episode_is_logged_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(episode, "make_chat_client", lambda *a, **k: _AlwaysCheckBalanceChat())
    monkeypatch.setattr(orchestrator, "expected_arguments_for_tool", _buggy_expected_arguments)
    out = tmp_path / "crash"
    summary = asyncio.run(run_episodes(RunConfig(episodes=30, seed=1, mode="agent", concurrency=4), out))

    errors = [json.loads(line) for line in (out / ERRORS_FILE).read_text().splitlines()]
    assert summary["errors"] == len(errors) > 0
    assert summary["episodes"] > 0
    assert summary["episodes"] + summary["errors"] + summary["dropped_llm"] == 30
    assert summary["dropped_llm"] == 0
    first = errors[0]
    assert first["error_type"] == "KeyError"
    assert first["tool_call"]["tool_name"] == "bank__get_balance"
    assert "account_id" not in first["mandate"]
    assert "Traceback" in first["traceback"]
