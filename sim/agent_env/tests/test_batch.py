from __future__ import annotations

import asyncio
import json
from collections import Counter

import httpx
import pytest

from discovery_sim.llm import chat as chat_module
from discovery_sim.llm.chat import DEFAULT_TEMPERATURE, ChatTurn, LmStudioChat, TritonAPIChat
from discovery_sim.sim import batch, episode
from discovery_sim.sim.batch import BatchConfig, plan_batch, run_batch
from discovery_sim.sim.run_config import RunConfig
from discovery_sim.sim.task_defs import agent_tasks, all_task_defs

GRID = {"name": "unit", "models": ["model-a", "model-b", "model-c"], "temperatures": [0.2, 0.7, 1.0],
        "episodes": 50, "max_episodes": 300, "base_seed": 100}


def _config(**overrides) -> BatchConfig:
    return BatchConfig(**{**GRID, **overrides, "models": tuple(overrides.get("models", GRID["models"])),
                          "temperatures": tuple(overrides.get("temperatures", GRID["temperatures"]))})


# ---------------------------------------------------------------- temperature knob

@pytest.mark.parametrize("client", ["lmstudio", "triton"])
def test_chat_clients_send_their_temperature(monkeypatch, client):
    sent = []
    real_client = httpx.AsyncClient

    def handler(request):
        sent.append(json.loads(request.content)["temperature"])
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    make = (lambda **kw: LmStudioChat(base_url="http://lm/v1", model="m", **kw)) if client == "lmstudio" else \
        (lambda **kw: TritonAPIChat(base_url="http://tr/v1", model="m", api_key="k", **kw))

    asyncio.run(make().complete([{"role": "user", "content": "hi"}], []))
    asyncio.run(make(temperature=0.9).complete([{"role": "user", "content": "hi"}], []))
    assert sent == [DEFAULT_TEMPERATURE, 0.9]


def test_run_config_temperature_reaches_the_chat_client(monkeypatch):
    seen = {}

    def fake_make_chat_client(backend, **kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(episode, "make_chat_client", fake_make_chat_client)
    episode.EpisodeRunner(config=RunConfig(temperature=0.85))._chat_for_episode()
    assert seen["temperature"] == 0.85


# ---------------------------------------------------------------- task-set knob

def test_core_task_set_is_the_original_tasks():
    assert [task.task_id for task in agent_tasks("core")] == [task.task_id for task in all_task_defs()]
    assert [task.task_id for task in agent_tasks()] == [task.task_id for task in agent_tasks("core")]
    with pytest.raises(ValueError, match="unknown task set"):
        agent_tasks("nope")


# ---------------------------------------------------------------- planning

def test_budget_picks_distinct_cells_covering_every_knob_value():
    for sample_seed in range(5):
        runs = plan_batch(_config(sample_seed=sample_seed))
        assert len(runs) == 6 and len({run.cell for run in runs}) == 6
        assert sorted(Counter(run.cell.model for run in runs).values()) == [2, 2, 2]
        assert sorted(Counter(run.cell.temperature for run in runs).values()) == [2, 2, 2]


def test_picks_are_repeatable_and_a_bigger_budget_keeps_them():
    six = [run.name for run in plan_batch(_config())]
    assert six == [run.name for run in plan_batch(_config())]
    assert [run.name for run in plan_batch(_config(max_episodes=450))][:6] == six
    assert [run.name for run in plan_batch(_config(sample_seed=1))] != six


def test_no_budget_runs_the_whole_grid_with_one_seed_per_cell():
    runs = plan_batch(_config(max_episodes=None))
    assert len(runs) == 9
    assert [run.seed for run in runs] == list(range(100, 109))
    assert runs[0].name == "unit-model-a-t0.2-core"


def test_bad_configs_are_rejected(tmp_path):
    path = tmp_path / "batch.json"
    path.write_text(json.dumps({**GRID, "task_sets": ["missing"]}))
    with pytest.raises(ValueError, match="Unknown task sets"):
        BatchConfig.from_file(path)
    path.write_text(json.dumps({**GRID, "typo": 1}))
    with pytest.raises(ValueError, match="unknown batch config keys"):
        BatchConfig.from_file(path)


# ---------------------------------------------------------------- running

class _DoneChat:
    """Answers without calling a tool, so each episode ends at once."""

    model_id = "fake"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def complete(self, messages, tools):
        return ChatTurn(content="Done.", tool_calls=[])


class _CheckBalanceChat(_DoneChat):
    async def complete(self, messages, tools):
        if any(message["role"] == "tool" for message in messages):
            return ChatTurn(content="Done.", tool_calls=[])
        return ChatTurn(content=None, tool_calls=[chat_module.ToolCallRequest(
            "c1", "bank__get_balance", {"account_id": "acct_synth_0031"})])


def test_run_batch_writes_runs_and_resumes(tmp_path, monkeypatch, capsys):
    made = []

    def fake_make_chat_client(backend, **kwargs):
        made.append(kwargs)
        return _CheckBalanceChat(**kwargs)

    monkeypatch.setattr(episode, "make_chat_client", fake_make_chat_client)
    monkeypatch.setattr(batch, "AGENT_ENV", tmp_path)
    config = _config(models=["model-a"], temperatures=[0.5], episodes=3, max_episodes=None)

    summary = asyncio.run(run_batch(config))
    assert summary["runs"] == 1 and summary["statuses"] == {"unit-model-a-t0.5-core": "done"}

    run_dir = tmp_path / "data/runs/unit-model-a-t0.5-core"
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert (manifest["temperature"], manifest["task_set"], manifest["seed"], manifest["episodes"]) == (0.5, "core", 100, 3)
    assert {kwargs["temperature"] for kwargs in made} == {0.5} and {kwargs["model"] for kwargs in made} == {"model-a"}
    assert (run_dir / "laya_train.jsonl").read_text().strip()

    batch_file = json.loads((tmp_path / "data/batches/unit.json").read_text())
    assert batch_file["runs"] == ["unit-model-a-t0.5-core"]
    assert batch_file["cells"][0] == {"run": "unit-model-a-t0.5-core", "model": "model-a", "temperature": 0.5,
                                      "task_set": "core", "seed": 100}

    made.clear()
    again = asyncio.run(run_batch(config))
    assert again["statuses"] == {"unit-model-a-t0.5-core": "skipped (done)"} and not made


def test_dry_run_calls_no_model_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(episode, "make_chat_client", lambda *a, **k: pytest.fail("dry run called a model"))
    monkeypatch.setattr(batch, "AGENT_ENV", tmp_path)
    summary = asyncio.run(run_batch(_config(), dry_run=True))
    assert summary["runs"] == 6 and summary["episodes"] == 300
    assert set(summary["statuses"].values()) == {"would run"}
    assert not (tmp_path / "data").exists()
