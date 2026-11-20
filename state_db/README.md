# State DB

A SQLite record of an agent run's **internal** state (the conversation, what the model sent, what came
back) and **external** state (what the tool received, the full ledger) after every step. It is the
groundwork for recovery: when a classifier flags a call, everything after the step before it can be
deleted and the state read back as of that step. [`../recovery`](../recovery/README.md) does exactly that with `--recovery`.

Standard library only (`sqlite3`). It takes plain dicts and knows nothing about the simulator;
`agent_env` calls it when a run uses `--state-db`.

## Setup

```bash
python3.12 -m venv state_db/.venv     # an arm64 Python, like the other venvs
state_db/.venv/bin/pip install -e "state_db[dev]"
agent_env/.venv/bin/pip install -e state_db        # so the simulator can import it
```

Tests: `cd state_db && .venv/bin/python -m pytest -q`.

## Files

A run with `--state-db` writes `state.db` to its run folder. It holds the run's **current** state, and
`delete_after` removes rows from it. The full history of a run (including anything later deleted here)
is the telemetry `mcp_hardening` writes to `telemetry/*.parquet`, which is append-only.

## Tables

Every write is a **step** with a run-wide `seq` (exact order, never ties) and a `wall_time`. Episodes
share the file and are told apart by `episode_id`.

| Table | Columns | State |
| --- | --- | --- |
| `steps` | `seq`, `episode_id`, `wall_time`, `kind` (`start` / `message` / `tool_call`) | the timeline |
| `messages` | `seq`, `episode_id`, `position`, `role`, `content` (JSON) | internal: the conversation |
| `tool_calls` | `seq`, `episode_id`, `call_id`, `call_index`, `tool`, `model_call`, `received_args`, `output`, `cause`, `output_correct`, `classifier_outputs`, `recover` | internal: `model_call`, `output`; external: `received_args`; the run's classifier outputs and recovery decision (empty without classifiers) |
| `accounts`, `payees`, `payments`, `transactions` | `seq`, `episode_id` + the ledger's fields | external: the full ledger after that step |

The ledger is snapshotted at the episode's `start` and after every tool call, so the ledger as of any
step is the latest snapshot at or before it.

## API (`state_db.StateStore`)

| Method | Does |
| --- | --- |
| `start_episode(episode_id, ledger)` | First step: the ledger before anything ran |
| `record_message(episode_id, position, message)` | One chat message |
| `record_tool_call(episode_id, call, ledger)` | One tool call and the ledger right after it |
| `delete_after(episode_id, seq)` | Delete everything the episode wrote after `seq` (other episodes untouched) |
| `seq_at(episode_id, wall_time)` | The last step at or before a wall-clock time |
| `ledger_at` / `messages_at` / `tool_calls_at(episode_id, seq)` | The state as of a step |

Every write method returns the new step's `seq`.

## Looking at a DB

```bash
sqlite3 agent_env/data/runs/<run>/state.db "select kind, count(*) from steps group by kind"
sqlite3 agent_env/data/runs/<run>/state.db "select episode_id, call_index, tool, cause from tool_calls limit 20"
```
