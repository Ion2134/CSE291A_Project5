"""SQLite record of a run's internal and external state, step by step.

Every write is a *step* with a run-wide ``seq`` (exact order, never ties) and a ``wall_time``. Each
step stores what changed: a conversation message, a tool call (internal: what the model sent and what
came back; external: what the tool received), and a full snapshot of the ledger after it.
``delete_after`` removes everything an episode did after a given step, which is what recovery will use.

The DB is the *current* state and may be cut back. The full history of a run lives in the telemetry
(``mcp_hardening``'s append-only Parquet spans), not here.

This module knows nothing about the simulator: it takes plain dicts. The ledger dict has four lists,
``accounts``, ``payees``, ``payments`` and ``transactions``, with the fields in ``LEDGER_TABLES``.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

# Ledger table -> its columns (besides seq and episode_id). Mirrors discovery_sim/ledger/store.py.
LEDGER_TABLES: dict[str, tuple[str, ...]] = {
    "accounts": ("account_id", "holder_name", "balance_cents", "is_primary"),
    "payees": ("payee_id", "name", "routing_hint"),
    "payments": ("payment_id", "from_account_id", "payee_id", "amount_cents", "status", "created_at"),
    "transactions": ("transaction_id", "account_id", "amount_cents", "description", "posted_at"),
}
# call_id: the model's tool-call id (telemetry's gen_ai.tool.call.id), so a row joins to its span.
# classifier_outputs / recover: what the run's classifiers said about the call and the recovery decision
# (None when the run lists no classifiers).
TOOL_CALL_COLUMNS = ("call_id", "call_index", "tool", "model_call", "received_args", "output", "cause", "output_correct",
                     "classifier_outputs", "recover")
# Columns stored as JSON text.
_JSON_COLUMNS = {"content", "model_call", "received_args", "output", "classifier_outputs"}

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS steps (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL,
    wall_time REAL NOT NULL,
    kind TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    seq INTEGER NOT NULL, episode_id TEXT NOT NULL, position INTEGER NOT NULL, role TEXT, content TEXT
);
CREATE TABLE IF NOT EXISTS tool_calls (
    seq INTEGER NOT NULL, episode_id TEXT NOT NULL, {", ".join(TOOL_CALL_COLUMNS)}
);
{"".join(f"CREATE TABLE IF NOT EXISTS {table} (seq INTEGER NOT NULL, episode_id TEXT NOT NULL, {', '.join(columns)});"
         for table, columns in LEDGER_TABLES.items())}
CREATE INDEX IF NOT EXISTS steps_by_episode ON steps (episode_id, seq);
"""
DATA_TABLES = ("messages", "tool_calls", *LEDGER_TABLES)


class StateStore:
    """One SQLite file for a whole run; episodes are told apart by ``episode_id``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        self._db.close()

    # Writes ----------------------------------------------------------------------------------------

    def start_episode(self, episode_id: str, ledger: dict) -> int:
        """The episode's first step: the ledger before anything ran."""
        with self._db:
            seq = self._step(episode_id, "start")
            self._write_ledger(seq, episode_id, ledger)
        return seq

    def record_message(self, episode_id: str, position: int, message: dict) -> int:
        """One conversation message (system, user, assistant or tool) at its position in the chat."""
        with self._db:
            seq = self._step(episode_id, "message")
            self._db.execute("INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
                             (seq, episode_id, position, message.get("role"), _dump(message)))
        return seq

    def record_tool_call(self, episode_id: str, call: dict, ledger: dict) -> int:
        """One tool call (keys from ``TOOL_CALL_COLUMNS``) and the ledger right after it."""
        values = [_dump(call.get(column)) if column in _JSON_COLUMNS else call.get(column)
                  for column in TOOL_CALL_COLUMNS]
        with self._db:
            seq = self._step(episode_id, "tool_call")
            self._db.execute(f"INSERT INTO tool_calls VALUES (?, ?, {', '.join('?' * len(values))})",
                             (seq, episode_id, *values))
            self._write_ledger(seq, episode_id, ledger)
        return seq

    def delete_after(self, episode_id: str, seq: int) -> int:
        """Delete every step this episode wrote after ``seq``; returns how many steps went."""
        with self._db:
            for table in DATA_TABLES:
                self._db.execute(f"DELETE FROM {table} WHERE episode_id = ? AND seq > ?", (episode_id, seq))
            deleted = self._db.execute("DELETE FROM steps WHERE episode_id = ? AND seq > ?",
                                       (episode_id, seq)).rowcount
        return deleted

    # Reads -----------------------------------------------------------------------------------------

    def seq_at(self, episode_id: str, wall_time: float) -> int | None:
        """The episode's last step at or before ``wall_time`` (None if it had not started)."""
        row = self._db.execute("SELECT MAX(seq) FROM steps WHERE episode_id = ? AND wall_time <= ?",
                               (episode_id, wall_time)).fetchone()
        return row[0]

    def ledger_at(self, episode_id: str, seq: int) -> dict:
        """The ledger as of step ``seq``: the latest snapshot at or before it."""
        row = self._db.execute("SELECT MAX(seq) FROM accounts WHERE episode_id = ? AND seq <= ?",
                               (episode_id, seq)).fetchone()
        if row[0] is None:
            return {table: [] for table in LEDGER_TABLES}
        return {table: [dict(zip(columns, values)) for values in self._db.execute(
                    f"SELECT {', '.join(columns)} FROM {table} WHERE episode_id = ? AND seq = ? ORDER BY rowid",
                    (episode_id, row[0]))]
                for table, columns in LEDGER_TABLES.items()}

    def messages_at(self, episode_id: str, seq: int) -> list[dict]:
        """The conversation as of step ``seq``, in order."""
        return [json.loads(content) for (content,) in self._db.execute(
            "SELECT content FROM messages WHERE episode_id = ? AND seq <= ? ORDER BY position, seq",
            (episode_id, seq))]

    def tool_calls_at(self, episode_id: str, seq: int) -> list[dict]:
        """The tool calls made up to step ``seq``, in order."""
        rows = self._db.execute(f"SELECT {', '.join(TOOL_CALL_COLUMNS)} FROM tool_calls "
                                "WHERE episode_id = ? AND seq <= ? ORDER BY seq", (episode_id, seq))
        return [{column: json.loads(value) if column in _JSON_COLUMNS and value is not None else value
                 for column, value in zip(TOOL_CALL_COLUMNS, row)} for row in rows]

    # Helpers ---------------------------------------------------------------------------------------

    def _step(self, episode_id: str, kind: str) -> int:
        return self._db.execute("INSERT INTO steps (episode_id, wall_time, kind) VALUES (?, ?, ?)",
                                (episode_id, time.time(), kind)).lastrowid  # type: ignore

    def _write_ledger(self, seq: int, episode_id: str, ledger: dict) -> None:
        for table, columns in LEDGER_TABLES.items():
            self._db.executemany(
                f"INSERT INTO {table} VALUES (?, ?, {', '.join('?' * len(columns))})",
                [(seq, episode_id, *(row.get(column) for column in columns)) for row in ledger.get(table, [])])


def _dump(value: Any) -> str | None:
    return None if value is None else json.dumps(value, default=str)
