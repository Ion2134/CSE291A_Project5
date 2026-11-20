"""Shared basics: the monitor's labels, row identity and grouping, file helpers and run bookkeeping."""
from __future__ import annotations

import hashlib
import json
import re
import site
import sys
import time
from collections import Counter
from pathlib import Path

# The banking monitor's three causes. ``none`` means the tool output was correct.
LABELS = ("none", "agent", "adversary")

# The two monitor questions every banking row answers.
MONITOR_QUESTIONS = {"output_correct", "cause"}

# The frozen injection detector used by the ``stacking`` monitor (relative to ``mlp_baseline/``).
DEFAULT_INJECTION_CHECKPOINT = Path("experiments/injection/seed42/model")

# Random bookkeeping ids in tool outputs; masked so they don't make equal inputs look different.
_BOOKKEEPING_ID = re.compile(r"\b(pay|txn)_synth_[0-9a-f]{8}\b")
_ACCOUNT_OR_PAYEE_ID = re.compile(r"\b(acct|payee)_synth_\w+\b")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_USER_MESSAGE_PREFIX = "user_message: "


# ---------------------------------------------------------------- files

def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_empty_dir(path: Path, message: str) -> None:
    """Refuse to write into a directory that already has files (outputs are never overwritten)."""
    if path.exists() and any(path.iterdir()):
        raise ValueError(message)


def require_isolated_venv() -> None:
    if sys.prefix == sys.base_prefix or site.ENABLE_USER_SITE:
        raise RuntimeError("Run MLP training inside the isolated project venv")


# ---------------------------------------------------------------- monitor rows

def input_id(row: dict) -> str:
    """Identity of a row's input (state + questions), with random bookkeeping ids masked."""
    state = _BOOKKEEPING_ID.sub(r"\1_synth_<id>", row["state"])
    text = json.dumps({"state": state, "questions": row["questions"]}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def request_group(row: dict) -> str:
    """The row's user request with account ids and numbers masked; splits never cut across a group."""
    request = next((line[len(_USER_MESSAGE_PREFIX):] for line in row["state"].splitlines()
                    if line.startswith(_USER_MESSAGE_PREFIX)), None)
    if request is None:
        raise ValueError("Each monitor state must contain a user_message line for grouped splitting")

    request = _ACCOUNT_OR_PAYEE_ID.sub(r"\1_synth_<id>", request.lower())
    request = _NUMBER.sub("<number>", request)
    return hashlib.sha256(request.encode()).hexdigest()


def cause_of(row: dict) -> str:
    return row["expected"]["cause"]


def cause_counts(rows) -> dict:
    """``{cause: count}`` sorted by cause name."""
    return dict(sorted(Counter(cause_of(row) for row in rows).items()))


def valid_monitor_labels(expected: dict) -> bool:
    """Both answers are present and agree: the output is correct exactly when the cause is ``none``."""
    return (set(expected) == MONITOR_QUESTIONS
            and type(expected["output_correct"]) is bool
            and expected["cause"] in LABELS
            and expected["output_correct"] == (expected["cause"] == "none"))


def check_monitor_row(row: dict, source: str, line: int, questions) -> None:
    """Raise if a row has invalid labels or a different question schema than ``questions``."""
    if not valid_monitor_labels(row["expected"]):
        raise ValueError(f"Invalid or inconsistent monitor labels at {source} row {line}")
    if questions is not None and row["questions"] != questions:
        raise ValueError("All runs must use the same monitor question schema")


# ---------------------------------------------------------------- run bookkeeping

class RunRecord:
    """A ``results.json`` that is rewritten as a run progresses, so a crash still leaves a record.

    Use as a context manager: it is saved on entry; on exit it gets ``status``/``phase`` set to
    ``complete`` (or ``failed`` plus the error) and ``elapsed_seconds``, and is saved again.
    """

    def __init__(self, path: Path, header: dict):
        self.path = path
        self.result = dict(header)
        self._started = 0.0

    def __enter__(self) -> "RunRecord":
        self._started = time.perf_counter()
        self.save()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        elapsed = time.perf_counter() - self._started
        if exc is None:
            self.result.update(status="complete", phase="complete", elapsed_seconds=elapsed)
        else:
            self.result.update(status="failed", phase="failed", error=f"{exc_type.__name__}: {exc}",
                               elapsed_seconds=elapsed)
        self.save()
        return False  # Never swallow the error.

    def update(self, **fields) -> None:
        self.result.update(fields)

    def set_phase(self, phase: str) -> None:
        self.result["phase"] = phase
        self.save()

    def save(self) -> None:
        write_json(self.path, self.result)
