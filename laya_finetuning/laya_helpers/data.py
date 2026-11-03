"""Load and validate the train/val/test splits, and tokenize them into Laya training items."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .common import LABELS, TASK_QUESTIONS, read_jsonl, sha256

_BOOKKEEPING_ID = re.compile(r"\b(pay|txn)_synth_[0-9a-f]{8}\b")
_ACCOUNT_OR_PAYEE_ID = re.compile(r"\b(acct|payee)_synth_\w+\b")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_USER_MESSAGE_PREFIX = "user_message: "


# ---------------------------------------------------------------- validation

def read_splits(paths: dict, task: str):
    """Load each split and validate labels, schema, duplicate inputs and group isolation.

    Returns ``(splits, report)``: split name -> rows, and split name -> path, hash, size, label counts.
    """
    splits, report = {}, {}
    seen_inputs, group_splits = {}, {}   # input signature -> where it was seen; group -> its split
    schema = None
    for name, path in paths.items():
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f"Empty {name} split")

        for index, row in enumerate(rows, 1):
            where = f"{name}:{index}"
            _check_row(row, task, where)

            if schema is None:
                schema = row["questions"]
            if row["questions"] != schema:
                raise ValueError(f"Question schema changed at {where}")

            signature = _input_signature(row)
            if signature in seen_inputs:
                raise ValueError(f"Duplicate input at {where}; already in {seen_inputs[signature]}")
            seen_inputs[signature] = where

            group = _group_of(row, task, where)
            if group in group_splits and group_splits[group] != name:
                raise ValueError(f"Group overlap between {group_splits[group]} and {name}")
            group_splits[group] = name

        splits[name] = rows
        report[name] = {"path": str(Path(path).resolve()), "sha256": sha256(path), "rows": len(rows),
                        "labels": {qid: dict(Counter(str(r["expected"][qid]) for r in rows)) for qid in schema}}

    for qid in schema:
        if {r["expected"][qid] for r in splits["train"]} != set(LABELS[qid]):
            raise ValueError(f"Training split must contain every {qid} class")
    return splits, report


def _check_row(row: dict, task: str, where: str) -> None:
    """The row has a state, exactly the task's questions, and valid, consistent answers."""
    if not isinstance(row.get("state"), str) or not row["state"].strip():
        raise ValueError(f"Missing state at {where}")

    questions, expected = row.get("questions"), row.get("expected")
    keys = TASK_QUESTIONS[task]
    if (not isinstance(questions, dict) or not isinstance(expected, dict)
            or set(questions) != keys or set(expected) != keys):
        raise ValueError(f"Wrong question/label keys for {task} at {where}")

    for qid in keys:
        if not _valid_answer(qid, expected[qid], questions[qid]):
            raise ValueError(f"Invalid {qid} labels/schema at {where}")
    if task == "banking" and expected["output_correct"] != (expected["cause"] == "none"):
        raise ValueError(f"Inconsistent correctness/cause at {where}")


def _valid_answer(qid: str, value, question: dict) -> bool:
    """``output_correct`` is a yes/no (``noul``) question; the others are multiple choice."""
    if qid == "output_correct":
        return type(value) is bool and question.get("type") == "noul"
    return (type(value) is str and value in LABELS[qid] and question.get("type") == "choice"
            and set(question.get("criteria", {})) == set(LABELS[qid]))


def _input_signature(row: dict) -> str:
    """Case- and whitespace-insensitive state, with random payment/transaction ids masked."""
    state = _BOOKKEEPING_ID.sub(r"\1_synth_<id>", row["state"])
    return " ".join(state.lower().split())


def _group_of(row: dict, task: str, where: str) -> str:
    """Banking rows group by user request; injection rows carry a similarity ``group_id``."""
    if task == "banking":
        return request_group(row["state"])
    group = row.get("group_id")
    if not isinstance(group, str):
        raise ValueError(f"Injection rows need group_id for similarity-group protection at {where}")
    return group


def request_group(state: str) -> str:
    """The user request with account ids and numbers masked (same grouping as ``mlp_baseline``)."""
    request = next((line[len(_USER_MESSAGE_PREFIX):] for line in state.splitlines()
                    if line.startswith(_USER_MESSAGE_PREFIX)), None)
    if request is None:
        raise ValueError("Banking states need a user_message line for request grouping")
    request = _ACCOUNT_OR_PAYEE_ID.sub(r"\1_synth_<id>", request.lower())
    return _NUMBER.sub("<number>", request)


# ---------------------------------------------------------------- tokenization

def prepare_items(splits: dict, checkpoint: Path, max_len: int, head_max_len: int):
    """Tokenize every split into Laya items, refusing any state that would be truncated.

    Returns ``(tokenizer, items, report)``.
    """
    from laya.train import items_from_rows
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True)
    items, report = {}, {}
    for name, rows in splits.items():
        longest, tightest_room = _check_no_truncation(tokenizer, name, rows, max_len, head_max_len)

        items[name], skipped = items_from_rows(tokenizer, rows, max_len, head_max_len)
        if any(skipped.values()) or len(items[name]) != sum(len(r["questions"]) for r in rows):
            raise ValueError(f"Laya skipped questions in {name}: {skipped}")
        report[name] = {"items": len(items[name]), "max_state_tokens": longest,
                        "min_state_room": tightest_room, "skipped": skipped}
    return tokenizer, items, report


def _check_no_truncation(tokenizer, name, rows, max_len, head_max_len):
    """Every state fits the room its questions leave; returns (longest state, tightest room)."""
    from laya.common import encode_text, state_room
    from laya.train import to_internal

    longest, tightest_room = 0, max_len
    for i, row in enumerate(rows, 1):
        length = len(encode_text(tokenizer, row["state"], add_special_tokens=False)["input_ids"])
        room = min(state_room(tokenizer, to_internal(qid, question), max_len, head_max_len)
                   for qid, question in row["questions"].items())
        if length > room:
            raise ValueError(f"{name}:{i} would truncate {length} state tokens to {room}; increase --max-len")
        longest, tightest_room = max(longest, length), min(tightest_room, room)
    return longest, tightest_room
