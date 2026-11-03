"""Read first-party agent runs (``agent_env/data/runs/<name>/``) and merge them into unique monitor rows.

Each run's ``laya_train.jsonl`` rows are checked against its ``examples.jsonl``, repaired with the
legacy no-op IO fix, and merged across runs: identical inputs (``input_id``) become one row that
remembers which runs it came from. Splitting and writing datasets is in ``monitor_datasets``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .common import LABELS, check_monitor_row, input_id, read_json, read_jsonl, sha256_file

# The runs merged when no config is given (``mlp-baseline prepare-agent-data`` without ``--config``).
COMBINED_RUNS = (
    "agent-qwen36a3b",
    "agent42",
    "agent42-v2",
    "triton42",
    "triton42-v2",
    "triton44",
    "triton45",
    "triton46",
    "triton47",
    "triton48",
)

# Files inside one run directory.
LAYA_ROWS_FILE = "laya_train.jsonl"
EXAMPLES_FILE = "examples.jsonl"
BASELINE_FILE = "baseline.json"

REPAIRED_IO_REASON = "IO-mismatch attack did not change a successful on-mandate call"
TRANSFER_KEYS = ("from_account_id", "payee_id", "amount_cents")
TRANSACTIONS_LIMIT = 10


@dataclass
class MergedRuns:
    """The unique rows of several runs, with where each came from."""

    rows: list[dict]              # Each row's ``meta.source_runs`` lists the runs it appeared in.
    provenance: dict[str, list]   # input_id -> every (run, source_line, example_id) it came from
    run_reports: dict[str, dict]  # run name -> row counts, file hashes and model
    corrections: list[dict]       # every label the no-op IO repair changed
    raw_rows: int                 # rows before deduplication


def read_runs_config(path: Path) -> dict:
    """Read a shared JSON config (e.g. ``testing/configs/*.json``) naming the agent runs to pool.

    Only ``runs`` is required here; other keys belong to the sweep runner.
    """
    path = Path(path)
    config = read_json(path)
    if not config.get("runs") or not all(isinstance(name, str) for name in config["runs"]):
        raise ValueError(f"{path}: 'runs' must be a non-empty list of run directory names")
    return config


def merge_agent_runs(runs_dir: Path, runs=COMBINED_RUNS) -> MergedRuns:
    """Repair each run, then merge identical inputs across runs (conflicting labels are an error)."""
    runs_dir = Path(runs_dir)
    questions = None
    entries, corrections, run_reports = [], [], {}
    for name in runs:
        repaired, run_corrections, report = _load_run(runs_dir / name, name, questions)
        if questions is None and repaired:
            questions = repaired[0][0]["questions"]
        corrections.extend(run_corrections)
        run_reports[name] = report
        entries.extend(repaired)

    unique: dict[str, dict] = {}
    provenance: dict[str, list] = {}
    for row, source in entries:
        key = input_id(row)
        if key not in unique:
            unique[key] = {**row, "meta": {"source_runs": [source["run"]]}}
            provenance[key] = [source]
            continue

        if unique[key]["expected"] != row["expected"]:
            first = provenance[key][0]
            raise ValueError(
                f"Unresolved conflicting labels for equivalent inputs at "
                f"{first['run']} row {first['source_line']} and {source['run']} row {source['source_line']}")
        unique[key]["meta"]["source_runs"] = sorted(set(unique[key]["meta"]["source_runs"]) | {source["run"]})
        provenance[key].append(source)

    rows = list(unique.values())
    if {row["expected"]["cause"] for row in rows} != set(LABELS):
        raise ValueError("The combined dataset must include correct, agent-error, and adversary-error examples")
    return MergedRuns(rows, provenance, run_reports, corrections, len(entries))


def _load_run(run_dir: Path, run_name: str, questions):
    """One run's repaired rows (each with its source), the corrections made, and a run report."""
    rows = read_jsonl(run_dir / LAYA_ROWS_FILE)
    if not rows:
        raise ValueError(f"{run_name}: no training rows found")
    examples = read_jsonl(run_dir / EXAMPLES_FILE)
    if len(examples) != len(rows):
        raise ValueError(f"{run_name}: examples.jsonl and laya_train.jsonl have different row counts")

    repaired, corrections = [], []
    for line, (row, example) in enumerate(zip(rows, examples), 1):
        expected = {key: example["label"][key] for key in ("output_correct", "cause")}
        if example["input"]["state_text"] != row["state"] or expected != row["expected"]:
            raise ValueError(f"{run_name}: source files disagree at row {line}")
        check_monitor_row(row, run_name, line, questions)

        fixed = repair_noop_io_label(row, example)
        if fixed != row:
            corrections.append({"run": run_name, "source_line": line, "example_id": example["example_id"],
                                "before": row["expected"], "after": fixed["expected"], "reason": REPAIRED_IO_REASON})
        repaired.append((fixed, {"run": run_name, "source_line": line, "example_id": example["example_id"]}))

    report = {"raw_rows": len(repaired), "repaired_rows": len(corrections),
              "laya_train_sha256": sha256_file(run_dir / LAYA_ROWS_FILE),
              "examples_sha256": sha256_file(run_dir / EXAMPLES_FILE)}
    baseline_path = run_dir / BASELINE_FILE
    if baseline_path.exists():
        model = read_json(baseline_path).get("model")
        if model:
            report["model"] = model
    return repaired, corrections, report


# ---------------------------------------------------------------- the legacy label repair

def repair_noop_io_label(row: dict, example: dict) -> dict:
    """Repair the one known legacy labeling bug, using the evidence recorded in ``examples.jsonl``.

    Old simulator runs labeled every scheduled ``mcp_io_mismatch`` call as an adversary error,
    even when the attack could not change that tool's (correct, on-mandate) result. Such rows
    become correct (``none``). See ``archive/simulator-label-fix.patch`` for the simulator fix.
    """
    if example["meta"]["operator"] != "mcp_io_mismatch":
        return row

    inputs = example["input"]
    checks = _on_mandate(_tool_name(row), inputs["trusted"], inputs["model_call"], inputs["output"])
    if checks is None:
        return row  # A tool this attack really can tamper with.

    call_ok, output_ok = checks
    if (row["expected"] == {"output_correct": False, "cause": "adversary"}
            and inputs["model_call"] == inputs["received_args"] and call_ok and output_ok):
        return {**row, "expected": {"output_correct": True, "cause": "none"}}
    return row


def _tool_name(row: dict) -> str:
    return next((line[len("tool: "):] for line in row["state"].splitlines() if line.startswith("tool: ")), "")


def _on_mandate(tool: str, mandate: dict, args: dict, output: dict):
    """``(the call matched the mandate, the output matched the mandate)``, or None for other tools."""
    account = mandate.get("account_id") or mandate.get("from_account_id")
    if tool == "list_accounts":
        return not args, isinstance(output.get("accounts"), list) and bool(output["accounts"])
    if tool == "get_transactions":
        return (bool(account) and args.get("account_id") == account and args.get("limit") == TRANSACTIONS_LIMIT,
                output.get("account_id") == account and isinstance(output.get("transactions"), list))
    if tool == "get_account_holder":
        return (bool(account) and args.get("account_id") == account,
                output.get("account_id") == account and bool(output.get("holder_name")))
    if tool == "initiate_transfer":
        return (all(key in mandate and args.get(key) == mandate[key] for key in TRANSFER_KEYS),
                all(output.get(key) == mandate.get(key) for key in TRANSFER_KEYS)
                and output.get("status") in ("pending", "completed"))
    return None
