from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any



def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _repair_noop_io_label(row: dict[str, Any], example: dict[str, Any]) -> dict[str, Any]:
    """Repair only the known legacy no-op IO attack, using recorded on-mandate evidence."""
    if example["meta"]["operator"] != "mcp_io_mismatch":
        return row
    tool = next((line[6:] for line in row["state"].splitlines() if line.startswith("tool: ")), "")
    inputs = example["input"]
    mandate, args, output = inputs["trusted"], inputs["model_call"], inputs["output"]
    account = mandate.get("account_id") or mandate.get("from_account_id")
    if tool == "list_accounts":
        call_ok = not args
        output_ok = isinstance(output.get("accounts"), list) and bool(output["accounts"])
    elif tool in ("get_account_holder", "get_transactions"):
        call_ok = bool(account) and args.get("account_id") == account
        if tool == "get_transactions":
            call_ok = call_ok and args.get("limit") == 10
            output_ok = output.get("account_id") == account and isinstance(output.get("transactions"), list)
        else:
            output_ok = output.get("account_id") == account and bool(output.get("holder_name"))
    elif tool == "initiate_transfer":
        keys = ("from_account_id", "payee_id", "amount_cents")
        call_ok = all(key in mandate and args.get(key) == mandate[key] for key in keys)
        output_ok = all(output.get(key) == mandate.get(key) for key in keys) and output.get("status") in ("pending", "completed")
    else:
        return row
    if (row["expected"] == {"output_correct": False, "cause": "adversary"}
            and args == inputs["received_args"] and call_ok and output_ok):
        return {**row, "expected": {"output_correct": True, "cause": "none"}}
    return row


def prepare_training_data(run_dir: Path, out_dir: Path | None = None) -> dict[str, Any]:
    """Write a repaired, deduplicated training copy and an audit trail; keep raw runs intact."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else run_dir / "prepared"
    if out_dir.resolve() == run_dir.resolve():
        raise ValueError("Use a separate output directory to preserve the raw run files")
    source_path = run_dir / "laya_train.jsonl"
    rows = _read_jsonl(source_path)
    if not rows:
        raise ValueError("No training rows found")
    examples_path = run_dir / "examples.jsonl"
    examples = _read_jsonl(examples_path) if examples_path.exists() else None
    if examples is not None and len(examples) != len(rows):
        raise ValueError("examples.jsonl and laya_train.jsonl have different row counts")

    cleaned: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    corrections = []
    seen: dict[str, int] = {}
    for line, original in enumerate(rows, 1):
        row = original
        example = examples[line - 1] if examples is not None else None
        if example is not None:
            expected = {key: example["label"][key] for key in ("output_correct", "cause")}
            if example["input"]["state_text"] != original["state"] or expected != original["expected"]:
                raise ValueError(f"Source files disagree at row {line}; no output written")
            row = _repair_noop_io_label(original, example)
            if row != original:
                corrections.append({
                    "source_line": line,
                    "example_id": example["example_id"],
                    "before": original["expected"],
                    "after": row["expected"],
                    "reason": "IO-mismatch attack did not change a successful on-mandate call",
                })

        # Expected answers are deliberately excluded: different answers to the
        # same input are a data error, not separate examples to keep or vote on.
        key = json.dumps({"state": row["state"], "questions": row["questions"]}, sort_keys=True)
        source = {"source_line": line}
        if example is not None:
            source["example_id"] = example["example_id"]
        if key in seen:
            index = seen[key]
            if cleaned[index]["expected"] != row["expected"]:
                first = provenance[index]["sources"][0]["source_line"]
                raise ValueError(f"Unresolved conflicting labels at rows {first} and {line}; no output written")
            provenance[index]["sources"].append(source)
        else:
            seen[key] = len(cleaned)
            cleaned.append(row)
            provenance.append({"clean_line": len(cleaned), "sources": [source]})

    report = {
        "source_file": source_path.name,
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "examples_sha256": hashlib.sha256(examples_path.read_bytes()).hexdigest() if examples is not None else None,
        "raw_rows": len(rows),
        "repaired_rows": len(corrections),
        "duplicate_rows_removed": len(rows) - len(cleaned),
        "unique_inputs": len(cleaned),
        "label_counts": dict(sorted(Counter(row["expected"]["cause"] for row in cleaned).items())),
        "remaining_conflicting_inputs": 0,
        "label_corrections": corrections,
        "baseline_note": "Raw run files and baseline.json are historical and unchanged; baseline metrics are not recomputed here.",
    }
    # Finish validation before writing anything, so unknown conflicts cannot
    # leave a partially cleaned dataset behind.
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "laya_train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in cleaned))
    (out_dir / "provenance.jsonl").write_text("".join(json.dumps(row) + "\n" for row in provenance))
    (out_dir / "data_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
