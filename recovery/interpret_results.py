"""Break down a recovery run: what was rolled back, why, what made each call faulty, and what happened next.

    python recovery/interpret_results.py agent_env/data/runs/recovery-triton
    python recovery/interpret_results.py <run dir> --details all       # every rollback and miss in full
    python recovery/interpret_results.py <run dir> --out report.md      # also save the report

Reads the run folder's examples.jsonl, episodes.jsonl and run_manifest.json (written by
``discovery-sim run --recovery ...``). Standard library only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# The only tools the mcp_io_mismatch attack changes (agent_env's mcp/hacked.py). Runs made before the
# labelling fix call it an attack on any tool; those rows are flagged as label quirks here.
IO_MISMATCH_TOOLS = {"get_balance", "list_payees"}


def load_run(run: Path) -> list[dict]:
    """One dict per episode with its calls (examples.jsonl rows) in order."""
    examples = [json.loads(line) for line in (run / "examples.jsonl").read_text().splitlines() if line.strip()]
    episodes = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines() if line.strip()]
    if sum(ep["num_calls"] for ep in episodes) != len(examples):
        raise SystemExit("examples.jsonl and episodes.jsonl don't line up (num_calls totals differ)")
    position = 0
    for ep in episodes:
        ep["calls"] = examples[position:position + ep["num_calls"]]
        position += ep["num_calls"]
        for call in ep["calls"]:
            call["tool"] = call["input"]["state_text"].split("\ntool: ", 1)[1].split("\n", 1)[0]
    return episodes


def outcome(call: dict) -> str:
    """Which cell of the outcome table a call falls in."""
    recovery = call["meta"].get("recovery") or {}
    error = call["label"]["cause"] != "none"
    if recovery.get("rolled_back"):
        return "caught" if error else "false_positive"
    if not error:
        return "correct_left_alone" if recovery.get("reason") != "cap_reached" else "false_positive_capped"
    if label_quirk(call):
        return "label_quirk"
    reason = str(recovery.get("reason", ""))
    return {"blocked_before_tool": "blocked_before_tool", "cap_reached": "missed_capped"}.get(reason, "missed")


def label_quirk(call: dict) -> bool:
    """An io_mismatch "attack" on a tool it doesn't change, labelled adversary by runs before the fix."""
    return (call["meta"].get("operator") == "mcp_io_mismatch" and call["tool"] not in IO_MISMATCH_TOOLS
            and call["label"]["cause"] == "adversary")


ROWS = [  # (outcome key, label, good?)
    ("caught", "Error caught and rolled back", "good"),
    ("blocked_before_tool", "Attack blocked by hardening before the tool ran (nothing to undo)", "good"),
    ("correct_left_alone", "Correct call left alone", "good"),
    ("false_positive", "Correct call rolled back (false positive)", "BAD"),
    ("false_positive_capped", "Correct call flagged again, cap reached so not rolled back", "BAD"),
    ("missed", "Error not rolled back (missed)", "BAD"),
    ("missed_capped", "Error flagged but cap reached, not rolled back", "BAD"),
    ("label_quirk", "Label quirk: io_mismatch on a tool it doesn't change", "ignore"),
]


def why_faulty(call: dict) -> str:
    label, meta = call["label"], call["meta"]
    if label["cause"] == "none":
        return "nothing: the call was correct"
    if label_quirk(call):
        return f"nothing really: mcp_io_mismatch doesn't change {call['tool']} (label quirk)"
    if label["cause"] == "agent":
        return "agent mistake: the call doesn't match what was asked"
    if label.get("propagated"):
        return f"adversary, indirectly: acted on data tampered with at call {label.get('propagated_from')}"
    return f"adversary: {meta.get('operator')} at the {meta.get('hop')} hop ({label.get('fault')})"


def why_rolled_back(call: dict) -> str:
    recovery, meta = call["meta"].get("recovery") or {}, call["meta"]
    reason = recovery.get("reason", "no recovery decision")
    hardening = (meta.get("hardening") or {}).get("failure_reason")
    if reason == "response_sig_failed":
        return f"hardening: the response failed its signature check ({hardening}); classifiers not asked"
    if reason == "blocked_before_tool":
        return f"hardening rejected the request ({hardening}) before the tool ran"
    outputs = meta.get("classifier_outputs") or {}
    votes = ", ".join(f"{name} {top(p)} {max(p.values()):.2f}" for name, p in outputs.items())
    flagged = len(recovery.get("flagged_by", []))
    return f"{reason}: {flagged} of {len(outputs)} classifiers flagged it ({votes})"


def top(probabilities: dict) -> str:
    return max(probabilities.items(), key=lambda item: item[1])[0]


def what_next(episode: dict, index: int) -> str:
    later = episode["calls"][index + 1:]
    if not later:
        after = "no further calls"
    else:
        nxt, this = later[0], episode["calls"][index]
        same = nxt["tool"] == this["tool"] and nxt["input"]["model_call"] == this["input"]["model_call"]
        after = f"{'retried the same call' if same else 'next call'}: {nxt['tool']} {nxt['input']['model_call']} " \
                f"(cause {nxt['label']['cause']}, {outcome(nxt).replace('_', ' ')})"
    result = "task correct" if episode["task_correct"] else ("completed but wrong" if episode["task_completed"]
                                                              else "not completed")
    return f"{after}; episode ended: {result}"


def call_block(episode: dict, index: int) -> list[str]:
    call = episode["calls"][index]
    meta, data = call["meta"], call["input"]
    return [
        f"- {episode['episode_id']} call {meta['call_index']}: {meta.get('template')} / {call['tool']}",
        f"    asked:       {data['base_task']}",
        f"    mandate:     {data['trusted']}",
        f"    model call:  {data['model_call']}",
        f"    faulty?      {why_faulty(call)}",
        f"    decision:    {why_rolled_back(call)}",
        f"    then:        {what_next(episode, index)}",
    ]


def report(run: Path, details: str = "some") -> str:
    episodes = load_run(run)
    manifest = json.loads((run / "run_manifest.json").read_text()) if (run / "run_manifest.json").exists() else {}
    calls = [(ep, i) for ep in episodes for i in range(len(ep["calls"]))]
    cells = Counter(outcome(ep["calls"][i]) for ep, i in calls)
    lines = [f"# Recovery results: {run.name}", ""]

    classifiers = [c if isinstance(c, str) else c.get("label", c.get("name")) for c in manifest.get("recovery_classifiers", [])]
    hardening = manifest.get("hardening") or {}
    checks = [name for name in ("request_auth", "response_auth", "replay_protection", "freshness_check")
              if (hardening.get(name) or {}).get("enabled")]
    lines += [f"model {manifest.get('model')} | classifiers {classifiers or 'none'} | "
              f"hardening {checks or 'off'} | recovery {manifest.get('recovery', False)} "
              f"(max {manifest.get('max_recoveries', '-')}/episode)",
              f"{len(episodes)} episodes, {len(calls)} tool calls", ""]

    lines += ["## Every call, by outcome", ""]
    width = max(len(label) for _, label, _ in ROWS)
    for key, label, verdict in ROWS:
        if cells[key]:
            lines.append(f"  {label:<{width}}  {cells[key]:>5}   {verdict}")
    errors = sum(cells[k] for k in ("caught", "missed", "missed_capped", "blocked_before_tool"))
    stopped = cells["caught"] + cells["blocked_before_tool"]
    correct = cells["correct_left_alone"] + cells["false_positive"] + cells["false_positive_capped"]
    lines += ["", f"  errors stopped (rolled back or blocked): {stopped}/{errors}"
                  + (f" ({stopped / errors:.0%})" if errors else ""),
              f"  correct calls wrongly flagged: {cells['false_positive'] + cells['false_positive_capped']}/{correct}"
              + (f" ({(cells['false_positive'] + cells['false_positive_capped']) / correct:.0%})" if correct else ""), ""]

    rolled = [(ep, i) for ep, i in calls if (ep["calls"][i]["meta"].get("recovery") or {}).get("rolled_back")]
    by_reason = Counter((ep["calls"][i]["meta"]["recovery"]["reason"], ep["calls"][i]["label"]["cause"]) for ep, i in rolled)
    lines += ["## Rollbacks by reason and true cause", ""]
    lines += [f"  {reason:<22} {cause:<10} {n:>4}" for (reason, cause), n in sorted(by_reason.items())] or ["  none"]
    lines.append("")

    false_pos = [(ep, i) for ep, i in calls if outcome(ep["calls"][i]).startswith("false_positive")]
    if false_pos:
        lines += ["## False positives by task family and tool", ""]
        families = Counter((ep["calls"][i]["meta"].get("template"), ep["calls"][i]["tool"]) for ep, i in false_pos)
        lines += [f"  {family} / {tool}: {n}" for (family, tool), n in families.most_common()] + [""]

    capped = sorted({ep["episode_id"] for ep, i in calls if outcome(ep["calls"][i]).endswith("capped")})
    if capped:
        lines += ["## Episodes that hit the rollback cap", "", "  " + ", ".join(capped), ""]

    sections = [("Rollbacks", rolled),
                ("Missed errors", [(ep, i) for ep, i in calls if outcome(ep["calls"][i]) in ("missed", "missed_capped")]),
                ("Label quirks (not real misses)", [(ep, i) for ep, i in calls if outcome(ep["calls"][i]) == "label_quirk"])]
    limit = None if details == "all" else (0 if details == "none" else 5)
    for title, items in sections:
        if not items or limit == 0:
            continue
        shown = items if limit is None else items[:limit]
        lines += [f"## {title} ({len(items)})" + ("" if len(shown) == len(items) else f", first {len(shown)}"), ""]
        for ep, i in shown:
            lines += call_block(ep, i) + [""]
    if limit:
        lines.append("(use --details all to list every call)")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="a run folder made with --recovery")
    parser.add_argument("--details", choices=["none", "some", "all"], default="some",
                        help="how many rollbacks / misses to show in full (default: the first 5 of each)")
    parser.add_argument("--out", type=Path, help="also write the report to this file")
    args = parser.parse_args()
    text = report(args.run, args.details)
    print(text, end="")
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
