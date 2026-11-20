"""Audit a run's telemetry Parquet against the run's own records.

    python scripts/verify_telemetry.py data/runs/hardening_sample [more run dirs...]
    python scripts/verify_telemetry.py --self-test /tmp/telemetry_selftest

The first form checks existing runs: completeness (one trace per episode, one span per tool call),
span tree structure and timing, per-call agreement with examples.jsonl, status/error semantics, and
counts against baseline.json.

``--self-test`` additionally generates runs that stress the logger (5000 fast episodes, 16
concurrent episodes with a fake async LM, reruns into the same directory, a crash mid-run, content
capture off) and audits them. No network needed. Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
from mcp_hardening import load_config, read_telemetry
from mcp_hardening import semconv as sc
from mcp_hardening.telemetry.exporter import SCHEMA

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)] if path.exists() else []


def audit(run: Path) -> None:
    manifest = json.loads((run / "run_manifest.json").read_text())
    tel_cfg = (manifest.get("hardening") or {}).get("telemetry") or {}
    tel = run / tel_cfg.get("dir", "telemetry")
    episodes = _jsonl(run / "episodes.jsonl")
    examples = _jsonl(run / "examples.jsonl")
    malformed = _jsonl(run / "malformed_calls.jsonl")
    errors = _jsonl(run / "errors.jsonl")
    baseline = json.loads((run / "baseline.json").read_text())
    rows = read_telemetry(tel).to_pylist()
    parts = sorted(tel.glob("part-*.parquet"))

    by_op: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_op[r[sc.GEN_AI_OPERATION_NAME]].append(r)
    agents, tools, chats = by_op["invoke_agent"], by_op["execute_tool"], by_op["chat"]
    print(
        f"  {run}: {len(parts)} part file(s), {len(rows)} rows "
        f"({len(agents)} invoke_agent, {len(chats)} chat, {len(tools)} execute_tool)"
    )

    # --- files / schema
    check(bool(parts), "telemetry part files exist")
    check(all(pq.read_schema(p).equals(SCHEMA) for p in parts), "every part file has the fixed schema")
    check(not set(by_op) - {"invoke_agent", "chat", "execute_tool"}, f"only GenAI operations present: {sorted(by_op)}")

    # --- completeness (crashed episodes have spans but no episodes.jsonl row)
    ep_ids = {e["episode_id"] for e in episodes}
    logged_ids = {a[sc.GEN_AI_CONVERSATION_ID] for a in agents}
    check(len(agents) == len(episodes) + len(errors), f"one invoke_agent per episode ({len(agents)} vs {len(episodes)} + {len(errors)} crashed)")
    check(len({r["trace_id"] for r in rows}) == len(agents), "one trace per episode")
    check(ep_ids <= logged_ids, "every episodes.jsonl episode_id has a trace (gen_ai.conversation.id)")
    agent_by_trace = {a["trace_id"]: a for a in agents}
    done_traces = {t for t, a in agent_by_trace.items() if a[sc.GEN_AI_CONVERSATION_ID] in ep_ids}
    real_tools = [t for t in tools if not t[sc.SIM_MALFORMED] and t["trace_id"] in done_traces]
    mal_tools = [t for t in tools if t[sc.SIM_MALFORMED] and t["trace_id"] in done_traces]
    check(len(real_tools) == len(examples), f"one execute_tool per executed call ({len(real_tools)} vs {len(examples)})")
    check(len(mal_tools) == len(malformed), f"one execute_tool per malformed call ({len(mal_tools)} vs {len(malformed)})")
    if manifest.get("mode") == "agent":
        check(len(chats) >= len(agents), "at least one chat span per episode")

    # --- tree structure and timing
    span_ids = {r["span_id"]: r for r in rows}
    check(all(a["parent_span_id"] is None for a in agents), "invoke_agent spans are roots")
    misparented = [
        r for r in rows
        if r[sc.GEN_AI_OPERATION_NAME] != "invoke_agent"
        and span_ids.get(r["parent_span_id"]) is not agent_by_trace.get(r["trace_id"])
    ]
    check(not misparented, f"every chat/tool span is a child of its own episode ({len(misparented)} misparented)")
    check(all(r["end_time_unix_nano"] >= r["start_time_unix_nano"] for r in rows), "end >= start for every span")
    outside = [
        r for r in rows
        if r["parent_span_id"] in span_ids
        and not (
            span_ids[r["parent_span_id"]]["start_time_unix_nano"] <= r["start_time_unix_nano"]
            and r["end_time_unix_nano"] <= span_ids[r["parent_span_id"]]["end_time_unix_nano"]
        )
    ]
    check(not outside, "children lie within their parent's time window")

    # --- per-call agreement with examples.jsonl (same order: episodes in file order, calls in time order)
    ep_by_trace = {a["trace_id"]: a[sc.GEN_AI_CONVERSATION_ID] for a in agents}
    tools_by_ep: dict[str, list[dict]] = collections.defaultdict(list)
    for t in sorted(real_tools, key=lambda r: r["start_time_unix_nano"]):
        tools_by_ep[ep_by_trace[t["trace_id"]]].append(t)
    it = iter(examples)
    mismatches: list[tuple] = []
    for e in episodes:
        exs = [next(it) for _ in range(e["num_calls"])]
        spans = tools_by_ep[e["episode_id"]]
        if len(exs) != len(spans):
            mismatches.append((e["episode_id"], "count", len(spans), len(exs)))
            continue
        for ex, t in zip(exs, spans):
            h = ex["meta"].get("hardening", {})
            want = {
                sc.SIM_CALL_INDEX: ex["meta"]["call_index"],
                sc.SIM_OPERATOR: ex["meta"]["operator"],
                sc.SIM_OUTPUT_CORRECT: ex["label"]["output_correct"],
                sc.SIM_CAUSE: ex["label"]["cause"],
                sc.HARDENING_FAILURE_REASON: h.get("failure_reason"),
                sc.HARDENING_REQUEST_VERIFIED: h.get("request_verified"),
                sc.HARDENING_RESPONSE_VERIFIED: h.get("response_verified"),
            }
            mismatches += [(e["episode_id"], k, t[k], v) for k, v in want.items() if t[k] != v]
    check(not mismatches, f"tool span attrs == examples.jsonl labels/hardening, call by call {mismatches[:3]}")

    # --- status / error semantics
    ep_by_id = {e["episode_id"]: e for e in episodes}
    bad = [
        a for a in agents
        if a[sc.GEN_AI_CONVERSATION_ID] in ep_by_id
        and a["status_code"] != ("OK" if ep_by_id[a[sc.GEN_AI_CONVERSATION_ID]]["task_completed"] else "ERROR")
    ]
    check(not bad, "invoke_agent status OK iff task_completed")
    hard_rows = [t for t in real_tools if t[sc.HARDENING_FAILURE_REASON]]
    check(
        all(t["status_code"] == "ERROR" and t[sc.ERROR_TYPE] == "mcp_hardening" for t in hard_rows),
        f"hardening-rejected calls -> ERROR, error.type=mcp_hardening ({len(hard_rows)} rows)",
    )
    check(
        all(t["status_code"] == "ERROR" and t[sc.ERROR_TYPE] == "malformed_tool_call" for t in mal_tools),
        f"malformed calls -> ERROR, error.type=malformed_tool_call ({len(mal_tools)} rows)",
    )
    check(all(t[sc.ERROR_TYPE] is None for t in real_tools if t["status_code"] == "OK"), "OK tool spans carry no error.type")
    check(all(t["status_code"] != "UNSET" for t in tools + chats), "no tool/chat span left with UNSET status")
    blocked = baseline.get("hardening_blocked_calls", {}).get("total", 0)
    check(len(hard_rows) == blocked, f"blocked-call count == baseline.json ({len(hard_rows)} vs {blocked})")

    # --- chat spans
    ok_chats = [c for c in chats if c["status_code"] == "OK"]
    if ok_chats:
        check(all(c[sc.GEN_AI_REQUEST_MODEL] and c[sc.GEN_AI_PROVIDER_NAME] for c in ok_chats), "model + provider recorded on chat spans")
        check(
            all(c[sc.GEN_AI_USAGE_INPUT_TOKENS] and c[sc.GEN_AI_USAGE_OUTPUT_TOKENS] is not None for c in ok_chats),
            "token usage recorded on every successful chat span",
        )
        attrs = [json.loads(c["attributes_json"]) for c in ok_chats]
        check(all(sc.GEN_AI_RESPONSE_FINISH_REASONS in a for a in attrs), "finish reasons recorded")
        if tel_cfg.get("capture_content", True):
            check(all(sc.GEN_AI_INPUT_MESSAGES in a for a in attrs), "input messages captured (capture_content on)")
    if tel_cfg.get("capture_content") is False:
        leaked = [r for r in rows if any(k in r["attributes_json"] for k in (sc.GEN_AI_INPUT_MESSAGES, sc.GEN_AI_TOOL_CALL_ARGUMENTS, sc.GEN_AI_TOOL_CALL_RESULT))]
        check(not leaked, "no content attributes (capture_content off)")


# ---------------------------------------------------------------------------------------------
# --self-test: generate runs that stress the logger, then audit them
# ---------------------------------------------------------------------------------------------


def _fake_chat_cls():
    from discovery_sim.llm.chat import ChatClient, ChatTurn, ToolCallRequest

    class FakeAsyncChat(ChatClient):
        """Async with random latency (interleaves concurrent episodes), OpenAI-style usage,
        sometimes malforms its first call."""

        model_id = "fake-model"
        provider_name = "fake"
        temperature = 0.2

        def __init__(self) -> None:
            self.n = 0

        async def complete(self, messages, tools):
            await asyncio.sleep(random.random() * 0.01)
            self.n += 1
            tool_msgs = [m for m in messages if m["role"] == "tool"]
            mandate = json.loads(messages[1]["content"].split("amounts): ")[1])
            acct = mandate.get("account_id") or mandate.get("from_account_id") or "acct_synth_0031"
            body = {"id": f"r{self.n}", "model": self.model_id, "usage": {"prompt_tokens": 100 + len(messages), "completion_tokens": 7}}
            if not tool_msgs and random.random() < 0.3:
                tc = ToolCallRequest(f"m{self.n}", "bank__get_balance", {}, raw_arguments="{}")
            elif not tool_msgs or (len(tool_msgs) == 1 and "error" in tool_msgs[0]["content"]):
                tc = ToolCallRequest(f"c{self.n}", "bank__get_balance", {"account_id": acct})
            elif "amount_cents" in mandate and not any("payment_id" in m["content"] for m in tool_msgs):
                args = {k: mandate[k] for k in ("from_account_id", "payee_id", "amount_cents")}
                tc = ToolCallRequest(f"c{self.n}", "payment__initiate_transfer", args)
            else:
                body["choices"] = [{"finish_reason": "stop"}]
                return ChatTurn(content="Done.", tool_calls=[], raw=body)
            body["choices"] = [{"finish_reason": "tool_calls"}]
            return ChatTurn(content=None, tool_calls=[tc], raw=body)

    return FakeAsyncChat


def self_test(out: Path) -> None:
    import discovery_sim.sim.episode as episode_mod
    from discovery_sim.sim.episode import run_episodes
    from discovery_sim.sim.run_config import RunConfig

    fake = _fake_chat_cls()
    cfg = load_config("all_on")
    small = replace(cfg, telemetry=replace(cfg.telemetry, flush_rows=37))

    print("\n[self-test 1] scripted, 5000 fast episodes (no spans may be dropped)")
    asyncio.run(run_episodes(RunConfig(episodes=5000, seed=1, mode="scripted", hardening=cfg, overwrite_output=True), out / "burst"))
    audit(out / "burst")

    print("\n[self-test 2] agent mode, 120 episodes, concurrency 16, fake async LM, flush_rows=37")
    episode_mod.make_chat_client = lambda *a, **k: fake()
    random.seed(0)
    asyncio.run(run_episodes(RunConfig(episodes=120, seed=11, mode="agent", concurrency=16, hardening=small, overwrite_output=True), out / "agent"))
    audit(out / "agent")
    sizes = [pq.read_metadata(p).num_rows for p in sorted((out / "agent" / "telemetry").glob("part-*.parquet"))]
    check(all(s == 37 for s in sizes[:-1]) and 0 < sizes[-1] <= 37, f"part files hold exactly flush_rows rows ({sizes})")

    print("\n[self-test 3] append-only across a rerun into the same directory")
    tel = out / "agent" / "telemetry"
    before = {p.name: p.read_bytes() for p in tel.glob("part-*.parquet")}
    asyncio.run(run_episodes(RunConfig(episodes=10, seed=12, mode="agent", hardening=small, overwrite_output=True), out / "agent"))
    after = {p.name: p.read_bytes() for p in tel.glob("part-*.parquet")}
    check(all(after.get(n) == b for n, b in before.items()), f"all {len(before)} earlier part files byte-identical")
    check(len(after) > len(before), f"rerun appended new parts ({len(before)} -> {len(after)})")

    print("\n[self-test 4] crash mid-run still flushes what was logged")

    class Boom(fake):
        async def complete(self, messages, tools):
            if self.n >= 5:
                raise KeyboardInterrupt  # escapes per-episode handling: aborts the whole run
            return await super().complete(messages, tools)

    boom = Boom()
    episode_mod.make_chat_client = lambda *a, **k: boom
    try:
        asyncio.run(run_episodes(RunConfig(episodes=20, seed=13, mode="agent", hardening=cfg, overwrite_output=True), out / "crash"))
    except KeyboardInterrupt:
        pass
    rows = read_telemetry(out / "crash" / "telemetry").to_pylist()
    check(sum(r[sc.GEN_AI_OPERATION_NAME] == "chat" for r in rows) >= 5, f"spans from before the crash were written ({len(rows)} rows)")

    print("\n[self-test 5] capture_content=false")
    off = replace(cfg, telemetry=replace(cfg.telemetry, capture_content=False))
    episode_mod.make_chat_client = lambda *a, **k: fake()
    asyncio.run(run_episodes(RunConfig(episodes=10, seed=14, mode="agent", hardening=off, overwrite_output=True), out / "nocontent"))
    audit(out / "nocontent")
    rows = read_telemetry(out / "nocontent" / "telemetry").to_pylist()
    check(not any("acct_synth" in r["attributes_json"] for r in rows), "no account ids anywhere in the telemetry")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, help="run directories to audit")
    parser.add_argument("--self-test", type=Path, metavar="DIR", help="generate stress runs in DIR and audit them")
    args = parser.parse_args()
    if not args.runs and not args.self_test:
        parser.error("give run directories and/or --self-test DIR")
    for run in args.runs:
        print(f"\n[audit] {run}")
        audit(run)
    if args.self_test:
        self_test(args.self_test)
    print(f"\n{'ALL CHECKS PASSED' if not FAILS else f'{len(FAILS)} FAILED:'}")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
