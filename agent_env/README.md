# discovery-sim

Local A2A 1.0 agents (chatbot, bank, payment) with MCP-backed bank and payment tools over a synthetic ledger.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run agents

```bash
discovery-sim agents
```

Default ports: chatbot `8001`, bank `8002`, payment `8003`.

## Run attack episodes (no HTTP servers required)

**Agent mode (default)** — LM Studio **tool-calling loop**: model → tool → tool result → model, until done or `--max-agent-steps`. Attacks are scheduled on task-aware tool rounds (see `attack_applied` in `episodes.jsonl` / `baseline.json`).

```bash
export LMSTUDIO_MODEL=your-model
discovery-sim run --episodes 20 --seed 42 --out data/runs/agent42
# Re-running the same --out creates a sibling dir (timestamp-seed) unless you pass --overwrite
```

**Scripted mode (legacy harness)** — template steps + injected attacks (fast, no LLM):

```bash
discovery-sim run --mode scripted --episodes 100 --seed 42 --out data/runs/scripted42
```

Optional: `--llm-samples N` applies only to **scripted** benign planning (strict JSON match).

Writes `episodes.jsonl`, `examples.jsonl`, `laya_train.jsonl`, `questions.json`, `baseline.json`, and `run_manifest.json`. Agent examples include `conversation`, `prior_calls`, `trusted`, and `ledger_snapshot`.

**Tasks (10 agent templates):** balance (two accounts), list accounts, transactions, holder lookup, three transfer variants, and two balance-then-pay flows. Scripted mode uses the same single-step catalog plus multi-step scenarios.

**Independent runs:** Reusing `--out` when `episodes.jsonl` already exists creates a sibling directory like `myrun-20261020T183045Z-seed42`. Pass `--overwrite` to replace files in place.

**Parallel episodes:** `--concurrency N` runs up to N episodes at once (separate ledgers; parallel HTTP calls to LM Studio). Example: `discovery-sim run --episodes 50 --concurrency 4 --seed 42 --out data/runs/parallel`. With `N>1`, per-episode random choices use a derived sub-seed so runs stay reproducible by `(seed, episode_index)`.

## Tests

```bash
pytest
```
