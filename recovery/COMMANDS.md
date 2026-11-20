# Recovery commands

Run from `agent_env/`. Each flag is on its own line so you can see (and change) every setting.

## Quick check: oracle (no API)

The agent follows the task's plan and the `oracle` classifier flags exactly the true errors, so this
shows recovery working on its own.

```bash
.venv/bin/discovery-sim run \
  --mode agent \
  --llm-backend oracle \
  --task-set hard \
  --episodes 40 \
  --state-db \
  --recovery \
  --max-recoveries 2 \
  --recovery-classifiers '["oracle"]' \
  --hardening-config all_on \
  --out data/runs/recovery-oracle
```

## Full run: Triton model, real classifiers, hardening

```bash
.venv/bin/discovery-sim run \
  --mode agent \
  --llm-backend triton \
  --llm-model api-glm-5.3-flash \
  --temperature 0.7 \
  --task-set hard \
  --episodes 50 \
  --concurrency 2 \
  --state-db \
  --recovery \
  --max-recoveries 2 \
  --recovery-classifiers '[
      "rules",
      {"name": "logreg_checks", "train": "testing/results/agent-v8-logreg/data/p42/f0/train.jsonl"},
      {"name": "logreg_memory", "train": "testing/results/agent-v8-logreg/data/p42/f0/train.jsonl"}
    ]' \
  --hardening-config all_on \
  --out data/runs/recovery-triton
```

| Flag | What it does |
| --- | --- |
| `--mode agent` | An LLM agent makes the tool calls (needed for recovery) |
| `--llm-backend` | `oracle` (scripted, no API), `triton` or `lmstudio` |
| `--llm-model` | Model id for `triton` / `lmstudio` |
| `--temperature` | Agent sampling temperature |
| `--task-set` | `core` or `hard` |
| `--episodes` | How many episodes |
| `--concurrency` | Episodes run in parallel |
| `--state-db` | Record every step to `state.db` (required by `--recovery`) |
| `--recovery` | Roll flagged calls back and let the agent carry on |
| `--max-recoveries` | At most this many rollbacks per episode (default 2) |
| `--recovery-classifiers` | JSON list of classifiers that vote on each call (see `README.md`) |
| `--hardening-config` | MCP signing preset: `default`, `all_on`, `all_off`, `request_only`, `response_only`, or a file |
| `--out` | Run folder |

## Look at the results

The whole breakdown in one go: every call sorted by outcome (caught, blocked, false positive, missed),
rollbacks by reason, false positives by task family, and for each rollback or miss what was asked, what
the model called, why it was faulty, why it was (or wasn't) rolled back, and what the agent did next:

```bash
python3 ../recovery/interpret_results.py \
  data/runs/recovery-triton \
  --details some \
  --out data/runs/recovery-triton/recovery_report.md
```

`--details` is `none` (summary only), `some` (first 5 rollbacks and misses, the default) or `all`.

The individual queries below show single pieces. Swap in your run folder.

Rollbacks, and how many decisions each reason produced:

```bash
python3 -c "import json; print(json.load(open('data/runs/recovery-oracle/baseline.json'))['recoveries'])"
python3 -c "import json; print(json.load(open('data/runs/recovery-triton/baseline.json'))['recoveries'])"
```

Each rolled-back call, with its true cause, the reason and who flagged it:

```bash
python3 -c "
import json
for line in open('data/runs/recovery-triton/examples.jsonl'):
    r = json.loads(line)
    rec = r['meta']['recovery']
    if rec.get('rolled_back'):
        print(r['meta']['call_index'], r['label']['cause'], rec['reason'], rec['flagged_by'])
"
```

Calls in the current state (rolled-back calls aren't here; compare with `examples.jsonl` minus the rollbacks):

```bash
sqlite3 data/runs/recovery-triton/state.db \
  "select count(*) from tool_calls"
```

Where the agent was told a call was blocked and carried on:

```bash
sqlite3 data/runs/recovery-triton/state.db \
  "select episode_id, position, substr(content, 1, 90)
   from messages
   where content like '%blocked by the monitor%'
   limit 10"
```
