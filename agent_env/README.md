# discovery-sim

A simulated banking stack for generating labeled training data for **Laya**, a monitor model that watches an agent's tool calls and flags silent failures.

## What this does

There are two models with different jobs:

- **The banking agent** (an LM served by LM Studio) receives a request such as "Pay payee_synth_0007 $50 from acct_synth_0031" and completes it by calling bank and payment tools. This is the system under test.
- **Laya** (the model you fine-tune on this data) does not do banking. It reads one tool call at a time and answers two questions: did this step produce the correct result, and if not, was the banking agent at fault or did something tamper with the call?

Each episode runs one banking task against a fresh synthetic ledger. Most episodes also inject one **attack** on one tool call, at one of three hops:

| Hop | What the attack does | Example operators |
|-----|----------------------|-------------------|
| `llm` | Rewrites the model's own tool arguments, simulating an agent mistake | `llm_wrong_account`, `llm_wrong_amount` |
| `a2a` | Intercepts the agent-to-agent message between the model and the bank or payment agent | `a2a_call_intercepted`, `a2a_toolset_mismatch` |
| `mcp` | Hacks the tool server so it returns wrong or injected output | `mcp_io_mismatch`, `mcp_output_injection` |

`benign` episodes have no attack. `hard_negative_reencode` is a hard negative: it is tagged as an attack on the A2A hop but leaves the call's meaning unchanged, so the correct label is still "output is correct."

Every tool call is labeled by comparing three things against the trusted mandate and the ledger: what the model asked for, what the tool actually received, and what the tool returned. Labels come from the simulator, not from a model.

The bank, payment, and chatbot agents speak A2A 1.0 over JSON-RPC, and bank and payment tools are served over MCP. `discovery-sim run` executes them in-process, so no HTTP servers are needed to generate data.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Generate data

**Agent mode (default):** the LM runs a real tool-calling loop (model, then tool, then tool result back to the model) until it stops calling tools or hits `--max-agent-steps`. Requires LM Studio running with a tool-capable model.

```bash
export LMSTUDIO_MODEL=your-model
discovery-sim run --episodes 50 --seed 42 --concurrency 4 --out data/runs/run_1
```

**Scripted mode:** fixed tool-call templates with injected attacks and no LM. Fast and deterministic, useful for smoke tests.

```bash
discovery-sim run --mode scripted --episodes 100 --seed 42 --out data/runs/scripted42
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--episodes N` | 50 | Number of episodes to run |
| `--seed S` | 0 | Random seed for task and attack selection |
| `--out DIR` | `data/runs/0` | Output directory |
| `--mode agent\|scripted` | `agent` | LM tool loop or fixed templates |
| `--concurrency N` | 1 | Run up to N episodes in parallel. With N > 1, each episode uses a sub-seed derived from `(seed, episode_index)`, so results are reproducible but differ from a sequential run with the same seed |
| `--max-agent-steps N` | 8 | Maximum model turns per episode |
| `--overwrite` | off | Write into `--out` even if it already has a run. Without this flag, an existing run is kept and the new one goes to a sibling directory such as `run_1-20261020T183045-seed42` |
| `--single-step-only` | off | Scripted mode only: skip multi-step scenarios |
| `--llm-samples N` | 0 | Scripted mode only: plan the first N benign episodes with the LM |
| `--llm-base-url`, `--llm-model` | env | Override `LMSTUDIO_BASE_URL` and `LMSTUDIO_MODEL` |

### Tasks

There are 10 banking tasks: balance lookup on two accounts, list accounts, transaction history, account holder lookup, three transfer variants, and two "check balance, then pay" flows. They are defined in `src/discovery_sim/sim/task_defs.py`.

## Generate prompt-gate data

`discovery-sim prompts` writes a labeled dataset of **initial user prompts** — safe and unsafe variants of the ten task templates — for training an input-gate classifier that screens the first user message before the agent runs. This covers *direct* prompt injection at the user hop only; indirect injection through tool outputs is what the episode-level Laya data covers. The episode engine is untouched.

```bash
# Deterministic only, fully reproducible
discovery-sim prompts --count 1000 --seed 42 --out data/runs/prompts42

# Add LM Studio paraphrases to diversify the safe pool
discovery-sim prompts --count 2000 --seed 42 --llm-paraphrases 20 --out data/runs/prompts42_llm
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--count N` | 1000 | Target number of rows, sampled ~50/50 safe/unsafe (deterministic safe pool is a few hundred rows; larger counts need `--llm-paraphrases` to stay balanced) |
| `--seed S` | 0 | Random seed for sampling, splits, and holdout selection |
| `--out DIR` | `data/runs/prompts_0` | Output directory |
| `--llm-paraphrases N` | 0 | Add N LM Studio paraphrases per template to the safe pool. Paraphrases that drop an account/payee id or drift into injection phrasing are rejected |
| `--llm-base-url`, `--llm-model` | env | Override `LMSTUDIO_BASE_URL` and `LMSTUDIO_MODEL` |
| `--overwrite` | off | Write into `--out` even if it has a run; otherwise a timestamped sibling is created |
| `--concurrency N` | 1 | Run up to N paraphrase calls in parallel. Row order is stable, so results match a sequential run modulo LM output variance. A tqdm bar tracks paraphrase progress |

### `prompt_train.jsonl`

One row per prompt:

```json
{"text": "...", "label": "safe", "attack_family": null, "base_task_id": "agent_transfer", "source": "mutation", "split": "train", "seed": 42}
```

- **`label`**: `safe` or `unsafe` — what the gate should answer.
- **`attack_family`**: one of `instruction_override`, `mandate_forgery`, `authority_claim`, `system_probe`, `delimiter_escape`, `encoding_trick` (`null` for safe rows).
- **`source`**: `template` (raw task text), `mutation` (tone/register rewrites), `hard_negative` (benign phrasing with bait words like "ignore" or "admin", labeled `safe`), `llm_paraphrase`, or `attack`.
- **`split`**: two whole templates are held out as `test` (stable per seed), so you can measure whether the classifier generalizes to unseen phrasings rather than memorizing templates; the rest split 80/20 into `train`/`val`.

`prompt_manifest.json` records the seed, counts by label/family/source/split, holdout template ids, and LM settings.

## Output files

Each run directory contains six files. There is one row per **episode** in `episodes.jsonl`, and one row per **tool call** in `examples.jsonl` and `laya_train.jsonl`, so an episode with two tool calls produces two training rows.

### `laya_train.jsonl` (training data)

The file to fine-tune Laya on. Each line is one tool call:

```json
{
  "state": "user_message: ...\ntrusted: {...}\nprior_calls: ...\nmodel_call: {...}\nreceived_args: {...}\noutput: {...}\n",
  "questions": { "output_correct": {...}, "cause": {...} },
  "expected": { "output_correct": false, "cause": "adversary" }
}
```

- **`state`** is the text Laya sees: the user request, the trusted mandate, earlier calls in the episode, a ledger snapshot, what the model called, the arguments the tool actually received, and the tool's output.
- **`questions`** is the same two-question rubric on every row (see `questions.json`).
- **`expected`** holds the answers:
  - `output_correct` (true/false): does this step's output match what the mandate called for?
  - `cause`: `none` if the output is correct; `agent` if the model asked for the wrong thing and the tool obeyed; `adversary` if the model asked correctly but the arguments or output were tampered with.

Rows from attacked episodes are not all negatives. Only the attacked call is usually wrong, and hard negatives are labeled correct on purpose.

### `questions.json`

A standalone copy of the two Laya questions: their instructions, answer types, and the meaning of each label. It has the same content as the `questions` field on every training row. Use it if your training pipeline wants the schema in a separate file. The simulator never reads it; to change the questions, edit `LAYA_QUESTIONS` in `src/discovery_sim/sim/episode.py`.

### `examples.jsonl` (full records for analysis)

The same tool calls as `laya_train.jsonl`, with more detail:

- **`input`**: structured versions of the state, including `conversation` (agent mode), `prior_calls`, `trusted`, `model_call`, `received_args`, `output`, and `ledger_snapshot`.
- **`label`**: `output_correct` and `cause`, plus `verdict` (`allow` or `deny`), `fault` (for example `io_mismatch`), and `severity`.
- **`meta`**: the operator and hop applied to this call, call index, seed, planner model, and whether the call came from a live LM.

Use this file for debugging, filtering, and stratified splits. Do not train on `meta`, since the operator name gives away the answer.

### `episodes.jsonl` (one row per episode)

Episode outcomes used for the baseline:

| Field | Meaning |
|-------|---------|
| `operator`, `fault_hop` | Attack chosen for the episode, and which hop it targets |
| `fault_call_index` | Tool round the attack was scheduled on |
| `attack_applied` | Whether the attack actually ran. False if the model never reached the scheduled round |
| `num_calls` | Number of tool calls the model made |
| `task_completed` | The agent finished its loop without an error |
| `task_correct` | Completed, every tool step was correct, and the ledger matches the mandate |
| `silent_wrong` | Completed but not correct: the agent finished confidently with a wrong result |
| `hard_fail` | Did not complete, because of an LM or tool error |
| `absorbed` | Attacked, but the episode still ended correct |

In agent mode `hard_fail` is almost always false. Attacks return plausible-looking data rather than errors, so the agent keeps going and the failure shows up as `silent_wrong`.

### `baseline.json` (Phase 1 metrics)

Aggregate rates over attacked (non-benign) episodes, with no recovery or monitoring in place. This is the number Laya is meant to improve on.

- `attack_applied_rate`: fraction of attacked episodes where the attack actually ran. It should be close to 1.0.
- `task_correct_rate`, `silent_wrong_rate`, `hard_fail_rate`: outcome rates over attacked episodes.
- `*_applied_only`: the same rates restricted to episodes where the attack ran.
- `by_operator`, `by_hop`: the same rates broken down by attack type and hop.
- Run settings: `mode`, `concurrency`, `max_agent_steps`, and `dropped_llm_mismatch` (scripted mode episodes dropped because the LM's plan didn't match the template).

### `run_manifest.json`

The settings used for the run: seed, episode count, mode, concurrency, max agent steps, task count, and output directory. Use it to reproduce or compare runs.

## Run the agents over HTTP

To run the A2A agents as standalone servers (not needed for data generation):

```bash
discovery-sim agents
```

Default ports: chatbot `8001`, bank `8002`, payment `8003`. Each serves its agent card at `/.well-known/agent-card.json` and JSON-RPC at `/rpc`.

## Tests

```bash
pytest
```
