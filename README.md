# CSE291A Project 5: monitoring a banking agent's tool calls

A banking agent (an LLM) completes requests by calling bank and payment tools, while a simulator
injects attacks into some calls. A **monitor** reads one tool call at a time and answers:

- **output_correct**: did this call produce the result the user's mandate called for?
- **cause**: if not, was it the `agent`'s mistake or an `adversary`'s tampering (`none` if correct)?

We generate labeled data with the simulator, train two kinds of monitors (small MLPs and a
fine-tuned Laya model), and compare them on the same cross-validation splits.

## Layout

The repo is grouped into three folders: the simulator and tester (`sim/`), the monitors (`classifiers/`),
and everything for undoing bad calls while an agent runs (`recovery/`). Each subfolder is its own
project with its own README (and venv where it has one); `CSE291A_Project5.code-workspace` opens them
all in VS Code with the right interpreter per folder.

| Folder | What it does | Details |
| --- | --- | --- |
| [`sim/agent_env/`](sim/agent_env/) | Simulator: runs the agent on banking tasks with attacks and writes labeled data | [README](sim/agent_env/README.md) |
| [`sim/testing/`](sim/testing/) | Unified tester: trains and scores every monitor over many splits and seeds | [README](sim/testing/README.md) |
| [`classifiers/mlp_baseline/`](classifiers/mlp_baseline/) | CPU baselines: the injection detector, the `mlp` / `stacking` / `rules` monitors, shared checks and metrics | [README](classifiers/mlp_baseline/README.md) |
| [`classifiers/logreg_baseline/`](classifiers/logreg_baseline/) | Logistic regression on the comparison features | [README](classifiers/logreg_baseline/README.md) |
| [`classifiers/knn_baseline/`](classifiers/knn_baseline/) | k-nearest-neighbours monitor | [README](classifiers/knn_baseline/README.md) |
| [`classifiers/laya_finetuning/`](classifiers/laya_finetuning/) | Fine-tunes the Laya monitor (GPU) | [README](classifiers/laya_finetuning/README.md) |
| [`recovery/`](recovery/) | Recovery classifiers, rollback rules, rollback, and a results interpreter | [README](recovery/README.md) |
| [`recovery/state_db/`](recovery/state_db/) | SQLite record of every step's internal and external state | [README](recovery/state_db/README.md) |
| [`recovery/mcp_hardening/`](recovery/mcp_hardening/) | Per-call request/response signing and turn-level telemetry | [README](recovery/mcp_hardening/README.md) |

Data flows left to right: `sim/agent_env/data/runs/<run>/` → merged by `mlp_baseline` → split by
`testing` → each model's own training command → one shared scorer.

## Setup

Each folder has its own Python 3.12 venv (they pin different packages). From the repository root:

```bash
python3.12 -m venv sim/agent_env/.venv && sim/agent_env/.venv/bin/pip install -e "sim/agent_env[dev]"
python3.12 -m venv classifiers/mlp_baseline/.venv && (cd classifiers/mlp_baseline && .venv/bin/pip install -r requirements.txt -r ../../sim/testing/requirements.txt)
python3.12 -m venv classifiers/laya_finetuning/.venv && classifiers/laya_finetuning/.venv/bin/pip install -r classifiers/laya_finetuning/requirements.txt
```

On a GPU machine, install the CUDA build of torch before Laya's requirements (see its README).
The tester runs in the `mlp_baseline` venv.

## Run each part

**1. Generate data** (`sim/agent_env/`; the agent needs an LLM, picked with `--llm-backend`)

| Backend | Where the model runs | Settings |
| --- | --- | --- |
| `lmstudio` (default) | Local [LM Studio](https://lmstudio.ai) server; load a model with tool calling and start the server | Shell env vars: `LMSTUDIO_MODEL` (the loaded model's id), `LMSTUDIO_BASE_URL` (default `http://127.0.0.1:1234/v1`) |
| `triton` | Hosted TritonAI API | `sim/agent_env/.env` (gitignored): `cp .env.example .env`, then fill in `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` |
| `oracle` | No model: always makes the correct (on-mandate) tool calls; offline and free | None |

`--llm-model` and `--llm-base-url` override either backend's settings. LM Studio's settings are read
from the shell only, not from `.env`.

```bash
cd sim/agent_env

# LM Studio (local)
export LMSTUDIO_MODEL=qwen/qwen3.6-35b-a3b
.venv/bin/discovery-sim run --llm-backend lmstudio --episodes 50 --seed 42 --out data/runs/my-local-run

# TritonAI (hosted; reads .env)
.venv/bin/discovery-sim run --llm-backend triton --episodes 50 --seed 42 --concurrency 2 --out data/runs/my-triton-run

# A model × temperature grid with a 300-episode budget ("backend" and "models" are set in the config)
.venv/bin/discovery-sim batch --config configs/batch-example.json --dry-run
```

**2. Evaluate the monitors over many splits** (`sim/testing/`). `agent-v5.json` is the latest config (12 agent runs); to add runs, copy it under a new `name` and extend its `runs` list.

```bash
classifiers/mlp_baseline/.venv/bin/python sim/testing/sweep.py run --config sim/testing/configs/agent-v5.json --model mlp
classifiers/mlp_baseline/.venv/bin/python sim/testing/sweep.py run --config sim/testing/configs/agent-v5.json --model stacking
classifiers/mlp_baseline/.venv/bin/python sim/testing/sweep.py run --config sim/testing/configs/agent-v5.json --model laya   # GPU
classifiers/mlp_baseline/.venv/bin/python sim/testing/sweep.py summarize --config sim/testing/configs/agent-v5.json
```

Results: `sim/testing/results/<config name>/summary/results.csv` (median, mean and std per model and metric).

**3. Train a model directly** (optional; the tester calls these for you)

```bash
cd classifiers/mlp_baseline && .venv/bin/mlp-baseline train-injection --data data/injection --out runs/seed42
classifiers/laya_finetuning/.venv/bin/python classifiers/laya_finetuning/train_laya.py --task banking --device cuda --output classifiers/laya_finetuning/runs/bank
```

## Tests

```bash
sim/agent_env/.venv/bin/python -m pytest -q sim/agent_env/tests
classifiers/mlp_baseline/.venv/bin/python -m pytest -q classifiers/mlp_baseline/tests sim/testing/tests
classifiers/laya_finetuning/.venv/bin/python -m pytest -q classifiers/laya_finetuning/tests      # offline, tiny model
```

## Notes

- Data is small: 10 request types and few agent-error rows, so single splits swing a lot. Report
  the tester's medians, not one split.
- Superseded experiments are kept in `classifiers/mlp_baseline/archive/` and `classifiers/laya_finetuning/archive/`.

## Data and licenses

The banking data is generated by our simulator (`sim/agent_env/data/runs/`). The injection detector is
trained on public datasets: Agentic Prompt-Injection 5K (CC BY 4.0) and the AgentGuard benchmark
(Apache 2.0), with ToolFault-Bench (CC BY 4.0) used only for evaluation. Attribution, pinned
revisions and license texts are in [`classifiers/mlp_baseline/sources/`](classifiers/mlp_baseline/sources/NOTICE.md).
