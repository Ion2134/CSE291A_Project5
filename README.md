# CSE291A Project 5: monitoring a banking agent's tool calls

A banking agent (an LLM) completes requests by calling bank and payment tools, while a simulator
injects attacks into some calls. A **monitor** reads one tool call at a time and answers:

- **output_correct**: did this call produce the result the user's mandate called for?
- **cause**: if not, was it the `agent`'s mistake or an `adversary`'s tampering (`none` if correct)?

We generate labeled data with the simulator, train two kinds of monitors (small MLPs and a
fine-tuned Laya model), and compare them on the same cross-validation splits.

## Layout

| Folder | What it does | Details |
| --- | --- | --- |
| [`agent_env/`](agent_env/) | Simulator: runs the agent on banking tasks with attacks and writes labeled data | [README](agent_env/README.md) |
| [`mlp_baseline/`](mlp_baseline/) | CPU baselines: the injection detector and the `mlp` / `stacking` monitors | [README](mlp_baseline/README.md) |
| [`laya_finetuning/`](laya_finetuning/) | Fine-tunes the Laya monitor (GPU) | [README](laya_finetuning/README.md) |
| [`testing/`](testing/) | Unified tester: trains and scores every monitor over many splits and seeds | [README](testing/README.md) |

Data flows left to right: `agent_env/data/runs/<run>/` → merged by `mlp_baseline` → split by
`testing` → each model's own training command → one shared scorer.

## Setup

Each folder has its own Python 3.12 venv (they pin different packages). From the repository root:

```bash
python3.12 -m venv agent_env/.venv && agent_env/.venv/bin/pip install -e "agent_env[dev]"
python3.12 -m venv mlp_baseline/.venv && (cd mlp_baseline && .venv/bin/pip install -r requirements.txt -r ../testing/requirements.txt)
python3.12 -m venv laya_finetuning/.venv && laya_finetuning/.venv/bin/pip install -r laya_finetuning/requirements.txt
```

On a GPU machine, install the CUDA build of torch before Laya's requirements (see its README).
The tester runs in the `mlp_baseline` venv.

## Run each part

**1. Generate data** (`agent_env/`; the agent needs an LLM, picked with `--llm-backend`)

| Backend | Where the model runs | Settings |
| --- | --- | --- |
| `lmstudio` (default) | Local [LM Studio](https://lmstudio.ai) server; load a model with tool calling and start the server | Shell env vars: `LMSTUDIO_MODEL` (the loaded model's id), `LMSTUDIO_BASE_URL` (default `http://127.0.0.1:1234/v1`) |
| `triton` | Hosted TritonAI API | `agent_env/.env` (gitignored): `cp .env.example .env`, then fill in `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` |
| `oracle` | No model: always makes the correct (on-mandate) tool calls; offline and free | None |

`--llm-model` and `--llm-base-url` override either backend's settings. LM Studio's settings are read
from the shell only, not from `.env`.

```bash
cd agent_env

# LM Studio (local)
export LMSTUDIO_MODEL=qwen/qwen3.6-35b-a3b
.venv/bin/discovery-sim run --llm-backend lmstudio --episodes 50 --seed 42 --out data/runs/my-local-run

# TritonAI (hosted; reads .env)
.venv/bin/discovery-sim run --llm-backend triton --episodes 50 --seed 42 --concurrency 2 --out data/runs/my-triton-run

# A model × temperature grid with a 300-episode budget ("backend" and "models" are set in the config)
.venv/bin/discovery-sim batch --config configs/batch-example.json --dry-run
```

**2. Evaluate the monitors over many splits** (`testing/`; add new run names to the config's `runs`)

```bash
mlp_baseline/.venv/bin/python testing/sweep.py run --config testing/configs/agent-v4.json --model mlp
mlp_baseline/.venv/bin/python testing/sweep.py run --config testing/configs/agent-v4.json --model stacking
mlp_baseline/.venv/bin/python testing/sweep.py run --config testing/configs/agent-v4.json --model laya   # GPU
mlp_baseline/.venv/bin/python testing/sweep.py summarize --config testing/configs/agent-v4.json
```

Results: `testing/results/<config name>/summary/results.csv` (median, mean and std per model and metric).

**3. Train a model directly** (optional; the tester calls these for you)

```bash
cd mlp_baseline && .venv/bin/mlp-baseline train-injection --data data/injection --out runs/seed42
laya_finetuning/.venv/bin/python laya_finetuning/train_laya.py --task banking --device cuda --output laya_finetuning/runs/bank
```

## Tests

```bash
agent_env/.venv/bin/python -m pytest -q agent_env/tests
mlp_baseline/.venv/bin/python -m pytest -q mlp_baseline/tests testing/tests
laya_finetuning/.venv/bin/python -m pytest -q laya_finetuning/tests      # offline, tiny model
```

## Notes

- Data is small: 10 request types and few agent-error rows, so single splits swing a lot. Report
  the tester's medians, not one split.
- Superseded experiments are kept in `mlp_baseline/archive/` and `laya_finetuning/archive/`.
