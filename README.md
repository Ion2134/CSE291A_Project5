# CSE291A Project 5: monitoring a banking agent's tool calls

A banking agent (an LLM) completes requests by calling bank and payment tools, while a simulator
injects attacks into some calls. A **monitor** reads one tool call at a time and answers:

- **output_correct**: did this call produce the result the user's mandate called for?
- **cause**: if not, was it the `agent`'s mistake or an `adversary`'s tampering (`none` if correct)?

We generate labeled data with the simulator, train and compare monitors on the same
cross-validation splits, and use them in the loop to roll bad calls back while the agent runs.

## Layout

Three groups: the simulator and tester (`sim/`), the monitors (`classifiers/`), and everything for
undoing bad calls while an agent runs (`recovery/`). Each subfolder is its own project with its own
README; `CSE291A_Project5.code-workspace` opens them all in VS Code with the right interpreter per folder.

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

Data flows: `sim/agent_env/data/runs/<run>/` → merged by `mlp_baseline` → split by `sim/testing` →
each model's own training command → one shared scorer.

## Setup

Use an **arm64** Python 3.12 (torch has no Intel-Mac builds past 2.2). From the repository root:

```bash
python3.12 -m venv classifiers/mlp_baseline/.venv     # also runs logreg, knn and the tester
classifiers/mlp_baseline/.venv/bin/pip install -e "classifiers/mlp_baseline[dev]" -r sim/testing/requirements.txt

python3.12 -m venv sim/agent_env/.venv
sim/agent_env/.venv/bin/pip install -e recovery/mcp_hardening -e recovery/state_db -e classifiers/mlp_baseline \
  -e recovery -e "sim/agent_env[dev]"

python3.12 -m venv recovery/.venv
recovery/.venv/bin/pip install -e classifiers/mlp_baseline -e recovery/state_db -e "recovery[dev]"

python3.12 -m venv recovery/state_db/.venv && recovery/state_db/.venv/bin/pip install -e "recovery/state_db[dev]"
```

Laya has its own venv and a GPU setup: see [its README](classifiers/laya_finetuning/README.md).
For the hosted LLM, copy `sim/agent_env/.env.example` to `.env` ([backends](sim/agent_env/README.md#model-backends)).

## Run each part

```bash
# 1. Generate data (agent + attacks); flags and backends: sim/agent_env/README.md
cd sim/agent_env && .venv/bin/discovery-sim run --llm-backend triton --task-set hard --episodes 50 --out data/runs/my-run

# 2. Score the monitors over many splits (from the repo root); configs: sim/testing/configs/
classifiers/mlp_baseline/.venv/bin/python sim/testing/sweep.py run --config sim/testing/configs/agent-v8-logreg.json --model all
classifiers/mlp_baseline/.venv/bin/python sim/testing/sweep.py summarize --config sim/testing/configs/agent-v8-logreg.json
```

**3. Recovery in the loop** (classifiers vote on each call, flagged calls are rolled back): the full
commands and how to read the results are in [`recovery/COMMANDS.md`](recovery/COMMANDS.md).

## Tests

```bash
M=classifiers/mlp_baseline/.venv/bin/python
(cd sim/agent_env && .venv/bin/python -m pytest -q)
$M -m pytest -q classifiers/mlp_baseline/tests classifiers/logreg_baseline/tests classifiers/knn_baseline/tests sim/testing/tests
(cd recovery && .venv/bin/python -m pytest -q)
(cd recovery/state_db && .venv/bin/python -m pytest -q)
(cd recovery/mcp_hardening && ../../sim/agent_env/.venv/bin/python -m pytest -q)
classifiers/laya_finetuning/.venv/bin/python -m pytest -q classifiers/laya_finetuning/tests   # offline, tiny model
```

## Notes

- The data is small (few request groups, few agent-error rows), so single splits swing a lot. Report
  the tester's medians, not one split.
- Superseded experiments are kept in `classifiers/mlp_baseline/archive/` and `classifiers/laya_finetuning/archive/`.

## Data and licenses

The banking data is generated by our simulator (`sim/agent_env/data/runs/`). The injection detector is
trained on public datasets: Agentic Prompt-Injection 5K (CC BY 4.0) and the AgentGuard benchmark
(Apache 2.0), with ToolFault-Bench (CC BY 4.0) used only for evaluation. Attribution, pinned
revisions and license texts are in [`classifiers/mlp_baseline/sources/`](classifiers/mlp_baseline/sources/NOTICE.md).
