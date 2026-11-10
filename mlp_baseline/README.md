# MLP baseline

A standalone CPU package with two jobs:

1. **The injection detector**: an MLP that classifies a message as **benign** or **prompt
   injection**, trained on public datasets. Its checkpoint is also the frozen base of the
   `stacking` model.
2. **The banking monitors** (rules, logistic regression, kNN, MLPs and `stacking`): they predict
   whether a banking tool call's output is correct and, if not, the cause (`none` / `agent` /
   `adversary`). These are trained and compared against Laya by the unified tester in
   [`../testing/`](../testing/), which calls this package's `fit-predict` command.

It does not import or edit `agent_env`, and has no Laya dependency.

## Layout

| Path | Contents |
| --- | --- |
| `src/mlp_baseline/` | The package (see [Code](#code) below) |
| `tests/` | Regression tests |
| `data/injection/` | Injection dataset: 8,893 train / 1,064 val / 1,178 test messages, plus ToolFault diagnostics; also Laya's `injection` task data |
| `experiments/injection/` | The trained injection detector: `summary.json` and `metrics.csv` across seeds, and per seed `seedN/{model,predictions,results.json,train.log,history.csv}` |
| `sources/` | Dataset cards, licenses and attribution for the public data |
| `archive/` | Earlier experiments and one-off notes, kept for the record ([`archive/README.md`](archive/README.md)) |
| `REPORT.md` | Method, results and limitations of the injection detector |
| `RELEASE.json` | File hashes checked by `mlp-baseline verify` |

## Install and check

Run from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m pytest -q tests
mlp-baseline verify
```

Use a fully isolated venv, without `--system-site-packages`. Training uses the CPU and four
threads by default. `verify` checks the file hashes in `RELEASE.json`, split separation, and
all 1,178 saved test probabilities for each of the three injection checkpoints.

## Injection detector

The primary checkpoint is seed 42, chosen before comparing test scores:

```bash
mlp-baseline predict --checkpoint experiments/injection/seed42/model \
  --data data/injection/test.jsonl --out runs/predictions.jsonl
```

Each input row needs `state` and `questions`; labels are optional. Load the whole `model/`
directory: inference needs its weights, vocabulary, IDF values, architecture and validation
calibration together.

To retrain (all processed data is committed, so no download is needed):

```bash
mlp-baseline train-injection --data data/injection --out runs/seed42 --seed 42 --epochs 40 --threads 4
```

`scripts/reproduce.sh` runs seeds 42, 43 and 44. The 40 epochs are a maximum: early stopping
uses patience 8, and the lowest validation cross entropy selects the checkpoint. To rebuild the
exact splits from the pinned public sources:

```bash
mlp-baseline prepare-injection --cache .cache/sources \
  --out runs/regenerated-data --download --seed 42 --similarity 0.9
```

Provenance, revisions, licenses and file hashes are in [`sources/NOTICE.md`](sources/NOTICE.md)
and [`data/injection/manifest.json`](data/injection/manifest.json).

## Banking monitor

Results come from the unified tester ([`../testing/README.md`](../testing/README.md)), which
builds cross-validation folds from the agent runs and calls, for each fold:

```bash
mlp-baseline fit-predict --model mlp --train train.jsonl --validation val.jsonl --test test.jsonl \
  --out runs/fold --seed 42 --k 5 --use-checks --view middleware
```

`--validation` rows are added to training, because the inner cross-validation already holds
data out. Correctness is `P(cause = none)`.

### What the monitor sees: the `state` string and the two views

Each row's `state` is a text block: `user_message`, `trusted` (the mandate), `prior_calls`, the
current `tool`, `model_call`, `received_args`, `output`, and `ledger_snapshot`. Two of those lines
are **privileged**: a monitor in the middleware between agent and tool client would not have them.

| `--view` | What the model sees |
| --- | --- |
| `full` (default) | Everything, including `ledger_snapshot` and `received_args`. An oracle: it shows what the privileged fields are worth, not a deployable score. |
| `middleware` | `ledger_snapshot` and `received_args` removed before any model sees the row (`checks.middleware_view`). The text models and the comparison features all get the same stripped input, so TF-IDF cannot read the ledger either. |

Without `received_args` the tool is assumed to have received the call as sent. Without the
ledger, the balance check compares a `get_balance` output with an earlier read of the same account
in the episode. Predictions are always keyed by the original `input_id`, so the tester can join them.
`tests/test_checks.py` fails if the view-level features or rules change when the privileged lines
are replaced with garbage.

### Comparison features (`checks.py`)

A TF-IDF bag of words cannot tell whether two fields are equal, yet the label is almost a
function of exactly that. `comparison_features` parses the state and returns 12 binary flags
(all zeros if the state cannot be parsed):

| Feature | Fires when |
| --- | --- |
| `parse_ok` | The state parsed |
| `call_account_mismatch`, `call_payee_mismatch`, `call_amount_mismatch` | The model's call differs from the trusted mandate (the **agent** deviated) |
| `call_any_mismatch` | Any of the three above |
| `args_differ_from_call` | `received_args` differs from `model_call` (needs the full view; always 0 in the middleware view) |
| `output_differs_from_args` | The output echoes an account, payee or amount that differs from the arguments |
| `output_has_private_key` | The output has an extra key starting with `_` |
| `output_injection_text` | A string in the output matches an injection phrase ("ignore previous", "system prompt", …) |
| `balance_differs_from_ledger` | A first `get_balance` disagrees with the ledger (full view) or with an earlier read of that account (middleware view) |
| `tool_is_transfer`, `tool_is_read` | Which kind of tool ran |

The `mlp` and `stacking` models append these to their text features with `--use-checks`.

### Models

Pass the model name to `fit-predict --model`; the tester uses the same names.

| Model | What it is | Trained on |
| --- | --- | --- |
| `rules` | `checks.rule_cause`, a hand-written cascade: any call mismatch → `agent`; else any integrity flag → `adversary`; else `none`. Fixed 0.97 confidence. | Nothing |
| `logreg_checks` | Multinomial logistic regression on the 12 flags. Learns the cascade from labels and gives real probabilities. Weights are saved in `results.json`. | Training rows |
| `mlp` | Three-class TF-IDF MLP. Inner group cross-validation picks the epoch count and fits temperature calibration on out-of-fold predictions; the final model is refit on all training rows. | Training rows |
| `mlp_embed` | The same MLP on a sentence embedding (`BAAI/bge-small-en-v1.5`, from the local Hugging Face cache) instead of TF-IDF | Training rows |
| `knn`, `knn_embed` | Cosine k-nearest-neighbours (`--neighbors`, default 5) on the MLP's input features; votes are weighted by closeness and inverse class frequency | Training rows |
| `stacking` | The frozen injection detector (`experiments/injection/seed42/model`) supplies P(injection); a logistic regression on that plus TF-IDF predicts the cause. It lets first-party labels teach that a resisted attack is `none`, not `adversary`. | Training rows (+ external data for the frozen base) |
| `rules_plus`, `logreg_memory` | `rules` / `logreg_checks` plus two extra checks, described below | Training rows |

Embeddings are cached on disk by (model, text) hash in `~/.cache/mlp_baseline_embeddings`
(`MLP_EMBED_CACHE` moves it). The model is loaded from the local cache only; nothing is downloaded.

### Read `rules_plus` and `logreg_memory` as upper bounds

They add (1) `balance_unseen`: a first-read balance never seen for that account in training rows
labelled `none`, and (2) `empty_transfer_output`: a finished transfer whose output is `{}`. The
balance table is built from the training fold only (the memory columns for training rows leave out
their own request group), so there is no test leakage. But:

- Both checks were found by inspecting the folds' errors, so their scores are partly in-sample.
- The balance check works because every simulated episode starts from the same seeded ledger. In a
  real deployment balances change after each transfer, and it would fail silently.
- The empty-transfer check rests on 3 distinct rows.

Headline the generic `rules` and `logreg_checks`; confirm the extra checks on a freshly generated pool
before relying on them.

Two commands build datasets from agent runs (`agent_env/data/runs/<name>/`), taking the run
list from a tester config:

```bash
mlp-baseline prepare-agent-data --runs ../agent_env/data/runs \
  --config ../testing/configs/agent-v4.json --out runs/combined --seed 42
mlp-baseline export-laya-banking --data runs/combined --out runs/laya-banking
```

`prepare-agent-data` checks each run's `laya_train.jsonl` against its `examples.jsonl`, repairs
a known legacy labeling bug (an IO-mismatch attack that could not change a correct result was
labeled an adversary error), merges and deduplicates rows across runs, and writes a
group-disjoint train/test split. `export-laya-banking` writes the same split in Laya's
train/val/test format.

## Code

| Module | What it does |
| --- | --- |
| `common.py` | Cause labels, row identity (`input_id`) and request grouping, file helpers, `RunRecord` |
| `mlp.py` | The MLP (TF-IDF or embedding input) and `TrainSettings`: train, save, load |
| `checks.py` | `middleware_view`, the 12 comparison features, and the `rules` cascade |
| `embeddings.py` | Cached sentence embeddings of a state (`mlp_embed`, `knn_embed`) |
| `variants.py` | `rules_plus`, `logreg_checks` / `logreg_memory` and the kNN |
| `metrics.py` | Banking-monitor metrics, temperature calibration, majority baseline |
| `splits.py` | Group-disjoint stratified partitions and cross-validation folds |
| `agent_runs.py` | Read agent runs, repair the legacy label bug, merge them into unique rows |
| `monitor_datasets.py` | Merged runs → train/test dataset (`prepare-agent-data`), checks, Laya export |
| `monitor_training.py` | Dispatches every monitor and runs `fit-predict` (one tester fold) |
| `stacking.py` | The stacked monitor on the frozen injection detector |
| `comparison.py` | The archived single-split comparison report (`run-agent-comparison`) |
| `injection_data.py` | Builds the injection dataset from the pinned public sources |
| `injection_model.py` | Loads, runs and scores the injection detector |
| `injection_training.py` | Trains and reports the injection detector (`train-injection`) |
| `banking_dev.py` | The archived banking-dev experiment (`prepare-banking`, `train-banking`) |
| `verify.py`, `cli.py` | `mlp-baseline verify` and the command line |

See [REPORT.md](REPORT.md) for the injection detector's results; false alarms remain material.
