# Laya fine-tuning handoff

Clone this branch, create a venv, run the dry run, then start training on a CUDA GPU. All new Laya code lives here. The branch also contains `mlp_baseline/`, including the shared public dataset and the completed MLP results.

This uses the **English `convaiinnovations/laya` checkpoint**, approximately **421M parameters**, with a ModernBERT-large encoder and Laya's decision head. It is the locally available Jev equivalent discussed in Discord. The base weights are pinned to `7b928d828b7b0e022f929d9bd2e44165aa270148`. Training uses the native `laya==0.4.21` API. [Model card](https://huggingface.co/convaiinnovations/laya), [native fine-tuning documentation](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune.md).

## Setup on the GPU machine

Python **3.12** is the tested version. Commands below assume Linux and one NVIDIA GPU. Multiple available GPUs are fine; this script uses one selected device.

```bash
git clone --branch laya-finetuning-handoff https://github.com/Ion2134/CSE291A_Project5.git
cd CSE291A_Project5
python3.12 -m venv laya_finetuning/.venv
source laya_finetuning/.venv/bin/activate
python -m pip install --upgrade pip

# CUDA 12.8 example: choose the build supported by your NVIDIA driver.
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r laya_finetuning/requirements.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

The official [PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/#v2100) also lists CUDA 12.6 and 13.0 builds for this version. `requirements.txt` keeps an already-installed matching CUDA build. Model download happens automatically from Hugging Face; the data is already committed. Set `HF_HOME` if you want to choose the download cache location.

## Choose the experiment

| Task | Default data | Train / validation / test | What it predicts |
| --- | --- | --- | --- |
| `injection` | `mlp_baseline/data/injection-v1/` | 8,893 / 1,064 / 1,178 messages | `benign` or `injection` |
| `banking` | `laya_finetuning/data/agent-combined-v4/` | 108 / 28 / 33 cases | `output_correct` and `cause`: none, agent, adversary |

**Use `injection` for comparison with the public-data MLP.** It combines Agentic Prompt-Injection 5K and AgentGuard with the same held-out splits and similarity groups as the MLP. The labels describe attempted injection, not whether an attack succeeded or whether a tool executed correctly. Source licenses and attribution are under [`../mlp_baseline/sources/`](../mlp_baseline/sources/); preparation details are in the [MLP report](../mlp_baseline/REPORT.md).

**Banking uses the same allowlisted agent-run merge as the MLP.** Edit `COMBINED_RUNS` in `prepare_banking_data.py` (keep it in sync with `mlp_baseline/.../agent_data.py`), then:

```bash
python laya_finetuning/prepare_banking_data.py \
  --runs agent_env/data/runs \
  --out laya_finetuning/data/agent-combined-v4
```

That merges/repairs/dedups the named runs, freezes the same test split the MLP uses, and carves val from the train pool. `train_laya.py --task banking` defaults to that directory. Legacy `data/banking-dev/` (single-run triton42-v2) remains as a schema example. Agent-error support is still thin; treat banking results as development checks. The script never combines banking and injection.

For multi-split comparisons with the MLP, use [`../testing/`](../testing/): it builds the folds, calls `train_laya.py --task banking --train … --validation … --test …` for each one, and scores Laya and the MLP the same way. `prepare_banking_data.py --config ../testing/configs/agent-v4.json` reads the run list from that config.

## Validate first; this does not train

```bash
python laya_finetuning/train_laya.py --task injection --dry-run
python -m pytest -q laya_finetuning/tests
```

Dry run downloads only configuration/tokenizer files if missing. It validates all split labels, question schemas, input duplicates, and cross-split group overlap, tokenizes every row, rejects state truncation or skipped questions, and prints counts, hashes, versions, and planned optimizer updates. It neither loads model weights nor creates a run directory. GPU access is unnecessary for dry run or the offline tests.

For custom data, supply **all three** files:

```bash
python laya_finetuning/train_laya.py --task banking --dry-run \
  --train /path/train.jsonl --validation /path/val.jsonl --test /path/test.jsonl
```

Rows use the existing `{state, questions, expected}` Laya JSONL format. Injection rows also need `group_id` from similarity grouping; banking states need their `user_message:` line for request grouping. Keep the held-out files fixed between models. Each training class must be present. Use the committed files as schema examples.

## Start a run

```bash
CUDA_VISIBLE_DEVICES=0 python laya_finetuning/train_laya.py \
  --task injection --device cuda --seed 42 \
  --output laya_finetuning/runs/injection-seed42
```

For the banking development run, change `--task banking` and give a different output directory. On Apple Silicon use `--device mps` (FP16 AMP stays off; Metal mixed precision is unreliable). Custom file flags work the same way during training. Run `--help` for every option. A different local native Laya checkpoint can be passed through `--model /path/to/checkpoint`; another Hub model also requires `--revision COMMIT_SHA`.

Default recipe:

- Full encoder and decision-head fine-tuning, supervised soft cross-entropy (`--loss soft-ce`). Native RLCD is available with `--loss rlcd` for a separate experiment.
- Four epochs; micro-batch 2; accumulation 8; nominal effective batch 16 **question items**. Banking contributes two items per case. The last accumulation window may be smaller.
- AdamW; encoder learning rate `2.5e-5`, head `1e-4`, weight decay `0.01`; cosine schedule to `1e-6`; gradient clipping at 1.
- Native CUDA FP16 mixed precision and gradient checkpointing. The maximum sequence budget is 1,024, including a head budget of 256; actual short messages use shorter sequences. These budgets are saved into the checkpoint.
- Best epoch selected by validation cross-entropy. Validation then fits temperature calibration; types with fewer than 20 examples or fewer than two target classes keep temperature 1. The banking validation set is too small to calibrate either type.
- The selected checkpoint is reloaded through `laya.load`, and the test split is evaluated once after selection/calibration. Native saved calibration is applied at inference. Inherited base calibration is cleared before training.

If memory is tight, try `--batch-size 1 --grad-accum 16 --eval-batch-size 1`. `--freeze-encoder` trains just the decision head and should be reported as a different experiment. Hardware memory requirements and CUDA execution have not been measured here. Seeds are recorded; exact numerical reproducibility across different GPUs is not promised.

## Outputs and sharing

Each run writes:

```text
model/                     best selected native checkpoint, with validation calibration
last_model/                final epoch weights, without calibration
run_manifest.json          arguments, parameter count, dataset hashes, versions, device, selection policy
training.log               captured trainer stdout/stderr
history.json, history.csv  per-epoch training loss and validation cross-entropy
calibration.json           validation-only fitted temperatures and skipped types
results.json               held-out metrics, best epoch, elapsed time, source breakdown
test_predictions.jsonl     labels and native answers/probabilities in test-file order
checkpoint_checksums.json  SHA-256 of exported best-checkpoint files
```

Metrics include accuracy, fixed-label macro/per-class F1, confusion matrices, class support and log loss; binary questions also get AUROC, Brier score, and false-positive/negative rates. For `output_correct`, the positive class is **correct output**. Native API probabilities are rounded to four decimal places; metrics normalize these returned probabilities.

Keep `results.json`, `run_manifest.json`, histories, calibration, and logs with the paper results. Run directories are gitignored so GPU checkpoints are not accidentally committed. Use shared storage for the native checkpoint directory. An interrupted run may leave completed epoch checkpoints and history; choose a new output directory and use `--model old-run/model` to start from saved weights. Optimizer/scaler/scheduler state is not saved, so this starts a new run rather than resuming an identical optimizer trajectory.

For inference from the repository root:

```python
import json
import laya

agent = laya.load("laya_finetuning/runs/injection-seed42/model",
                  device="cuda", fast=False, compile=False)
with open("mlp_baseline/data/injection-v1/test.jsonl") as handle:
    row = json.loads(next(handle))
print(agent.predict(row["state"], row["questions"])["answers"])
```

## What was checked before handoff

The isolated project venv passed **10 tests**, including real native training on a tiny random local encoder, encoder/head weight updates, frozen-encoder behavior, checkpoint export/reload, metrics, rejected duplicate/group leakage, invalid labels/schema, rejected truncation, calibration eligibility, and a dry run that cannot load weights or train. Real datasets also passed dry runs with the pinned pretrained tokenizer: no truncated states and no skipped questions. Machine-readable evidence is in [`verification/`](verification/).

**No pretrained Laya fine-tuning or GPU training was run for this handoff.** Tiny-model tests establish pipeline wiring, not Laya quality or CUDA behavior. The GPU owner still needs to execute the real run and record its results.
