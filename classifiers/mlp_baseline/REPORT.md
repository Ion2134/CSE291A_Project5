# MLP baseline report

This is a CPU neural-network baseline for **benign versus prompt-injection message classification**.
It is an auxiliary security task. Its scores do not measure banking tool-call correctness,
attack success, recovery actions, or agent-versus-adversary attribution. The separate banking
experiment below is a small-data development check.

## Data and split protocol

The main corpus combines 4,935 Agentic Prompt-Injection 5K messages with 6,200 AgentGuard messages.
The exact processed data and splits are committed under `data/injection/`.

| Partition | Total | Benign | Injection | Agentic 5K | AgentGuard |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 8,893 | 5,756 | 3,137 | 3,923 | 4,970 |
| Validation | 1,064 | 644 | 420 | 467 | 597 |
| Test | 1,178 | 785 | 393 | 545 | 633 |

Existing Agentic 5K splits were retained. AgentGuard messages were assigned by deterministic
similarity groups. Grouping uses NFKC/case/whitespace normalization and character 3-5-gram
TF-IDF cosine similarity at 0.90, with connected components kept together. Any group overlapping
an existing holdout would protect test before validation before training. No exclusions were
needed in this version. There are no shared exact inputs or similarity groups across splits.
This check does not guarantee that every semantically related template is separated.

Grouping features are used only to create partitions. Classifier vocabulary and IDF are fit
only on training text. Labels, source names, attack categories, and author explanations are
metadata and never classifier input. Source revisions, hashes, original cards, and dataset
licenses are preserved in `sources/` and the data manifest.

## Model and training

- Word TF-IDF with unigrams/bigrams and up to 4,096 features.
- MLP: 4,096 -> 128 ReLU -> dropout 0.1 -> 64 ReLU -> 2 logits.
- **532,802 parameters**; trained from scratch on local Mac CPU, 4 threads.
- Class-weighted cross entropy; AdamW, learning rate 0.001, weight decay 0.01, batch size 64.
- Maximum 40 epochs; patience 8; lowest validation cross entropy selects the checkpoint.
- Temperature fitting uses validation only. The threshold is 0.5; test data is not used to fit it.
- Fixed dataset split seed 42; training seeds 42, 43, and 44. Seed 42 remains the primary model.
  No seed was selected by test performance.

All three runs were repeated after relocating the code into this standalone directory.
Their prediction-quality metrics matched the original runs exactly. Console logs and structured
per-epoch histories are committed in `experiments/injection/seedN/`; checkpoints include calibration and vocabulary.

## Main held-out results

| Training seed | Best epoch | Epochs completed | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 42 | 14 | 22 | 0.9253 | 0.9198 |
| 43 | 20 | 28 | 0.9177 | 0.9119 |
| 44 | 7 | 15 | 0.9032 | 0.8973 |

Across seeds: accuracy **0.9154 +/- 0.0112**, macro F1 **0.9097 +/- 0.0114** (sample standard
deviation, three seeds; not a confidence interval).

The primary seed-42 model obtains accuracy **0.9253**, macro F1 **0.9198**, injection precision
**0.8197**, injection recall **0.9949**, and AUROC **0.9916**. Its test confusion matrix has rows
as true labels and columns as predictions, ordered `[benign, injection]`:

| True label | Predicted benign | Predicted injection |
| --- | ---: | ---: |
| Benign | 699 | 86 |
| Injection | 2 | 391 |

The primary model flags **10.96%** of benign messages and misses **0.51%** of injections.
Always predicting benign yields accuracy 0.6664 and macro F1
0.3999 on these same test examples.

### Primary model by source

| Test source | N | Accuracy | Macro F1 | False-alarm rate |
| --- | ---: | ---: | ---: | ---: |
| AgentGuard | 633 | 0.9921 | 0.9866 | 0.58% |
| Agentic 5K | 545 | 0.8477 | 0.8426 | 31.32% |

The aggregate score hides a substantial benign-hard-negative weakness on Agentic 5K.
These differences are retained in the report rather than averaged away.

## External ToolFault diagnostic

ToolFault is evaluation-only; none of its messages enters the vocabulary, training set,
checkpoint selection, or temperature fit. The diagnostic asks only whether the final tool result
contains prompt injection. It does **not** evaluate ToolFault's 13-way fault taxonomy or recovery.
Inputs contain user request, final tool name, and final result. Fault types other than
`prompt_injection` are negatives for injection presence, not assertions of correct execution.

On its official test split (441 cases, 41 injection / 400 other): primary accuracy **0.8073**,
macro F1 **0.6802**, injection F1 **0.4785**, recall **0.9512**, and false-alarm rate **20.75%**.
This weaker precision under a different input distribution is a material limitation.
Official validation (81 cases) is also reported separately, without fitting on it.

## Banking development experiment

The original synthetic run has 85 tool-call rows. Two labels incorrectly marked unchanged
non-balance tool results as adversary failures. Cleanup corrects those labels and removes 27
exact duplicates, leaving 58 inputs. Normalizing random payment/transaction IDs removes six
more equivalent cases. Request-group splitting gives 31 train / 10 validation / 11 test cases.

The separate three-class MLP predicts `none`, `agent`, or `adversary`, with correctness probability
from `P(cause=none)`. Its development test accuracy is **0.8182** and cause macro F1 **0.5185**.
The test has only one agent-error example and two adversary-error examples. These scores are
not reliable evidence of final banking-monitor quality. More independent banking examples are
still required; binary injection data must not be relabeled into missing causal ground truth.

## Artifacts and verification

All under `experiments/injection/`:

- `seed42/model`, `seed43/model`, `seed44/model`: complete, reloadable MLP bundles.
- `seedN/predictions/`: per-example main-test and ToolFault predictions, including false alarms.
- `seedN/train.log`, `seedN/history.csv`: console output and per-epoch losses from standalone reproduction.
- `seedN/results.json`, `summary.json`, `metrics.csv`: per-seed results, the cross-seed summary, and a compact CSV.

The banking development metrics are in `archive/banking-dev/report/`. `RELEASE.json` holds
checksums for code, data, models, logs, reports, and source notices.

At release, the standalone package passed **18 focused tests** and the unchanged simulator **44 tests**.
Release verification also reloads all three exported injection checkpoints and checks every
saved test probability. The package has no runtime dependency on `agent_env` or Laya.

## Limits

Both main corpora are synthetic; high scores on these distributions do not establish deployment
safety. Source/template effects and false positives remain substantial. Only three training
seeds were used. Hardware timing is a local warm single-request measurement and is not a
cross-hardware comparison. AFTraj was audited but not used because its failure-localization
annotations do not provide equivalent injection labels.
