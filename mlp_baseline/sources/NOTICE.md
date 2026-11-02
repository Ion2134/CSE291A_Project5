# Dataset attribution and transformations

This package redistributes processed examples from the following public datasets. The original
dataset cards are preserved next to this notice. Exact source revisions and file hashes appear
in `data/injection-v1/manifest.json` and `sources/attribution.json`.

| Dataset and attribution | Revision | Declared data license | Use |
| --- | --- | --- | --- |
| [Agentic Prompt-Injection 5K — Enes Deniz](https://huggingface.co/datasets/3nesdeniz/agentic-prompt-injection-5k) | `e658d94dcafc133e3a4483f9f9aa13507f3848c4` | [CC BY 4.0](licenses/CC-BY-4.0.txt) | Training, validation, and test |
| [AgentGuard Inter-Agent Benchmark — AgentGuard / Nizba](https://huggingface.co/datasets/Nizba/agentguard-benchmark-v1) | `4951a8a0a04591e25b71d61fedd3fef1fd69c3a4` | [Apache 2.0](licenses/Apache-2.0.txt) | Training, validation, and test |
| [ToolFault-Bench — Atharva Vichare](https://huggingface.co/datasets/ATTY57/toolfault-bench) | `57554086b6a5122471e95c7cc3872f61b566c91f` | [CC BY 4.0](licenses/CC-BY-4.0.txt) | Separate diagnostic evaluation only |

The following modifications were made to prepare the redistributed JSONL files:

- Message text was mapped to `state`; a fixed injection-choice rubric was added under `questions`.
- Source labels 0/1 were mapped to `benign`/`injection` under `expected`.
- Source identifiers and attack-family annotations were retained as metadata, outside model inputs.
- Near-message groups were computed with normalized character n-gram similarity at cosine 0.90.
  Existing Agentic 5K splits take priority over new group assignments; overlaps would be excluded.
- AgentGuard groups were assigned deterministic train/validation/test partitions using seed 42.
- ToolFault diagnostic inputs were serialized from the user request, tool name, and last result.
  Only its `prompt_injection` cases are positive for injection presence. Other faults remain negative
  for this binary question, not assertions that execution was correct.

The archived cards contain the upstream descriptions and citations. The source names and marks
identify those works; the package does not imply endorsement by the dataset authors.

The banking data is a copy of this repository's synthetic `agent_env/data/runs/triton42-v2` run
at commit `4c257a3ce3a6f92a7bc5940d30f4d14815779b27`. Its prepared copy repairs two mislabeled
no-op attack examples and deduplicates inputs. Original records and labels are preserved under
`data/banking/triton42-v2/raw`; the correction trail is in `prepared/data_report.json` and
`prepared/provenance.jsonl`.
