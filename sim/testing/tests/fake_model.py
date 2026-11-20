"""Stand-in for a model's training command: writes test predictions in that model's format.

It puts probability 0.8 on the true cause, so scores are predictable.
"""
import argparse
import json
from pathlib import Path

LABELS = ("none", "agent", "adversary")

parser = argparse.ArgumentParser()
parser.add_argument("--format", choices=["mlp", "laya"], required=True)
parser.add_argument("--test", required=True)
parser.add_argument("--out")
parser.add_argument("--output")
args, _ = parser.parse_known_args()

out = Path(args.out or args.output)
out.mkdir(parents=True)
lines = []
for index, line in enumerate(Path(args.test).read_text().splitlines()):
    row = json.loads(line)
    cause = {label: (0.8 if label == row["expected"]["cause"] else 0.1) for label in LABELS}
    if args.format == "mlp":
        lines.append({"input_id": row["meta"]["input_id"], "expected": row["expected"],
                      "cause_probabilities": cause, "correct_probability": cause["none"]})
    else:
        lines.append({"row": index, "meta": row["meta"], "expected": row["expected"],
                      "answers": {"cause": {"type": "choice", "probabilities": cause},
                                  "output_correct": {"type": "noul", "noul": cause["none"]}}})
(out / "test_predictions.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
if args.format == "laya":
    (out / "model").mkdir()  # Stands in for a checkpoint the tester should delete.
