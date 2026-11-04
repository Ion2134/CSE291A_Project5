"""Constants, the files one run writes, and small helpers shared by the Laya scripts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

LAYA_DIR = Path(__file__).resolve().parents[1]   # laya_finetuning/
REPO = LAYA_DIR.parent

BASE_MODEL = "convaiinnovations/laya"
BASE_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"

# Answer labels per question id, in the order Laya's probabilities are read.
LABELS = {"injection": ["benign", "injection"], "cause": ["none", "agent", "adversary"],
          "output_correct": [False, True]}
TASK_QUESTIONS = {"injection": {"injection"}, "banking": {"cause", "output_correct"}}

SPLITS = ("train", "val", "test")
DEFAULT_DATA = {"injection": REPO / "mlp_baseline/data/injection", "banking": LAYA_DIR / "data/agent-combined-v4"}

# Package versions recorded in every run manifest.
RECORDED_PACKAGES = ("laya", "torch", "transformers", "numpy", "scikit-learn", "safetensors", "huggingface-hub")


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass(frozen=True)
class RunFiles:
    """Everything one training run writes under ``out``."""

    out: Path

    @property
    def training_log(self) -> Path:
        return self.out / "training.log"

    @property
    def manifest(self) -> Path:
        return self.out / "run_manifest.json"

    @property
    def history_json(self) -> Path:
        return self.out / "history.json"

    @property
    def history_csv(self) -> Path:
        return self.out / "history.csv"

    @property
    def model(self) -> Path:
        """The best epoch's checkpoint, calibrated after training."""
        return self.out / "model"

    @property
    def last_model(self) -> Path:
        """The final epoch's weights, without calibration."""
        return self.out / "last_model"

    @property
    def calibration(self) -> Path:
        return self.out / "calibration.json"

    @property
    def test_predictions(self) -> Path:
        return self.out / "test_predictions.jsonl"

    @property
    def results(self) -> Path:
        return self.out / "results.json"

    @property
    def checksums(self) -> Path:
        return self.out / "checkpoint_checksums.json"


class Tee:
    """A stream that writes to two streams, used to copy stdout/stderr into ``training.log``."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.stream.flush()
        self.log.flush()
