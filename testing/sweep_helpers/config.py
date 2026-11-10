"""The sweep config, the paths of everything the sweep writes, and small file helpers."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The tester reuses mlp_baseline's merge, grouping and scoring code (never its training code).
sys.path.insert(0, str(REPO / "mlp_baseline" / "src"))

MODELS = ("mlp", "mlp_embed", "stacking", "rules", "rules_plus", "knn", "knn_embed", "logreg_checks", "logreg_memory", "laya")


# ---------------------------------------------------------------- config

@dataclass(frozen=True)
class SplitSettings:
    """How the pooled rows are cut into cross-validation folds."""

    seeds: tuple[int, ...]       # One distinct fold assignment per split seed.
    k: int                       # Folds per split seed.
    max_test_fraction: float     # Largest share of all rows one test fold may hold.
    laya_val_fraction: float     # Share of training groups held out for Laya's validation.


@dataclass(frozen=True)
class ModelSettings:
    """How to call one model's own training command."""

    name: str
    python: str                          # Interpreter to launch with; defaults to this process.
    training_seeds: tuple[int, ...]
    args: tuple[str, ...] = ()           # Extra flags passed straight to the command.
    device: str = "cuda"                 # Laya only.
    keep_checkpoints: bool = False       # Laya only; checkpoints are large.
    raw: dict = field(default_factory=dict, compare=False)  # As written in the config.

    @classmethod
    def from_config(cls, name: str, raw: dict, default_seeds) -> "ModelSettings":
        # Optional: omit `python` (or set null/"") to use whatever is running the sweep.
        python = raw.get("python") or sys.executable
        return cls(name=name, python=python,
                   training_seeds=tuple(raw.get("training_seeds", default_seeds)),
                   args=tuple(raw.get("args", ())), device=raw.get("device", "cuda"),
                   keep_checkpoints=raw.get("keep_checkpoints", False), raw=raw)

    @property
    def is_laya(self) -> bool:
        return self.name == "laya"


def load_config(path: Path) -> dict:
    """Read a sweep config and check that its required keys are present."""
    config = read_json(path)

    missing = [key for key in ("name", "runs_dir", "runs", "split", "training_seeds", "models")
               if key not in config]
    missing += [f"split.{key}" for key in ("seeds", "k", "max_test_fraction", "laya_val_fraction")
                if key not in config.get("split", {})]
    if missing:
        raise ValueError(f"Config is missing: {', '.join(missing)}")

    unknown = set(config["models"]) - set(MODELS)
    if unknown:
        raise ValueError(f"Unknown models in config: {sorted(unknown)}")
    return config


# ---------------------------------------------------------------- what gets written where

@dataclass(frozen=True)
class Unit:
    """One training run: one model on one fold of one split seed, with one training seed."""

    model: str
    split_seed: int
    fold: int
    training_seed: int | None  # None for the majority baseline, which does not train.

    def __str__(self) -> str:
        return f"{self.model} p{self.split_seed} f{self.fold} t{self.training_seed}"

    @property
    def dir_name(self) -> str:
        return f"p{self.split_seed}_f{self.fold}_t{self.training_seed}"

    @classmethod
    def from_dir_name(cls, model: str, name: str) -> "Unit":
        split_seed, fold, seed = (int(part[1:]) for part in name.split("_"))  # e.g. p42_f0_t43
        return cls(model, split_seed, fold, seed)


@dataclass(frozen=True)
class FoldFiles:
    """The three files every model receives for one fold."""

    dir: Path

    @property
    def train(self) -> Path:
        return self.dir / "train.jsonl"

    @property
    def val(self) -> Path:
        return self.dir / "val.jsonl"

    @property
    def test(self) -> Path:
        return self.dir / "test.jsonl"


@dataclass(frozen=True)
class UnitFiles:
    """What the tester keeps for one unit."""

    dir: Path

    @property
    def native(self) -> Path:
        """The model's own output directory."""
        return self.dir / "native"

    @property
    def log(self) -> Path:
        return self.dir / "log.txt"

    @property
    def predictions(self) -> Path:
        return self.dir / "predictions.jsonl"

    @property
    def run_config(self) -> Path:
        return self.dir / "run_config.json"

    @property
    def metrics(self) -> Path:
        """Written last, so its presence marks the unit as done."""
        return self.dir / "metrics.json"

    def is_done(self) -> bool:
        return self.metrics.exists()

    def start_fresh(self) -> None:
        """Remove anything left from an interrupted attempt."""
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True)


@dataclass
class Sweep:
    """One config plus the repository it runs in."""

    config: dict
    repo: Path = REPO

    def __post_init__(self):
        split = self.config["split"]
        self.split = SplitSettings(seeds=tuple(split["seeds"]), k=split["k"],
                                   max_test_fraction=split["max_test_fraction"],
                                   laya_val_fraction=split["laya_val_fraction"])
        self.models = {name: ModelSettings.from_config(name, raw, self.config["training_seeds"])
                       for name, raw in self.config["models"].items()}

    @property
    def name(self) -> str:
        return self.config["name"]

    @property
    def out(self) -> Path:
        return self.repo / "testing" / "results" / self.name

    def resolve(self, path: str) -> Path:
        """Config paths are relative to the repository root."""
        return Path(path) if Path(path).is_absolute() else self.repo / path

    def fold_files(self, split_seed: int, fold: int) -> FoldFiles:
        return FoldFiles(self.out / "data" / f"p{split_seed}" / f"f{fold}")

    def model_dir(self, model: str) -> Path:
        return self.out / "runs" / model

    def unit_files(self, unit: Unit) -> UnitFiles:
        return UnitFiles(self.model_dir(unit.model) / unit.dir_name)

    def data_fingerprint(self) -> str:
        """Covers everything that changes the pooled rows or the folds."""
        return fingerprint({key: self.config[key] for key in ("runs_dir", "runs", "split")})


# ---------------------------------------------------------------- file helpers

def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_json(path: Path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_csv(path: Path, rows: list[dict]) -> None:
    with Path(path).open("w", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
