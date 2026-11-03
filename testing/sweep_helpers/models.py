"""Run each model's own training command on the fold files and collect its test predictions.

The tester never trains anything itself:

- ``mlp`` / ``stacking`` run ``mlp-baseline fit-predict`` in the mlp_baseline environment
- ``laya`` runs ``laya_finetuning/train_laya.py --task banking`` in the Laya environment

Rerunning the sweep skips every unit that already finished.
"""
from __future__ import annotations

import shlex
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

from tqdm import tqdm

from .config import ModelSettings, Sweep, Unit, UnitFiles, read_json, read_jsonl, write_json
from .summary import Predictions, save_predictions, score

# Both training commands write their test predictions under this name.
NATIVE_PREDICTIONS = "test_predictions.jsonl"
# Checkpoint folders train_laya.py saves; removed after scoring unless keep_checkpoints is set.
LAYA_CHECKPOINTS = ("model", "last_model")
# run_unit's status for a unit that finished in an earlier sweep.
SKIPPED = "skipped (done)"


def run_models(sweep: Sweep, model_names, manifest: dict, *, split_seeds=None, folds=None,
               training_seeds=None, dry_run=False) -> None:
    """Run every planned unit of each model with a progress bar (dry runs print the commands)."""
    for name in model_names:
        if name not in sweep.models:
            raise ValueError(f"No settings for '{name}' in the config")
        settings = sweep.models[name]
        units = list(planned_units(settings, manifest, split_seeds, folds, training_seeds))

        if dry_run:
            for unit in units:
                print(f"{unit}: {run_unit(sweep, unit, dry_run=True)}")
            continue

        check_model_settings(sweep, settings)
        statuses = []
        with tqdm(units, desc=name, unit="run") as progress:
            for unit in progress:
                progress.set_postfix_str(str(unit))
                statuses.append(run_unit(sweep, unit))
        print(f"{name}: {len(units) - statuses.count(SKIPPED)} trained, {statuses.count(SKIPPED)} already done")


def planned_units(settings: ModelSettings, manifest: dict, split_seeds, folds, training_seeds) -> Iterator[Unit]:
    """Every (split seed, fold, training seed) for one model, minus any the filters leave out."""
    def wanted(value, allowed):
        return allowed is None or value in allowed

    for split in manifest["splits"]:
        for fold in split["folds"]:
            for seed in settings.training_seeds:
                if (wanted(split["split_seed"], split_seeds) and wanted(fold["fold"], folds)
                        and wanted(seed, training_seeds)):
                    yield Unit(settings.name, split["split_seed"], fold["fold"], seed)


def run_unit(sweep: Sweep, unit: Unit, *, dry_run=False) -> str:
    """Train and score one unit; returns a short status."""
    files = sweep.unit_files(unit)
    if files.is_done():
        return SKIPPED

    command = build_command(sweep, unit, files.native)
    if dry_run:
        return "would run: " + shlex.join(command)

    files.start_fresh()
    elapsed = run_logged(command, cwd=sweep.repo, log=files.log, what=str(unit))

    settings = sweep.models[unit.model]
    predictions = read_native_predictions(settings, files.native)
    if settings.is_laya and not settings.keep_checkpoints:
        remove_laya_checkpoints(files.native)

    save_unit(sweep, unit, files, command, predictions, elapsed)
    return f"done in {elapsed:.1f}s"


def build_command(sweep: Sweep, unit: Unit, native: Path) -> list[str]:
    """The model's own training command for one unit, with the shared fold files."""
    settings = sweep.models[unit.model]
    fold = sweep.fold_files(unit.split_seed, unit.fold)

    # A configured path (not a bare command name like "python3") is relative to the repo.
    python = str(sweep.resolve(settings.python)) if "/" in settings.python else settings.python
    data = ["--train", str(fold.train), "--validation", str(fold.val), "--test", str(fold.test),
            "--seed", str(unit.training_seed)]

    if settings.is_laya:
        return [python, "laya_finetuning/train_laya.py", "--task", "banking", *data,
                "--device", settings.device, "--output", str(native), *settings.args]
    return [python, "-m", "mlp_baseline.cli", "fit-predict", "--model", settings.name, *data,
            "--out", str(native), *settings.args]


def run_logged(command: list[str], *, cwd: Path, log: Path, what: str) -> float:
    """Run a command with its output in ``log``; returns the elapsed seconds."""
    started = time.perf_counter()
    with log.open("w") as stream:
        process = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode != 0:
        raise RuntimeError(f"{what} failed; see {log}")
    return time.perf_counter() - started


def read_native_predictions(settings: ModelSettings, native: Path) -> Predictions:
    """Convert a model's own prediction file into input_id -> (cause probabilities, P(correct))."""
    predictions = {}
    for row in read_jsonl(native / NATIVE_PREDICTIONS):
        if settings.is_laya:
            answers = row["answers"]
            predictions[row["meta"]["input_id"]] = (answers["cause"]["probabilities"],
                                                    float(answers["output_correct"]["noul"]))
        else:
            predictions[row["input_id"]] = (row["cause_probabilities"], float(row["correct_probability"]))
    return predictions


def remove_laya_checkpoints(native: Path) -> None:
    for name in LAYA_CHECKPOINTS:
        shutil.rmtree(native / name, ignore_errors=True)


def save_unit(sweep: Sweep, unit: Unit, files: UnitFiles, command: list[str],
              predictions: Predictions, elapsed: float) -> None:
    """Write the unit's predictions, how it was run, and finally its metrics."""
    save_predictions(files.predictions, predictions)
    write_json(files.run_config, {
        "unit": str(unit),
        "command": command,
        "settings": sweep.models[unit.model].raw,
        "elapsed_seconds": elapsed,
    })

    test_rows = read_jsonl(sweep.fold_files(unit.split_seed, unit.fold).test)
    write_json(files.metrics, score(test_rows, predictions))  # Last: marks the unit as done.


def check_model_settings(sweep: Sweep, settings: ModelSettings) -> None:
    """Refuse to mix runs of one model that were made with different settings."""
    path = sweep.model_dir(settings.name) / "settings.json"

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, settings.raw)
    elif read_json(path) != settings.raw:
        raise ValueError(f"{settings.name} settings changed since the runs in {path.parent}; "
                         "delete that folder or use a new config name")
