"""Score predictions the same way for every model and write the summary.

The headline is one number per model and metric: the median (and mean) over all repeats.
A repeat is one split seed trained with one training seed; its k folds together cover
every row once, so they are scored as one set.

    summary/results.csv   median, mean and std per model and metric  <- the results
    summary/SUMMARY.md    the same as a table, plus model-vs-model differences
    summary/raw/          every repeat, every single fold, and every paired difference
"""
from __future__ import annotations

import itertools
from collections import Counter

import numpy as np

from .config import MODELS, Sweep, Unit, read_jsonl, write_csv, write_json, write_jsonl
from mlp_baseline.common import LABELS
from mlp_baseline.metrics import evaluate

# input_id -> ({cause: probability}, P(output correct))
Predictions = dict[str, tuple[dict[str, float], float]]
# (split seed, training seed): all k folds of one split, trained with one seed
Repeat = tuple[int, "int | None"]

METRICS = ("cause_macro_f1", "cause_accuracy", "error_auroc", "error_detection_f1",
           "false_alarm_rate", "missed_error_rate", "error_type_accuracy",
           "correctness_brier", "correctness_ece")
LOWER_IS_BETTER = {"false_alarm_rate", "missed_error_rate", "correctness_brier", "correctness_ece"}
MARKDOWN_PAIRED_METRICS = ("cause_macro_f1", "error_auroc", "false_alarm_rate")
MAJORITY = "majority"


# ---------------------------------------------------------------- predictions and scoring

def save_predictions(path, predictions: Predictions) -> None:
    write_jsonl(path, [{"input_id": key, "cause_probabilities": cause, "correct_probability": correct}
                       for key, (cause, correct) in predictions.items()])


def load_predictions(path) -> Predictions:
    return {row["input_id"]: (row["cause_probabilities"], row["correct_probability"])
            for row in read_jsonl(path)}


def score(rows: list[dict], predictions: Predictions) -> dict:
    """``mlp_baseline.metrics.evaluate`` on rows tagged with ``meta.input_id``."""
    chosen = [predictions[row["meta"]["input_id"]] for row in rows]
    cause = np.array([[probs[label] for label in LABELS] for probs, _ in chosen])
    correct = np.array([ok for _, ok in chosen])
    return evaluate(rows, cause, correct)


def majority_predictions(train_rows: list[dict], test_rows: list[dict]) -> Predictions:
    """Always predict the most common training cause."""
    label = Counter(row["expected"]["cause"] for row in train_rows).most_common(1)[0][0]
    cause = {name: float(name == label) for name in LABELS}
    return {row["meta"]["input_id"]: (cause, cause["none"]) for row in test_rows}


# ---------------------------------------------------------------- summary

def summarize(sweep: Sweep, manifest: dict) -> dict:
    """Score every finished unit and write the headline results plus the raw tables."""
    split_seeds = [split["split_seed"] for split in manifest["splits"]]
    test_rows = {(p, f): read_jsonl(sweep.fold_files(p, f).test)
                 for p in split_seeds for f in range(sweep.split.k)}
    units = load_units(sweep, split_seeds)

    pooled_scores = {model: pool_folds(found, test_rows, sweep.split.k) for model, found in units.items()}
    fold_scores = {model: {unit: score(test_rows[(unit.split_seed, unit.fold)], predictions)
                           for unit, predictions in found.items()}
                   for model, found in units.items()}

    summary = {
        "config": sweep.name,
        "pool": manifest["pool"],
        "models": {model: {"repeats": len(scores),
                           "metrics": {key: stats([m[key] for m in scores.values()]) for key in METRICS}}
                   for model, scores in pooled_scores.items()},
        "paired": paired_differences(pooled_scores),
    }
    write_summary(sweep, summary, pooled_scores, fold_scores)
    return summary


def load_units(sweep: Sweep, split_seeds: list[int]) -> dict[str, dict[Unit, Predictions]]:
    """Predictions of every finished unit, plus the majority baseline for every fold."""
    units = {MAJORITY: {}}
    for p in split_seeds:
        for f in range(sweep.split.k):
            files = sweep.fold_files(p, f)
            train = read_jsonl(files.train) + read_jsonl(files.val)
            units[MAJORITY][Unit(MAJORITY, p, f, None)] = majority_predictions(train, read_jsonl(files.test))

    for model in MODELS:
        model_dir = sweep.model_dir(model)
        unit_dirs = sorted(path for path in model_dir.iterdir() if path.is_dir()) if model_dir.exists() else []
        found = {}
        for unit_dir in unit_dirs:
            unit = Unit.from_dir_name(model, unit_dir.name)
            files = sweep.unit_files(unit)
            if files.is_done():
                found[unit] = load_predictions(files.predictions)
        if found:
            units[model] = found
    return units


def pool_folds(found: dict[Unit, Predictions], test_rows: dict, k: int) -> dict[Repeat, dict]:
    """Score each complete repeat as one set. Incomplete repeats only appear in raw/per_fold.csv."""
    by_repeat: dict[Repeat, dict[int, Predictions]] = {}
    for unit, predictions in found.items():
        by_repeat.setdefault((unit.split_seed, unit.training_seed), {})[unit.fold] = predictions

    pooled = {}
    for (p, t), folds in sorted(by_repeat.items(), key=str):
        if len(folds) == k:
            rows = [row for f in range(k) for row in test_rows[(p, f)]]
            predictions = {key: value for fold in folds.values() for key, value in fold.items()}
            pooled[(p, t)] = score(rows, predictions)
    return pooled


def stats(values: list) -> dict:
    values = [value for value in values if value is not None]
    if not values:
        return {"n": 0}

    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {"n": len(values), "median": float(median), "q1": float(q1), "q3": float(q3),
            "min": float(min(values)), "max": float(max(values)), "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def paired_differences(pooled_scores: dict[str, dict[Repeat, dict]]) -> list[dict]:
    """first − second on the same repeats, for every pair of models."""
    rows = []
    for first, second in itertools.combinations(pooled_scores, 2):
        pairs = matched_repeats(pooled_scores[first], pooled_scores[second])

        for key in METRICS:
            differences = [a[key] - b[key] for a, b in pairs if a[key] is not None and b[key] is not None]
            if not differences:
                continue

            first_better = [d < 0 if key in LOWER_IS_BETTER else d > 0 for d in differences]
            rows.append({"first": first, "second": second, "metric": key, "pairs": len(differences),
                         "median_difference": float(np.median(differences)),
                         "min_difference": float(min(differences)),
                         "max_difference": float(max(differences)),
                         "first_better_rate": float(np.mean(first_better))})
    return rows


def matched_repeats(first: dict[Repeat, dict], second: dict[Repeat, dict]) -> list[tuple[dict, dict]]:
    """Scores of both models on the same repeats. The majority baseline (seed None) matches any seed."""
    def lookup(scores, p, t):
        return scores.get((p, t)) or scores.get((p, None))

    seeded = first if any(t is not None for _, t in first) else second
    pairs = [(lookup(first, p, t), lookup(second, p, t)) for p, t in sorted(seeded, key=str)]
    return [(a, b) for a, b in pairs if a and b]


# ---------------------------------------------------------------- output files

def write_summary(sweep: Sweep, summary: dict, pooled_scores: dict, fold_scores: dict) -> None:
    directory = sweep.out / "summary"
    raw = directory / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    write_csv(directory / "results.csv", results_rows(summary))
    write_json(directory / "summary.json", summary)
    (directory / "SUMMARY.md").write_text(markdown(summary))

    write_csv(raw / "repeats.csv", [
        {"model": model, "split_seed": p, "training_seed": t, **metric_columns(metrics)}
        for model, by_repeat in pooled_scores.items()
        for (p, t), metrics in sorted(by_repeat.items(), key=lambda item: str(item[0]))])
    write_csv(raw / "per_fold.csv", [
        {"model": model, "split_seed": unit.split_seed, "fold": unit.fold,
         "training_seed": unit.training_seed, **metric_columns(metrics)}
        for model, by_unit in fold_scores.items()
        for unit, metrics in sorted(by_unit.items(), key=lambda item: str(item[0]))])
    write_csv(raw / "paired.csv", summary["paired"])


def results_rows(summary: dict) -> list[dict]:
    """One row per model and metric: median, mean and std over the repeats (rounded for reading)."""
    def rounded(value):
        return None if value is None else round(value, 4)

    return [{"model": model, "metric": key, "median": rounded(stat.get("median")),
             "mean": rounded(stat.get("mean")), "std": rounded(stat.get("std")), "repeats": entry["repeats"]}
            for model, entry in summary["models"].items()
            for key, stat in entry["metrics"].items()]


def metric_columns(metrics: dict) -> dict:
    return {"samples": metrics["samples"], **{key: metrics[key] for key in METRICS}}


def markdown(summary: dict) -> str:
    def cell(stat):
        return f"{stat['median']:.3f} ({stat['mean']:.3f} ± {stat['std']:.3f})" if stat["n"] else "–"

    pool = summary["pool"]
    lines = [f"# Sweep results: {summary['config']}", "",
             f"Pool: {pool['rows']} rows in {pool['groups']} request groups, labels {pool['labels']}.",
             "Each cell is the **median** over all repeats, with (mean ± std).", "",
             "| Model | Repeats | " + " | ".join(METRICS) + " |",
             "| --- | ---: | " + " | ".join("---:" for _ in METRICS) + " |"]
    for model, entry in summary["models"].items():
        lines.append(f"| {model} | {entry['repeats']} | "
                     + " | ".join(cell(entry["metrics"][key]) for key in METRICS) + " |")

    lines += ["", "## Model vs model on the same repeats (median of first − second)", "",
              "| First | Second | Metric | Repeats | Median diff | First better |",
              "| --- | --- | --- | ---: | ---: | ---: |"]
    lines += [f"| {row['first']} | {row['second']} | {row['metric']} | {row['pairs']} | "
              f"{row['median_difference']:+.3f} | {row['first_better_rate']:.0%} |"
              for row in summary["paired"] if row["metric"] in MARKDOWN_PAIRED_METRICS]

    lines += ["", "## Caveats", "",
              "- Laya trains on fewer groups than the MLP and stacking: it holds out a fixed validation set, "
              "while they use inner cross-validation on all training groups.",
              "- Laya's banking validation sets have fewer than 20 rows, so Laya keeps temperature 1 "
              "(uncalibrated); compare Brier score and ECE with that in mind.",
              "- Each repeat combines the k separately trained fold models.",
              "- Development-scale data: treat these as comparisons, not benchmark claims.", ""]
    return "\n".join(lines)
