#!/usr/bin/env python3
"""Single-device, native Laya supervised fine-tuning. See README before running."""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE_MODEL = "convaiinnovations/laya"
BASE_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
LABELS = {"injection": ["benign", "injection"], "cause": ["none", "agent", "adversary"],
          "output_correct": [False, True]}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def request_group(state):
    request = next((line[14:] for line in state.splitlines() if line.startswith("user_message: ")), None)
    if request is None:
        raise ValueError("Banking states need a user_message line for request grouping")
    request = re.sub(r"\b(acct|payee)_synth_\w+\b", r"\1_synth_<id>", request.lower())
    return re.sub(r"\d+(?:\.\d+)?", "<number>", request)


def read_splits(paths, task):
    """Validate labels, duplicate inputs, schema, and request/similarity groups."""
    splits, report, seen, groups = {}, {}, {}, {}
    schema = None
    for name, path in paths.items():
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"Empty {name} split")
        for index, row in enumerate(rows, 1):
            where = f"{name}:{index}"
            if not isinstance(row.get("state"), str) or not row["state"].strip():
                raise ValueError(f"Missing state at {where}")
            questions, expected = row.get("questions"), row.get("expected")
            keys = {"injection"} if task == "injection" else {"cause", "output_correct"}
            if not isinstance(questions, dict) or not isinstance(expected, dict) or set(questions) != keys or set(expected) != keys:
                raise ValueError(f"Wrong question/label keys for {task} at {where}")
            for qid in keys:
                value = expected[qid]
                if qid == "output_correct":
                    valid = type(value) is bool and questions[qid].get("type") == "noul"
                else:
                    valid = (type(value) is str and value in LABELS[qid]
                             and questions[qid].get("type") == "choice"
                             and set(questions[qid].get("criteria", {})) == set(LABELS[qid]))
                if not valid:
                    raise ValueError(f"Invalid {qid} labels/schema at {where}")
            if task == "banking" and expected["output_correct"] != (expected["cause"] == "none"):
                raise ValueError(f"Inconsistent correctness/cause at {where}")
            if schema is None:
                schema = questions
            if questions != schema:
                raise ValueError(f"Question schema changed at {where}")
            state = re.sub(r"\b(pay|txn)_synth_[0-9a-f]{8}\b", r"\1_synth_<id>", row["state"])
            signature = " ".join(state.lower().split())
            if signature in seen:
                raise ValueError(f"Duplicate input at {where}; already in {seen[signature]}")
            seen[signature] = where
            group = request_group(row["state"]) if task == "banking" else row.get("group_id")
            if task == "injection" and not isinstance(group, str):
                raise ValueError(f"Injection rows need group_id for similarity-group protection at {where}")
            if group in groups and groups[group] != name:
                raise ValueError(f"Group overlap between {groups[group]} and {name}")
            groups[group] = name
        splits[name] = rows
        report[name] = {"path": str(Path(path).resolve()), "sha256": sha256(path), "rows": len(rows),
                        "labels": {qid: dict(Counter(str(r["expected"][qid]) for r in rows)) for qid in schema}}
    for qid in schema:
        if {r["expected"][qid] for r in splits["train"]} != set(LABELS[qid]):
            raise ValueError(f"Training split must contain every {qid} class")
    return splits, report


def resolve_model(model, revision, dry_run):
    if Path(model).is_dir():
        return Path(model).resolve()
    from huggingface_hub import snapshot_download
    patterns = ["rl_agent_config.json", "tokenizer/*", "encoder/config.json"]
    if not dry_run:
        patterns.append("model.safetensors")
    return Path(snapshot_download(repo_id=model, revision=revision, allow_patterns=patterns))


def prepare_items(splits, checkpoint, max_len, head_max_len):
    from laya.common import encode_text, state_room
    from laya.train import items_from_rows, to_internal
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True)
    items, report = {}, {}
    for name, rows in splits.items():
        maximum, minimum_room = 0, max_len
        for i, row in enumerate(rows, 1):
            length = len(encode_text(tok, row["state"], add_special_tokens=False)["input_ids"])
            room = min(state_room(tok, to_internal(qid, q), max_len, head_max_len)
                       for qid, q in row["questions"].items())
            if length > room:
                raise ValueError(f"{name}:{i} would truncate {length} state tokens to {room}; increase --max-len")
            maximum, minimum_room = max(maximum, length), min(minimum_room, room)
        items[name], skipped = items_from_rows(tok, rows, max_len, head_max_len)
        if any(skipped.values()) or len(items[name]) != sum(len(r["questions"]) for r in rows):
            raise ValueError(f"Laya skipped questions in {name}: {skipped}")
        report[name] = {"items": len(items[name]), "max_state_tokens": maximum,
                        "min_state_room": minimum_room, "skipped": skipped}
    return tok, items, report


def validation_loss(records):
    losses = []
    for _, logits, target, _ in records:
        values = np.asarray(logits, dtype=np.float64)
        logp = values - np.max(values)
        logp -= np.log(np.exp(logp).sum())
        losses.append(float(-np.dot(target, logp)))
    result = float(np.mean(losses))
    if not math.isfinite(result):
        raise ValueError("Non-finite validation loss; inspect the training log")
    return result


def fit_calibration(records, minimum=20):
    """Fit on validation only; small or single-class types stay at temperature 1."""
    from laya.calibrate import fit_temperature_map
    by_type = defaultdict(list)
    for record in records:
        by_type[int(record[0])].append(record)
    eligible, skipped = [], {}
    for qtype, part in by_type.items():
        if len(part) < minimum or len({int(np.argmax(r[2])) for r in part}) < 2:
            skipped[str(qtype)] = {"n": len(part), "reason": "need >=20 validation items and >=2 target classes"}
        else:
            eligible.extend(part)
    fitted = fit_temperature_map(eligible)
    return {**fitted, "skipped_types": skipped, "source": "validation only", "minimum_items": minimum}


def evaluate(agent, rows, output_file, batch_size):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score
    answers = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        outputs = agent.predict_batch([r["state"] for r in chunk], chunk[0]["questions"], batch_size=batch_size)
        answers.extend(output["answers"] for output in outputs)
    metrics = {}
    with Path(output_file).open("w", encoding="utf-8") as handle:
        for index, (row, answer) in enumerate(zip(rows, answers)):
            handle.write(json.dumps({"row": index, "meta": row.get("meta"), "expected": row["expected"], "answers": answer}) + "\n")
    for qid in rows[0]["questions"]:
        labels = LABELS[qid]
        y = np.array([labels.index(r["expected"][qid]) for r in rows])
        if qid == "output_correct":
            p = np.array([[1 - a[qid]["noul"], a[qid]["noul"]] for a in answers])
        else:
            p = np.array([[a[qid]["probabilities"][label] for label in labels] for a in answers])
        p = np.clip(p, 1e-7, 1)
        p /= p.sum(axis=1, keepdims=True)
        pred = p.argmax(axis=1)
        result = {"rows": len(y), "labels": labels, "accuracy": float(accuracy_score(y, pred)),
                  "macro_f1": float(f1_score(y, pred, labels=list(range(len(labels))), average="macro", zero_division=0)),
                  "per_class_f1": f1_score(y, pred, labels=list(range(len(labels))), average=None, zero_division=0).tolist(),
                  "class_support": np.bincount(y, minlength=len(labels)).tolist(),
                  "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(labels)))).tolist(),
                  "log_loss": float(log_loss(y, p, labels=list(range(len(labels)))))}
        if len(labels) == 2:
            result.update(auroc=float(roc_auc_score(y, p[:, 1])) if len(set(y)) == 2 else None,
                          brier=float(np.mean((p[:, 1] - y) ** 2)),
                          false_positive_rate=float(np.mean(pred[y == 0] == 1)) if np.any(y == 0) else None,
                          false_negative_rate=float(np.mean(pred[y == 1] == 0)) if np.any(y == 1) else None)
        metrics[qid] = result
    # Source breakdown exposes the domain gap hidden by an aggregate score.
    if "injection" in metrics:
        sources = sorted({r.get("meta", {}).get("source", "unknown") for r in rows})
        metrics["by_source"] = {}
        for source in sources:
            indices = [i for i, r in enumerate(rows) if r.get("meta", {}).get("source", "unknown") == source]
            y = [LABELS["injection"].index(rows[i]["expected"]["injection"]) for i in indices]
            pred = [int(answers[i]["injection"]["probabilities"]["injection"] >
                        answers[i]["injection"]["probabilities"]["benign"]) for i in indices]
            metrics["by_source"][source] = {"rows": len(y), "accuracy": float(accuracy_score(y, pred)),
                                            "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0))}
    return metrics


class Tee:
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


def execute(args):
    from laya.train import TrainConfig, calibration_records, load_checkpoint, optimizer_updates, save_checkpoint, train_model
    if args.revision is None:
        if args.model == BASE_MODEL:
            args.revision = BASE_REVISION
        elif not Path(args.model).is_dir():
            raise ValueError("Specify --revision with a commit SHA for a different Hub model")
    if args.max_len <= args.head_max_len or args.eval_batch_size < 1:
        raise ValueError("Need --max-len > --head-max-len and a positive --eval-batch-size")
    if args.seed < 0 or args.seed >= 2 ** 32:
        raise ValueError("Seed must be in [0, 2**32)")
    config = TrainConfig(epochs=args.epochs, micro_batch=args.batch_size, grad_accum=args.grad_accum,
                         encoder_lr=args.encoder_lr, head_lr=args.head_lr, weight_decay=args.weight_decay,
                         freeze_encoder=args.freeze_encoder, loss=args.loss, seed=args.seed,
                         max_len=args.max_len, head_max_len=args.head_max_len, amp=args.device.startswith("cuda"),
                         gradient_checkpointing=True, calib_frac=0, calib_max=0, log_every=args.log_every)
    config.validate()
    for value in (args.encoder_lr, args.head_lr):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Learning rates must be finite and positive")
    root = (HERE.parent / "mlp_baseline/data/injection-v1" if args.task == "injection"
            else HERE / "data/agent-combined-v4")
    paths = {"train": args.train or root / "train.jsonl", "val": args.validation or root / "val.jsonl",
             "test": args.test or root / "test.jsonl"}
    if any((args.train, args.validation, args.test)) and not all((args.train, args.validation, args.test)):
        raise ValueError("Custom data requires --train, --validation, and --test together")
    splits, data_report = read_splits(paths, args.task)
    checkpoint = resolve_model(args.model, args.revision, args.dry_run)
    cfg = json.loads((checkpoint / "rl_agent_config.json").read_text())
    if cfg.get("option_layout", "sequential") != "sequential":
        raise ValueError("This handoff supports sequential checkpoints only")
    tok, items, token_report = prepare_items(splits, checkpoint, args.max_len, args.head_max_len)
    manifest = {"task": args.task, "base_model": args.model, "revision": args.revision,
                "resolved_checkpoint": str(checkpoint), "config": asdict(config), "data": data_report,
                "tokenization": token_report, "script_sha256": sha256(__file__), "seed": args.seed,
                "optimizer_updates": optimizer_updates(len(items["train"]), args.batch_size, args.grad_accum, args.epochs),
                "versions": {name: importlib.metadata.version(name) for name in
                             ("laya", "torch", "transformers", "numpy", "scikit-learn", "safetensors", "huggingface-hub")},
                "selection": "minimum validation cross-entropy; test evaluated once after selection and calibration",
                "warnings": (
                    [f"Banking data has {sum(len(splits[name]) for name in splits)} unique cases "
                     "across 10 request groups; development results only."]
                    if args.task == "banking" else [])}
    if args.dry_run:
        print(json.dumps({"dry_run": True, **manifest}, indent=2))
        return manifest
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable. Install a matching CUDA PyTorch build on the GPU machine. Use --dry-run here.")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ValueError("MPS unavailable on this PyTorch build/macOS. Use --device cpu or --dry-run.")
    if device.type not in {"cuda", "cpu", "mps"}:
        raise ValueError("Supported devices: cuda[:index], mps, or cpu (cpu is intended for tiny smoke tests)")
    out = Path(args.output) if args.output else HERE / f"runs/laya-{args.task}-seed{args.seed}"
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}; choose a new run directory")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "training.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            started = time.monotonic()
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed)
            model, tok, cfg = load_checkpoint(str(checkpoint))
            cfg.update(max_len=args.max_len, head_max_len=args.head_max_len, temperature=[1., 1., 1.])
            cfg.pop("temperature_by_options", None)
            manifest["parameters"] = sum(p.numel() for p in model.parameters())
            manifest["device"] = str(device)
            if device.type == "cuda":
                manifest["gpu"] = torch.cuda.get_device_name(device)
            elif device.type == "mps":
                manifest["gpu"] = "Apple MPS"
            else:
                manifest["gpu"] = None
            write_json(out / "run_manifest.json", manifest)
            print(json.dumps({"task": args.task, "parameters": manifest["parameters"], "device": str(device),
                              "training_items": len(items["train"]), "config": asdict(config)}, indent=2), flush=True)
            best, history = {"loss": float("inf"), "epoch": None}, []

            def epoch_end(epoch, loss):
                if not math.isfinite(loss):
                    raise ValueError("Non-finite training loss; inspect AMP settings and learning rates")
                records = calibration_records(model, tok, items["val"], device, args.max_len,
                                              args.head_max_len, batch_size=args.eval_batch_size)
                val_loss = validation_loss(records)
                history.append({"epoch": epoch + 1, "train_loss": loss, "validation_cross_entropy": val_loss})
                print(f"epoch {epoch + 1}: validation CE={val_loss:.6f}", flush=True)
                if val_loss < best["loss"]:
                    best.update(loss=val_loss, epoch=epoch + 1)
                    save_checkpoint(model, tok, cfg, str(out / "model"))
                write_json(out / "history.json", history)
                with (out / "history.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(history[0]), lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(history)
                model.train()
                if args.freeze_encoder:
                    model.encoder.eval()

            train_model(model, tok, items["train"], config, device, args.max_len, args.head_max_len,
                        on_epoch_end=epoch_end)
            save_checkpoint(model, tok, cfg, str(out / "last_model"))
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            model, tok, cfg = load_checkpoint(str(out / "model"))
            model.to(device)
            records = calibration_records(model, tok, items["val"], device, args.max_len,
                                          args.head_max_len, batch_size=args.eval_batch_size)
            calibration = fit_calibration(records)
            cfg.update(temperature=calibration["temperature"], temperature_by_options=calibration["temperature_by_options"])
            save_checkpoint(model, tok, cfg, str(out / "model"))
            write_json(out / "calibration.json", {**calibration, "validation_sha256": data_report["val"]["sha256"]})
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            import laya
            agent = laya.load(str(out / "model"), device=str(device), fast=False, compile=False)
            metrics = evaluate(agent, splits["test"], out / "test_predictions.jsonl", args.eval_batch_size)
            report = {"task": args.task, "best_epoch": best["epoch"], "best_validation_cross_entropy": best["loss"],
                      "epochs_completed": len(history), "elapsed_seconds": time.monotonic() - started,
                      "checkpoint_reload_verified": True, "test": metrics, "warnings": manifest["warnings"]}
            write_json(out / "results.json", report)
            write_json(out / "checkpoint_checksums.json", {str(p.relative_to(out)): sha256(p)
                       for p in sorted((out / "model").rglob("*")) if p.is_file()})
            print(json.dumps(report, indent=2), flush=True)
            return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--task", choices=["injection", "banking"], required=True)
    result.add_argument("--model", default=BASE_MODEL)
    result.add_argument("--revision", help="Pinned Hub commit; defaults to the documented English Laya checkpoint")
    result.add_argument("--train", type=Path)
    result.add_argument("--validation", type=Path)
    result.add_argument("--test", type=Path)
    result.add_argument("--output", help="Empty/new directory; defaults to laya_finetuning/runs/laya-TASK-seedSEED")
    result.add_argument("--device", default="cuda", help="cuda[:index], mps (Apple Silicon), or cpu")
    result.add_argument("--dry-run", action="store_true", help="Validate/tokenize all splits without loading weights or training")
    result.add_argument("--epochs", type=int, default=4)
    result.add_argument("--batch-size", type=int, default=2)
    result.add_argument("--grad-accum", type=int, default=8)
    result.add_argument("--eval-batch-size", type=int, default=4)
    result.add_argument("--encoder-lr", type=float, default=2.5e-5)
    result.add_argument("--head-lr", type=float, default=1e-4)
    result.add_argument("--weight-decay", type=float, default=0.01)
    result.add_argument("--max-len", type=int, default=1024)
    result.add_argument("--head-max-len", type=int, default=256)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--loss", choices=["soft-ce", "rlcd"], default="soft-ce")
    result.add_argument("--freeze-encoder", action="store_true", help="Train only the decision head; full fine-tuning is the default")
    result.add_argument("--log-every", type=int, default=100)
    return result


if __name__ == "__main__":
    execute(parser().parse_args())
