"""One Laya fine-tuning run: check the arguments, validate and tokenize the data, then (unless it is a
dry run) train, keep the best epoch, calibrate on validation, score test, and write every report."""
from __future__ import annotations

import contextlib
import csv
import gc
import importlib.metadata
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .common import (BASE_MODEL, BASE_REVISION, DEFAULT_DATA, LAYA_DIR, RECORDED_PACKAGES, RunFiles, Tee, read_json,
                     sha256, write_json)
from .data import prepare_items, read_splits
from .evaluation import evaluate, fit_calibration, validation_loss

ENTRY_SCRIPT = LAYA_DIR / "train_laya.py"   # Hashed into every run manifest.
SUPPORTED_DEVICES = {"cuda", "cpu", "mps"}
SELECTION = "minimum validation cross-entropy; test evaluated once after selection and calibration"


def run_training(args) -> dict:
    """The whole run. A dry run stops after tokenization and returns the manifest; otherwise the results."""
    _resolve_revision(args)
    _check_arguments(args)
    config = _train_config(args)

    splits, data_report = read_splits(_data_paths(args), args.task)
    checkpoint = resolve_model(args.model, args.revision, args.dry_run)
    if read_json(checkpoint / "rl_agent_config.json").get("option_layout", "sequential") != "sequential":
        raise ValueError("This handoff supports sequential checkpoints only")
    _, items, token_report = prepare_items(splits, checkpoint, args.max_len, args.head_max_len)

    manifest = _manifest(args, config, checkpoint, splits, items, data_report, token_report)
    if args.dry_run:
        print(json.dumps({"dry_run": True, **manifest}, indent=2))
        return manifest

    device = _device(args.device)
    files = RunFiles(Path(args.output) if args.output else LAYA_DIR / f"runs/laya-{args.task}-seed{args.seed}")
    if files.out.exists() and any(files.out.iterdir()):
        raise ValueError(f"Output directory is not empty: {files.out}; choose a new run directory")
    files.out.mkdir(parents=True, exist_ok=True)

    with files.training_log.open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            return _train_and_report(args, config, device, checkpoint, splits, items, data_report, manifest, files)


# ---------------------------------------------------------------- checks and setup

def _resolve_revision(args) -> None:
    """The documented base model gets its pinned revision; another Hub model must name one."""
    if args.revision is None:
        if args.model == BASE_MODEL:
            args.revision = BASE_REVISION
        elif not Path(args.model).is_dir():
            raise ValueError("Specify --revision with a commit SHA for a different Hub model")


def _check_arguments(args) -> None:
    if args.max_len <= args.head_max_len or args.eval_batch_size < 1:
        raise ValueError("Need --max-len > --head-max-len and a positive --eval-batch-size")
    if args.seed < 0 or args.seed >= 2 ** 32:
        raise ValueError("Seed must be in [0, 2**32)")


def _train_config(args):
    from laya.train import TrainConfig

    config = TrainConfig(epochs=args.epochs, micro_batch=args.batch_size, grad_accum=args.grad_accum,
                         encoder_lr=args.encoder_lr, head_lr=args.head_lr, weight_decay=args.weight_decay,
                         freeze_encoder=args.freeze_encoder, loss=args.loss, seed=args.seed,
                         max_len=args.max_len, head_max_len=args.head_max_len, amp=args.device.startswith("cuda"),
                         gradient_checkpointing=True, calib_frac=0, calib_max=0, log_every=args.log_every)
    config.validate()
    for value in (args.encoder_lr, args.head_lr):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Learning rates must be finite and positive")
    return config


def _data_paths(args) -> dict:
    """``--train/--validation/--test`` together, or the task's default data directory."""
    root = DEFAULT_DATA[args.task]
    paths = {"train": args.train or root / "train.jsonl", "val": args.validation or root / "val.jsonl",
             "test": args.test or root / "test.jsonl"}
    if any((args.train, args.validation, args.test)) and not all((args.train, args.validation, args.test)):
        raise ValueError("Custom data requires --train, --validation, and --test together")
    return paths


def resolve_model(model, revision, dry_run) -> Path:
    """A local checkpoint directory, or a pinned Hub snapshot (without weights for a dry run)."""
    if Path(model).is_dir():
        return Path(model).resolve()

    from huggingface_hub import snapshot_download
    patterns = ["rl_agent_config.json", "tokenizer/*", "encoder/config.json"]
    if not dry_run:
        patterns.append("model.safetensors")
    return Path(snapshot_download(repo_id=model, revision=revision, allow_patterns=patterns))


def _manifest(args, config, checkpoint, splits, items, data_report, token_report) -> dict:
    from laya.train import optimizer_updates

    warnings = []
    if args.task == "banking":
        warnings.append(f"Banking data has {sum(len(splits[name]) for name in splits)} unique cases "
                        "across 10 request groups; development results only.")
    return {"task": args.task, "base_model": args.model, "revision": args.revision,
            "resolved_checkpoint": str(checkpoint), "config": asdict(config), "data": data_report,
            "tokenization": token_report, "script_sha256": sha256(ENTRY_SCRIPT), "seed": args.seed,
            "optimizer_updates": optimizer_updates(len(items["train"]), args.batch_size, args.grad_accum, args.epochs),
            "versions": {name: importlib.metadata.version(name) for name in RECORDED_PACKAGES},
            "selection": SELECTION,
            "warnings": warnings}


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable. Install a matching CUDA PyTorch build on the GPU machine. Use --dry-run here.")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ValueError("MPS unavailable on this PyTorch build/macOS. Use --device cpu or --dry-run.")
    if device.type not in SUPPORTED_DEVICES:
        raise ValueError("Supported devices: cuda[:index], mps, or cpu (cpu is intended for tiny smoke tests)")
    return device


def _gpu_name(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple MPS"
    return None


# ---------------------------------------------------------------- training, calibration, evaluation

def _train_and_report(args, config, device, checkpoint, splits, items, data_report, manifest, files: RunFiles):
    from laya.train import calibration_records, load_checkpoint, save_checkpoint, train_model

    started = time.monotonic()
    _seed_everything(args.seed, device)

    model, tokenizer, cfg = load_checkpoint(str(checkpoint))
    cfg.update(max_len=args.max_len, head_max_len=args.head_max_len, temperature=[1., 1., 1.])
    cfg.pop("temperature_by_options", None)  # Inherited base calibration is cleared before training.

    manifest.update(parameters=sum(p.numel() for p in model.parameters()), device=str(device), gpu=_gpu_name(device))
    write_json(files.manifest, manifest)
    print(json.dumps({"task": args.task, "parameters": manifest["parameters"], "device": str(device),
                      "training_items": len(items["train"]), "config": asdict(config)}, indent=2), flush=True)

    # Train; after each epoch the tracker scores validation and saves the best epoch to files.model.
    tracker = EpochTracker(model, tokenizer, cfg, items["val"], device, args, files)
    train_model(model, tokenizer, items["train"], config, device, args.max_len, args.head_max_len,
                on_epoch_end=tracker)
    save_checkpoint(model, tokenizer, cfg, str(files.last_model))
    tracker.model = None  # Drop every reference so the training model is freed before calibration.
    del model
    _free_memory(device)

    # Calibrate the best epoch on validation only.
    model, tokenizer, cfg = load_checkpoint(str(files.model))
    model.to(device)
    records = calibration_records(model, tokenizer, items["val"], device, args.max_len, args.head_max_len,
                                  batch_size=args.eval_batch_size)
    calibration = fit_calibration(records)
    cfg.update(temperature=calibration["temperature"], temperature_by_options=calibration["temperature_by_options"])
    save_checkpoint(model, tokenizer, cfg, str(files.model))
    write_json(files.calibration, {**calibration, "validation_sha256": data_report["val"]["sha256"]})
    del model
    _free_memory(device)

    # Score test once, through the same native loader a user would use.
    import laya
    agent = laya.load(str(files.model), device=str(device), fast=False, compile=False)
    metrics = evaluate(agent, splits["test"], files.test_predictions, args.eval_batch_size)

    report = {"task": args.task, "best_epoch": tracker.best_epoch,
              "best_validation_cross_entropy": tracker.best_loss,
              "epochs_completed": len(tracker.history), "elapsed_seconds": time.monotonic() - started,
              "checkpoint_reload_verified": True, "test": metrics, "warnings": manifest["warnings"]}
    write_json(files.results, report)
    write_json(files.checksums, {str(p.relative_to(files.out)): sha256(p)
                                 for p in sorted(files.model.rglob("*")) if p.is_file()})
    print(json.dumps(report, indent=2), flush=True)
    return report


class EpochTracker:
    """Laya's end-of-epoch callback: score validation, keep the best checkpoint, write the history."""

    def __init__(self, model, tokenizer, cfg, val_items, device, args, files: RunFiles):
        self.model, self.tokenizer, self.cfg = model, tokenizer, cfg
        self.val_items, self.device, self.args, self.files = val_items, device, args, files
        self.best_loss, self.best_epoch = float("inf"), None
        self.history = []

    def __call__(self, epoch: int, loss: float) -> None:
        from laya.train import calibration_records, save_checkpoint

        if not math.isfinite(loss):
            raise ValueError("Non-finite training loss; inspect AMP settings and learning rates")
        records = calibration_records(self.model, self.tokenizer, self.val_items, self.device, self.args.max_len,
                                      self.args.head_max_len, batch_size=self.args.eval_batch_size)
        val_loss = validation_loss(records)
        self.history.append({"epoch": epoch + 1, "train_loss": loss, "validation_cross_entropy": val_loss})
        print(f"epoch {epoch + 1}: validation CE={val_loss:.6f}", flush=True)

        if val_loss < self.best_loss:
            self.best_loss, self.best_epoch = val_loss, epoch + 1
            save_checkpoint(self.model, self.tokenizer, self.cfg, str(self.files.model))
        self._write_history()

        # Scoring put the model in eval mode; hand it back to training (the encoder stays frozen if asked).
        self.model.train()
        if self.args.freeze_encoder:
            self.model.encoder.eval()

    def _write_history(self) -> None:
        write_json(self.files.history_json, self.history)
        with self.files.history_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.history[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.history)


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _free_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
