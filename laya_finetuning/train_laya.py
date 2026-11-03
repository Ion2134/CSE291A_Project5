#!/usr/bin/env python3
"""Single-device, native Laya supervised fine-tuning. See README before running.

    python laya_finetuning/train_laya.py --task banking --device cuda --seed 42 --output laya_finetuning/runs/bank

The work happens in ``laya_helpers/`` (data checks, training, calibration, evaluation).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from laya_helpers.common import BASE_MODEL
from laya_helpers.training import run_training


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # What to train on.
    result.add_argument("--task", choices=["injection", "banking"], required=True)
    result.add_argument("--model", default=BASE_MODEL)
    result.add_argument("--revision", help="Pinned Hub commit; defaults to the documented English Laya checkpoint")
    result.add_argument("--train", type=Path)
    result.add_argument("--validation", type=Path)
    result.add_argument("--test", type=Path)

    # Where and how to run.
    result.add_argument("--output", help="Empty/new directory; defaults to laya_finetuning/runs/laya-TASK-seedSEED")
    result.add_argument("--device", default="cuda", help="cuda[:index], mps (Apple Silicon), or cpu")
    result.add_argument("--dry-run", action="store_true",
                        help="Validate/tokenize all splits without loading weights or training")

    # Training recipe.
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
    result.add_argument("--freeze-encoder", action="store_true",
                        help="Train only the decision head; full fine-tuning is the default")
    result.add_argument("--log-every", type=int, default=100)
    return result


def main(argv=None) -> dict:
    return run_training(parser().parse_args(argv))


if __name__ == "__main__":
    main()
