import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SPEC = importlib.util.spec_from_file_location("train_laya_handoff", Path(__file__).resolve().parents[1] / "train_laya.py")
train = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = train
SPEC.loader.exec_module(train)

QUESTIONS = {"injection": {"type": "choice", "instructions": "Classify this message.",
                            "criteria": {"benign": "ordinary request", "injection": "malicious override"}}}
BANK_QUESTIONS = {
    "cause": {"type": "choice", "instructions": "Who caused the error?", "criteria": {
        "none": "correct output", "agent": "wrong model call", "adversary": "tampered tool output"}},
    "output_correct": {"type": "noul", "instructions": "Is the output correct?", "labels": {
        "false": "output does not match", "true": "output matches"}},
}


@pytest.fixture
def data(tmp_path):
    def create(task="injection"):
        paths = {}
        for split in ("train", "val", "test"):
            rows = []
            for i in range(6):
                if task == "injection":
                    expected = {"injection": "benign" if i % 2 == 0 else "injection"}
                    state = f"{split} request {i} " + ("ordinary balance check" if i % 2 == 0 else "override instructions now")
                    question = QUESTIONS
                else:
                    cause = ["none", "agent", "adversary"][i % 3]
                    expected = {"cause": cause, "output_correct": cause == "none"}
                    # Different words, rather than random IDs, make independent request groups.
                    state = f"user_message: {split} {'alpha beta gamma delta epsilon zeta'.split()[i]} request\noutput: {cause}"
                    question = BANK_QUESTIONS
                rows.append({"state": state, "questions": question, "expected": expected,
                             "group_id": f"{split}-{i}", "meta": {"source": "fixture"}})
            path = tmp_path / f"{task}-{split}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            paths[split] = path
        return paths
    return create


@pytest.fixture
def checkpoint(tmp_path):
    from laya.common import DecisionModel
    from laya.train import save_checkpoint
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
    words = ("[PAD] [UNK] [CLS] [SEP] [MASK] benign injection ordinary request malicious override "
             "none agent adversary correct output wrong model call tampered tool does not match true false "
             "Classify this message Who caused the error Is instructions now train val test "
             "alpha beta gamma delta epsilon zeta user_message balance check").split()
    vocab = {word: i for i, word in enumerate(dict.fromkeys(words))}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]",
                                 cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    torch.manual_seed(7)
    encoder = BertModel(BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1,
                                  num_attention_heads=4, intermediate_size=64, max_position_embeddings=2048))
    model = DecisionModel(encoder, head_layers=1)
    cfg = {"encoder": "tiny-offline-fixture", "head_layers": 1, "max_len": 1024, "head_max_len": 256,
           "temperature": [1, 1, 1], "act_costs": {"escalate": 0.5}}
    path = tmp_path / "checkpoint"
    save_checkpoint(model, tok, cfg, str(path))
    return path


def arguments(checkpoint, paths, output, task="injection", extra=()):
    return train.parser().parse_args([
        "--task", task, "--model", str(checkpoint), "--device", "cpu", "--epochs", "2",
        "--batch-size", "2", "--grad-accum", "2", "--eval-batch-size", "2",
        "--train", str(paths["train"]), "--validation", str(paths["val"]), "--test", str(paths["test"]),
        "--output", str(output), *extra])


@pytest.mark.parametrize("task,freeze", [("injection", False), ("injection", True), ("banking", False)])
def test_native_training_export_and_reload(tmp_path, checkpoint, data, task, freeze):
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    initial = load_file(str(checkpoint / "model.safetensors"))
    paths, out = data(task), tmp_path / "run"
    args = arguments(checkpoint, paths, out, task, ["--freeze-encoder"] if freeze else [])
    result = train.execute(args)
    assert result["checkpoint_reload_verified"]
    assert result["epochs_completed"] == 2
    assert result["best_epoch"] in [1, 2]
    assert result["test"]["injection" if task == "injection" else "cause"]["rows"] == 6
    final = load_file(str(out / "last_model/model.safetensors"))
    encoder_changed = any(not torch.equal(v, final[k]) for k, v in initial.items() if k.startswith("encoder."))
    head_changed = any(not torch.equal(v, final[k]) for k, v in initial.items() if not k.startswith("encoder."))
    assert encoder_changed is not freeze
    assert head_changed
    assert len((out / "test_predictions.jsonl").read_text().splitlines()) == 6
    assert len(json.loads((out / "history.json").read_text())) == 2
    assert "epoch 2" in (out / "training.log").read_text()
    cfg = json.loads((out / "model/rl_agent_config.json").read_text())
    assert cfg["temperature"] == [1, 1, 1]  # Tiny validation sets cannot support calibration.
    for rel, expected in json.loads((out / "checkpoint_checksums.json").read_text()).items():
        assert train.sha256(out / rel) == expected
    with pytest.raises(ValueError, match="not empty"):
        train.execute(args)


def test_dry_run_never_loads_weights_or_trains(tmp_path, checkpoint, data, monkeypatch):
    import laya.train as native
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run attempted to load weights or train")
    monkeypatch.setattr(native, "load_checkpoint", forbidden)
    monkeypatch.setattr(native, "train_model", forbidden)
    # No weights are required for dry-run tokenization.
    (checkpoint / "model.safetensors").unlink()
    out = tmp_path / "dry-run"
    args = arguments(checkpoint, data(), out, extra=["--dry-run"])
    result = train.execute(args)
    assert result["tokenization"]["train"]["items"] == 6
    assert not out.exists()


@pytest.mark.parametrize("failure", ["duplicate", "group", "label", "schema"])
def test_invalid_data_rejected(data, failure):
    paths = data()
    rows = [json.loads(line) for line in paths["val"].read_text().splitlines()]
    first = json.loads(paths["train"].read_text().splitlines()[0])
    if failure == "duplicate":
        rows[0]["state"] = first["state"].upper()
    elif failure == "group":
        rows[0]["group_id"] = first["group_id"]
    elif failure == "label":
        rows[0]["expected"]["injection"] = "unknown"
    else:
        rows[0]["questions"] = copy.deepcopy(QUESTIONS)
        rows[0]["questions"]["injection"]["instructions"] = "Changed task"
    paths["val"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Duplicate|overlap|Invalid|schema"):
        train.read_splits(paths, "injection")


def test_truncation_rejected(checkpoint, data):
    paths = data()
    rows = [json.loads(line) for line in paths["train"].read_text().splitlines()]
    rows[0]["state"] = "request " * 400
    paths["train"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    splits, _ = train.read_splits(paths, "injection")
    with pytest.raises(ValueError, match="would truncate"):
        train.prepare_items(splits, checkpoint, max_len=128, head_max_len=64)


def test_calibration_uses_only_supported_types():
    records = [(0, np.array([2., 0.]), np.array([1., 0.]), 2)] * 10
    result = train.fit_calibration(records)
    assert result["temperature"] == [1, 1, 1]
    assert result["skipped_types"]["0"]["n"] == 10
    records = [(0, np.array([2., 0.]), np.eye(2)[i % 2], 2) for i in range(24)]
    result = train.fit_calibration(records)
    assert not result["skipped_types"]
    assert all(np.isfinite(result["temperature"]))
