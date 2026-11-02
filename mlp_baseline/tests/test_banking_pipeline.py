import json

from mlp_baseline.banking import load_banking_mlp, run_banking


def test_banking_pipeline_is_standalone_and_reloads(tmp_path):
    from pathlib import Path
    import numpy as np
    source = Path(__file__).resolve().parents[1] / "data/banking/triton42-v2/raw"
    result = run_banking(source, tmp_path / "bank", epochs=3, threads=1)
    assert result["status"] == "complete"
    assert result["development_only"] is True
    assert result["training"]["checkpoint_reload_verified"]
    assert result["data"]["audit"]["repaired_rows"] == 2
    assert sum(part["rows"] for part in result["data"]["splits"].values()) == 52
    cases = [json.loads(line) for line in (tmp_path / "bank/splits/test.jsonl").read_text().splitlines()]
    cause, correct = load_banking_mlp(tmp_path / "bank/mlp").predict(cases)
    assert cause.shape == (len(cases), 3)
    np.testing.assert_allclose(cause.sum(axis=1), 1)
    assert len(correct) == len(cases)
