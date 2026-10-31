from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_MARKER = "episodes.jsonl"


def resolve_run_dir(requested: Path, seed: int, *, overwrite: bool, marker: str = RUN_MARKER) -> Path:
    """Use ``requested`` when empty; otherwise create a sibling dir so runs do not clobber."""
    requested = Path(requested)
    if overwrite or not (requested / marker).exists():
        return requested
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    name = requested.name or "run"
    return requested.parent / f"{name}-{stamp}-seed{seed}"


def write_run_manifest(out_dir: Path, payload: dict[str, Any]) -> None:
    import json

    (out_dir / "run_manifest.json").write_text(json.dumps(payload, indent=2))
