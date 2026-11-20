from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds

from mcp_hardening.telemetry.exporter import SCHEMA


def read_telemetry(path: str | Path) -> pa.Table:
    """Read every part file in a telemetry directory as one table (empty table if none yet)."""
    path = Path(path)
    if not path.is_dir() or not any(path.glob("part-*.parquet")):
        return SCHEMA.empty_table()
    return ds.dataset(path, format="parquet", schema=SCHEMA).to_table()
