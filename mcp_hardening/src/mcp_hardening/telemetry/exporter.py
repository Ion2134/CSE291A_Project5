"""Append-only Parquet exporter for OpenTelemetry spans.

Parquet files are immutable, so "append" means: each flush writes a *new* part file
(``part-00000-1a2b3c4d.parquet``, ``part-00001-...``) and never reopens an existing one (files are
created with ``O_EXCL``). Every part shares one fixed schema, so the directory reads as a single
table: ``pyarrow.dataset.dataset(dir, format="parquet")`` or ``pandas.read_parquet(dir)``.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from mcp_hardening.telemetry import semconv

# Attributes promoted to typed top-level columns (everything is also kept in attributes_json).
PROMOTED: dict[str, pa.DataType] = {
    semconv.GEN_AI_OPERATION_NAME: pa.string(),
    semconv.GEN_AI_CONVERSATION_ID: pa.string(),
    semconv.GEN_AI_AGENT_NAME: pa.string(),
    semconv.GEN_AI_PROVIDER_NAME: pa.string(),
    semconv.GEN_AI_REQUEST_MODEL: pa.string(),
    semconv.GEN_AI_RESPONSE_MODEL: pa.string(),
    semconv.GEN_AI_USAGE_INPUT_TOKENS: pa.int64(),
    semconv.GEN_AI_USAGE_OUTPUT_TOKENS: pa.int64(),
    semconv.GEN_AI_TOOL_NAME: pa.string(),
    semconv.GEN_AI_TOOL_CALL_ID: pa.string(),
    semconv.ERROR_TYPE: pa.string(),
    semconv.HARDENING_REQUEST_SIGNED: pa.bool_(),
    semconv.HARDENING_REQUEST_VERIFIED: pa.bool_(),
    semconv.HARDENING_RESPONSE_VERIFIED: pa.bool_(),
    semconv.HARDENING_FAILURE_REASON: pa.string(),
    semconv.HARDENING_FAILURE_STAGE: pa.string(),
    semconv.SIM_TASK_ID: pa.string(),
    semconv.SIM_OPERATOR: pa.string(),
    semconv.SIM_HOP: pa.string(),
    semconv.SIM_CALL_INDEX: pa.int64(),
    semconv.SIM_MALFORMED: pa.bool_(),
    semconv.SIM_OUTPUT_CORRECT: pa.bool_(),
    semconv.SIM_CAUSE: pa.string(),
}

SCHEMA = pa.schema(
    [
        pa.field("trace_id", pa.string(), nullable=False),
        pa.field("span_id", pa.string(), nullable=False),
        pa.field("parent_span_id", pa.string()),
        pa.field("name", pa.string(), nullable=False),
        pa.field("kind", pa.string()),
        pa.field("start_time_unix_nano", pa.int64()),
        pa.field("end_time_unix_nano", pa.int64()),
        pa.field("duration_ms", pa.float64()),
        pa.field("status_code", pa.string()),
        pa.field("status_description", pa.string()),
        pa.field("service_name", pa.string()),
        *[pa.field(name, dtype) for name, dtype in PROMOTED.items()],
        pa.field("attributes_json", pa.string()),
        pa.field("events_json", pa.string()),
    ]
)

_PART_RE = re.compile(r"^part-(\d+)-[0-9a-f]+\.parquet$")


def _hex(value: int, width: int) -> str:
    return format(value, f"0{width}x")


def _coerce(value: Any, dtype: pa.DataType) -> Any:
    if value is None:
        return None
    try:
        if pa.types.is_boolean(dtype):
            return bool(value)
        if pa.types.is_int64(dtype):
            return int(value)
        if pa.types.is_string(dtype):
            return value if isinstance(value, str) else json.dumps(value, default=str)
    except (TypeError, ValueError):
        return None
    return value


def span_to_row(span: ReadableSpan) -> dict[str, Any]:
    ctx = span.get_span_context()
    attrs = dict(span.attributes or {})
    start, end = span.start_time, span.end_time
    row: dict[str, Any] = {
        "trace_id": _hex(ctx.trace_id, 32),
        "span_id": _hex(ctx.span_id, 16),
        "parent_span_id": _hex(span.parent.span_id, 16) if span.parent else None,
        "name": span.name,
        "kind": span.kind.name if span.kind else None,
        "start_time_unix_nano": start,
        "end_time_unix_nano": end,
        "duration_ms": (end - start) / 1e6 if start is not None and end is not None else None,
        "status_code": span.status.status_code.name,
        "status_description": span.status.description,
        "service_name": (span.resource.attributes.get("service.name") if span.resource else None),
        "attributes_json": json.dumps(attrs, default=list, sort_keys=True),
        "events_json": json.dumps(
            [
                {"name": e.name, "timestamp": e.timestamp, "attributes": dict(e.attributes or {})}
                for e in span.events
            ],
            default=list,
        ),
    }
    for name, dtype in PROMOTED.items():
        row[name] = _coerce(attrs.get(name), dtype)
    return row


class ParquetSpanExporter(SpanExporter):
    def __init__(self, out_dir: str | Path, flush_rows: int = 500) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.flush_rows = max(1, flush_rows)
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._seq = self._next_seq()
        self._shutdown = False
        self.files_written: list[Path] = []

    def _next_seq(self) -> int:
        seqs = [int(m.group(1)) for p in self.out_dir.iterdir() if (m := _PART_RE.match(p.name))]
        return max(seqs) + 1 if seqs else 0

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self._shutdown:
            return SpanExportResult.FAILURE
        rows = [span_to_row(s) for s in spans]
        with self._lock:
            self._buffer.extend(rows)
            # Full part files of exactly flush_rows rows; any remainder waits for the next flush.
            while len(self._buffer) >= self.flush_rows:
                self._write_locked(self.flush_rows)
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        with self._lock:
            self._write_locked(len(self._buffer))
        return True

    def shutdown(self) -> None:
        with self._lock:
            self._write_locked(len(self._buffer))
            self._shutdown = True

    def _write_locked(self, n: int) -> None:
        if n <= 0 or not self._buffer:
            return
        chunk = self._buffer[:n]
        table = pa.Table.from_pylist(chunk, schema=SCHEMA)
        path = self.out_dir / f"part-{self._seq:05d}-{uuid.uuid4().hex[:8]}.parquet"
        # "xb": create-only. An existing file is never opened for writing.
        with path.open("xb") as fh:
            pq.write_table(table, fh)
        self._seq += 1
        del self._buffer[:n]
        self.files_written.append(path)
