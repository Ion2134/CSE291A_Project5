"""OpenTelemetry GenAI turn-level telemetry with an append-only Parquet exporter.

``semconv`` is import-light (no SDK), so the middleware can use it without pulling in tracing.
"""

from mcp_hardening.telemetry import semconv

__all__ = ["semconv", "GenAITelemetry", "ParquetSpanExporter", "read_telemetry"]


def __getattr__(name: str):
    if name == "GenAITelemetry":
        from mcp_hardening.telemetry.tracing import GenAITelemetry

        return GenAITelemetry
    if name == "ParquetSpanExporter":
        from mcp_hardening.telemetry.exporter import ParquetSpanExporter

        return ParquetSpanExporter
    if name == "read_telemetry":
        from mcp_hardening.telemetry.reader import read_telemetry

        return read_telemetry
    raise AttributeError(name)
