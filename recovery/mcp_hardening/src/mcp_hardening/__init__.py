"""MCP hardening: per-call two-way signing, signed user mandates, and OpenTelemetry GenAI Parquet telemetry."""

from mcp_hardening.config import (
    CONFIG_ENV_VAR,
    HardeningConfig,
    load_config,
    preset_names,
)
from mcp_hardening.errors import FAILURE_REASONS, HardeningError
from mcp_hardening.keys import Keyring, generate_encrypted_api_key
from mcp_hardening.mandate import MANDATE_KEY, sign_mandate, verify_mandate
from mcp_hardening.middleware import (
    ENVELOPE_KEY,
    ClientMiddleware,
    HardenedToolServer,
    HardeningRuntime,
    ServerMiddleware,
    envelope_from,
    failure_reason_from_message,
)
from mcp_hardening.replay import NonceCache
from mcp_hardening.telemetry import semconv
from mcp_hardening.telemetry.reader import read_telemetry
from mcp_hardening.telemetry.tracing import GenAITelemetry

__all__ = [
    "CONFIG_ENV_VAR",
    "ENVELOPE_KEY",
    "FAILURE_REASONS",
    "ClientMiddleware",
    "GenAITelemetry",
    "HardenedToolServer",
    "HardeningConfig",
    "HardeningError",
    "HardeningRuntime",
    "Keyring",
    "MANDATE_KEY",
    "NonceCache",
    "ServerMiddleware",
    "envelope_from",
    "failure_reason_from_message",
    "generate_encrypted_api_key",
    "load_config",
    "preset_names",
    "read_telemetry",
    "semconv",
    "sign_mandate",
    "verify_mandate",
]
