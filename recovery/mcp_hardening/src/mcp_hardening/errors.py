from __future__ import annotations

# Every reason a hardening check can fail. Kept as plain strings so they serialize into logs/telemetry.
MISSING_ENVELOPE = "missing_envelope"
UNKNOWN_KID = "unknown_kid"
BAD_REQUEST_SIG = "bad_request_sig"
REPLAYED_NONCE = "replayed_nonce"
STALE_TIMESTAMP = "stale_timestamp"
BAD_RESPONSE_SIG = "bad_response_sig"
MISSING_RESPONSE_SIG = "missing_response_sig"
# Signed user mandates (mandate.py)
MISSING_MANDATE = "missing_mandate"
BAD_MANDATE_SIG = "bad_mandate_sig"
MANDATE_EXPIRED = "mandate_expired"
MANDATE_VIOLATION = "mandate_violation"
MANDATE_USED = "mandate_used"

FAILURE_REASONS = (
    MISSING_ENVELOPE,
    UNKNOWN_KID,
    BAD_REQUEST_SIG,
    REPLAYED_NONCE,
    STALE_TIMESTAMP,
    BAD_RESPONSE_SIG,
    MISSING_RESPONSE_SIG,
    MISSING_MANDATE,
    BAD_MANDATE_SIG,
    MANDATE_EXPIRED,
    MANDATE_VIOLATION,
    MANDATE_USED,
)


class HardeningError(ValueError):
    """A hardening check rejected a tool call. Subclasses ValueError so existing MCP/A2A error paths
    (which map ValueError to a failed task / JSON-RPC error) fail the call without extra handling."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"mcp_hardening:{reason}" + (f" ({detail})" if detail else ""))
