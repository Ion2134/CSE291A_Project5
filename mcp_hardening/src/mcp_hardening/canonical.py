from __future__ import annotations

import json
from typing import Any

PROTOCOL_VERSION = "v1"


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace. Both sides MAC exactly this string."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def request_mac_input(kid: str, tool: str, nonce: str, ts: int, arguments: dict[str, Any]) -> bytes:
    parts = [PROTOCOL_VERSION, "req", kid, tool, nonce, str(ts), canonical_json(arguments or {})]
    return "|".join(parts).encode("utf-8")


def response_mac_input(
    kid: str, tool: str, nonce: str, structured_content: Any, is_error: bool
) -> bytes:
    parts = [
        PROTOCOL_VERSION,
        "resp",
        kid,
        tool,
        nonce,
        canonical_json(structured_content if structured_content is not None else {}),
        "1" if is_error else "0",
    ]
    return "|".join(parts).encode("utf-8")
