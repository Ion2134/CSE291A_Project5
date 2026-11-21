"""JSON toggle config. Each mechanism is its own block and can be switched on/off independently.

Example (``configs/all_on.json``)::

    {
      "version": 1,
      "request_auth":      {"enabled": true},
      "response_auth":     {"enabled": true},
      "replay_protection": {"enabled": true, "nonce_ttl_s": 300, "max_nonces": 100000},
      "freshness_check":   {"enabled": true, "window_s": 120},
      "mandate_binding":   {"enabled": false, "protected_tools": ["initiate_transfer"], "max_age_s": 3600,
                            "single_use": false},
      "telemetry":         {"enabled": true, "capture_content": true, "flush_rows": 500, "dir": "telemetry"}
    }
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from importlib import resources
from pathlib import Path
from typing import Any

CONFIG_VERSION = 1
CONFIG_ENV_VAR = "MCP_HARDENING_CONFIG"


@dataclass(frozen=True)
class RequestAuthConfig:
    enabled: bool = False


@dataclass(frozen=True)
class ResponseAuthConfig:
    enabled: bool = False


@dataclass(frozen=True)
class ReplayProtectionConfig:
    enabled: bool = False
    nonce_ttl_s: float = 300.0
    max_nonces: int = 100_000


@dataclass(frozen=True)
class FreshnessCheckConfig:
    enabled: bool = False
    window_s: float = 120.0


@dataclass(frozen=True)
class MandateBindingConfig:
    """Signed user mandates (``mandate.py``): protected tools run only if the user's mandate allows the call."""

    enabled: bool = False
    protected_tools: tuple[str, ...] = ("initiate_transfer",)
    max_age_s: float = 3600.0
    # Each mandate allows a protected tool at most ``max_uses`` times (set per rule in the mandate,
    # default 1), counted when a call actually runs. Off: a mandate can be used any number of times.
    single_use: bool = False


@dataclass(frozen=True)
class TelemetryConfig:
    enabled: bool = True
    capture_content: bool = True
    flush_rows: int = 500
    dir: str = "telemetry"


_BLOCKS: dict[str, type] = {
    "request_auth": RequestAuthConfig,
    "response_auth": ResponseAuthConfig,
    "replay_protection": ReplayProtectionConfig,
    "freshness_check": FreshnessCheckConfig,
    "mandate_binding": MandateBindingConfig,
    "telemetry": TelemetryConfig,
}


@dataclass(frozen=True)
class HardeningConfig:
    request_auth: RequestAuthConfig = field(default_factory=RequestAuthConfig)
    response_auth: ResponseAuthConfig = field(default_factory=ResponseAuthConfig)
    replay_protection: ReplayProtectionConfig = field(default_factory=ReplayProtectionConfig)
    freshness_check: FreshnessCheckConfig = field(default_factory=FreshnessCheckConfig)
    mandate_binding: MandateBindingConfig = field(default_factory=MandateBindingConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    @property
    def security_enabled(self) -> bool:
        """True if any call-security mechanism is on (telemetry doesn't count)."""
        return (
            self.request_auth.enabled
            or self.response_auth.enabled
            or self.replay_protection.enabled
            or self.freshness_check.enabled
            or self.mandate_binding.enabled
        )

    def to_dict(self) -> dict[str, Any]:
        def plain(block: Any) -> dict[str, Any]:  # tuples as lists, so a to_dict() equals its JSON round trip
            return {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(block).items()}

        return {"version": CONFIG_VERSION, **{name: plain(getattr(self, name)) for name in _BLOCKS}}

    @classmethod
    def from_dict(cls, data: Any, source: str = "<config>") -> HardeningConfig:
        if not isinstance(data, dict):
            raise ValueError(f"{source}: config must be a JSON object, got {type(data).__name__}")
        unknown = sorted(set(data) - set(_BLOCKS) - {"version"})
        if unknown:
            raise ValueError(f"{source}: unknown key(s) {unknown}; expected {sorted(_BLOCKS)}")
        version = data.get("version", CONFIG_VERSION)
        if version != CONFIG_VERSION or isinstance(version, bool):
            raise ValueError(f"{source}: $.version must be {CONFIG_VERSION}, got {version!r}")
        blocks = {
            name: _parse_block(block_cls, data[name], f"{source}: $.{name}")
            for name, block_cls in _BLOCKS.items()
            if name in data
        }
        return cls(**blocks)


def _parse_block(block_cls: type, raw: Any, where: str) -> Any:
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be an object, got {type(raw).__name__}")
    specs = {f.name: f for f in fields(block_cls)}
    unknown = sorted(set(raw) - set(specs))
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; expected {sorted(specs)}")
    defaults = block_cls()
    values: dict[str, Any] = {}
    for key, value in raw.items():
        expected = type(getattr(defaults, key))
        if expected is bool:
            ok = isinstance(value, bool)
        elif expected is float:
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
            value = float(value) if ok else value
        elif expected is int:
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif expected is tuple:  # a JSON list of strings
            ok = isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value)
            value = tuple(value) if ok else value
        else:
            ok = isinstance(value, expected)
        if not ok:
            wanted = "a list of strings" if expected is tuple else expected.__name__
            raise ValueError(f"{where}.{key} must be {wanted}, got {type(value).__name__}")
        if expected in (int, float) and value <= 0:
            raise ValueError(f"{where}.{key} must be > 0, got {value}")
        values[key] = value
    return block_cls(**values)


def preset_names() -> list[str]:
    root = resources.files("mcp_hardening") / "configs"
    return sorted(p.name.removesuffix(".json") for p in root.iterdir() if p.name.endswith(".json"))


def load_config(path: str | os.PathLike[str] | None = None) -> HardeningConfig:
    """Load a config. Resolution: explicit ``path`` → ``$MCP_HARDENING_CONFIG`` → built-in defaults.

    ``path`` may also name a packaged preset (``default``, ``all_on``, ``request_only``, ...).
    """
    target = path if path is not None else os.environ.get(CONFIG_ENV_VAR)
    if target is None or str(target) == "":
        return HardeningConfig()
    file = Path(target)
    if not file.is_file():
        preset = resources.files("mcp_hardening") / "configs" / f"{target}.json"
        if not preset.is_file():
            raise FileNotFoundError(
                f"hardening config {str(target)!r} not found (not a file or one of the presets {preset_names()})"
            )
        return HardeningConfig.from_dict(json.loads(preset.read_text()), source=f"preset:{target}")
    try:
        data = json.loads(file.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{file}: invalid JSON: {exc}") from exc
    return HardeningConfig.from_dict(data, source=str(file))
