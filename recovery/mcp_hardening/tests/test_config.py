from __future__ import annotations

import json

import pytest

from mcp_hardening import HardeningConfig, load_config, preset_names


def test_presets_load():
    assert {"default", "all_on", "all_off", "request_only", "response_only"} <= set(preset_names())
    assert load_config("default") == HardeningConfig()
    on = load_config("all_on")
    assert on.security_enabled and on.request_auth.enabled and on.freshness_check.enabled
    only = load_config("request_only")
    assert only.request_auth.enabled and not only.response_auth.enabled
    assert not load_config("all_off").telemetry.enabled


def test_default_resolution(monkeypatch, tmp_path):
    monkeypatch.delenv("MCP_HARDENING_CONFIG", raising=False)
    assert load_config() == HardeningConfig()
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"version": 1, "response_auth": {"enabled": True}}))
    monkeypatch.setenv("MCP_HARDENING_CONFIG", str(path))
    cfg = load_config()
    assert cfg.response_auth.enabled and not cfg.request_auth.enabled  # missing blocks take defaults


def test_roundtrip():
    cfg = load_config("all_on")
    assert HardeningConfig.from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize(
    "data,match",
    [
        ({"version": 2}, "version"),
        ({"bogus": {}}, "unknown key"),
        ({"request_auth": {"enabled": "yes"}}, r"request_auth\.enabled must be bool"),
        ({"request_auth": {"enabld": True}}, "unknown key"),
        ({"freshness_check": {"window_s": -1}}, "must be > 0"),
        ({"replay_protection": {"max_nonces": 1.5}}, "must be int"),
        ({"telemetry": []}, "must be an object"),
        ([], "JSON object"),
    ],
)
def test_strict_validation(data, match):
    with pytest.raises(ValueError, match=match):
        HardeningConfig.from_dict(data)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.json")


def test_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_config(path)
