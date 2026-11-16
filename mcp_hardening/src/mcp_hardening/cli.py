from __future__ import annotations

import argparse
import json
import sys

from mcp_hardening.config import load_config, preset_names
from mcp_hardening.keys import API_KEY_ENC_ENV, MASTER_KEY_ENV, Keyring, generate_encrypted_api_key


def _cmd_keygen(_: argparse.Namespace) -> None:
    master, enc = generate_encrypted_api_key()
    kid = Keyring.from_encrypted(master, enc).kid
    print(f"# mcp-hardening key (kid={kid}). Add to agent_env/.env (gitignored):")
    print(f"{MASTER_KEY_ENV}={master}")
    print(f"{API_KEY_ENC_ENV}={enc}")


def _cmd_show_config(args: argparse.Namespace) -> None:
    cfg = load_config(args.path)
    print(json.dumps(cfg.to_dict(), indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mcp-hardening")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("keygen", help="Generate a master key and an encrypted API key").set_defaults(func=_cmd_keygen)
    show = sub.add_parser("show-config", help=f"Validate and print a config (file path or preset: {preset_names()})")
    show.add_argument("path", nargs="?", default=None)
    show.set_defaults(func=_cmd_show_config)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
