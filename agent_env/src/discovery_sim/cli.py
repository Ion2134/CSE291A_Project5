from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import uvicorn

from discovery_sim.agents.bank import BankAgent
from discovery_sim.agents.chatbot import ChatbotAgent
from discovery_sim.agents.payment import PaymentAgent
from discovery_sim.ledger.store import SyntheticLedger
from discovery_sim.sim.episode import run_episodes
from discovery_sim.sim.run_config import RunConfig


async def _serve_all(host: str, chatbot_port: int, bank_port: int, payment_port: int) -> None:
    ledger = SyntheticLedger()
    servers = [
        uvicorn.Server(
            uvicorn.Config(
                ChatbotAgent(f"http://{host}:{chatbot_port}").starlette_app(),
                host=host,
                port=chatbot_port,
                log_level="info",
            )
        ),
        uvicorn.Server(
            uvicorn.Config(
                BankAgent(f"http://{host}:{bank_port}", ledger=ledger).starlette_app(),
                host=host,
                port=bank_port,
                log_level="info",
            )
        ),
        uvicorn.Server(
            uvicorn.Config(
                PaymentAgent(f"http://{host}:{payment_port}", ledger=ledger).starlette_app(),
                host=host,
                port=payment_port,
                log_level="info",
            )
        ),
    ]
    await asyncio.gather(*(server.serve() for server in servers))


def _cmd_agents(args: argparse.Namespace) -> None:
    if args.agent == "all":
        asyncio.run(_serve_all(args.host, args.chatbot_port, args.bank_port, args.payment_port))
        return

    ports = {
        "chatbot": args.chatbot_port,
        "bank": args.bank_port,
        "payment": args.payment_port,
    }
    port = ports[args.agent]
    if args.agent == "chatbot":
        agent = ChatbotAgent(f"http://{args.host}:{port}")
    elif args.agent == "bank":
        agent = BankAgent(f"http://{args.host}:{port}")
    else:
        agent = PaymentAgent(f"http://{args.host}:{port}")

    uvicorn.run(agent.starlette_app(), host=args.host, port=port, log_level="info")


def _cmd_run(args: argparse.Namespace) -> None:
    config = RunConfig(
        episodes=args.episodes,
        seed=args.seed,
        mode=args.mode,
        max_agent_steps=args.max_agent_steps,
        llm_samples=args.llm_samples,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        multi_step=not args.single_step_only,
        overwrite_output=args.overwrite,
        concurrency=max(1, args.concurrency),
    )
    summary = asyncio.run(run_episodes(config, Path(args.out)))
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discovery simulation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    agents = sub.add_parser("agents", help="Start A2A HTTP agents")
    agents.add_argument(
        "--agent",
        choices=["chatbot", "bank", "payment", "all"],
        default="all",
    )
    agents.add_argument("--host", default="127.0.0.1")
    agents.add_argument("--chatbot-port", type=int, default=8001)
    agents.add_argument("--bank-port", type=int, default=8002)
    agents.add_argument("--payment-port", type=int, default=8003)
    agents.set_defaults(func=_cmd_agents)

    run = sub.add_parser("run", help="Run seeded attack episodes and write dataset artifacts")
    run.add_argument("--episodes", type=int, default=50)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out", default="data/runs/0")
    run.add_argument(
        "--mode",
        choices=["agent", "scripted"],
        default="agent",
        help="agent: LM tool-calling loop; scripted: template-driven harness (legacy)",
    )
    run.add_argument("--max-agent-steps", type=int, default=8, help="Max model↔tool rounds per episode")
    run.add_argument(
        "--llm-samples",
        type=int,
        default=0,
        help="First N episodes plan tool calls via LM Studio (benign only; mismatches dropped)",
    )
    run.add_argument("--llm-base-url", default=None, help="OpenAI-compatible base URL (default LM Studio)")
    run.add_argument("--llm-model", default=None, help="Model id (default LMSTUDIO_MODEL env or local-model)")
    run.add_argument(
        "--single-step-only",
        action="store_true",
        help="Disable multi-step scenarios (balance then transfer)",
    )
    run.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace artifacts in --out if they already exist (default: create a new timestamped directory)",
    )
    run.add_argument(
        "--concurrency",
        type=int,
        default=1,
        metavar="N",
        help="Run up to N episodes in parallel (LM inference). Default 1 (sequential).",
    )
    run.set_defaults(func=_cmd_run)

    args = parser.parse_args()
    args.func(args)


def agents_main() -> None:
    """Backward-compatible entry point for discovery-sim-agents."""
    import sys

    sys.argv = [sys.argv[0], "agents", *sys.argv[1:]]
    main()


if __name__ == "__main__":
    main()
