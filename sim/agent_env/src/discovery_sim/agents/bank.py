from __future__ import annotations

from discovery_sim.agents.base import McpBackedAgent
from discovery_sim.ledger.store import SyntheticLedger
from discovery_sim.mcp.server import McpToolServer, bank_mcp_server


class BankAgent(McpBackedAgent):
    agent_name = "Synth Bank"
    agent_description = "Read-only banking data over synthetic accounts (MCP-backed)."

    def __init__(
        self,
        public_base_url: str,
        ledger: SyntheticLedger | None = None,
        mcp: McpToolServer | None = None,
    ) -> None:
        self._ledger = ledger or SyntheticLedger()
        mcp = mcp or bank_mcp_server(self._ledger)
        super().__init__(public_base_url, mcp, skill_tag="bank")

    @property
    def ledger(self) -> SyntheticLedger:
        return self._ledger
