from __future__ import annotations

from discovery_sim.agents.base import McpBackedAgent
from discovery_sim.ledger.store import SyntheticLedger
from discovery_sim.mcp.server import McpToolServer, payment_mcp_server


class PaymentAgent(McpBackedAgent):
    agent_name = "Synth Payment"
    agent_description = "Payment and payee operations over the synthetic ledger (MCP-backed)."

    def __init__(
        self,
        public_base_url: str,
        ledger: SyntheticLedger | None = None,
        mcp: McpToolServer | None = None,
    ) -> None:
        self._ledger = ledger or SyntheticLedger()
        mcp = mcp or payment_mcp_server(self._ledger)
        super().__init__(public_base_url, mcp, skill_tag="payment")

    @property
    def ledger(self) -> SyntheticLedger:
        return self._ledger
