from __future__ import annotations

from typing import Any

import httpx


class A2AClient:
    """Minimal JSON-RPC client for SendMessage and agent card fetch."""

    def __init__(self, rpc_url: str, timeout: float = 30.0) -> None:
        self.rpc_url = rpc_url.rstrip("/")
        if not self.rpc_url.endswith("/rpc"):
            self.rpc_url = f"{self.rpc_url}/rpc"
        self._timeout = timeout

    async def get_agent_card(self) -> dict[str, Any]:
        base = self.rpc_url[: -len("/rpc")]
        url = f"{base}/.well-known/agent-card.json"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    async def send_message(
        self,
        message: dict[str, Any],
        configuration: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "SendMessage",
            "params": {"message": message, "configuration": configuration or {}},
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(self.rpc_url, json=payload)
            response.raise_for_status()
            body = response.json()
        if "error" in body:
            raise RuntimeError(body["error"])
        return body["result"]
