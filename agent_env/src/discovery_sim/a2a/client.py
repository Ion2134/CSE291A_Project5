from __future__ import annotations

import copy
from typing import Any

import httpx
from mcp_hardening import ENVELOPE_KEY, HardeningError, HardeningRuntime, failure_reason_from_message


class A2AClient:
    """Minimal JSON-RPC client for SendMessage, MCP tools/call, and agent card fetch.

    With a ``runtime`` whose security is enabled, every call is signed by the client-side
    middleware and every result's signature is checked; a rejected call raises ``HardeningError``.
    """

    def __init__(
        self,
        rpc_url: str,
        timeout: float = 30.0,
        runtime: HardeningRuntime | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.rpc_url = rpc_url.rstrip("/")
        if not self.rpc_url.endswith("/rpc"):
            self.rpc_url = f"{self.rpc_url}/rpc"
        self.base_url = self.rpc_url[: -len("/rpc")]
        self._timeout = timeout
        self._runtime = runtime if runtime is not None and runtime.enabled else None
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self._timeout, transport=self._transport)

    async def get_agent_card(self) -> dict[str, Any]:
        url = f"{self.base_url}/.well-known/agent-card.json"
        async with self._client() as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    async def _rpc(self, url: str, method: str, params: dict[str, Any]) -> dict[str, Any]:
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        async with self._client() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            body = response.json()
        if "error" in body:
            reason = failure_reason_from_message(str(body["error"].get("message", "")))
            if reason:
                raise HardeningError(reason, "rejected by server")
            raise RuntimeError(body["error"])
        return body["result"]

    async def send_message(
        self,
        message: dict[str, Any],
        configuration: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        envelope = None
        tool = None
        if self._runtime is not None:
            message = copy.deepcopy(message)
            metadata = message.setdefault("metadata", {})
            tool = str(metadata.get("skillId", ""))
            arguments = next(
                (p["data"] for p in message.get("parts", []) if isinstance(p.get("data"), dict)), {}
            )
            envelope = self._runtime.client.sign(tool, arguments)
            metadata[ENVELOPE_KEY] = envelope
        result = await self._rpc(
            self.rpc_url, "SendMessage", {"message": message, "configuration": configuration or {}}
        )
        if self._runtime is not None:
            task = result.get("task") or {}
            status = task.get("status") or {}
            reason = failure_reason_from_message(str(status.get("message", "")))
            if status.get("state") == "failed" and reason:
                raise HardeningError(reason, "rejected by server")
            artifacts = task.get("artifacts") or []
            if artifacts:
                artifact = artifacts[0]
                data = next((p.get("data") for p in artifact.get("parts", []) if "data" in p), {})
                self._runtime.client.verify_response(
                    str(tool),
                    envelope,
                    {"structuredContent": data, "isError": False, "_meta": artifact.get("metadata")},
                )
        return result

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """MCP ``tools/call`` against the agent's ``/mcp`` endpoint (signed + verified when enabled)."""
        params: dict[str, Any] = {"name": name, "arguments": dict(arguments or {})}
        envelope = None
        if self._runtime is not None:
            params, envelope = self._runtime.client.attach(params)
        result = await self._rpc(f"{self.base_url}/mcp", "tools/call", params)
        if self._runtime is not None:
            self._runtime.client.verify_response(name, envelope, result)
        return result
