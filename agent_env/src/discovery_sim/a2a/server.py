from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Awaitable

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from discovery_sim.a2a.types import (
    AgentCard,
    SendMessageRequest,
    SendMessageResponse,
    parse_send_message_params,
)

SendMessageHandler = Callable[[SendMessageRequest], Awaitable[SendMessageResponse]]


def create_a2a_app(
    card: AgentCard,
    on_send_message: SendMessageHandler,
    mcp_tools_list: Callable[[], dict[str, Any]] | None = None,
    mcp_tools_call: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
) -> Starlette:
    card_body = card.to_dict()

    async def agent_card(_: Request) -> JSONResponse:
        return JSONResponse(card_body)

    async def jsonrpc(request: Request) -> Response:
        body = await request.json()
        if body.get("jsonrpc") != "2.0":
            return JSONResponse(
                {"jsonrpc": "2.0", "id": body.get("id"), "error": {"code": -32600, "message": "Invalid Request"}},
                status_code=400,
            )
        req_id = body.get("id")
        method = body.get("method")
        params = body.get("params") or {}

        if method == "SendMessage":
            try:
                send_req = parse_send_message_params(params)
            except (TypeError, KeyError) as exc:
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32602, "message": f"Invalid params: {exc}"},
                    }
                )
            try:
                response = await on_send_message(send_req)
            except ValueError as exc:
                task_id = str(uuid.uuid4())
                from discovery_sim.a2a.types import failed_task

                response = SendMessageResponse(task=failed_task(task_id, str(exc)))
            return JSONResponse(
                {"jsonrpc": "2.0", "id": req_id, "result": response.to_result_dict()}
            )

        if method in ("tools/list", "tools/call"):
            if mcp_tools_list is None or mcp_tools_call is None:
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32601, "message": "Method not found"},
                    }
                )
            if method == "tools/list":
                return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": mcp_tools_list()})
            try:
                result = await mcp_tools_call(params)
            except ValueError as exc:
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32000, "message": str(exc)},
                    }
                )
            return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": result})

        return JSONResponse(
            {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}
        )

    routes = [
        Route("/.well-known/agent-card.json", agent_card, methods=["GET"]),
        Route("/rpc", jsonrpc, methods=["POST"]),
    ]
    if mcp_tools_list is not None:
        routes.append(Route("/mcp", jsonrpc, methods=["POST"]))

    return Starlette(routes=routes)
