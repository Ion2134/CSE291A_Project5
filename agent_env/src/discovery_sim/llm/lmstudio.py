from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

from discovery_sim.llm.base import LlmPlanner, PlannedToolCall, PlannerResult
from discovery_sim.sim.context import AgentContext


SYSTEM_PROMPT = """You plan one banking tool call at a time.
Respond with a single JSON object only, no markdown:
{"agent":"bank"|"payment","tool":"<tool_name>","arguments":{...}}

Bank tools: get_balance, list_accounts, get_transactions, get_account_holder
Payment tools: initiate_transfer, get_payment_status, cancel_payment, add_payee
Use account ids like acct_synth_0031 and payee ids like payee_synth_0007.
Amounts are in cents in arguments."""


class LmStudioPlanner(LlmPlanner):
    model_id = "lmstudio"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")).rstrip("/")
        self.model = model or os.environ.get("LMSTUDIO_MODEL", "local-model")
        self.model_id = self.model
        self.timeout = timeout

    async def plan(self, context: AgentContext, step_hint: dict[str, Any] | None = None) -> PlannerResult:
        user_block = _format_context(context, step_hint)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_block},
            ],
            "temperature": 0.2,
        }
        url = f"{self.base_url}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                body = response.json()
        except Exception as exc:
            return PlannerResult(call=None, error=str(exc), model_id=self.model)

        choices = body.get("choices") or []
        if not choices:
            return PlannerResult(call=None, error="empty choices", model_id=self.model)
        raw = choices[0].get("message", {}).get("content") or ""
        parsed = _parse_json_object(raw)
        if parsed is None:
            return PlannerResult(call=None, raw_text=raw, error="failed to parse JSON", model_id=self.model)
        try:
            call = PlannedToolCall(
                agent=parsed["agent"],
                tool_name=parsed["tool"],
                arguments=dict(parsed.get("arguments") or {}),
            )
        except (KeyError, TypeError) as exc:
            return PlannerResult(call=None, raw_text=raw, error=str(exc), model_id=self.model)
        return PlannerResult(call=call, raw_text=raw, model_id=self.model)


def _format_context(context: AgentContext, step_hint: dict[str, Any] | None) -> str:
    lines = [
        f"User request: {context.user_message}",
        f"Trusted mandate (authoritative): {json.dumps(context.trusted)}",
    ]
    if context.prior_calls:
        lines.append("Prior tool calls in this session:")
        for i, prior in enumerate(context.prior_calls):
            lines.append(
                f"  [{i}] {prior.llm_call.tool_name} model_call={prior.llm_call.arguments} "
                f"output={prior.output}"
            )
    if step_hint:
        lines.append(f"Hint for this step only (ground truth for tests): {json.dumps(step_hint)}")
    return "\n".join(lines)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
