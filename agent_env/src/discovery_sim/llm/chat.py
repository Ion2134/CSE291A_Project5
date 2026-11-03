from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx

@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any]
    # Exactly what the model sent, and why it couldn't be parsed into ``arguments`` (if it couldn't).
    raw_arguments: str | None = None
    parse_error: str | None = None


@dataclass
class ChatTurn:
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    raw: dict[str, Any] | None = None
    error: str | None = None


# agent_env/.env (chat.py -> llm -> discovery_sim -> src -> agent_env)
DEFAULT_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"

# Default sampling temperature for every chat backend (and the scripted-mode LM planner).
# Agent-mode runs can override it with ``--temperature``.
DEFAULT_TEMPERATURE = 0.2

ChatBackend = Literal["lmstudio", "triton"]
CHAT_BACKENDS: tuple[str, ...] = ("lmstudio", "triton")


def make_chat_client(backend: ChatBackend, base_url: str | None = None, model: str | None = None,
                     temperature: float = DEFAULT_TEMPERATURE) -> ChatClient:
    if backend == "lmstudio":
        return LmStudioChat(base_url=base_url, model=model, temperature=temperature)
    if backend == "triton":
        return TritonAPIChat(base_url=base_url, model=model, temperature=temperature)
    raise ValueError(f"unknown chat backend {backend!r}; expected one of {CHAT_BACKENDS}")

def _parse_chat_completion(body: dict[str, Any]) -> ChatTurn:
    choices = body.get("choices") or []
    if not choices:
        return ChatTurn(content=None, error="empty choices")
    message = choices[0].get("message") or {}
    tool_calls_raw = message.get("tool_calls") or []
    tool_calls: list[ToolCallRequest] = []
    for tc in tool_calls_raw:
        fn = tc.get("function") or {}
        args_text = fn.get("arguments") or "{}"
        raw_args = args_text if isinstance(args_text, str) else json.dumps(args_text)
        parse_error: str | None = None
        try:
            parsed = json.loads(args_text) if isinstance(args_text, str) else args_text
        except json.JSONDecodeError as exc:
            parsed, parse_error = {}, f"arguments are not valid JSON: {exc}"
        if parse_error is None and not isinstance(parsed, dict):
            parsed, parse_error = {}, f"arguments must be a JSON object, got {type(parsed).__name__}"
        tool_calls.append(
            ToolCallRequest(
                id=str(tc.get("id", "")),
                name=str(fn.get("name", "")),
                arguments=parsed,
                raw_arguments=raw_args,
                parse_error=parse_error,
            )
        )
    return ChatTurn(
        content=message.get("content"),
        tool_calls=tool_calls,
        raw=body,
    )

def load_dotenv(path: Path = DEFAULT_ENV_PATH) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


class ChatClient(ABC):
    model_id: str = "unknown"

    @abstractmethod
    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ChatTurn:
        raise NotImplementedError


class OpenAICompatibleChat(ChatClient):
    """POSTs to an OpenAI-style ``/chat/completions`` endpoint; subclasses set the URL, model and auth."""

    temperature: float = DEFAULT_TEMPERATURE

    def _headers(self) -> dict[str, str]:
        return {}

    def _describe_http_error(self, exc: httpx.HTTPStatusError) -> str:
        return str(exc)

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ChatTurn:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        url = f"{self.base_url}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload, headers=self._headers())
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPStatusError as exc:
            return ChatTurn(content=None, error=self._describe_http_error(exc))
        except Exception as exc:
            return ChatTurn(content=None, error=str(exc))
        return _parse_chat_completion(body)


class LmStudioChat(OpenAICompatibleChat):
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self.base_url = (base_url or os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")).rstrip("/")
        self.model = model or os.environ.get("LMSTUDIO_MODEL", "local-model")
        self.timeout = timeout
        self.temperature = temperature
        self.model_id = self.model


class TritonAPIChat(OpenAICompatibleChat):
    """OpenAI-compatible TritonAI endpoint. Reads OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL,
    falling back to agent_env/.env (gitignored) for any that aren't already set."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        load_dotenv()
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://tritonai-api.ucsd.edu/v1")).rstrip("/")
        self.model = model or os.environ.get("OPENAI_MODEL", "api-muse-glimmer-30b")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout
        self.temperature = temperature
        self.model_id = self.model
        if not self.base_url or not self.api_key or not self.model:
            raise ValueError("TritonAPIChatClient needs OPENAI_BASE_URL, OPENAI_API_KEY and OPENAI_MODEL (env or .env)")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _describe_http_error(self, exc: httpx.HTTPStatusError) -> str:
        return f"{exc.response.status_code}: {exc.response.text[:500]}"


class ScriptChatClient(ChatClient):
    """Canned tool-call sequence for tests (no network)."""

    model_id = "script"

    def __init__(self, turns: list[ChatTurn]) -> None:
        self._turns = list(turns)
        self._index = 0

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ChatTurn:
        if self._index >= len(self._turns):
            return ChatTurn(content="Done.", tool_calls=[])
        turn = self._turns[self._index]
        self._index += 1
        return turn