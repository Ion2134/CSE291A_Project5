"""The chat client waits and retries when the API rate-limits or flakes, and says so when it gives up."""

import asyncio

import httpx
import pytest

from discovery_sim.llm import chat
from discovery_sim.llm.chat import LmStudioChat

OK = {"choices": [{"message": {"content": "hi"}}]}


@pytest.fixture
def api(monkeypatch):
    """Serve ``api.responses`` in order; record the requests and the waits instead of sleeping."""

    class Api:
        responses: list[httpx.Response | Exception] = []
        requests = 0
        waits: list[float] = []

    def handle(request):
        Api.requests += 1
        response = Api.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    real_client = httpx.AsyncClient
    monkeypatch.setattr(chat.httpx, "AsyncClient",
                        lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs))

    async def no_wait(seconds):
        Api.waits.append(seconds)

    monkeypatch.setattr(chat.OpenAICompatibleChat, "_sleep", staticmethod(no_wait))
    return Api


def complete():
    return asyncio.run(LmStudioChat(base_url="http://test/v1", model="m").complete([], []))


def test_rate_limit_waits_retry_after_then_succeeds(api):
    limited = httpx.Response(429, headers={"retry-after": "60"}, text="Rate limit exceeded")
    api.responses = [limited, limited, httpx.Response(200, json=OK)]
    turn = complete()
    assert turn.error is None and turn.content == "hi"
    assert api.requests == 3 and api.waits == [60, 60]


def test_expired_key_is_not_retried(api, capsys):
    api.responses = [httpx.Response(401, text="Expired Key")]
    turn = complete()
    assert turn.error and api.requests == 1 and api.waits == []
    assert "API call failed" in capsys.readouterr().err


def test_server_errors_back_off_then_give_up_loudly(api, capsys):
    api.responses = [httpx.Response(500, text="boom") for _ in range(chat.OpenAICompatibleChat.max_retries + 1)]
    turn = complete()
    assert turn.error and api.requests == chat.OpenAICompatibleChat.max_retries + 1
    assert api.waits == [2, 4, 8, 16, 32]
    assert "API call failed" in capsys.readouterr().err


def test_dropped_connection_is_retried(api):
    api.responses = [httpx.ConnectError("refused"), httpx.Response(200, json=OK)]
    assert complete().content == "hi"
    assert api.requests == 2 and api.waits == [2]
