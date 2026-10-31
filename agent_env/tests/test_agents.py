from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from discovery_sim.agents.bank import BankAgent
from discovery_sim.agents.chatbot import ChatbotAgent
from discovery_sim.agents.payment import PaymentAgent
from discovery_sim.ledger.store import SyntheticLedger


def _send_message(client: TestClient, skill_id: str, arguments: dict | None = None, text: str | None = None):
    parts = []
    if text is not None:
        parts.append({"text": text})
    if arguments is not None:
        parts.append({"data": arguments, "mediaType": "application/json"})
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "SendMessage",
        "params": {
            "message": {
                "role": "user",
                "metadata": {"skillId": skill_id},
                "parts": parts,
            }
        },
    }
    return client.post("/rpc", json=payload)


def test_agent_card_lists_skills():
    agent = BankAgent("http://127.0.0.1:8002")
    client = TestClient(agent.starlette_app())
    card = client.get("/.well-known/agent-card.json").json()
    assert card["name"] == "Synth Bank"
    skill_ids = {s["id"] for s in card["skills"]}
    assert skill_ids == {"get_balance", "list_accounts", "get_transactions", "get_account_holder"}


def test_bank_get_balance():
    agent = BankAgent("http://127.0.0.1:8002")
    client = TestClient(agent.starlette_app())
    response = _send_message(client, "get_balance", {"account_id": "acct_synth_0031"})
    assert response.status_code == 200
    task = response.json()["result"]["task"]
    assert task["status"]["state"] == "completed"
    part = task["artifacts"][0]["parts"][0]["data"]
    assert part["balance_cents"] == 120_000


def test_payment_transfer_updates_shared_ledger():
    ledger = SyntheticLedger()
    bank = BankAgent("http://127.0.0.1:8002", ledger=ledger)
    payment = PaymentAgent("http://127.0.0.1:8003", ledger=ledger)
    bank_client = TestClient(bank.starlette_app())
    pay_client = TestClient(payment.starlette_app())

    before = _send_message(bank_client, "get_balance", {"account_id": "acct_synth_0031"}).json()
    before_cents = before["result"]["task"]["artifacts"][0]["parts"][0]["data"]["balance_cents"]

    transfer = _send_message(
        pay_client,
        "initiate_transfer",
        {
            "from_account_id": "acct_synth_0031",
            "payee_id": "payee_synth_0007",
            "amount_cents": 5000,
        },
    ).json()
    payment_id = transfer["result"]["task"]["artifacts"][0]["parts"][0]["data"]["payment_id"]

    after = _send_message(bank_client, "get_balance", {"account_id": "acct_synth_0031"}).json()
    after_cents = after["result"]["task"]["artifacts"][0]["parts"][0]["data"]["balance_cents"]
    assert after_cents == before_cents - 5000

    status = _send_message(pay_client, "get_payment_status", {"payment_id": payment_id}).json()
    assert status["result"]["task"]["artifacts"][0]["parts"][0]["data"]["status"] == "completed"


def test_chatbot_answer_stub():
    agent = ChatbotAgent("http://127.0.0.1:8001")
    client = TestClient(agent.starlette_app())
    response = _send_message(client, "chat.answer", text="Hello")
    assert response.status_code == 200
    data = response.json()["result"]["task"]["artifacts"][0]["parts"][0]["data"]
    assert data["echo"] == "Hello"
    assert "stub" in data["reply"].lower()


def test_mcp_tools_list_on_bank():
    agent = BankAgent("http://127.0.0.1:8002")
    client = TestClient(agent.starlette_app())
    response = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    assert response.status_code == 200
    tools = response.json()["result"]["tools"]
    assert any(t["name"] == "get_balance" for t in tools)
