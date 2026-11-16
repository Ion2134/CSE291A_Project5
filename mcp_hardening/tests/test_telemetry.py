from __future__ import annotations

from mcp_hardening import GenAITelemetry, HardeningRuntime, Keyring, load_config, read_telemetry, semconv
from mcp_hardening.config import TelemetryConfig
from mcp_hardening.errors import HardeningError


def _names(table):
    return table.column("name").to_pylist()


def test_append_only_parts(tmp_path):
    tel = GenAITelemetry(TelemetryConfig(flush_rows=1000), tmp_path)
    with tel.agent_span("bank-agent", "ep_1"):
        with tel.chat_span("m", "lmstudio", temperature=0.2, input_messages=[{"role": "user", "content": "hi"}]) as s:
            tel.end_chat(s, response={"usage": {"prompt_tokens": 5, "completion_tokens": 2}, "choices": [{"finish_reason": "tool_calls"}]})
    tel.flush()
    first = sorted(tmp_path.joinpath("telemetry").glob("part-*.parquet"))
    assert len(first) == 1
    before = first[0].read_bytes()

    with tel.agent_span("bank-agent", "ep_2"):
        with tel.tool_span("get_balance", "call_1", arguments={"account_id": "a"}) as s:
            tel.end_tool(s, error_type="mcp_hardening", error="mcp_hardening:bad_request_sig")
    tel.shutdown()
    tel.shutdown()  # idempotent

    parts = sorted(tmp_path.joinpath("telemetry").glob("part-*.parquet"))
    assert len(parts) == 2
    assert parts[0].read_bytes() == before  # earlier part never rewritten

    table = read_telemetry(tmp_path / "telemetry")
    rows = table.to_pylist()
    assert len(rows) == 4
    chat = next(r for r in rows if r[semconv.GEN_AI_OPERATION_NAME] == "chat")
    assert chat[semconv.GEN_AI_USAGE_INPUT_TOKENS] == 5 and chat["status_code"] == "OK"
    tool = next(r for r in rows if r[semconv.GEN_AI_OPERATION_NAME] == "execute_tool")
    assert tool["status_code"] == "ERROR" and tool[semconv.ERROR_TYPE] == "mcp_hardening"
    agent = next(r for r in rows if r["span_id"] == tool["parent_span_id"])
    assert agent[semconv.GEN_AI_CONVERSATION_ID] == "ep_2"


def test_new_exporter_continues_sequence(tmp_path):
    for _ in range(2):
        tel = GenAITelemetry(TelemetryConfig(), tmp_path)
        with tel.agent_span("a", "ep"):
            pass
        tel.shutdown()
    names = sorted(p.name for p in tmp_path.joinpath("telemetry").glob("part-*.parquet"))
    assert [n[:10] for n in names] == ["part-00000", "part-00001"]


def test_disabled_writes_nothing(tmp_path):
    tel = GenAITelemetry(TelemetryConfig(enabled=False), tmp_path)
    with tel.agent_span("a", "ep"), tel.tool_span("t", "c") as s:
        tel.end_tool(s, result={"x": 1})
    tel.shutdown()
    assert not tmp_path.joinpath("telemetry").exists()
    assert read_telemetry(tmp_path / "telemetry").num_rows == 0


def test_content_capture_toggle(tmp_path):
    tel = GenAITelemetry(TelemetryConfig(capture_content=False), tmp_path)
    with tel.tool_span("t", "c", arguments={"secret": 1}) as s:
        tel.end_tool(s, result={"x": 1})
    tel.shutdown()
    row = read_telemetry(tmp_path / "telemetry").to_pylist()[0]
    assert "secret" not in row["attributes_json"]


def test_middleware_records_on_active_span(tmp_path):
    tel = GenAITelemetry(TelemetryConfig(), tmp_path)
    rt = HardeningRuntime.build(load_config("all_on"), Keyring.ephemeral_key())

    class Server:
        def tools_list(self):
            return {"tools": []}

        def tools_call(self, params):
            return {"structuredContent": {"ok": True}, "isError": False}

    with tel.tool_span("get_balance", "c1") as s:
        signed, env = rt.client.attach({"name": "get_balance", "arguments": {}})
        result = rt.wrap(Server()).tools_call(signed)
        rt.client.verify_response("get_balance", env, result)
        tel.end_tool(s, result=result["structuredContent"])
    with tel.tool_span("get_balance", "c2") as s:
        try:
            rt.wrap(Server()).tools_call({"name": "get_balance", "arguments": {}})
        except HardeningError as exc:
            tel.end_tool(s, error_type="mcp_hardening", error=str(exc))
    tel.shutdown()
    rows = {r[semconv.GEN_AI_TOOL_CALL_ID]: r for r in read_telemetry(tmp_path / "telemetry").to_pylist()}
    assert rows["c1"][semconv.HARDENING_REQUEST_VERIFIED] is True
    assert rows["c1"][semconv.HARDENING_RESPONSE_VERIFIED] is True
    assert rows["c2"][semconv.HARDENING_REQUEST_VERIFIED] is False
    assert rows["c2"][semconv.HARDENING_FAILURE_REASON] == "missing_envelope"
