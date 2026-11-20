# mcp-hardening

Middleware for MCP-shaped tool calls with two parts:

1. **Two-way per-call signing.** It uses an API key stored encrypted. Each call costs two HMAC-SHA256 checks, with no I/O and no asymmetric crypto.
2. **Turn-level telemetry.** It follows the OpenTelemetry GenAI semantic conventions and writes **append-only Parquet**.

The package doesn't depend on the simulator; `sim/agent_env` imports it.

## Install

```bash
sim/agent_env/.venv/bin/pip install -e "recovery/mcp_hardening[dev]"   # agent_env's venv, from the repo root
```

## Toggles: one JSON config

Each mechanism is its own block, and **any combination is valid**:

```json
{
  "version": 1,
  "request_auth":      {"enabled": true},
  "response_auth":     {"enabled": true},
  "replay_protection": {"enabled": true, "nonce_ttl_s": 300, "max_nonces": 100000},
  "freshness_check":   {"enabled": true, "window_s": 120},
  "telemetry":         {"enabled": true, "capture_content": true, "flush_rows": 500, "dir": "telemetry"}
}
```

| Mechanism | What it does when on | Failure reason |
|---|---|---|
| `request_auth` | The client signs `tool + arguments + nonce + ts`. The server rejects a missing or wrong signature **before the handler runs**. | `missing_envelope`, `unknown_kid`, `bad_request_sig` |
| `response_auth` | The real server signs `structuredContent` bound to the request nonce. The client rejects tampered, injected or stripped output. | `missing_response_sig`, `bad_response_sig` |
| `replay_protection` | The server rejects a nonce it has already accepted. Checked last, so forged calls can't burn a valid nonce. | `replayed_nonce` |
| `freshness_check` | The server rejects requests whose timestamp is outside `window_s`. | `stale_timestamp` |
| `telemetry` | Writes OTel GenAI spans to `<run>/<dir>/part-*.parquet`. `capture_content` controls whether messages and tool arguments/results are recorded. | n/a |

**Rules for blocks and validation:**
- A block you leave out keeps its default: everything off except telemetry.
- Validation is strict: unknown keys, wrong types and a bad `version` all raise an error that names the JSON path.

**Where the config comes from**, in order:
1. An explicit path.
2. `$MCP_HARDENING_CONFIG`.
3. Built-in defaults.

A path can also name a packaged preset: `default`, `all_on`, `all_off`, `request_only` or `response_only` (see `src/mcp_hardening/configs/`).

```bash
mcp-hardening show-config all_on        # validate + print a config
```

**Any failed check fails the tool call.** It raises `HardeningError`, a `ValueError` subclass with `.reason`. Over HTTP the failure comes back as a JSON-RPC error.

## Keys

The API key is never stored in plaintext:

```bash
mcp-hardening keygen     # prints two lines; put them in sim/agent_env/.env (gitignored)
# MCP_HARDENING_MASTER_KEY=...   Fernet key-encryption key
# MCP_HARDENING_API_KEY_ENC=...  the API key, Fernet-encrypted
```

**How the key is used:**
- The key is decrypted once when the runtime is built.
- HKDF-SHA256 derives separate request and response HMAC subkeys from it.
- The `kid` (first 8 hex characters of the key's sha256) identifies the key in every envelope.
- If hardening is on and no key is configured:
  - In-process simulation falls back to an **ephemeral** in-memory key and logs a warning.
  - HTTP agents refuse to start, because separate processes have to share the real key.

## Wire format

The envelope travels in MCP's `_meta` extension field:

```
request  params._meta.mcp_hardening = {v, kid, nonce, ts, sig?}
         sig = HMAC(k_req,  "v1|req|kid|tool|nonce|ts|canonical(arguments)")
response result._meta.mcp_hardening = {v, kid, nonce, sig}
         sig = HMAC(k_resp, "v1|resp|kid|tool|nonce|canonical(structuredContent)|isError")
```

`canonical()` means JSON with sorted keys and no whitespace.

## API

```python
from mcp_hardening import HardeningRuntime, load_config

rt = HardeningRuntime.build(load_config("all_on"))   # once per process/run
server = rt.wrap(real_tool_server)                   # server side: verify → handler → sign
params, env = rt.client.attach({"name": "get_balance", "arguments": {...}})   # client side
result = server.tools_call(params)
rt.client.verify_response("get_balance", env, result)
```

`rt.wrap()` returns the server unchanged when all four security toggles are off. Wrap the **real** server first, and put any proxy or attack layer outside it. That way anything changed after signing fails `bad_response_sig`.

## Telemetry

```python
from mcp_hardening import GenAITelemetry, read_telemetry

tel = GenAITelemetry(config.telemetry, out_dir)        # its own TracerProvider (not global)
with tel.agent_span("banking-agent", episode_id) as ep:
    with tel.chat_span(model, provider, input_messages=msgs) as s:
        tel.end_chat(s, response=openai_body)
    with tel.tool_span("get_balance", call_id, arguments=args) as s:
        tel.end_tool(s, result=output)                 # or error_type=..., error=...
    tel.end_agent(ep, completed=True)
tel.shutdown()                                          # flushes the last part file
table = read_telemetry(out_dir / "telemetry")          # all parts as one pyarrow Table
```

**Spans.** Each run produces one trace per episode: an `invoke_agent` span, with a `chat {model}` child for each LM turn and an `execute_tool {tool}` child for each tool call (rejected and malformed calls included).

**Columns.** The typed columns are:
- Span identity and timing: `trace_id`, `span_id`, `parent_span_id`, `name`, timestamps, `duration_ms`.
- Outcome: `status_code` (`OK` / `ERROR`), `status_description`.
- Promoted attributes:
  - `gen_ai.operation.name`, `gen_ai.conversation.id`
  - `gen_ai.request.model`, `gen_ai.provider.name`
  - `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`
  - `gen_ai.tool.name`, `gen_ai.tool.call.id`, `error.type`
  - `mcp_hardening.request.verified`, `mcp_hardening.response.verified`, `mcp_hardening.failure_reason`
  - `discovery_sim.*` labels

Every attribute is also kept in `attributes_json`.

**Append-only writes.** Each flush writes a **new** `part-NNNNN-xxxxxxxx.parquet`, created with exclusive create. Existing files are never rewritten. A new exporter in the same directory continues the sequence numbering.

The server middleware records its verification results on the **active span**, so they land on the matching `execute_tool` row.

## Tests

```bash
cd recovery/mcp_hardening && ../../sim/agent_env/.venv/bin/python -m pytest -q
```
