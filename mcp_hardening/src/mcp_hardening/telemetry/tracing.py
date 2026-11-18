"""Turn-level OpenTelemetry GenAI spans.

Span tree per episode::

    invoke_agent {agent}             gen_ai.operation.name=invoke_agent, gen_ai.conversation.id
      ├─ chat {model}                one per LM turn
      ├─ execute_tool {tool}         one per tool call (incl. malformed / rejected calls)
      └─ ...

Each run gets its own ``TracerProvider`` (no global provider), so concurrent runs and tests don't
share exporters. When telemetry is disabled every helper still works but records nothing.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from mcp_hardening.config import TelemetryConfig
from mcp_hardening.telemetry import semconv
from mcp_hardening.telemetry.exporter import ParquetSpanExporter


def _json(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def _set(span: Span, attrs: dict[str, Any]) -> None:
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, (dict, list)) and not (
            isinstance(value, list) and all(isinstance(v, str) for v in value)
        ):
            value = _json(value)
        span.set_attribute(key, value)


class GenAITelemetry:
    def __init__(
        self,
        config: TelemetryConfig,
        out_dir: str | Path | None = None,
        service_name: str = "discovery-sim",
    ) -> None:
        self.config = config
        self.exporter: ParquetSpanExporter | None = None
        self._provider: TracerProvider | None = None
        if config.enabled and out_dir is not None:
            self.exporter = ParquetSpanExporter(Path(out_dir) / config.dir, flush_rows=config.flush_rows)
            self._provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
            # Synchronous: every finished span goes straight into the exporter's buffer. The
            # BatchSpanProcessor's bounded queue (2048) silently drops spans when a fast run outpaces
            # its background thread, which is unacceptable for an audit log. Batching into part files
            # is done by the exporter itself (flush_rows), so this costs only a list append per span.
            self._provider.add_span_processor(SimpleSpanProcessor(self.exporter))
            self.tracer: trace.Tracer = self._provider.get_tracer("mcp_hardening.telemetry")
        else:
            self.tracer = trace.NoOpTracer()

    @property
    def enabled(self) -> bool:
        return self._provider is not None

    @property
    def capture_content(self) -> bool:
        return self.enabled and self.config.capture_content

    @property
    def out_dir(self) -> Path | None:
        return self.exporter.out_dir if self.exporter else None

    def flush(self) -> None:
        """Push queued spans to the exporter, then write them out as a new part file."""
        if self._provider is not None:
            self._provider.force_flush()
            assert self.exporter is not None
            self.exporter.force_flush()

    def shutdown(self) -> None:
        """Flush buffered spans to a final part file. Safe to call more than once."""
        if self._provider is not None:
            self._provider.shutdown()
            self._provider = None

    # ---- spans -------------------------------------------------------------------------------

    @contextmanager
    def agent_span(
        self, agent_name: str, conversation_id: str, attributes: dict[str, Any] | None = None
    ) -> Iterator[Span]:
        with self.tracer.start_as_current_span(
            f"{semconv.OP_INVOKE_AGENT} {agent_name}", kind=SpanKind.INTERNAL
        ) as span:
            _set(
                span,
                {
                    semconv.GEN_AI_OPERATION_NAME: semconv.OP_INVOKE_AGENT,
                    semconv.GEN_AI_AGENT_NAME: agent_name,
                    semconv.GEN_AI_CONVERSATION_ID: conversation_id,
                    **(attributes or {}),
                },
            )
            yield span

    def end_agent(
        self,
        span: Span,
        *,
        completed: bool,
        error: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Close out an episode: OK if the agent finished its loop, ERROR (with why) if it didn't."""
        if attributes:
            _set(span, attributes)
        if completed:
            span.set_status(Status(StatusCode.OK))
        else:
            _set(span, {semconv.ERROR_TYPE: "episode_incomplete"})
            span.set_status(Status(StatusCode.ERROR, (error or "agent did not complete the task")[:500]))

    @contextmanager
    def chat_span(
        self,
        model: str,
        provider: str,
        *,
        temperature: float | None = None,
        input_messages: list[dict[str, Any]] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        with self.tracer.start_as_current_span(f"{semconv.OP_CHAT} {model}", kind=SpanKind.CLIENT) as span:
            attrs: dict[str, Any] = {
                semconv.GEN_AI_OPERATION_NAME: semconv.OP_CHAT,
                semconv.GEN_AI_PROVIDER_NAME: provider,
                semconv.GEN_AI_REQUEST_MODEL: model,
                semconv.GEN_AI_REQUEST_TEMPERATURE: temperature,
                **(attributes or {}),
            }
            if self.capture_content and input_messages is not None:
                attrs[semconv.GEN_AI_INPUT_MESSAGES] = input_messages
            _set(span, attrs)
            yield span

    def end_chat(
        self,
        span: Span,
        *,
        response: dict[str, Any] | None = None,
        output_message: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Record an OpenAI-style chat completion body (usage, finish reasons) or an error."""
        if error:
            _set(span, {semconv.ERROR_TYPE: "chat_error"})
            span.set_status(Status(StatusCode.ERROR, error[:500]))
            return
        body = response or {}
        usage = body.get("usage") or {}
        choices = body.get("choices") or []
        _set(
            span,
            {
                semconv.GEN_AI_RESPONSE_ID: body.get("id"),
                semconv.GEN_AI_RESPONSE_MODEL: body.get("model"),
                semconv.GEN_AI_RESPONSE_FINISH_REASONS: [
                    str(c.get("finish_reason")) for c in choices if c.get("finish_reason")
                ]
                or None,
                semconv.GEN_AI_USAGE_INPUT_TOKENS: usage.get("prompt_tokens", usage.get("input_tokens")),
                semconv.GEN_AI_USAGE_OUTPUT_TOKENS: usage.get("completion_tokens", usage.get("output_tokens")),
            },
        )
        if self.capture_content and output_message is not None:
            _set(span, {semconv.GEN_AI_OUTPUT_MESSAGES: [output_message]})
        span.set_status(Status(StatusCode.OK))

    @contextmanager
    def tool_span(
        self,
        tool_name: str,
        call_id: str | None,
        *,
        arguments: Any = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        with self.tracer.start_as_current_span(
            f"{semconv.OP_EXECUTE_TOOL} {tool_name}", kind=SpanKind.INTERNAL
        ) as span:
            attrs: dict[str, Any] = {
                semconv.GEN_AI_OPERATION_NAME: semconv.OP_EXECUTE_TOOL,
                semconv.GEN_AI_TOOL_NAME: tool_name,
                semconv.GEN_AI_TOOL_CALL_ID: call_id or None,
                semconv.GEN_AI_TOOL_TYPE: "function",
                **(attributes or {}),
            }
            if self.capture_content and arguments is not None:
                attrs[semconv.GEN_AI_TOOL_CALL_ARGUMENTS] = arguments if isinstance(arguments, str) else _json(arguments)
            _set(span, attrs)
            yield span

    def end_tool(
        self,
        span: Span,
        *,
        result: Any = None,
        error_type: str | None = None,
        error: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if attributes:
            _set(span, attributes)
        if self.capture_content and result is not None:
            _set(span, {semconv.GEN_AI_TOOL_CALL_RESULT: _json(result)})
        if error_type or error:
            _set(span, {semconv.ERROR_TYPE: error_type or "tool_error"})
            span.set_status(Status(StatusCode.ERROR, (error or error_type or "")[:500]))
        else:
            span.set_status(Status(StatusCode.OK))
