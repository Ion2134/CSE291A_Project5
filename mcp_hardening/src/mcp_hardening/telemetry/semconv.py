"""Attribute names. ``gen_ai.*`` / ``error.type`` follow the OpenTelemetry GenAI semantic conventions;
``mcp_hardening.*`` and ``discovery_sim.*`` are project extensions (namespaced per the OTel guidance)."""

from __future__ import annotations

# --- OpenTelemetry GenAI semantic conventions ---
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
GEN_AI_TOOL_TYPE = "gen_ai.tool.type"
GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"
ERROR_TYPE = "error.type"

# gen_ai.operation.name values
OP_INVOKE_AGENT = "invoke_agent"
OP_CHAT = "chat"
OP_EXECUTE_TOOL = "execute_tool"

# --- mcp_hardening extensions ---
HARDENING_KID = "mcp_hardening.kid"
HARDENING_REQUEST_SIGNED = "mcp_hardening.request.signed"
HARDENING_REQUEST_VERIFIED = "mcp_hardening.request.verified"
HARDENING_RESPONSE_VERIFIED = "mcp_hardening.response.verified"
HARDENING_FAILURE_REASON = "mcp_hardening.failure_reason"
HARDENING_FAILURE_STAGE = "mcp_hardening.failure_stage"

# --- discovery_sim extensions (set by the simulator) ---
SIM_EPISODE_INDEX = "discovery_sim.episode.index"
SIM_TASK_ID = "discovery_sim.task.id"
SIM_OPERATOR = "discovery_sim.operator"
SIM_HOP = "discovery_sim.hop"
SIM_CALL_INDEX = "discovery_sim.call_index"
SIM_STEP = "discovery_sim.step"
SIM_MALFORMED = "discovery_sim.malformed"
SIM_OUTPUT_CORRECT = "discovery_sim.label.output_correct"
SIM_CAUSE = "discovery_sim.label.cause"
SIM_TASK_COMPLETED = "discovery_sim.task_completed"
SIM_TASK_CORRECT = "discovery_sim.task_correct"
