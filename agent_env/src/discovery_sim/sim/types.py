from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


Hop = Literal["llm", "a2a", "mcp"]
Cause = Literal["none", "agent", "adversary"]


class OperatorId(str, Enum):
    BENIGN = "benign"
    LLM_WRONG_ACCOUNT = "llm_wrong_account"
    LLM_WRONG_AMOUNT = "llm_wrong_amount"
    A2A_CALL_INTERCEPTED = "a2a_call_intercepted"
    A2A_TOOLSET_MISMATCH = "a2a_toolset_mismatch"
    MCP_IO_MISMATCH = "mcp_io_mismatch"
    MCP_OUTPUT_INJECTION = "mcp_output_injection"
    HARD_NEGATIVE_REENCODE = "hard_negative_reencode"
    TAINT_PRIOR_OUTPUT = "taint_prior_output"


@dataclass
class BaseTask:
    task_id: str
    template: str
    user_message: str
    mandate: dict[str, Any]
    target_agent: Literal["bank", "payment"]
    tool_name: str
    correct_arguments: dict[str, Any]

    def describe(self) -> str:
        return f"{self.user_message} | mandate={self.mandate}"


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, Any]
    agent: Literal["bank", "payment"]


@dataclass
class AttackSpec:
    operator: OperatorId
    hop: Hop | None
    fault: str | None
    hard_negative: bool = False


@dataclass
class CallRecord:
    base_task: BaseTask
    call_index: int
    llm_call: ToolCall
    a2a_tool: str
    a2a_arguments: dict[str, Any]
    mcp_arguments: dict[str, Any]
    output: dict[str, Any]
    attack: AttackSpec
    hidden_adversary: bool = False
    effect_manifest: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class Labels:
    output_correct: bool
    cause: Cause
    verdict: Literal["allow", "deny", "review"]
    fault: str | None
    severity: Literal["info", "low", "high", "critical"]
    # Bookkeeping for us, not a monitor label: the call was wrong only because it used data an
    # earlier call's tampering corrupted (its cause is then ``adversary``).
    propagated: bool = False
    propagated_from: int | None = None


@dataclass
class LabeledExample:
    example_id: str
    input_state: dict[str, Any]
    labels: Labels
    meta: dict[str, Any]


@dataclass
class EpisodeOutcome:
    episode_id: str
    seed: int
    operator: OperatorId
    task_completed: bool
    task_correct: bool
    silent_wrong: bool
    hard_fail: bool
    absorbed: bool
    fault_agent: str
    fault_tool: str
    fault_hop: Hop | None
    fault_call_index: int
    failure_call_index: int | None
    records: list[CallRecord]
    examples: list[LabeledExample]
    attack_applied: bool = False
    malformed_calls: list[dict[str, Any]] = field(default_factory=list)
    model_failure: bool = False
