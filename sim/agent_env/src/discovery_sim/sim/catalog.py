from __future__ import annotations

import uuid

from discovery_sim.sim.task_defs import single_step_base_tasks
from discovery_sim.sim.types import BaseTask, ToolCall


def task_templates() -> list[BaseTask]:
    return single_step_base_tasks()


def tool_call_for_task(task: BaseTask) -> ToolCall:
    return ToolCall(
        tool_name=task.tool_name,
        arguments=dict(task.correct_arguments),
        agent=task.target_agent,
    )


def new_task_id() -> str:
    return f"task_{uuid.uuid4().hex[:8]}"
