from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentSkill:
    id: str
    name: str
    description: str
    tags: list[str]
    examples: list[str] = field(default_factory=list)


@dataclass
class AgentInterface:
    url: str
    protocol_binding: str = "JSONRPC"
    protocol_version: str = "1.0"


@dataclass
class AgentCapabilities:
    streaming: bool = False
    push_notifications: bool = False


@dataclass
class AgentCard:
    name: str
    description: str
    version: str
    supported_interfaces: list[AgentInterface]
    capabilities: AgentCapabilities
    default_input_modes: list[str]
    default_output_modes: list[str]
    skills: list[AgentSkill]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "supportedInterfaces": [
                {
                    "url": i.url,
                    "protocolBinding": i.protocol_binding,
                    "protocolVersion": i.protocol_version,
                }
                for i in self.supported_interfaces
            ],
            "capabilities": {
                "streaming": self.capabilities.streaming,
                "pushNotifications": self.capabilities.push_notifications,
            },
            "defaultInputModes": self.default_input_modes,
            "defaultOutputModes": self.default_output_modes,
            "skills": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "tags": s.tags,
                    **({"examples": s.examples} if s.examples else {}),
                }
                for s in self.skills
            ],
        }


@dataclass
class Message:
    role: str
    parts: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    message_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "parts": self.parts}
        if self.metadata:
            out["metadata"] = self.metadata
        if self.message_id:
            out["messageId"] = self.message_id
        return out


@dataclass
class SendMessageRequest:
    message: Message
    configuration: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    id: str
    status: dict[str, Any]
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "artifacts": self.artifacts,
        }


@dataclass
class SendMessageResponse:
    task: Task | None = None
    message: Message | None = None

    def to_result_dict(self) -> dict[str, Any]:
        if self.task is not None:
            return {"task": self.task.to_dict()}
        if self.message is not None:
            return {"message": self.message.to_dict()}
        raise ValueError("SendMessageResponse must contain task or message")


def parse_send_message_params(params: dict[str, Any]) -> SendMessageRequest:
    raw = params.get("message") or {}
    parts = raw.get("parts") or []
    message = Message(
        role=raw.get("role", "user"),
        parts=parts,
        metadata=dict(raw.get("metadata") or {}),
        message_id=raw.get("messageId"),
    )
    return SendMessageRequest(
        message=message,
        configuration=dict(params.get("configuration") or {}),
        metadata=dict(params.get("metadata") or {}),
    )


def data_part(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"data": arguments, "mediaType": "application/json"}


def text_part(text: str) -> dict[str, Any]:
    return {"text": text}


def extract_json_from_message(message: Message) -> dict[str, Any] | None:
    for part in message.parts:
        if "data" in part:
            data = part["data"]
            if isinstance(data, dict):
                return data
    return None


def extract_text_from_message(message: Message) -> str:
    chunks: list[str] = []
    for part in message.parts:
        if "text" in part:
            chunks.append(str(part["text"]))
    return "\n".join(chunks)


def skill_id_from_message(message: Message) -> str | None:
    sid = message.metadata.get("skillId")
    if isinstance(sid, str) and sid:
        return sid
    return None


def completed_task(task_id: str, result: dict[str, Any]) -> Task:
    return Task(
        id=task_id,
        status={"state": "completed"},
        artifacts=[
            {
                "artifactId": f"{task_id}-result",
                "parts": [data_part(result)],
            }
        ],
    )


def failed_task(task_id: str, message: str) -> Task:
    return Task(
        id=task_id,
        status={"state": "failed", "message": message},
        artifacts=[],
    )
