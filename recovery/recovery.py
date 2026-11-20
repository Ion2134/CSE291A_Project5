"""Recovery: decide whether to undo a tool call, and roll the recorded state back to just before it.

The rules (``RecoveryPolicy.decide``), in order:

    hardening rejected the call before the tool ran  -> no rollback  ("blocked_before_tool": nothing changed)
    hardening rejected the response after it ran     -> roll back    ("response_sig_failed")
    the classifiers' majority flags the call         -> roll back    ("classifiers")
    otherwise                                         -> no rollback ("not_flagged" / "no_classifiers")

Past ``max_recoveries`` rollbacks in an episode, a call that would be rolled back is not
("cap_reached"), so an agent repeating the same call cannot loop forever.

``rollback`` cuts a state store (``state_db.StateStore``, duck-typed) back to a step and returns the
ledger and conversation as of that step; the caller puts them back in place. This module knows nothing
about the simulator and uses only the standard library.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

# The tool reply the agent gets instead of the rolled-back call's output.
BLOCKED_REPLY = {"error": "blocked by the monitor; state restored to before this call"}

REQUEST_STAGE = "request"    # hardening refused the call before the tool ran
RESPONSE_STAGE = "response"  # the tool ran, then the response failed its signature check


class StateStore(Protocol):
    def delete_after(self, episode_id: str, seq: int) -> int: ...
    def ledger_at(self, episode_id: str, seq: int) -> dict: ...
    def messages_at(self, episode_id: str, seq: int) -> list[dict]: ...


@dataclass
class Decision:
    recover: bool
    reason: str
    flagged_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"recover": self.recover, "reason": self.reason, "flagged_by": list(self.flagged_by)}


@dataclass
class RecoveryPolicy:
    """One per episode: applies the rules and counts the rollbacks it allowed."""

    max_recoveries: int = 2
    recoveries: int = 0

    def decide(self, hardening_failure_stage: str | None, classifier_decision: dict | None) -> Decision:
        """``hardening_failure_stage``: ``"request"``, ``"response"`` or None (no rejection).
        ``classifier_decision``: ``{"recover", "flagged_by"}`` from the classifiers' vote, or None."""
        if hardening_failure_stage == REQUEST_STAGE:
            return Decision(False, "blocked_before_tool")
        if hardening_failure_stage == RESPONSE_STAGE:
            decision = Decision(True, "response_sig_failed")
        elif classifier_decision is None:
            return Decision(False, "no_classifiers")
        elif classifier_decision.get("recover"):
            decision = Decision(True, "classifiers", list(classifier_decision.get("flagged_by", [])))
        else:
            return Decision(False, "not_flagged", list(classifier_decision.get("flagged_by", [])))
        if self.recoveries >= self.max_recoveries:
            return Decision(False, "cap_reached", decision.flagged_by)
        self.recoveries += 1
        return decision


@dataclass
class Rollback:
    ledger: dict          # the ledger as of the rollback point (the four collections)
    messages: list[dict]  # the conversation as of the rollback point
    deleted_steps: int


def rollback(store: StateStore, episode_id: str, to_seq: int) -> Rollback:
    """Delete everything ``episode_id`` recorded after step ``to_seq`` and return its state as of that step."""
    deleted = store.delete_after(episode_id, to_seq)
    return Rollback(store.ledger_at(episode_id, to_seq), store.messages_at(episode_id, to_seq), deleted)
