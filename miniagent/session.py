"""Session + context management.

A Session is one independent conversation "window". The SessionManager keeps
many of them keyed by session_id, so user A's window-1 (weather + todo) and
window-2 (weekly report + todo) never interfere: separate history, separate
todo list, separate memory.

Context we keep per session:
  - history: the running list of turns (user / assistant-thought / observation).
    This is what gets replayed to the LLM so it can follow-up and remember
    prior state, including tool results.
  - todos: session-scoped tool state.
  - memory: durable one-line facts the agent chose to remember (see MEMORY
    recall/placement in the README). Always injected into the system prompt.
  - summary: a compressed digest of old history once we exceed the token budget.

Context compression (basic): when estimated tokens exceed the budget, we fold
the oldest turns into `summary` and keep only the most recent turns verbatim.
Memory facts are never dropped.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant", "observation"]


@dataclass
class Turn:
    role: Role
    content: str
    ts: float = field(default_factory=time.time)


def estimate_tokens(text: str) -> int:
    """Rough token estimate. ~4 chars/token for latin, but CJK is denser, so we
    use a conservative max(chars/4, chars/2-for-cjk-ish). Good enough to decide
    when to compress; we are not billing on it."""
    if not text:
        return 0
    # Count CJK codepoints separately (roughly 1 token each).
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    other = len(text) - cjk
    return cjk + max(1, other // 4)


@dataclass
class Session:
    session_id: str
    title: str = ""
    history: list[Turn] = field(default_factory=list)
    todos: list[str] = field(default_factory=list)
    memory: list[str] = field(default_factory=list)
    summary: str = ""
    created_at: float = field(default_factory=time.time)

    def add(self, role: Role, content: str) -> None:
        self.history.append(Turn(role, content))

    def remember(self, fact: str) -> None:
        fact = fact.strip()
        if fact and fact not in self.memory:
            self.memory.append(fact)

    # --- context assembly -----------------------------------------------------

    def estimated_tokens(self) -> int:
        base = estimate_tokens(self.summary) + sum(estimate_tokens(f) for f in self.memory)
        return base + sum(estimate_tokens(t.content) for t in self.history)

    def build_messages(self) -> list[dict[str, Any]]:
        """Render history into Anthropic-style messages.

        We collapse our 3 internal roles into the API's 2 roles:
          - user            -> user
          - assistant       -> assistant  (the model's thoughts/decisions)
          - observation     -> user       (tool results, prefixed so the model
                                           can tell them apart from real input)
        A leading summary (if any) is injected as a user preface.
        """
        msgs: list[dict[str, Any]] = []
        if self.summary:
            msgs.append({"role": "user", "content": f"[Earlier conversation summary]\n{self.summary}"})
        for t in self.history:
            if t.role == "assistant":
                msgs.append({"role": "assistant", "content": t.content})
            elif t.role == "observation":
                msgs.append({"role": "user", "content": f"[Tool result]\n{t.content}"})
            else:  # user
                msgs.append({"role": "user", "content": t.content})
        # The API requires the first message to be from the user; summary/obs
        # both map to user, and a session always starts with a user turn, so
        # this holds. Merge accidental consecutive same-role msgs defensively.
        return _merge_consecutive(msgs)


def _merge_consecutive(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for m in msgs:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append(dict(m))
    return merged


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str, title: str = "") -> Session:
        s = self._sessions.get(session_id)
        if s is None:
            s = Session(session_id=session_id, title=title or session_id)
            self._sessions[session_id] = s
        return s

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def ids(self) -> list[str]:
        return list(self._sessions)
