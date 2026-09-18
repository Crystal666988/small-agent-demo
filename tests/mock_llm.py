"""A scripted mock LLM for deterministic tests.

Two modes:
  - ScriptedLLM(queue): pops a canned raw-text reply per complete() call,
    regardless of input. Simplest for linear flows.
  - RoutedLLM(fn): calls fn(system, messages) -> raw text, so a test can react
    to what the agent actually sent (e.g. return a tool result then a final).

Neither touches the network, so tests run offline and fast.
"""
from __future__ import annotations

from typing import Any, Callable


class ScriptedLLM:
    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    def complete(self, system: str, messages: list[dict[str, Any]], *, max_tokens: int = 1024) -> str:
        self.calls.append((system, messages))
        if not self._replies:
            raise AssertionError("ScriptedLLM ran out of scripted replies")
        return self._replies.pop(0)


class RoutedLLM:
    def __init__(self, fn: Callable[[str, list[dict[str, Any]]], str]):
        self._fn = fn
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    def complete(self, system: str, messages: list[dict[str, Any]], *, max_tokens: int = 1024) -> str:
        self.calls.append((system, messages))
        return self._fn(system, messages)
