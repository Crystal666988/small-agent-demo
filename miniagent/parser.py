"""Parse the LLM's textual output into a structured decision.

Protocol (see prompts.py): the model replies with a SINGLE JSON object:

    {"thought": "...", "tool_call": {"name": "...", "arguments": {...}}}
  or
    {"thought": "...", "final_answer": "..."}

Real models wrap JSON in ```json fences, add prose, or emit minor glitches.
This parser is defensive: it locates the JSON object, tolerates code fences and
leading/trailing text, and returns a typed Decision. On unrecoverable garbage it
raises ParseError with the raw text, which the runtime feeds back to the model
as an error observation so it can self-correct.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


class ParseError(ValueError):
    pass


@dataclass
class Decision:
    thought: str
    # Exactly one of the two below is set.
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    final_answer: str | None = None
    remember: str | None = None  # optional durable fact the model asked to store

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1) if m else text


def _find_json_object(text: str) -> str:
    """Return the first balanced {...} block, respecting strings/escapes."""
    start = text.find("{")
    if start == -1:
        raise ParseError(f"no JSON object found in output: {text!r}")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ParseError(f"unbalanced JSON braces in output: {text!r}")


def parse_decision(text: str) -> Decision:
    """Parse raw LLM text into a Decision, or raise ParseError."""
    candidate = _find_json_object(_strip_fences(text))
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON ({exc}); extracted={candidate!r}") from exc

    if not isinstance(data, dict):
        raise ParseError(f"expected a JSON object, got {type(data).__name__}")

    thought = str(data.get("thought", "")).strip()
    remember = data.get("remember")
    remember = str(remember).strip() if remember else None

    tool_call = data.get("tool_call")
    final = data.get("final_answer")

    if tool_call is not None:
        if not isinstance(tool_call, dict) or "name" not in tool_call:
            raise ParseError(f"tool_call must be an object with a 'name': {tool_call!r}")
        args = tool_call.get("arguments", {})
        if not isinstance(args, dict):
            raise ParseError(f"tool_call.arguments must be an object: {args!r}")
        return Decision(
            thought=thought, tool_name=str(tool_call["name"]), tool_args=args, remember=remember
        )

    if final is not None:
        return Decision(thought=thought, final_answer=str(final), remember=remember)

    raise ParseError(
        "decision must contain either 'tool_call' or 'final_answer'; "
        f"got keys {list(data.keys())}"
    )
