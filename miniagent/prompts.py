"""System prompt construction.

The prompt pins down the strict JSON decision protocol the parser expects,
lists the available tools (rendered from their schemas), and injects the
session's durable memory facts so the agent can recall them at every step.
"""
from __future__ import annotations

from .session import Session
from .tools import ToolRegistry

_PROTOCOL = """\
You are a minimal tool-using agent. On EVERY turn you MUST reply with a SINGLE
JSON object and nothing else. Two shapes are allowed:

1) Call a tool:
{
  "thought": "<brief reasoning about what to do next>",
  "tool_call": {"name": "<tool name>", "arguments": { ... }}
}

2) Give the final answer to the user:
{
  "thought": "<brief reasoning>",
  "final_answer": "<the answer text for the user>"
}

Rules:
- Output raw JSON only. No markdown fences, no text before or after.
- Call a tool ONLY when you need external computation or data. For greetings,
  clarifications, or things you already know, answer directly.
- After you receive a "[Tool result]", decide again: call another tool or give
  the final answer.
- Use tool arguments that exactly match the tool's parameter schema.
- To persist a durable fact for later turns, add the tool_call name "todo" or
  simply state it; the runtime also lets you remember facts by including a
  top-level "remember": "<fact>" field alongside your JSON (optional).
"""


def build_system_prompt(session: Session, registry: ToolRegistry) -> str:
    parts = [_PROTOCOL, "", "AVAILABLE TOOLS (JSON schemas):", registry.render_schemas()]
    if session.memory:
        facts = "\n".join(f"- {m}" for m in session.memory)
        parts += ["", "REMEMBERED FACTS (durable memory for this session):", facts]
    if session.title:
        parts += ["", f"This conversation's topic: {session.title}"]
    return "\n".join(parts)
