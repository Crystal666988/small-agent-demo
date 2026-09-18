"""Tool abstraction + registry.

Each tool declares a name, a human-readable description, and a JSON-schema-like
parameter spec. The registry renders all specs into the system prompt so the
LLM can decide, purely from the schema, which tool to call and with what args.
"""
from __future__ import annotations

import json
from typing import Any, Callable


class ToolError(Exception):
    """Raised by a tool when it cannot fulfil a call (bad args, upstream fail)."""


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[[dict[str, Any]], str],
        *,
        required: list[str] | None = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters  # {arg_name: {"type":..., "description":...}}
        self.required = required or []
        self._handler = handler

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            },
        }

    def _validate(self, args: dict[str, Any]) -> None:
        missing = [r for r in self.required if r not in args or args[r] in ("", None)]
        if missing:
            raise ToolError(f"missing required argument(s): {', '.join(missing)}")

    def run(self, args: dict[str, Any]) -> str:
        self._validate(args)
        return self._handler(args)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def render_schemas(self) -> str:
        """Pretty JSON of all tool schemas, embedded in the system prompt."""
        return json.dumps([t.schema() for t in self._tools.values()], indent=2, ensure_ascii=False)
