"""Built-in tools: calculator, search (mock), weather (mock), todo.

The todo tool is session-scoped: its state lives on the Session object, so two
windows keep independent todo lists. `build_registry` wires everything up and
takes a `todo_store` callable that yields the current session's todo list.
"""
from __future__ import annotations

import ast
import operator
from typing import Any, Callable

from .base import Tool, ToolError, ToolRegistry

# ----------------------------- calculator -----------------------------------
# A safe arithmetic evaluator: parse to AST and only allow numeric operators.
# No eval(), so no code-injection surface.

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError("only numeric literals are allowed")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval_node(node.operand))
    raise ToolError("unsupported expression element")


def _calculator(args: dict[str, Any]) -> str:
    expr = str(args["expression"])
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_node(tree.body)
    except ToolError:
        raise
    except Exception as exc:  # noqa: BLE001 - report any parse/eval issue to the model
        raise ToolError(f"could not evaluate {expr!r}: {exc}") from exc
    # Render integers without a trailing .0 for readability.
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expr} = {result}"


# ------------------------------- search (mock) -------------------------------
# Deterministic canned results so tests are reproducible and no network needed.

_MOCK_SEARCH = {
    "python": "Python is a high-level programming language created by Guido van Rossum (1991).",
    "agent": "An AI agent is a system that perceives inputs and takes actions via tools to reach goals.",
    "redis": "Redis is an in-memory key-value store often used as a cache and message broker.",
}


def _search(args: dict[str, Any]) -> str:
    query = str(args["query"]).lower()
    for key, val in _MOCK_SEARCH.items():
        if key in query:
            return f"[mock-search] {val}"
    return f"[mock-search] No curated result for {args['query']!r}. (This is a mock search tool.)"


# ------------------------------ weather (mock) -------------------------------

_MOCK_WEATHER = {
    "guangzhou": ("Guangzhou", 28, "cloudy"),
    "广州": ("广州", 28, "多云"),
    "xiamen": ("Xiamen", 26, "sunny"),
    "厦门": ("厦门", 26, "晴"),
    "beijing": ("Beijing", 12, "windy"),
    "北京": ("北京", 12, "有风"),
}


def _weather(args: dict[str, Any]) -> str:
    city = str(args["city"]).strip()
    info = _MOCK_WEATHER.get(city.lower()) or _MOCK_WEATHER.get(city)
    if not info:
        return f"[mock-weather] No data for {city!r}; assume mild, 20C, clear. (mock)"
    name, temp, cond = info
    return f"[mock-weather] {name}: {temp}C, {cond}."


# --------------------------------- todo --------------------------------------
# Session-scoped. The handler closes over a getter returning the live list.

def _make_todo(todo_store: Callable[[], list[str]]) -> Callable[[dict[str, Any]], str]:
    def _todo(args: dict[str, Any]) -> str:
        action = str(args.get("action", "")).lower()
        todos = todo_store()
        if action == "add":
            item = str(args.get("item", "")).strip()
            if not item:
                raise ToolError("action 'add' requires a non-empty 'item'")
            todos.append(item)
            return f"Added todo #{len(todos)}: {item}"
        if action == "list":
            if not todos:
                return "Todo list is empty."
            return "\n".join(f"{i+1}. {t}" for i, t in enumerate(todos))
        if action == "done":
            idx = args.get("index")
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                raise ToolError("action 'done' requires an integer 'index'")
            if not 1 <= idx <= len(todos):
                raise ToolError(f"index {idx} out of range (1..{len(todos)})")
            removed = todos.pop(idx - 1)
            return f"Completed and removed: {removed}"
        raise ToolError("action must be one of: add, list, done")

    return _todo


# ------------------------------- registry ------------------------------------

def build_registry(todo_store: Callable[[], list[str]]) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool(
        name="calculator",
        description="Evaluate a basic arithmetic expression (+, -, *, /, %, **, //).",
        parameters={"expression": {"type": "string", "description": "e.g. '23 * 17 + 5'"}},
        required=["expression"],
        handler=_calculator,
    ))
    reg.register(Tool(
        name="search",
        description="Search the web for a query. (Mocked: returns canned/deterministic results.)",
        parameters={"query": {"type": "string", "description": "search keywords"}},
        required=["query"],
        handler=_search,
    ))
    reg.register(Tool(
        name="weather",
        description="Get current weather for a city. (Mocked.)",
        parameters={"city": {"type": "string", "description": "city name, e.g. 'Guangzhou'"}},
        required=["city"],
        handler=_weather,
    ))
    reg.register(Tool(
        name="todo",
        description="Manage the current session's todo list.",
        parameters={
            "action": {"type": "string", "description": "one of: add | list | done"},
            "item": {"type": "string", "description": "todo text (for action=add)"},
            "index": {"type": "integer", "description": "1-based index (for action=done)"},
        },
        required=["action"],
        handler=_make_todo(todo_store),
    ))
    return reg
