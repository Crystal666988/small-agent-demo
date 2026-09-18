"""Tests for the tool registry and built-in tools."""
import pytest

from miniagent.tools import ToolError, build_registry


def make_reg(store=None):
    store = store if store is not None else []
    return build_registry(todo_store=lambda: store), store


def test_registry_lists_all_tools():
    reg, _ = make_reg()
    assert set(reg.names()) == {"calculator", "search", "weather", "todo"}


def test_registry_schema_has_required_fields():
    reg, _ = make_reg()
    schema = reg.get("calculator").schema()
    assert schema["name"] == "calculator"
    assert "expression" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["expression"]


def test_calculator_basic():
    reg, _ = make_reg()
    assert reg.get("calculator").run({"expression": "23*17+5"}) == "23*17+5 = 396"


def test_calculator_integer_division_and_power():
    reg, _ = make_reg()
    assert reg.get("calculator").run({"expression": "2**10"}) == "2**10 = 1024"


def test_calculator_rejects_code_injection():
    reg, _ = make_reg()
    with pytest.raises(ToolError):
        reg.get("calculator").run({"expression": "__import__('os').system('echo hi')"})


def test_missing_required_arg_raises():
    reg, _ = make_reg()
    with pytest.raises(ToolError):
        reg.get("calculator").run({})


def test_search_mock_deterministic():
    reg, _ = make_reg()
    out = reg.get("search").run({"query": "what is python"})
    assert "Python" in out and out.startswith("[mock-search]")


def test_weather_mock():
    reg, _ = make_reg()
    assert "Guangzhou" in reg.get("weather").run({"city": "Guangzhou"})


def test_todo_add_list_done_flow():
    reg, store = make_reg()
    todo = reg.get("todo")
    assert "Added todo #1" in todo.run({"action": "add", "item": "buy milk"})
    assert "buy milk" in todo.run({"action": "list"})
    assert "Completed" in todo.run({"action": "done", "index": 1})
    assert todo.run({"action": "list"}) == "Todo list is empty."


def test_todo_bad_action():
    reg, _ = make_reg()
    with pytest.raises(ToolError):
        reg.get("todo").run({"action": "frobnicate"})
