"""Tests for the core agent runtime loop, using the mock LLM (offline)."""
import json

from miniagent.config import Config
from miniagent.runtime import Agent
from tests.mock_llm import RoutedLLM, ScriptedLLM


def make_config(**over):
    base = dict(base_url="http://x", auth_token="t", model="m",
                max_context_tokens=8000, max_steps=6)
    base.update(over)
    return Config(**base)


def test_direct_answer_no_tool():
    llm = ScriptedLLM(['{"thought":"just greet","final_answer":"Hello there!"}'])
    agent = Agent(llm, make_config())
    res = agent.run_turn("s1", "hi")
    assert res.stopped_reason == "final"
    assert res.answer == "Hello there!"
    assert res.steps == 1


def test_tool_then_final_loop():
    # step1: call calculator; step2: after seeing result, answer.
    llm = ScriptedLLM([
        '{"thought":"need math","tool_call":{"name":"calculator","arguments":{"expression":"23*17"}}}',
        '{"thought":"got it","final_answer":"23*17 is 391"}',
    ])
    agent = Agent(llm, make_config())
    res = agent.run_turn("s1", "what is 23*17?")
    assert res.stopped_reason == "final"
    assert "391" in res.answer
    assert res.steps == 2
    # The tool result must have been fed back into context.
    _, messages = llm.calls[1]
    assert any("391" in m["content"] for m in messages)


def test_session_isolation():
    """Two windows keep independent todos even for the same user."""
    def route(system, messages):
        # Look at the latest user message to decide what to do.
        last = messages[-1]["content"]
        if "add weather todo" in last:
            return '{"thought":"add","tool_call":{"name":"todo","arguments":{"action":"add","item":"check umbrella"}}}'
        if "add report todo" in last:
            return '{"thought":"add","tool_call":{"name":"todo","arguments":{"action":"add","item":"draft weekly report"}}}'
        # After a tool result, just finalize.
        return '{"thought":"done","final_answer":"ok"}'

    agent = Agent(RoutedLLM(route), make_config())
    agent.run_turn("win-1", "add weather todo")
    agent.run_turn("win-2", "add report todo")

    s1 = agent.sessions.get("win-1")
    s2 = agent.sessions.get("win-2")
    assert s1.todos == ["check umbrella"]
    assert s2.todos == ["draft weekly report"]


def test_parse_error_recovery():
    # First reply is garbage; runtime should feed the error back and the model
    # recovers on the second reply.
    llm = ScriptedLLM([
        "I will not follow the protocol.",
        '{"thought":"ok now","final_answer":"recovered"}',
    ])
    agent = Agent(llm, make_config())
    res = agent.run_turn("s1", "hello")
    assert res.answer == "recovered"
    assert res.steps == 2


def test_unknown_tool_recovery():
    llm = ScriptedLLM([
        '{"thought":"try","tool_call":{"name":"nope","arguments":{}}}',
        '{"thought":"fallback","final_answer":"handled unknown tool"}',
    ])
    agent = Agent(llm, make_config())
    res = agent.run_turn("s1", "do something")
    assert res.answer == "handled unknown tool"
    # The observation about the unknown tool must be in context.
    _, messages = llm.calls[1]
    assert any("unknown tool" in m["content"] for m in messages)


def test_tool_error_recovery():
    llm = ScriptedLLM([
        '{"thought":"bad math","tool_call":{"name":"calculator","arguments":{"expression":"1/0"}}}',
        '{"thought":"explain","final_answer":"cannot divide by zero"}',
    ])
    agent = Agent(llm, make_config())
    res = agent.run_turn("s1", "compute 1/0")
    assert "divide" in res.answer
    _, messages = llm.calls[1]
    assert any("ERROR" in m["content"] for m in messages)


def test_max_steps_guard():
    # Model keeps calling a tool forever; runtime must stop at max_steps.
    forever = '{"thought":"again","tool_call":{"name":"calculator","arguments":{"expression":"1+1"}}}'
    llm = ScriptedLLM([forever] * 10)
    agent = Agent(llm, make_config(max_steps=3))
    res = agent.run_turn("s1", "loop please")
    assert res.stopped_reason == "max_steps"
    assert res.steps == 3


def test_followup_remembers_prior_state():
    """A pure-conversation follow-up sees earlier turns in context."""
    llm = ScriptedLLM([
        '{"thought":"answer","final_answer":"Your name is noted as Sam."}',
        '{"thought":"recall","final_answer":"You told me your name is Sam."}',
    ])
    agent = Agent(llm, make_config())
    agent.run_turn("s1", "My name is Sam.")
    agent.run_turn("s1", "What is my name?")
    # Second call's messages must contain the first exchange.
    _, messages = llm.calls[1]
    joined = " ".join(m["content"] for m in messages)
    assert "Sam" in joined


def test_remember_field_populates_memory():
    llm = ScriptedLLM(['{"thought":"note it","final_answer":"noted","remember":"prefers metric units"}'])
    agent = Agent(llm, make_config())
    agent.run_turn("s1", "I prefer metric")
    assert "prefers metric units" in agent.sessions.get("s1").memory
