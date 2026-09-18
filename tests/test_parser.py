"""Tests for the LLM output parser."""
import json

import pytest

from miniagent.parser import ParseError, parse_decision


def test_parse_plain_tool_call():
    d = parse_decision('{"thought":"need math","tool_call":{"name":"calculator","arguments":{"expression":"1+1"}}}')
    assert d.tool_name == "calculator"
    assert d.tool_args == {"expression": "1+1"}
    assert not d.is_final


def test_parse_final_answer():
    d = parse_decision('{"thought":"done","final_answer":"42"}')
    assert d.is_final
    assert d.final_answer == "42"


def test_parse_strips_code_fences_and_prose():
    raw = 'Sure!\n```json\n{"thought":"t","final_answer":"hi"}\n```\nHope that helps.'
    d = parse_decision(raw)
    assert d.final_answer == "hi"


def test_parse_nested_braces_in_strings():
    # A JSON string value that itself contains braces must not confuse the scanner.
    raw = '{"thought":"use {this}","final_answer":"result is {ok}"}'
    d = parse_decision(raw)
    assert d.final_answer == "result is {ok}"


def test_parse_remember_field():
    d = parse_decision('{"thought":"t","final_answer":"ok","remember":"user likes tea"}')
    assert d.remember == "user likes tea"


def test_parse_missing_both_raises():
    with pytest.raises(ParseError):
        parse_decision('{"thought":"nothing here"}')


def test_parse_no_json_raises():
    with pytest.raises(ParseError):
        parse_decision("I refuse to output JSON")


def test_parse_bad_tool_call_shape():
    with pytest.raises(ParseError):
        parse_decision('{"thought":"t","tool_call":{"arguments":{}}}')  # no name
