"""Tests for session context assembly and compression."""
from miniagent.compressor import maybe_compress
from miniagent.session import Session, estimate_tokens
from tests.mock_llm import ScriptedLLM


def test_estimate_tokens_counts_cjk_denser():
    assert estimate_tokens("中文字符") >= 4
    assert estimate_tokens("") == 0


def test_build_messages_maps_roles_and_merges():
    s = Session("s1")
    s.add("user", "hi")
    s.add("assistant", '{"thought":"t","tool_call":{"name":"x"}}')
    s.add("observation", "tool said 5")
    msgs = s.build_messages()
    # observation becomes a user message tagged [Tool result]
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[2]["role"] == "user"
    assert "[Tool result]" in msgs[2]["content"]


def test_summary_injected_as_first_user_message():
    s = Session("s1")
    s.summary = "earlier: user asked about weather"
    s.add("user", "and now?")
    msgs = s.build_messages()
    assert msgs[0]["role"] == "user"
    assert "summary" in msgs[0]["content"].lower()


def test_compression_folds_old_turns():
    s = Session("s1")
    # Create many long turns to blow the budget.
    for i in range(20):
        s.add("user", f"message number {i} " * 50)
        s.add("assistant", f"reply number {i} " * 50)

    before = len(s.history)
    llm = ScriptedLLM(["SUMMARY: the user sent 20 messages about testing."])
    changed = maybe_compress(s, llm, budget_tokens=200, keep_recent=6)

    assert changed is True
    assert s.summary.startswith("SUMMARY")
    assert len(s.history) == 6  # only recent kept
    assert len(s.history) < before


def test_compression_noop_when_small():
    s = Session("s1")
    s.add("user", "short")
    llm = ScriptedLLM([])  # must not be called
    assert maybe_compress(s, llm, budget_tokens=8000) is False


def test_compression_fallback_when_llm_fails():
    from miniagent.llm import LLMError

    class FailingLLM:
        def complete(self, *a, **k):
            raise LLMError("down")

    s = Session("s1")
    for i in range(20):
        s.add("user", f"long message {i} " * 50)
        s.add("assistant", f"long reply {i} " * 50)

    changed = maybe_compress(s, FailingLLM(), budget_tokens=200, keep_recent=4)
    assert changed is True
    assert s.summary  # fallback produced a non-empty truncated summary
    assert len(s.history) == 4
