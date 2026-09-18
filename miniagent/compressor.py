"""Basic context compression.

When a session's estimated token count exceeds the budget, we summarize the
OLDEST turns into `session.summary` and drop them from `history`, keeping the
most recent `keep_recent` turns verbatim. Memory facts and the running summary
are preserved.

We ask the LLM for the summary (it's good at it). If the LLM call fails, we fall
back to a naive truncated concatenation so the agent keeps working offline / on
error. "Complex compression" (semantic dedup, importance scoring) is explicitly
out of scope per the assignment.
"""
from __future__ import annotations

from .llm import LLMError
from .session import Session

_SUMMARY_SYSTEM = (
    "You compress conversation history. Given earlier turns, produce a concise "
    "summary (<=150 words) capturing: the user's goals, decisions made, tool "
    "results that still matter, and any open follow-ups. Omit chit-chat. "
    "Write plain text, no preamble."
)


def maybe_compress(session: Session, llm, budget_tokens: int, *, keep_recent: int = 6) -> bool:
    """Compress if over budget. Returns True if compression happened."""
    if session.estimated_tokens() <= budget_tokens:
        return False
    if len(session.history) <= keep_recent:
        return False  # nothing old enough to fold away

    old = session.history[:-keep_recent]
    recent = session.history[-keep_recent:]

    transcript = "\n".join(f"{t.role.upper()}: {t.content}" for t in old)
    prior = f"Existing summary:\n{session.summary}\n\n" if session.summary else ""
    user_msg = f"{prior}Turns to fold in:\n{transcript}"

    try:
        new_summary = llm.complete(
            _SUMMARY_SYSTEM,
            [{"role": "user", "content": user_msg}],
            max_tokens=400,
        ).strip()
    except LLMError:
        # Offline / error fallback: keep a truncated concatenation.
        joined = (session.summary + " " + transcript).strip()
        new_summary = joined[:1200] + (" ..." if len(joined) > 1200 else "")

    session.summary = new_summary
    session.history = recent
    return True
