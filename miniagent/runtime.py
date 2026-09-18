"""The core Agent Runtime loop (implemented from scratch, no agent framework).

One user turn drives this loop:

    receive user input
      -> ask LLM for a decision (think)
      -> if final_answer: return it to the user
      -> if tool_call: run the tool, append the result as an observation,
         and loop again
      -> repeat until final answer or max_steps reached

Cross-cutting concerns handled here:
  - session isolation (each Session has its own history/todos/memory)
  - context management (history replayed each step; compression when over budget)
  - durable memory (Decision.remember -> session.memory)
  - exception handling (parse errors and tool errors become observations that
    the model can recover from, instead of crashing the loop)
  - tracing (every step logged)
"""
from __future__ import annotations

from dataclasses import dataclass

from .compressor import maybe_compress
from .config import Config
from .llm import LLMError
from .parser import ParseError, parse_decision
from .prompts import build_system_prompt
from .session import Session, SessionManager
from .tools import ToolError, ToolRegistry, build_registry
from .trace import Tracer


@dataclass
class AgentResult:
    answer: str
    steps: int
    stopped_reason: str  # "final" | "max_steps" | "error"


class Agent:
    def __init__(self, llm, config: Config, *, echo_trace: bool = False):
        self.llm = llm
        self.config = config
        self.echo_trace = echo_trace
        self.sessions = SessionManager()

    def _registry_for(self, session: Session) -> ToolRegistry:
        # Bind the todo tool to THIS session's list, guaranteeing isolation.
        return build_registry(todo_store=lambda: session.todos)

    def run_turn(self, session_id: str, user_input: str, *, title: str = "") -> AgentResult:
        session = self.sessions.get_or_create(session_id, title=title)
        tracer = Tracer(session_id, echo=self.echo_trace)
        registry = self._registry_for(session)

        session.add("user", user_input)
        tracer.emit("user_input", text=user_input)

        # Compress before we start if history has grown large.
        if maybe_compress(session, self.llm, self.config.max_context_tokens):
            tracer.emit("compressed", summary_tokens=len(session.summary))

        for step in range(1, self.config.max_steps + 1):
            system = build_system_prompt(session, registry)
            messages = session.build_messages()

            try:
                raw = self.llm.complete(system, messages, max_tokens=1024)
            except LLMError as exc:
                tracer.emit("llm_error", error=str(exc))
                msg = f"Sorry, the model call failed: {exc}"
                session.add("assistant", msg)
                return AgentResult(msg, step, "error")

            tracer.emit("llm_raw", step=step, raw=raw[:2000])

            # --- parse the decision, recovering from malformed output ---------
            try:
                decision = parse_decision(raw)
            except ParseError as exc:
                tracer.emit("parse_error", step=step, error=str(exc))
                # Feed the error back so the model can correct its format.
                session.add("assistant", raw)
                session.add(
                    "observation",
                    f"ERROR: your output was not valid per the protocol ({exc}). "
                    "Reply again with a single valid JSON object.",
                )
                continue

            # Record the model's thought/decision as an assistant turn.
            session.add("assistant", raw)
            if decision.remember:
                session.remember(decision.remember)
                tracer.emit("remember", fact=decision.remember)

            if decision.is_final:
                tracer.emit("final", step=step, answer=decision.final_answer)
                return AgentResult(decision.final_answer or "", step, "final")

            # --- tool call ----------------------------------------------------
            tool = registry.get(decision.tool_name or "")
            if tool is None:
                tracer.emit("unknown_tool", step=step, name=decision.tool_name)
                session.add(
                    "observation",
                    f"ERROR: unknown tool {decision.tool_name!r}. "
                    f"Available tools: {', '.join(registry.names())}.",
                )
                continue

            tracer.emit("tool_call", step=step, name=tool.name, args=decision.tool_args)
            try:
                result = tool.run(decision.tool_args)
            except ToolError as exc:
                tracer.emit("tool_error", step=step, name=tool.name, error=str(exc))
                session.add("observation", f"ERROR from tool {tool.name}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 - never let a tool crash the loop
                tracer.emit("tool_crash", step=step, name=tool.name, error=repr(exc))
                session.add("observation", f"ERROR: tool {tool.name} crashed: {exc}")
                continue

            tracer.emit("tool_result", step=step, name=tool.name, result=result[:2000])
            session.add("observation", result)
            # loop continues: model sees the observation and decides again

        # Ran out of steps without a final answer.
        tracer.emit("max_steps", steps=self.config.max_steps)
        fallback = "I reached the maximum number of reasoning steps without a final answer."
        session.add("assistant", fallback)
        return AgentResult(fallback, self.config.max_steps, "max_steps")
