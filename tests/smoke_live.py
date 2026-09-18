"""Live smoke test against the real LLM API (NOT part of the offline suite).

Run manually:  python -m tests.smoke_live
Requires ANTHROPIC_AUTH_TOKEN / ANTHROPIC_BASE_URL in the environment or .env.

Exercises a real tool-using loop and a follow-up in the same session.
"""
from miniagent.config import Config
from miniagent.llm import LLMClient
from miniagent.runtime import Agent


def main() -> int:
    config = Config.load()
    if not config.auth_token:
        print("skip: no auth token")
        return 0
    agent = Agent(LLMClient(config), config, echo_trace=True)

    print("\n=== turn 1: math (should call calculator) ===")
    r = agent.run_turn("live-1", "What is 23 * 17 + 4? Use the calculator.")
    print("ANSWER:", r.answer, "| steps:", r.steps, "| reason:", r.stopped_reason)

    print("\n=== turn 2: weather + todo in same session ===")
    r = agent.run_turn("live-1", "What's the weather in Guangzhou, and add a todo to bring an umbrella.")
    print("ANSWER:", r.answer, "| steps:", r.steps)

    print("\n=== turn 3: follow-up (pure conversation, recall) ===")
    r = agent.run_turn("live-1", "What todos do I have so far?")
    print("ANSWER:", r.answer, "| steps:", r.steps)

    print("\nsession todos:", agent.sessions.get("live-1").todos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
