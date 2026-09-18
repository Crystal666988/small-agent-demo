"""Interactive multi-window CLI.

Demonstrates the session isolation requirement: user A can open window 1
("check weather, note a todo") and window 2 ("write weekly report, note a
todo") and switch between them freely without cross-talk.

Commands:
  /new [title]        create and switch to a new window (session)
  /switch <id>        switch to an existing window
  /sessions           list all windows
  /todos              show current window's todo list
  /memory             show current window's remembered facts
  /trace              toggle live trace echo
  /help               show commands
  /quit               exit
Anything else is sent to the agent as user input.
"""
from __future__ import annotations

import sys

from .config import Config
from .llm import LLMClient
from .runtime import Agent

BANNER = """\
mini-agent  (from-scratch minimal agent runtime)
Type a message, or /help for commands. Two windows never share state.
"""


def _new_id(n: int) -> str:
    return f"win-{n}"


def main(argv: list[str] | None = None) -> int:
    config = Config.load()
    if not config.auth_token:
        print("ERROR: ANTHROPIC_AUTH_TOKEN is not set. Copy .env.example to .env "
              "and fill it in.", file=sys.stderr)
        return 2

    llm = LLMClient(config)
    agent = Agent(llm, config)

    print(BANNER)
    counter = 1
    current = _new_id(counter)
    agent.sessions.get_or_create(current, title="window 1")
    echo = False

    while True:
        try:
            line = input(f"[{current}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return 0

        if not line:
            continue

        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            cmd = parts[0]
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit"):
                print("bye.")
                return 0
            if cmd == "/help":
                print(__doc__)
            elif cmd == "/new":
                counter += 1
                current = _new_id(counter)
                agent.sessions.get_or_create(current, title=arg or f"window {counter}")
                print(f"created and switched to {current} ({arg or 'window ' + str(counter)})")
            elif cmd == "/switch":
                if agent.sessions.get(arg) is None:
                    print(f"no such window: {arg}. Use /sessions to list.")
                else:
                    current = arg
                    print(f"switched to {current}")
            elif cmd == "/sessions":
                for sid in agent.sessions.ids():
                    s = agent.sessions.get(sid)
                    marker = "*" if sid == current else " "
                    print(f" {marker} {sid}  title={s.title!r}  turns={len(s.history)} todos={len(s.todos)}")
            elif cmd == "/todos":
                s = agent.sessions.get(current)
                print("\n".join(f"{i+1}. {t}" for i, t in enumerate(s.todos)) or "(empty)")
            elif cmd == "/memory":
                s = agent.sessions.get(current)
                print("\n".join(f"- {m}" for m in s.memory) or "(no remembered facts)")
            elif cmd == "/trace":
                echo = not echo
                agent.echo_trace = echo
                print(f"trace echo {'on' if echo else 'off'}")
            else:
                print(f"unknown command: {cmd} (try /help)")
            continue

        # Normal message -> run one agent turn in the current window.
        result = agent.run_turn(current, line)
        print(f"\n{result.answer}\n")


if __name__ == "__main__":
    raise SystemExit(main())
