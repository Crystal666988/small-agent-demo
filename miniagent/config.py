"""Configuration loading.

Reads settings from a local .env file (simple KEY=VALUE parser, no dependency)
and from real environment variables. Environment variables win over .env so
that CI / shell overrides are respected.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a .env file without clobbering existing vars."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # .env fills gaps only; a real env var always wins.
        os.environ.setdefault(key, value)


@dataclass
class Config:
    base_url: str
    auth_token: str
    model: str
    max_context_tokens: int
    max_steps: int
    provider: str = "anthropic"  # "anthropic" | "openai" (OpenAI-compatible, e.g. DeepSeek)

    @classmethod
    def load(cls, dotenv_path: str | None = None) -> "Config":
        root = Path(__file__).resolve().parent.parent
        _load_dotenv(Path(dotenv_path) if dotenv_path else root / ".env")

        # Provider selection. Default anthropic; set MINIAGENT_PROVIDER=openai for
        # any OpenAI-compatible endpoint (DeepSeek, Moonshot, local vLLM, ...).
        provider = os.environ.get("MINIAGENT_PROVIDER", "anthropic").lower()

        if provider == "openai":
            base_url = os.environ.get(
                "OPENAI_BASE_URL", "https://api.deepseek.com"
            ).rstrip("/")
            token = os.environ.get("OPENAI_API_KEY", "")
            default_model = "deepseek-chat"
        else:
            base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
            token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
            default_model = "claude-haiku-4-5-20251001"

        model = os.environ.get("MINIAGENT_MODEL", default_model)
        max_ctx = int(os.environ.get("MINIAGENT_MAX_CONTEXT_TOKENS", "8000"))
        max_steps = int(os.environ.get("MINIAGENT_MAX_STEPS", "8"))
        return cls(base_url, token, model, max_ctx, max_steps, provider)

    @property
    def messages_url(self) -> str:
        return f"{self.base_url}/v1/messages"

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"
