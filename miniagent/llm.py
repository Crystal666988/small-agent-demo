"""LLM client for an Anthropic-compatible /v1/messages endpoint.

Deliberately thin: it only knows how to send a system prompt + a list of
{role, content} messages and return the assistant's *text*. We do NOT use the
provider's native tool-calling; the agent's own parser (parser.py) extracts
tool calls from the text. This keeps the "LLM output parsing" logic squarely
inside our runtime, as required.

The client exposes a `complete(system, messages) -> str` method. Tests inject
a fake object with the same method, so the runtime never touches the network.
"""
from __future__ import annotations

import json
import time
from typing import Any

import requests

from .config import Config


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, config: Config, *, timeout: int = 60, max_retries: int = 3):
        self.config = config
        self.timeout = timeout
        self.max_retries = max_retries

    def _headers(self) -> dict[str, str]:
        # Send both auth styles; compatible proxies accept one or the other.
        return {
            "x-api-key": self.config.auth_token,
            "Authorization": f"Bearer {self.config.auth_token}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def complete(self, system: str, messages: list[dict[str, Any]], *, max_tokens: int = 1024) -> str:
        """Send one completion request and return the assistant text.

        Retries transient network / 5xx / 429 errors with exponential backoff.
        Raises LLMError on unrecoverable failure so the runtime can degrade
        gracefully instead of crashing.
        """
        payload = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.config.messages_url,
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    return self._extract_text(resp.json())
                # 429 / 5xx are retryable; 4xx (except 429) are not.
                if resp.status_code != 429 and resp.status_code < 500:
                    raise LLMError(f"LLM HTTP {resp.status_code}: {resp.text[:300]}")
                last_err = LLMError(f"LLM HTTP {resp.status_code}: {resp.text[:200]}")
            except requests.RequestException as exc:
                last_err = exc
            time.sleep(min(2 ** attempt, 8))
        raise LLMError(f"LLM request failed after {self.max_retries} attempts: {last_err}")

    @staticmethod
    def _extract_text(body: dict[str, Any]) -> str:
        """Pull concatenated text out of an Anthropic messages response."""
        parts = body.get("content", [])
        if not isinstance(parts, list):
            raise LLMError(f"Unexpected LLM response shape: {body}")
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        if not text.strip():
            raise LLMError(f"LLM returned no text content: {body}")
        return text
