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

    @property
    def _is_openai(self) -> bool:
        return self.config.provider == "openai"

    def _headers(self) -> dict[str, str]:
        if self._is_openai:
            return {
                "Authorization": f"Bearer {self.config.auth_token}",
                "content-type": "application/json",
            }
        # Anthropic: send both auth styles; compatible proxies accept either.
        return {
            "x-api-key": self.config.auth_token,
            "Authorization": f"Bearer {self.config.auth_token}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _build_request(self, system: str, messages: list[dict[str, Any]], max_tokens: int):
        """Return (url, payload) for the configured provider.

        Anthropic takes `system` as a top-level field. OpenAI-compatible APIs
        (DeepSeek etc.) expect the system prompt as the first message with
        role=system, so we prepend it there.
        """
        if self._is_openai:
            oai_messages = [{"role": "system", "content": system}, *messages]
            return self.config.chat_completions_url, {
                "model": self.config.model,
                "max_tokens": max_tokens,
                "messages": oai_messages,
            }
        return self.config.messages_url, {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }

    def complete(self, system: str, messages: list[dict[str, Any]], *, max_tokens: int = 1024) -> str:
        """Send one completion request and return the assistant text.

        Retries transient network / 5xx / 429 errors with exponential backoff.
        Raises LLMError on unrecoverable failure so the runtime can degrade
        gracefully instead of crashing.
        """
        url, payload = self._build_request(system, messages, max_tokens)
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    url,
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

    def _extract_text(self, body: dict[str, Any]) -> str:
        """Pull assistant text out of a provider response."""
        if self._is_openai:
            try:
                text = body["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError):
                raise LLMError(f"Unexpected OpenAI-style response: {body}")
            if not text.strip():
                raise LLMError(f"LLM returned no text content: {body}")
            return text
        parts = body.get("content", [])
        if not isinstance(parts, list):
            raise LLMError(f"Unexpected LLM response shape: {body}")
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        if not text.strip():
            raise LLMError(f"LLM returned no text content: {body}")
        return text
