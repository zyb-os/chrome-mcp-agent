"""
providers.py

LLM providers for the Chrome MCP agent.
Mirrors the ProxyProvider from the existing browser-agent but adapted
for this agent (sync HTTP via httpx, no Playwright dependency).
"""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LlmRetryExhausted(Exception):
    def __init__(self, msg: str, attempts: int = 0, last_error: str = "") -> None:
        super().__init__(msg)
        self.attempts = attempts
        self.last_error = last_error


class ProxyProvider:
    """Routes all LLM calls through the orchestrator's /api/v1/llm/complete endpoint."""

    def __init__(
        self,
        proxy_url: str,
        agent_id: str,
        model: str = "claude-sonnet-4-6",
        retry_count: int = 3,
        retry_delay_s: float = 5.0,
    ) -> None:
        self._proxy_url = proxy_url
        self._agent_id = agent_id
        self.model = model
        self._retry_count = retry_count
        self._retry_delay_s = retry_delay_s

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> dict:
        """
        Synchronous LLM completion via the orchestrator proxy.
        Returns the full Anthropic-format response dict.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools

        last_error: str = ""
        for attempt in range(self._retry_count):
            try:
                resp = httpx.post(
                    self._proxy_url,
                    json=payload,
                    headers={"X-Agent-Id": self._agent_id},
                    timeout=120.0,
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "LLM proxy attempt %d/%d failed: %s",
                    attempt + 1,
                    self._retry_count,
                    exc,
                )
                if attempt < self._retry_count - 1:
                    time.sleep(self._retry_delay_s * (1 + attempt * 0.5))

        raise LlmRetryExhausted(
            f"All {self._retry_count} LLM proxy attempts failed. Last: {last_error}",
            attempts=self._retry_count,
            last_error=last_error,
        )

    def update_retry_settings(self, retry_count: int, retry_delay_s: float) -> None:
        self._retry_count = retry_count
        self._retry_delay_s = retry_delay_s
