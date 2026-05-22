"""
agent.py

LLM-driven agent loop for the Chrome MCP agent.
Uses ChromeController (→ Chrome extension) instead of Playwright.
The agent is fully async — no separate browser thread needed.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from chrome_controller import ChromeController
from providers import ProxyProvider, LlmRetryExhausted
from tools import TOOL_DEFINITIONS, TaskComplete, FollowupRequired, execute_tool

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 60
MAX_HISTORY_MESSAGES = 24   # user+assistant pairs kept in context
MAX_SCREENSHOTS_IN_CONTEXT = 1

SYSTEM_PROMPT = """You are an AI agent controlling a real Chrome browser via a set of tools.
Your goal is to complete the user's task by interacting with web pages.

## Guidelines

1. **Always start with a screenshot** to see the current page state before deciding what to do.
2. **Use coordinates precisely.** Pixel coordinates come from screenshots — top-left is (0,0).
3. **After every click**, take a screenshot to confirm what changed.
4. **For forms**: click the input field first to focus it, then call type_text.
5. **For searches**: type the query, then press Enter (or click the search button).
6. **Use find_element** to get exact center coordinates for known CSS selectors.
7. **Use get_page_text** to read long articles or extract structured text without images.
8. **Use evaluate_js** for advanced interactions (e.g. reading hidden data, triggering events).
9. **Be efficient** — avoid redundant screenshots or unnecessary waits.
10. **Call task_complete** with a clear summary once the task is done.
11. **Call ask_user** only when truly blocked with no way to proceed.

## Important
- The viewport coordinates are in pixels relative to the visible Chrome window.
- Dynamic pages (SPAs, React, Vue) may take a moment to update after clicks — use wait() if needed.
- If a page seems unchanged after clicking, try waiting 1-2 seconds and take another screenshot.
"""


class BudgetExhausted(Exception):
    pass


class ChromeAgent:
    """
    Stateful LLM agent that drives Chrome through a tool-call loop.
    Each call to run_task() is independent (fresh message history).
    """

    def __init__(
        self,
        chrome: ChromeController,
        provider: ProxyProvider,
    ) -> None:
        self._chrome = chrome
        self._provider = provider
        # ThreadPoolExecutor for the synchronous LLM call
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")

    def run_task(
        self,
        task: str,
        followup_answers: dict[str, str] | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        """
        Synchronous wrapper — runs the async loop.
        Returns a dict with keys: success, summary (or followup_required, question, or error).
        """
        import asyncio
        return asyncio.run(self._run_async(task, followup_answers, timeout_s))

    async def _run_async(
        self,
        task: str,
        followup_answers: dict[str, str] | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        import asyncio

        deadline = time.monotonic() + timeout_s

        # Build initial user message
        user_content = task
        if followup_answers:
            user_content += "\n\nAdditional context from user:\n"
            for q, a in followup_answers.items():
                user_content += f"Q: {q}\nA: {a}\n"

        messages: list[dict] = [{"role": "user", "content": user_content}]
        screenshot_count = 0

        for iteration in range(MAX_ITERATIONS):
            if time.monotonic() >= deadline:
                raise BudgetExhausted(f"Task timed out after {timeout_s:.0f}s")

            # LLM call is synchronous — run in executor to avoid blocking event loop
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    self._executor,
                    lambda msgs=messages: self._provider.complete(
                        messages=msgs,
                        tools=TOOL_DEFINITIONS,
                        system=SYSTEM_PROMPT,
                        max_tokens=4096,
                    ),
                )
            except LlmRetryExhausted as exc:
                return {"success": False, "error": str(exc)}

            content: list[dict] = response.get("content", [])
            stop_reason: str = response.get("stop_reason", "")

            # Add assistant turn to history
            messages.append({"role": "assistant", "content": content})

            # If no tool calls → natural completion
            if stop_reason == "end_turn" or not any(
                c.get("type") == "tool_use" for c in content
            ):
                text = " ".join(
                    c.get("text", "") for c in content if c.get("type") == "text"
                ).strip()
                return {"success": True, "summary": text or "Task completed successfully."}

            # Execute all tool calls in this turn
            tool_results: list[dict] = []
            for block in content:
                if block.get("type") != "tool_use":
                    continue

                tool_name: str = block["name"]
                tool_input: dict = block.get("input", {})
                tool_use_id: str = block["id"]

                try:
                    result_text, screenshot_b64 = await execute_tool(
                        tool_name, tool_input, self._chrome
                    )
                except TaskComplete as tc:
                    return {"success": True, "summary": tc.summary}
                except FollowupRequired as fr:
                    return {
                        "success": True,
                        "followup_required": True,
                        "question": fr.question,
                    }

                # Build tool_result content list
                result_content: list[dict] = [{"type": "text", "text": result_text}]

                if screenshot_b64:
                    screenshot_count += 1
                    if screenshot_count > MAX_SCREENSHOTS_IN_CONTEXT:
                        _drop_old_screenshots(messages)
                    result_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    })

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_content,
                })

            messages.append({"role": "user", "content": tool_results})

            # Trim history to avoid unbounded context growth.
            # Always keep the first user message (the original task).
            if len(messages) > MAX_HISTORY_MESSAGES * 2:
                messages = messages[:1] + messages[-(MAX_HISTORY_MESSAGES * 2 - 1):]

        return {
            "success": False,
            "error": f"Reached max iterations ({MAX_ITERATIONS}) without completing the task.",
        }


def _drop_old_screenshots(messages: list[dict]) -> None:
    """
    Walk the message history and replace all image blocks except the most
    recent one with a text placeholder, keeping context size bounded.
    """
    positions: list[tuple] = []

    for i, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for j, block in enumerate(content):
            if isinstance(block, dict):
                if block.get("type") == "image":
                    positions.append((i, j, None))
                elif block.get("type") == "tool_result":
                    sub_content = block.get("content") or []
                    for k, sub in enumerate(sub_content):
                        if isinstance(sub, dict) and sub.get("type") == "image":
                            positions.append((i, j, k))

    # Replace everything except the last screenshot
    for pos in positions[:-1]:
        i, j, k = pos
        placeholder = {"type": "text", "text": "[screenshot removed to save context]"}
        if k is None:
            messages[i]["content"][j] = placeholder
        else:
            messages[i]["content"][j]["content"][k] = placeholder
