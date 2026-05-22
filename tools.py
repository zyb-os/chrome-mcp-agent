"""
tools.py

Tool definitions (JSON Schema) and dispatcher for the Chrome MCP agent.
The LLM calls these tools; the dispatcher translates them into ChromeController commands.
"""
from __future__ import annotations

import asyncio
from typing import Any

from chrome_controller import ChromeController, ChromeNotConnectedError, ChromeCommandError

# ── Tool definitions (Anthropic tool-use format) ─────────────────────────

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "navigate",
        "description": "Navigate the browser to a URL and wait for the page to load.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to navigate to, including https:// (e.g. https://google.com)",
                }
            },
            "required": ["url"],
        },
    },
    {
        "name": "screenshot",
        "description": (
            "Take a screenshot of the current browser viewport. "
            "Always take a screenshot first to see what is on screen before deciding what to do next."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "click",
        "description": (
            "Click at specific pixel coordinates in the browser viewport. "
            "Use screenshot first to identify element coordinates. "
            "After clicking, a new screenshot is automatically taken."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X coordinate (pixels from left edge)"},
                "y": {"type": "integer", "description": "Y coordinate (pixels from top edge)"},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "type_text",
        "description": (
            "Type text into the currently focused element. "
            "Always click the input field first to focus it, then call type_text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to type"},
                "clear_first": {
                    "type": "boolean",
                    "description": "Clear existing text before typing (default: false)",
                    "default": False,
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "key_press",
        "description": (
            "Press a keyboard key. Common values: Enter, Tab, Escape, "
            "ArrowDown, ArrowUp, ArrowLeft, ArrowRight, Backspace, Delete, Space."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key name (e.g. Enter, Tab, Escape, ArrowDown)"}
            },
            "required": ["key"],
        },
    },
    {
        "name": "scroll",
        "description": "Scroll the page up or down.",
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "Direction to scroll",
                },
                "amount": {
                    "type": "integer",
                    "description": "Scroll units (1–10). Each unit is ~120px. Default: 3.",
                    "default": 3,
                },
            },
            "required": ["direction"],
        },
    },
    {
        "name": "hover",
        "description": "Move the mouse cursor to specific coordinates to reveal tooltips or dropdown menus.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X coordinate"},
                "y": {"type": "integer", "description": "Y coordinate"},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "wait",
        "description": "Pause execution for a number of seconds. Use after navigation or clicking to wait for content to load.",
        "input_schema": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "number",
                    "description": "Seconds to wait (0.5–15). Default: 2.",
                    "default": 2,
                }
            },
            "required": ["seconds"],
        },
    },
    {
        "name": "go_back",
        "description": "Navigate back to the previous page in browser history.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_page_text",
        "description": (
            "Get the full visible text content of the current page as plain text. "
            "Useful for reading long-form content or extracting data without image analysis."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "evaluate_js",
        "description": (
            "Execute JavaScript in the page context and return the result. "
            "Use for complex DOM queries, extracting data, or interactions that simple clicks cannot achieve."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "JavaScript expression to evaluate. Must return a serialisable value.",
                }
            },
            "required": ["code"],
        },
    },
    {
        "name": "find_element",
        "description": (
            "Find an element by CSS selector and return its text, center coordinates, and attributes. "
            "Use the returned centerX/centerY to click the element precisely."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector (e.g. '#submit-btn', 'input[name=q]', '.product-title')",
                }
            },
            "required": ["selector"],
        },
    },
    {
        "name": "task_complete",
        "description": "Signal that the task is fully complete. Call this when you have finished all required steps.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Clear summary of what was accomplished and any key results or findings",
                }
            },
            "required": ["summary"],
        },
    },
    {
        "name": "ask_user",
        "description": (
            "Ask the user a clarifying question when you cannot proceed without more information. "
            "Use sparingly — only when truly blocked."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The specific question to ask the user",
                }
            },
            "required": ["question"],
        },
    },
]


# ── Sentinel exceptions ───────────────────────────────────────────────────

class TaskComplete(Exception):
    def __init__(self, summary: str) -> None:
        self.summary = summary
        super().__init__(summary)


class FollowupRequired(Exception):
    def __init__(self, question: str) -> None:
        self.question = question
        super().__init__(question)


# ── Tool dispatcher ───────────────────────────────────────────────────────

async def execute_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    chrome: ChromeController,
) -> tuple[str, str | None]:
    """
    Execute a browser tool and return (result_text, screenshot_b64_or_None).

    Raises TaskComplete or FollowupRequired for terminal tools.
    All other exceptions are caught and returned as error text.
    """
    try:
        if tool_name == "navigate":
            result = await chrome.navigate(tool_input["url"])
            await asyncio.sleep(1.0)  # Let page settle after load event
            url = await chrome.get_url()
            title = await chrome.get_title()
            return f"Navigated to: {url}\nPage title: {title}", None

        elif tool_name == "screenshot":
            b64 = await chrome.screenshot()
            return "Screenshot taken.", b64

        elif tool_name == "click":
            x, y = int(tool_input["x"]), int(tool_input["y"])
            await chrome.click(x, y)
            await asyncio.sleep(0.8)
            b64 = await chrome.screenshot()
            return f"Clicked at ({x}, {y}).", b64

        elif tool_name == "type_text":
            text = tool_input["text"]
            clear_first = bool(tool_input.get("clear_first", False))
            await chrome.type_text(text, clear_first=clear_first)
            return f"Typed text ({len(text)} chars).", None

        elif tool_name == "key_press":
            key = tool_input["key"]
            await chrome.key_press(key)
            await asyncio.sleep(0.5)
            return f"Pressed key: {key}", None

        elif tool_name == "scroll":
            direction = tool_input["direction"]
            amount = int(tool_input.get("amount", 3))
            await chrome.scroll(direction, amount)
            await asyncio.sleep(0.4)
            b64 = await chrome.screenshot()
            return f"Scrolled {direction} ({amount} units).", b64

        elif tool_name == "hover":
            x, y = int(tool_input["x"]), int(tool_input["y"])
            await chrome.hover(x, y)
            await asyncio.sleep(0.3)
            return f"Hovered at ({x}, {y}).", None

        elif tool_name == "wait":
            seconds = float(tool_input.get("seconds", 2))
            seconds = max(0.1, min(seconds, 30.0))
            await chrome.wait(seconds)
            return f"Waited {seconds:.1f}s.", None

        elif tool_name == "go_back":
            await chrome.go_back()
            await asyncio.sleep(1.2)
            url = await chrome.get_url()
            return f"Navigated back. Now at: {url}", None

        elif tool_name == "get_page_text":
            text = await chrome.get_page_text()
            if len(text) > 8000:
                text = text[:8000] + "\n\n[...truncated, page has more content]"
            return f"Page text content:\n{text}", None

        elif tool_name == "evaluate_js":
            code = tool_input["code"]
            result = await chrome.evaluate_js(code)
            return f"JavaScript result: {result}", None

        elif tool_name == "find_element":
            selector = tool_input["selector"]
            result = await chrome.find_element(selector)
            if result.get("found"):
                return (
                    f"Element found — tag: {result.get('tag')}, "
                    f"text: {result.get('text', '')!r}, "
                    f"value: {result.get('value', '')!r}, "
                    f"type: {result.get('type', '')}, "
                    f"center: ({result.get('centerX')}, {result.get('centerY')}), "
                    f"rect: {result.get('rect')}"
                ), None
            return f"Element not found for selector: {selector!r}", None

        elif tool_name == "task_complete":
            raise TaskComplete(tool_input["summary"])

        elif tool_name == "ask_user":
            raise FollowupRequired(tool_input["question"])

        else:
            return f"Unknown tool: {tool_name!r}", None

    except (TaskComplete, FollowupRequired):
        raise
    except ChromeNotConnectedError as exc:
        return (
            f"Error: Chrome extension is not connected. {exc}\n"
            "Make sure the Chrome extension is loaded and shows 'Connected' status."
        ), None
    except ChromeCommandError as exc:
        return f"Chrome command error in {tool_name!r}: {exc}", None
    except Exception as exc:
        return f"Unexpected error in {tool_name!r}: {exc}", None
