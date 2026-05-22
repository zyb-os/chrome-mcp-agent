"""
orchestrator_client.py

Connects the chrome-mcp-agent to the agent-orchestrator, implementing
the same protocol as the existing browser-agent.

Protocol:
  ✓ POST /api/v1/agents/register — stable UUID, capability schema, required_settings
  ✓ WS /ws/{agent_id} — connect immediately after registration
  ✓ Close code 4004 — re-register then reconnect
  ✓ Exponential-backoff auto-reconnect (cap: 60 s)
  ✓ Heartbeat every 15 s — status, load, metrics, chrome_connected flag
  ✓ task_request → run ChromeAgent.run_task in thread → task_response
  ✓ Respects timeout_ms hint
  ✓ status_update on task start/finish
  ✓ settings_push → live model/retry updates
  ✓ task_cancel → cancel active task
  ✓ Graceful shutdown on SIGINT/SIGTERM
"""
from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx
import websockets
import websockets.exceptions

from agent import ChromeAgent
from chrome_controller import ChromeController
from providers import ProxyProvider

logger = logging.getLogger(__name__)

_AGENT_ID_FILE = Path(".agent_id")
_HEARTBEAT_INTERVAL = 15.0
_MAX_BACKOFF = 60.0
_DRAIN_TIMEOUT = 120.0


def _stable_agent_id() -> str:
    if _AGENT_ID_FILE.exists():
        aid = _AGENT_ID_FILE.read_text().strip()
        if aid:
            return aid
    aid = str(uuid.uuid4())
    _AGENT_ID_FILE.write_text(aid)
    logger.info("Generated new stable agent ID: %s", aid)
    return aid


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _envelope(
    sender_id: str,
    msg_type: str,
    payload: dict,
    recipient_id: str | None = None,
    correlation_id: str | None = None,
) -> str:
    return json.dumps({
        "id": str(uuid.uuid4()),
        "type": msg_type,
        "sender_id": sender_id,
        "recipient_id": recipient_id,
        "payload": payload,
        "timestamp": _now_iso(),
        "correlation_id": correlation_id,
    })


# ── Registration payload ──────────────────────────────────────────────────

REGISTRATION_PAYLOAD = {
    "name": "chrome-mcp-agent",
    "description": (
        "AI-powered browser agent that controls a real Chrome browser via a Chrome extension. "
        "Supports modern SPAs, dynamic pages, and multi-step web automation. "
        "Requires the Chrome MCP Agent extension to be installed in Chrome."
    ),
    "version": "1.0.0",
    "default_prompt": "",
    "capabilities": [
        {
            "name": "browse_web",
            "description": (
                "Navigate websites, click elements, fill forms, take screenshots, "
                "extract content, and complete multi-step browsing tasks using a real "
                "Chrome browser. Compatible with SPAs, JavaScript-heavy sites, and "
                "pages requiring login or complex interactions."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Natural language description of the browsing task",
                    },
                    "followup_answers": {
                        "type": "object",
                        "description": "Answers to questions from a previous attempt (for resuming tasks)",
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["task"],
            },
            "tags": [
                "browser", "chrome", "web", "automation", "ai", "research",
                "scrape", "navigate", "screenshot", "form", "click", "extract",
            ],
            "cost": {"type": "per_call", "estimated_cost_usd": 0.015},
        }
    ],
    "tags": ["browser", "chrome", "mcp", "ai", "extension"],
    "required_settings": [
        {
            "key": "chrome_agent_model",
            "label": "LLM Model",
            "type": "string",
            "required": False,
            "description": (
                "Model used by the Chrome agent for browser automation. "
                "Overrides the global default model when set. "
                "Examples: claude-opus-4-5, claude-sonnet-4-6, claude-haiku-4-5"
            ),
            "default": "claude-sonnet-4-6",
        },
        {
            "key": "chrome_agent_ws_port",
            "label": "Extension WebSocket Port",
            "type": "integer",
            "required": False,
            "description": "Port the Chrome extension connects to (default: 8765)",
            "default": 8765,
        },
        {
            "key": "chrome_agent_timeout_s",
            "label": "Task Timeout (seconds)",
            "type": "integer",
            "required": False,
            "description": "Maximum seconds allowed per browse_web task (default: 120)",
            "default": 120,
        },
        {
            "key": "browser_llm_retry_count",
            "label": "LLM Retry Count",
            "type": "integer",
            "required": False,
            "description": "Number of times to retry a failed LLM call (default: 3)",
            "default": 3,
        },
        {
            "key": "browser_llm_retry_delay_s",
            "label": "LLM Retry Delay (seconds)",
            "type": "integer",
            "required": False,
            "description": "Seconds between LLM retry attempts (default: 5)",
            "default": 5,
        },
    ],
}


# ── Main client ───────────────────────────────────────────────────────────

class OrchestratorClient:
    def __init__(self, orchestrator_url: str = "http://localhost:8000") -> None:
        self._base = orchestrator_url.rstrip("/")
        self._ws_base = self._base.replace("http://", "ws://").replace("https://", "wss://")
        self._agent_id = _stable_agent_id()
        self._http = httpx.AsyncClient(timeout=15.0)

        # Settings received from orchestrator
        self._common_settings: dict = {}

        # Chrome extension WebSocket server
        self._chrome: ChromeController | None = None
        self._agent: ChromeAgent | None = None
        self._provider: ProxyProvider | None = None

        # Task management
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="task")
        self._active_task: asyncio.Task | None = None
        self._task_sem = asyncio.Semaphore(1)

        # Status & metrics
        self._status = "starting"
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._total_duration_ms = 0.0
        self._start_time = time.monotonic()
        self._shutting_down = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self._shutdown()))

        # Start Chrome controller (WebSocket server for extension)
        port = 8765  # default; will be overridden after registration if agent_settings say otherwise
        self._chrome = ChromeController(port=port)
        await self._chrome.start()
        logger.info(
            "Chrome controller started. Install the chrome-extension in Chrome and it will connect automatically."
        )

        await self._connect_loop()

    # ── Registration ──────────────────────────────────────────────────────

    async def _register(self) -> None:
        url = f"{self._base}/api/v1/agents/register"
        logger.info("Registering with orchestrator at %s …", url)
        payload = {**REGISTRATION_PAYLOAD, "id": self._agent_id}
        resp = await self._http.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        self._agent_id = data["agent_id"]
        # Merge common + agent settings (agent-specific wins)
        self._common_settings = {
            **data.get("common_settings", {}),
            **data.get("agent_settings", {}),
        }
        self._init_provider()
        logger.info("Registered — agent_id=%s", self._agent_id)

    def _init_provider(self) -> None:
        model = (
            self._common_settings.get("chrome_agent_model")
            or self._common_settings.get("model")
            or self._common_settings.get("default_model")
            or "claude-sonnet-4-6"
        )
        retry_count = int(self._common_settings.get("browser_llm_retry_count") or 3)
        retry_delay = float(self._common_settings.get("browser_llm_retry_delay_s") or 5.0)
        self._provider = ProxyProvider(
            proxy_url=f"{self._base}/api/v1/llm/complete",
            agent_id=self._agent_id,
            model=model,
            retry_count=retry_count,
            retry_delay_s=retry_delay,
        )
        assert self._chrome is not None
        self._agent = ChromeAgent(chrome=self._chrome, provider=self._provider)
        logger.info("LLM provider initialised — model: %s", model)

    # ── Connection loop ───────────────────────────────────────────────────

    async def _connect_loop(self) -> None:
        backoff = 1.0
        while not self._shutting_down:
            try:
                await self._register()
                ws_url = f"{self._ws_base}/ws/{self._agent_id}"
                logger.info("Connecting to orchestrator WS: %s", ws_url)
                async with websockets.connect(ws_url) as ws:
                    backoff = 1.0
                    await self._run_session(ws)
            except websockets.exceptions.ConnectionClosed as exc:
                code = exc.rcvd.code if exc.rcvd else None
                if code == 4004:
                    logger.warning("Unknown agent_id (4004) — re-registering …")
                elif code == 4003:
                    logger.warning("Agent disabled (4003) — will retry")
                    backoff = max(backoff, 30.0)
                elif self._shutting_down:
                    break
                else:
                    logger.warning("WS closed (code=%s) — retry in %.0fs", code, backoff)
            except Exception as exc:
                if self._shutting_down:
                    break
                logger.warning("Connection error: %s — retry in %.0fs", exc, backoff)

            if not self._shutting_down:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF)

    # ── Session ───────────────────────────────────────────────────────────

    async def _run_session(self, ws) -> None:
        self._status = "available"
        logger.info("WebSocket session active")
        try:
            await asyncio.gather(
                self._heartbeat_loop(ws),
                self._recv_loop(ws),
            )
        finally:
            self._status = "offline"

    # ── Heartbeat ─────────────────────────────────────────────────────────

    async def _heartbeat_loop(self, ws) -> None:
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            try:
                n = self._tasks_completed + self._tasks_failed
                await ws.send(_envelope(
                    self._agent_id,
                    "heartbeat",
                    {
                        "status": self._status,
                        "current_load": 1.0 if self._active_task else 0.0,
                        "active_tasks": 1 if self._active_task else 0,
                        "expected_wait_time_ms": 30_000 if self._active_task else 0,
                        "metrics": {
                            "tasks_completed": self._tasks_completed,
                            "tasks_failed": self._tasks_failed,
                            "avg_response_time_ms": (
                                round(self._total_duration_ms / n, 1) if n else 0.0
                            ),
                            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
                            "chrome_connected": self._chrome.is_connected if self._chrome else False,
                        },
                    },
                ))
            except Exception as exc:
                logger.debug("Heartbeat failed: %s", exc)
                return

    # ── Receive loop ──────────────────────────────────────────────────────

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            mtype = msg.get("type", "")
            try:
                if mtype == "task_request":
                    asyncio.create_task(self._handle_task_request(ws, msg))
                elif mtype == "settings_push":
                    self._handle_settings_push(msg)
                elif mtype == "task_cancel":
                    self._handle_task_cancel()
                elif mtype in ("agent_registered", "agent_offline", "broadcast",
                               "discovery_response", "error"):
                    logger.debug("← [%s] %s", mtype, json.dumps(msg.get("payload", {}))[:120])
                else:
                    logger.debug("← unhandled message type: %r", mtype)
            except Exception as exc:
                logger.error("Error handling %r: %s", mtype, exc)

    # ── Task handling ─────────────────────────────────────────────────────

    async def _handle_task_request(self, ws, msg: dict) -> None:
        payload = msg.get("payload", {})
        capability = payload.get("capability")
        req_id = msg.get("id")
        sender_id = msg.get("sender_id")
        timeout_ms = float(payload.get("timeout_ms") or 120_000)

        if capability != "browse_web":
            await self._send_msg(ws, "task_response", {
                "success": False, "error": f"Unknown capability: {capability!r}",
            }, recipient_id=sender_id, correlation_id=req_id)
            return

        if self._active_task and not self._active_task.done():
            await self._send_msg(ws, "task_response", {
                "success": False, "error": "Agent is busy with another task",
            }, recipient_id=sender_id, correlation_id=req_id)
            return

        input_data = payload.get("input_data", {})
        task = (input_data.get("task") or "").strip()
        followup_answers = input_data.get("followup_answers") or None
        if not isinstance(followup_answers, dict):
            followup_answers = None

        if not task:
            await self._send_msg(ws, "task_response", {
                "success": False, "error": "input_data.task must be a non-empty string",
            }, recipient_id=sender_id, correlation_id=req_id)
            return

        timeout_s = timeout_ms / 1000.0
        agent_timeout_s = float(
            self._common_settings.get("chrome_agent_timeout_s") or timeout_s
        )
        effective_timeout = min(timeout_s, agent_timeout_s)

        async def _run_and_reply() -> None:
            async with self._task_sem:
                self._status = "busy"
                await self._send_msg(ws, "status_update", {"status": "busy"})
                t0 = time.monotonic()
                try:
                    assert self._agent is not None
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self._executor,
                        lambda: self._agent.run_task(
                            task=task,
                            followup_answers=followup_answers,
                            timeout_s=effective_timeout,
                        ),
                    )
                    duration_ms = (time.monotonic() - t0) * 1000

                    if result.get("followup_required"):
                        self._tasks_completed += 1
                        output = {
                            "followup_request": {
                                "question": result["question"],
                                "question_id": str(uuid.uuid4()),
                                "answer_format": "text",
                            }
                        }
                        await self._send_msg(ws, "task_response", {
                            "success": True,
                            "output_data": output,
                            "duration_ms": round(duration_ms, 1),
                        }, recipient_id=sender_id, correlation_id=req_id)
                    elif result.get("success"):
                        self._tasks_completed += 1
                        self._total_duration_ms += duration_ms
                        await self._send_msg(ws, "task_response", {
                            "success": True,
                            "output_data": {"summary": result.get("summary", "")},
                            "duration_ms": round(duration_ms, 1),
                        }, recipient_id=sender_id, correlation_id=req_id)
                    else:
                        self._tasks_failed += 1
                        await self._send_msg(ws, "task_response", {
                            "success": False,
                            "output_data": {},
                            "error": result.get("error", "Unknown error"),
                            "duration_ms": round(duration_ms, 1),
                        }, recipient_id=sender_id, correlation_id=req_id)

                except asyncio.CancelledError:
                    self._tasks_failed += 1
                    duration_ms = (time.monotonic() - t0) * 1000
                    await self._send_msg(ws, "task_response", {
                        "success": False,
                        "output_data": {},
                        "error": "Task was cancelled",
                        "duration_ms": round(duration_ms, 1),
                    }, recipient_id=sender_id, correlation_id=req_id)
                except Exception as exc:
                    self._tasks_failed += 1
                    duration_ms = (time.monotonic() - t0) * 1000
                    logger.exception("Unhandled error in task execution")
                    await self._send_msg(ws, "task_response", {
                        "success": False,
                        "output_data": {},
                        "error": str(exc),
                        "duration_ms": round(duration_ms, 1),
                    }, recipient_id=sender_id, correlation_id=req_id)
                finally:
                    self._active_task = None
                    self._status = "draining" if self._shutting_down else "available"
                    await self._send_msg(ws, "status_update", {"status": self._status})

        self._active_task = asyncio.create_task(_run_and_reply())

    def _handle_settings_push(self, msg: dict) -> None:
        settings = msg.get("payload", {}).get("settings", {})
        if not settings:
            return
        self._common_settings.update(settings)
        if self._provider is None:
            return
        new_model = (
            settings.get("chrome_agent_model")
            or settings.get("model")
            or settings.get("default_model")
        )
        if new_model:
            self._provider.model = new_model
            logger.info("settings_push: model updated → %s", new_model)
        rc = settings.get("browser_llm_retry_count")
        rd = settings.get("browser_llm_retry_delay_s")
        if rc is not None or rd is not None:
            self._provider.update_retry_settings(
                int(rc) if rc is not None else self._provider._retry_count,
                float(rd) if rd is not None else self._provider._retry_delay_s,
            )
        logger.debug("settings_push: applied %d setting(s)", len(settings))

    def _handle_task_cancel(self) -> None:
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
            logger.info("Active task cancelled via task_cancel message")

    # ── Graceful shutdown ─────────────────────────────────────────────────

    async def _shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        logger.info("Shutting down — waiting for active task to finish …")
        self._status = "draining"

        deadline = time.monotonic() + _DRAIN_TIMEOUT
        while self._active_task and not self._active_task.done():
            if time.monotonic() > deadline:
                logger.warning("Drain timeout — cancelling active task")
                self._active_task.cancel()
                break
            await asyncio.sleep(0.5)

        try:
            await self._http.delete(f"{self._base}/api/v1/agents/{self._agent_id}")
        except Exception as exc:
            logger.warning("Deregistration failed: %s", exc)

        if self._chrome:
            await self._chrome.stop()
        await self._http.aclose()
        self._executor.shutdown(wait=False)
        logger.info("Shutdown complete")

    # ── Helpers ───────────────────────────────────────────────────────────

    async def _send_msg(
        self,
        ws,
        msg_type: str,
        payload: dict,
        recipient_id: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        try:
            await ws.send(_envelope(self._agent_id, msg_type, payload, recipient_id, correlation_id))
        except Exception as exc:
            logger.debug("Failed to send %s: %s", msg_type, exc)
