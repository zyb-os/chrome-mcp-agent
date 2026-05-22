"""
main.py — Chrome MCP Agent entry point.

Starts a WebSocket server for the Chrome extension to connect to,
then registers with the orchestrator and runs the agent loop.

Usage:
    python main.py [--orchestrator-url http://localhost:8000]
"""
import argparse
import asyncio
import logging
import os

from orchestrator_client import OrchestratorClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chrome MCP Agent")
    parser.add_argument(
        "--orchestrator-url",
        default=os.getenv("ORCHESTRATOR_URL", "http://localhost:8000"),
        help="Agent Orchestrator base URL (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    client = OrchestratorClient(orchestrator_url=args.orchestrator_url)
    asyncio.run(client.start())


if __name__ == "__main__":
    main()
