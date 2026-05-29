#!/usr/bin/env python
"""Drive the kglite-docs MCP server over a real stdio session.

Boots `python -m kglite_docs.mcp_server --db <db>` as a subprocess, does
the MCP `initialize` handshake, then runs a sequence of tool calls
against the *same long-lived server* (so warm state persists across
calls — exactly like a Claude Desktop attachment).

Usage:
    mcp_session.py --db /tmp/kb.kgl --steps steps.json

`steps.json` is a list of {"tool": <name>, "args": {...}} objects, e.g.:
    [
      {"tool": "document", "args": {"action": "ingest", "path": "/abs/a.pdf"}},
      {"tool": "search",   "args": {"query": "training objective", "top_k": 5}}
    ]

Prints per-call wall-clock latency and a trimmed result so you can see
exactly where (if anywhere) a call stalls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _trim(obj: object, limit: int = 8000) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    return s if len(s) <= limit else s[:limit] + f"… [+{len(s) - limit} chars]"


async def run(db: str, steps: list[dict]) -> int:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "kglite_docs.mcp_server", "--db", db, "--log-level", "WARNING"],
    )
    t0 = time.monotonic()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print(f"[init] MCP initialize returned in {time.monotonic() - t0:.2f}s")
            tools = await session.list_tools()
            print(f"[init] {len(tools.tools)} tools available: "
                  f"{', '.join(t.name for t in tools.tools)}")
            # A real MCP client surfaces server instructions to the agent
            # on connect — print them so a blind session sees the same
            # orientation (the happy path, the index step, etc.).
            instructions = getattr(init, "instructions", None)
            if instructions:
                print(f"\n[server instructions]\n{instructions}\n")
            else:
                print()
            for i, step in enumerate(steps, 1):
                tool, args = step["tool"], step.get("args", {})
                t = time.monotonic()
                try:
                    res = await session.call_tool(tool, args)
                    dt = time.monotonic() - t
                    payload = [
                        (c.text if getattr(c, "type", None) == "text" else f"<{c.type}>")
                        for c in res.content
                    ]
                    flag = "  <-- SLOW" if dt > 30 else ""
                    print(f"[{i}] {tool}({_trim(args, 160)}) -> {dt:.2f}s{flag}")
                    print(f"     {_trim(' '.join(payload))}\n")
                except Exception as exc:
                    dt = time.monotonic() - t
                    print(f"[{i}] {tool} FAILED after {dt:.2f}s: {exc}\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--steps", required=True, help="path to steps JSON file")
    a = ap.parse_args()
    steps = json.loads(Path(a.steps).read_text())
    return asyncio.run(run(a.db, steps))


if __name__ == "__main__":
    sys.exit(main())
