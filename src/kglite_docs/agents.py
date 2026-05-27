"""LLM caller abstraction used by the workflow demo and any user-built
agent loop.

Two backends:

- **anthropic_sdk** — if `ANTHROPIC_API_KEY` is set (and the `anthropic`
  package installed), uses the official SDK. Best for production.
- **claude_cli** — shells out to the `claude -p` CLI. Reuses the user's
  existing Claude Code auth — no separate API key required. Good for
  one-off scripts and demos.

Pick automatically with `default_caller()`, or pass a specific one to
`call_agent()`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Protocol


class AgentCaller(Protocol):
    """One-shot LLM interface. Stateless; each call is independent."""

    def __call__(self, prompt: str, *, system: str = "", model: str = "sonnet") -> str: ...


def call_sdk(prompt: str, *, system: str = "", model: str = "claude-sonnet-4-6") -> str:
    """Anthropic SDK path. Requires `ANTHROPIC_API_KEY` and `anthropic`."""
    try:
        import anthropic  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "anthropic SDK not installed. pip install anthropic"
        ) from e
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system or "You are a helpful assistant.",
        messages=[{"role": "user", "content": prompt}],
    )
    parts = []
    for block in msg.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "".join(parts).strip()


def call_cli(prompt: str, *, system: str = "", model: str = "sonnet") -> str:
    """`claude -p` subprocess. Reuses the user's existing Claude Code auth.
    Slower per call than the SDK (~3-5s startup) but needs no API key."""
    if not shutil.which("claude"):
        raise RuntimeError("claude CLI not found on PATH")
    args = ["claude", "-p", "--bare", "--model", model]
    if system:
        args.extend(["--append-system-prompt", system])
    proc = subprocess.run(
        args,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI failed (exit {proc.returncode}): {proc.stderr[:500]}")
    return proc.stdout.strip()


def default_caller() -> AgentCaller:
    """Pick `call_sdk` if an API key is available, else `call_cli`."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # noqa: F401
            return call_sdk
        except ImportError:
            pass
    if shutil.which("claude"):
        return call_cli
    raise RuntimeError(
        "No LLM caller available — set ANTHROPIC_API_KEY (with `pip install anthropic`) "
        "or install the `claude` CLI."
    )


def call_agent(
    prompt: str,
    *,
    system: str = "",
    model: str = "sonnet",
    caller: AgentCaller | None = None,
) -> str:
    """One-shot agent call. Returns the text response."""
    fn = caller or default_caller()
    return fn(prompt, system=system, model=model)
