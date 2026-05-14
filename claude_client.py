"""
claude_client.py
----------------
Shared Anthropic configuration for the chat-model slots (ingest
normalization, cluster labeling, the interview agent). Anthropic has no
embedding API, so embeddings continue to use OpenAI's — that client is
constructed at the call sites that need it.

Env vars:
- ANTHROPIC_API_KEY (required)
"""

import os
import random
import threading
from typing import Optional
from anthropic import Anthropic

CHAT_MODEL = "claude-sonnet-4-6"


# Per-request timeout for the Anthropic client. The SDK default is 10
# minutes; 180s keeps real failures visible to the user inside ~3 minutes
# while leaving headroom for long tool-use turns.
CHAT_TIMEOUT_SECONDS = 600.0

# SDK retries are off so each 429 surfaces immediately at the call site
# instead of disappearing into several silent minutes of internal backoff.
# Call sites do their own retry using the constants below, logging every
# wait so the operator (or the user, via on_message) sees what's happening.
CHAT_MAX_RETRIES = 0

# Explicit retry policy used by call sites when they catch RateLimitError.
RATE_LIMIT_MAX_RETRIES = 8

# Global semaphore: caps the total number of concurrent Anthropic API calls
# across all batch workers (ingest chunks + OCR batches + cluster labeling).
# Without this, 4 files × 4 chunks/file = 16 simultaneous calls → rate limit storm.
# Interactive paths (agent, research agent) don't acquire this semaphore so
# they're never blocked by background ingestion.
MAX_ANTHROPIC_CONCURRENT = 3
ANTHROPIC_SEMAPHORE = threading.Semaphore(MAX_ANTHROPIC_CONCURRENT)


def rate_limit_pause(attempt: int, exc: Exception) -> float:
    """Return seconds to wait before retry attempt `attempt` (0-indexed).

    Respects the Retry-After header when Anthropic provides it. Falls back
    to exponential backoff (15s → 30s → 60s) with ±20% jitter so workers
    don't all wake up at the same moment.
    """
    retry_after = None
    resp = getattr(exc, "response", None)
    if resp is not None:
        ra = getattr(resp, "headers", {}).get("retry-after")
        if ra:
            try:
                retry_after = float(ra)
            except (TypeError, ValueError):
                pass
    if retry_after is not None:
        return retry_after + random.uniform(1, 3)
    base = min(60.0, 15.0 * (2 ** attempt))
    return base + random.uniform(0, base * 0.2)


def chat_client(api_key: Optional[str] = None) -> Anthropic:
    """Anthropic SDK client for the chat-model slots."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
    return Anthropic(
        api_key=key,
        timeout=CHAT_TIMEOUT_SECONDS,
        max_retries=CHAT_MAX_RETRIES,
    )


def thinking_enabled() -> bool:
    """Whether extended thinking should be enabled for the chat models.

    Controlled by the ENABLE_THINKING env var. Default is off — extended
    thinking is higher quality but materially slower. Set ENABLE_THINKING=
    true to re-enable. Note: extended thinking is incompatible with
    forced tool_choice ("any"/"tool"), so callers that force a specific
    tool ignore this flag.
    """
    return (os.environ.get("ENABLE_THINKING") or "").strip().lower() in {"1", "true", "yes", "on"}


# Token budget for extended thinking when enabled. Sonnet 4.6 wants
# budget_tokens < max_tokens; 8k of thinking leaves room for substantive
# output on a 32k cap.
THINKING_BUDGET_TOKENS = 8000


def thinking_param() -> dict:
    """Top-level `thinking` parameter for messages.create.

    Returns {"type": "enabled", "budget_tokens": ...} when ENABLE_THINKING
    is on, {"type": "disabled"} otherwise. Sonnet 4.6 does not support
    Opus 4.7's "adaptive" mode — only enabled/disabled.
    """
    if thinking_enabled():
        return {"type": "enabled", "budget_tokens": THINKING_BUDGET_TOKENS}
    return {"type": "disabled"}
