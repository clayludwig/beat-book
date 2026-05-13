"""
ingest.py
---------
Multi-format ingestion pipeline: raw bytes / URLs → normalized story dicts.

Two stages:
1. extract_text(filename, raw_bytes) → plain text/markdown
   Format dispatcher. markitdown handles docx/pdf/html/pptx/xlsx/rtf.
   stdlib handles txt/md/json. Unknown extensions get a utf-8 best-effort.

2. normalize(text, source_label, anthropic_key) → list[Story]
   Claude Sonnet 4.6 tool-use call with a strict schema. Documents that fit
   in one window get a single call; larger documents are split into
   overlapping windows, processed concurrently, and deduplicated on merge.
   The LLM only infers metadata (title/date/author) and returns verbatim
   markers for each story's body — the body itself is sliced from the
   original text, never LLM-rewritten.

A wrapper, ingest_source(), runs both stages and packages the result for
the /ingest route.
"""

from __future__ import annotations

import concurrent.futures
import html
import ipaddress
import json
import logging
import re
import socket
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import anthropic
import httpx

from claude_client import (
    CHAT_MODEL,
    RATE_LIMIT_MAX_RETRIES,
    RATE_LIMIT_PAUSE_SECONDS,
    chat_client,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MAX_FILE_BYTES = 25 * 1024 * 1024            # 25 MB per file
URL_FETCH_TIMEOUT = 15.0                     # seconds
URL_USER_AGENT = "BeatBookBuilder/1.0 (+https://github.com/clayludwig/beat-book)"
NORMALIZE_MODEL = CHAT_MODEL
# Max tokens per LLM call. Sized to fit marker data for up to ~30 stories
# per chunk.
NORMALIZE_MAX_TOKENS = 32768
# Number of attempts per chunk when the model returns no tool call. With
# tool_choice forced this should be rare, but one retry covers transients.
NORMALIZE_MAX_ATTEMPTS = 2

# Chunked normalization: documents over WINDOW_SIZE get split into overlapping
# windows and processed concurrently. WINDOW_OVERLAP must be larger than any
# expected single-story body so a story spanning a boundary is fully contained
# in one of the two adjacent windows.
WINDOW_SIZE = 160_000           # primary chunk size in chars
WINDOW_OVERLAP = 15_000          # chars of overlap between adjacent chunks
# Soft safety cap on LLM calls per document. Sized to comfortably cover
# the largest input the 25 MB raw-file cap could produce after extraction
# (≈50 MB of HTML-stripped text at ~160 KB per chunk).
MAX_CHUNKS = 500
# Per-document chunk concurrency. Multiple chunks of the same document
# fan out to Anthropic in parallel; combined with app.py:_INGEST_CONCURRENCY
# this caps the peak in-flight request count. The SDK's 429 handling and
# our own RateLimitError retry loop ride out brief concurrent-limit spikes.
NORMALIZE_CONCURRENCY = 8
# No inter-chunk pause — every second of sleep is a second off the 1–3 min
# pipeline budget. Rate-limit retries handle the rare overshoot.
INTER_CHUNK_PAUSE_SECONDS = 0.0
# Record separator emitted by _extract_json for top-level JSON lists.
# When present in the extracted text, _make_chunks packs whole records
# into each chunk so a story body is never split across two chunks.
RECORD_SEPARATOR = "\n\n---\n\n"

# Extensions markitdown converts to readable markdown.
_MARKITDOWN_EXTS = {
    ".docx", ".doc",
    ".pdf",
    ".html", ".htm",
    ".pptx", ".ppt",
    ".xlsx", ".xls",
    ".csv",
    ".rtf",
    ".epub",
}
# Extensions we read directly as utf-8 text.
_TEXT_EXTS = {".txt", ".md", ".markdown", ".log"}


# ─────────────────────────────────────────────────────────────────────────────
# TYPES
# ─────────────────────────────────────────────────────────────────────────────


class IngestError(Exception):
    """Raised when a single source can't be ingested. Surfaced to the user."""


@dataclass
class Story:
    """A single normalized story. Output of stage 2."""
    title: str
    content: str
    date: str = ""
    author: str = ""
    link: str = ""
    confidence: str = "medium"   # "high" | "medium" | "low"
    reasoning: str = ""          # one-sentence justification for the preview UI

    def to_pipeline_dict(self) -> dict:
        """Shape consumed by pipeline.py / agent.py / citation_matcher.py."""
        out = {"title": self.title, "content": self.content}
        if self.date:
            out["date"] = self.date
        if self.author:
            out["author"] = self.author
        if self.link:
            out["link"] = self.link
        return out

    def to_preview_dict(self) -> dict:
        """Shape sent to the preview UI (includes metadata fields)."""
        return {
            "title": self.title,
            "content": self.content,
            "date": self.date,
            "author": self.author,
            "link": self.link,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


@dataclass
class IngestedSource:
    """Result of ingesting one file or one URL."""
    source_label: str            # filename or URL
    kind: str                    # "file" | "url"
    stories: list[Story] = field(default_factory=list)
    excluded: bool = False
    skip_reason: str = ""
    extract_error: str = ""
    char_count: int = 0
    truncated: bool = False

    def to_preview_dict(self) -> dict:
        return {
            "source_label": self.source_label,
            "kind": self.kind,
            "stories": [s.to_preview_dict() for s in self.stories],
            "excluded": self.excluded,
            "skip_reason": self.skip_reason,
            "extract_error": self.extract_error,
            "char_count": self.char_count,
            "truncated": self.truncated,
        }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — EXTRACT
# ─────────────────────────────────────────────────────────────────────────────


def _ext_of(filename: str) -> str:
    return Path(filename).suffix.lower()


def _decode_text(raw: bytes) -> str:
    """Best-effort utf-8 decode with replacement for stray bytes."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def _clean_inline_html(s: str) -> str:
    """Unescape HTML entities and strip tags, preserving paragraph breaks.

    Applied to bodies pulled out of JSON/RSS records where the value commonly
    contains raw HTML. Keeps the text in a shape the LLM can read directly,
    so the markers it returns will match the extracted text.
    """
    if not s:
        return s
    # Decode entities twice in case of double-encoding (&amp;lt; → &lt; → <).
    text = html.unescape(html.unescape(s))
    # Block tags → paragraph breaks before stripping the rest.
    text = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", text)
    text = re.sub(
        r"(?i)</\s*(p|div|li|h[1-6]|blockquote|tr|article|section)\s*>",
        "\n\n", text,
    )
    text = re.sub(
        r"(?i)<\s*(p|div|li|h[1-6]|blockquote|tr|article|section)(\s[^>]*)?>",
        "\n\n", text,
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_with_markitdown(filename: str, raw: bytes) -> str:
    """Convert binary/office formats to markdown via markitdown.

    markitdown reads from a path, so we write the bytes to a NamedTemporaryFile
    with the correct suffix and let it dispatch on extension.
    """
    from markitdown import MarkItDown

    suffix = _ext_of(filename) or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()
        md = MarkItDown()
        result = md.convert(tmp.name)
        return (result.text_content or "").strip()


def _indent(text: str, by: str = "  ") -> str:
    return "\n".join((by + ln) if ln else ln for ln in text.split("\n"))


def _render_value(v) -> str:
    """Render any JSON value as readable text. No field names are
    privileged — every key is labeled, every value rendered recursively.
    Strings get HTML stripped so Stage 2's verbatim-marker resolver can
    find them in the source."""
    if v is None or v == "" or v == [] or v == {}:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return _clean_inline_html(v)
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        rendered = [s for s in (_render_value(x) for x in v) if s]
        if not rendered:
            return ""
        if all("\n" not in s and len(s) < 60 for s in rendered):
            return ", ".join(rendered)
        return "\n".join(f"- {_indent(s).lstrip()}" for s in rendered)
    if isinstance(v, dict):
        parts = []
        for k, sub in v.items():
            sub_text = _render_value(sub)
            if not sub_text:
                continue
            if "\n" in sub_text:
                parts.append(f"{k}:\n{_indent(sub_text)}")
            else:
                parts.append(f"{k}: {sub_text}")
        return "\n".join(parts)
    return str(v)


def _extract_json(raw: bytes) -> str:
    """Render JSON as readable text the LLM can scan for stories.

    Schema-agnostic — no field names are special. A top-level list is
    split with `---` separators so each item becomes a candidate story
    boundary; anything else is rendered as one document. HTML embedded
    in string values is stripped so Stage 2's verbatim markers resolve
    against the rendered text.
    """
    text = _decode_text(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Not valid JSON — return raw text; the LLM can still try.
        return text

    if isinstance(data, list):
        rendered = [r for r in (_render_value(item) for item in data) if r]
        return "\n\n---\n\n".join(rendered)
    return _render_value(data)


def extract_text(filename: str, raw: bytes) -> str:
    """Stage 1. Dispatch on extension to produce clean text/markdown."""
    if len(raw) > MAX_FILE_BYTES:
        raise IngestError(
            f"{filename}: file is {len(raw) / 1_048_576:.1f} MB; the limit is "
            f"{MAX_FILE_BYTES / 1_048_576:.0f} MB."
        )

    ext = _ext_of(filename)

    if ext == ".json":
        return _extract_json(raw)

    if ext in _TEXT_EXTS:
        return _decode_text(raw).strip()

    if ext in _MARKITDOWN_EXTS:
        try:
            return _extract_with_markitdown(filename, raw)
        except Exception as e:
            raise IngestError(f"{filename}: failed to extract — {type(e).__name__}: {e}") from e

    # Unknown — try utf-8, then markitdown as a hail-mary
    decoded = _decode_text(raw).strip()
    if decoded and "\x00" not in decoded[:1000]:
        return decoded
    try:
        return _extract_with_markitdown(filename, raw)
    except Exception as e:
        raise IngestError(
            f"{filename}: unsupported file type {ext!r}; could not extract text."
        ) from e


# ─────────────────────────────────────────────────────────────────────────────
# URL FETCHING (with SSRF protection)
# ─────────────────────────────────────────────────────────────────────────────


def _is_blocked_ip(host: str) -> bool:
    """Resolve hostname; reject loopback / private / link-local / multicast."""
    try:
        # getaddrinfo returns all A/AAAA records — check every one.
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return True  # unresolvable → block

    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return True
    return False


def fetch_url(url: str) -> Tuple[str, bytes]:
    """Fetch a URL, return (suggested_filename, raw_bytes).

    Refuses non-http(s) schemes and private/loopback addresses.
    Honors MAX_FILE_BYTES.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise IngestError(f"{url}: only http and https URLs are accepted.")
    if not parsed.hostname:
        raise IngestError(f"{url}: URL is missing a hostname.")
    if _is_blocked_ip(parsed.hostname):
        raise IngestError(f"{url}: URL resolves to a private or unreachable address.")

    headers = {"User-Agent": URL_USER_AGENT, "Accept": "*/*"}
    try:
        with httpx.Client(
            timeout=URL_FETCH_TIMEOUT,
            follow_redirects=True,
            headers=headers,
        ) as client:
            resp = client.get(url)
    except httpx.HTTPError as e:
        raise IngestError(f"{url}: fetch failed — {type(e).__name__}: {e}") from e

    if resp.status_code >= 400:
        raise IngestError(f"{url}: server returned HTTP {resp.status_code}.")

    body = resp.content
    if len(body) > MAX_FILE_BYTES:
        raise IngestError(
            f"{url}: response is {len(body) / 1_048_576:.1f} MB; the limit is "
            f"{MAX_FILE_BYTES / 1_048_576:.0f} MB."
        )

    # Decide a filename for extension dispatch.
    content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
    type_to_ext = {
        "text/html": ".html",
        "application/xhtml+xml": ".html",
        "text/plain": ".txt",
        "text/markdown": ".md",
        "application/json": ".json",
        "application/pdf": ".pdf",
        "application/msword": ".doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "application/rtf": ".rtf",
        "text/rtf": ".rtf",
    }
    suggested_ext = type_to_ext.get(content_type, "")
    path_part = Path(parsed.path).name or parsed.hostname
    if suggested_ext and not path_part.lower().endswith(suggested_ext):
        path_part = f"{path_part}{suggested_ext}" if path_part else f"page{suggested_ext}"
    elif not _ext_of(path_part):
        path_part = f"{path_part}.html"  # default for unknown content-type

    return path_part, body


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — NORMALIZE (LLM)
# ─────────────────────────────────────────────────────────────────────────────


_NORMALIZE_TOOL = {
    "name": "register_stories",
    "description": (
        "Register the news stories you found in the document. "
        "Return is_news_content=false when the document is not news content "
        "(e.g., reporter notes, invoices, transcripts of meetings) — in that "
        "case return an empty stories list and explain in skip_reason."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "is_news_content": {
                "type": "boolean",
                "description": "True if the document contains at least one news article or substantive story-like piece of journalism.",
            },
            "skip_reason": {
                "type": "string",
                "description": "If is_news_content is false, one short sentence explaining what the document looks like instead. Empty otherwise.",
            },
            "stories": {
                "type": "array",
                "description": "One entry per distinct news story in the document. Split multi-story documents (e.g., a notebook with several articles). Leave empty if is_news_content is false.",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The story's headline. If no headline is present, write a 6-10 word descriptive title.",
                        },
                        "date": {
                            "type": "string",
                            "description": "Publication date as YYYY-MM-DD. Empty string if not present in the document.",
                        },
                        "author": {
                            "type": "string",
                            "description": "Byline author. Empty string if not present. Do not put the publication name here.",
                        },
                        "link": {
                            "type": "string",
                            "description": "Source URL if present in the document. Empty string otherwise.",
                        },
                        "body_starts_with": {
                            "type": "string",
                            "description": "The first 30-80 characters of this story's body — COPIED VERBATIM from the document. This must appear exactly in the document so the server can locate the body's beginning. Start with the first sentence of the article body itself, not the title or byline.",
                        },
                        "body_ends_with": {
                            "type": "string",
                            "description": "The last 30-80 characters of this story's body — COPIED VERBATIM from the document. This must appear exactly in the document so the server can locate the body's end.",
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "high = metadata is explicit in the text. medium = metadata is reasonably inferred. low = metadata is a guess.",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "One sentence explaining how you identified this as a story and where the metadata came from.",
                        },
                    },
                    "required": [
                        "title", "date", "author", "link",
                        "body_starts_with", "body_ends_with",
                        "confidence", "reasoning",
                    ],
                },
            },
        },
        "required": ["is_news_content", "skip_reason", "stories"],
    },
}


_NORMALIZE_SYSTEM = (
    "You are a document parser for the Beat Book Builder, a tool that helps "
    "journalists turn a collection of source articles into a reporting guide. "
    "You receive a single document — extracted from any format (Word, PDF, "
    "HTML, plain text, markdown, JSON, scraped web pages) — and you identify "
    "the discrete news stories inside it.\n\n"
    "Rules:\n"
    "- One document may contain ZERO, ONE, or MANY stories. Split multi-story "
    "documents using their natural boundaries (headings, separator lines, "
    "byline blocks).\n"
    "- For each story, extract title, date (YYYY-MM-DD or empty), author "
    "(byline only, not publication name), and a source URL if present.\n"
    "- body_starts_with: a VERBATIM 30-80 character snippet from the document "
    "marking where this story's body begins. It must be the literal opening "
    "text of the article body (the first sentence) — NOT the title, byline, "
    "or any separator line above it.\n"
    "- body_ends_with: a VERBATIM 30-80 character snippet from the document "
    "marking where this story's body ends. It must be the last text of this "
    "article before the next article (or end of document).\n"
    "- The snippets must appear exactly in the document — character-for-"
    "character. The server will locate them with a substring search, so "
    "any paraphrasing or whitespace difference breaks the match.\n"
    "- If the document is not news content (meeting notes, an invoice, raw "
    "data, an empty file), set is_news_content=false, leave stories empty, "
    "and put a short explanation in skip_reason.\n"
    "- Do NOT rewrite or summarize content. The server uses your snippets "
    "to slice the original text verbatim.\n"
    "- You MUST call the register_stories tool. Do not respond with prose."
)


def _normalize_for_match(s: str) -> str:
    """Collapse whitespace so fuzzy substring searches survive small reformats."""
    return re.sub(r"\s+", " ", s).strip()


def _resolve_marker_offset(text: str, marker: str, *, after: int = 0) -> int:
    """Find `marker` in `text` starting at `after`, with whitespace-tolerant fallback.

    Returns -1 if not found.
    """
    marker = (marker or "").strip()
    if not marker:
        return -1

    # 1. Exact substring match.
    pos = text.find(marker, after)
    if pos >= 0:
        return pos

    # 2. Whitespace-normalized search — collapse runs of whitespace on both
    # sides and find the marker in the normalized text, then map the index
    # back to the original.
    norm_text = _normalize_for_match(text[after:])
    norm_marker = _normalize_for_match(marker)
    norm_pos = norm_text.find(norm_marker)
    if norm_pos >= 0:
        # Walk the original text counting normalized chars until we reach
        # norm_pos, then return the matching offset in the original.
        i, j = after, 0
        prev_space = False
        while i < len(text) and j < norm_pos:
            ch = text[i]
            if ch.isspace():
                if not prev_space:
                    j += 1
                    prev_space = True
            else:
                j += 1
                prev_space = False
            i += 1
        return i

    # 3. First-five-words fallback — handles minor LLM paraphrasing.
    first_words = " ".join(marker.split()[:5])
    if first_words and first_words != marker:
        pos = text.find(first_words, after)
        if pos >= 0:
            return pos

    return -1


def _slice_body(
    text: str,
    body_starts_with: str,
    body_ends_with: str,
    *,
    upper_bound: Optional[int] = None,
) -> str:
    """Locate the body inside `text` using the LLM's start/end snippets.

    `upper_bound`, when set, is the maximum end offset the body may extend
    to. This is how the caller prevents one story's body from bleeding into
    the next when the end marker fails to resolve cleanly — the caller knows
    where the next story starts and passes that as the ceiling.
    """
    start = _resolve_marker_offset(text, body_starts_with)
    if start < 0:
        return ""

    cap = upper_bound if upper_bound is not None else len(text)
    cap = min(cap, len(text))

    end_marker = (body_ends_with or "").strip()
    end_pos = _resolve_marker_offset(text, end_marker, after=start)
    if end_pos < 0:
        end = cap
    else:
        end = end_pos + len(end_marker)
        end = min(end, cap)

    if end <= start:
        return ""
    return text[start:end].strip()


def _make_chunks(text: str) -> list[str]:
    """Split text into chunks of at most WINDOW_SIZE chars.

    When the text carries record separators from a structured source
    (top-level JSON lists rendered by _extract_json), pack whole records
    into each chunk so a story body is never split across chunks — no
    overlap needed because records are pre-delimited.

    For unstructured text without separators, fall back to fixed-size
    overlapping windows so a body straddling the cut between window N
    and window N+1 is still fully contained in at least one of them.
    """
    if len(text) <= WINDOW_SIZE:
        return [text]

    if RECORD_SEPARATOR in text:
        records = text.split(RECORD_SEPARATOR)
        sep_len = len(RECORD_SEPARATOR)
        chunks: list[str] = []
        current: list[str] = []
        current_size = 0
        for rec in records:
            added_size = len(rec) + (sep_len if current else 0)
            if current and current_size + added_size > WINDOW_SIZE:
                chunks.append(RECORD_SEPARATOR.join(current))
                current = [rec]
                current_size = len(rec)
            else:
                current.append(rec)
                current_size += added_size
        if current:
            chunks.append(RECORD_SEPARATOR.join(current))
        return chunks

    chunks = []
    step = WINDOW_SIZE - WINDOW_OVERLAP
    start = 0
    while start < len(text):
        end = min(start + WINDOW_SIZE, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def _dedup_key(story: Story) -> str:
    """Stable key for deduplicating stories that show up in two overlapping
    chunks. Title + first 200 chars of body is specific enough to distinguish
    different events that happen to share a headline."""
    title = re.sub(r"\s+", " ", story.title.lower().strip())
    body_prefix = re.sub(r"\s+", " ", story.content[:200].lower().strip())
    return f"{title}||{body_prefix}"


def _normalize_chunk(
    text: str,
    source_label: str,
    anthropic_key: str,
    *,
    link_hint: str = "",
    allow_full_doc_fallback: bool = True,
) -> Tuple[list[Story], bool, str]:
    """Single LLM call on a chunk of text (which may be the whole document).

    allow_full_doc_fallback: when the LLM reports exactly one story but its
    markers fail to resolve, fall back to using this chunk's full text as
    that story's body. Safe only for whole-document calls; in chunked mode
    it would splice in content from adjacent stories.
    """
    client = chat_client(anthropic_key)

    user_prefix = f"Source label: {source_label}\n"
    if link_hint:
        user_prefix += f"Source URL: {link_hint}\n"
    user_prefix += f"Text length (chars): {len(text)}\n\n"
    user_prefix += "----- BEGIN DOCUMENT -----\n"
    user_suffix = "\n----- END DOCUMENT -----"

    tool_use_block = None
    for attempt in range(NORMALIZE_MAX_ATTEMPTS):
        # Inner retry loop catches RateLimitError (Anthropic concurrent-
        # connection 429s) and logs every wait at WARNING so the operator
        # can see what's happening instead of staring at a silent spinner.
        resp = None
        for rl_attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
            try:
                # Extended thinking is incompatible with forced tool_choice,
                # so we don't pass `thinking` here — omission == disabled.
                resp = client.messages.create(
                    model=NORMALIZE_MODEL,
                    max_tokens=NORMALIZE_MAX_TOKENS,
                    system=_NORMALIZE_SYSTEM,
                    messages=[
                        {"role": "user", "content": user_prefix + text + user_suffix},
                    ],
                    tools=[_NORMALIZE_TOOL],
                    tool_choice={"type": "tool", "name": "register_stories"},
                )
                break
            except anthropic.RateLimitError as e:
                if rl_attempt >= RATE_LIMIT_MAX_RETRIES:
                    raise IngestError(
                        f"{source_label}: rate limit not cleared after "
                        f"{RATE_LIMIT_MAX_RETRIES} retries — {e}"
                    ) from e
                logger.warning(
                    "Rate limited on %s; waiting %ds before retry %d/%d.",
                    source_label, RATE_LIMIT_PAUSE_SECONDS,
                    rl_attempt + 1, RATE_LIMIT_MAX_RETRIES,
                )
                time.sleep(RATE_LIMIT_PAUSE_SECONDS)
            except Exception as e:
                raise IngestError(
                    f"{source_label}: LLM normalization failed — {type(e).__name__}: {e}"
                ) from e

        assert resp is not None  # loop above either sets resp or raises

        tool_use_block = next(
            (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_use_block is not None:
            break
        if attempt + 1 < NORMALIZE_MAX_ATTEMPTS:
            logger.warning(
                "Chunk for %s returned no tool call on attempt %d/%d; retrying.",
                source_label, attempt + 1, NORMALIZE_MAX_ATTEMPTS,
            )

    if tool_use_block is None:
        raise IngestError(
            f"{source_label}: LLM did not return structured stories after "
            f"{NORMALIZE_MAX_ATTEMPTS} attempts."
        )

    payload = tool_use_block.input if isinstance(tool_use_block.input, dict) else {}

    is_news = bool(payload.get("is_news_content", False))
    skip_reason = (payload.get("skip_reason") or "").strip()
    raw_stories = payload.get("stories") or []

    # Drop non-dict entries defensively — strict tool-use should prevent this
    # but it's cheap insurance against a malformed input matching the schema.
    dropped_malformed = sum(1 for r in raw_stories if not isinstance(r, dict))
    if dropped_malformed:
        logger.warning(
            "Dropped %d non-dict entries from %s's stories array.",
            dropped_malformed, source_label,
        )
    raw_stories = [r for r in raw_stories if isinstance(r, dict)]

    if not is_news or not raw_stories:
        return [], False, skip_reason or "No news stories found in this document."

    stories: list[Story] = []
    use_full_doc_fallback = allow_full_doc_fallback and len(raw_stories) == 1

    # Pre-resolve every story's start position so each story's body can be
    # capped at the next story's start. Without this cap, a missing or
    # too-permissive end marker would silently extend the body into the
    # next article — a bug that surfaced as bleed-through in saved beat
    # books (e.g. a "council update" article whose body contained an
    # adjacent "today in history" column).
    start_positions: list[int] = []
    for raw in raw_stories:
        start_positions.append(
            _resolve_marker_offset(text, raw.get("body_starts_with") or "")
        )
    sorted_starts = sorted(s for s in start_positions if s >= 0)

    def _upper_bound_for(start: int) -> int:
        # First sorted start strictly greater than this story's start.
        for s in sorted_starts:
            if s > start:
                return s
        return len(text)

    for idx, raw in enumerate(raw_stories):
        body_start = raw.get("body_starts_with") or ""
        body_end = raw.get("body_ends_with") or ""
        start = start_positions[idx]
        upper = _upper_bound_for(start) if start >= 0 else None

        content = _slice_body(text, body_start, body_end, upper_bound=upper)

        if len(content) < 20 and use_full_doc_fallback:
            logger.warning(
                "Marker lookup for %s failed (got %d chars); falling back to full text.",
                source_label, len(content),
            )
            content = text.strip()

        if not content:
            logger.warning("Dropping story with empty body from %s.", source_label)
            continue

        title = (raw.get("title") or "").strip()
        if not title:
            first_line = next((ln for ln in content.splitlines() if ln.strip()), "")
            title = first_line[:80] if first_line else source_label

        stories.append(Story(
            title=title,
            content=content,
            date=(raw.get("date") or "").strip(),
            author=(raw.get("author") or "").strip(),
            link=(raw.get("link") or link_hint or "").strip(),
            confidence=(raw.get("confidence") or "medium").strip(),
            reasoning=(raw.get("reasoning") or "").strip(),
        ))

    if not stories:
        return [], False, skip_reason or "No story bodies could be extracted from this document."

    return stories, True, skip_reason


def normalize(
    text: str,
    source_label: str,
    anthropic_key: str,
    *,
    link_hint: str = "",
) -> Tuple[list[Story], bool, str]:
    """Stage 2. Run extracted text through Claude Sonnet 4.6 to produce
    structured stories. Single LLM call for documents that fit in one window;
    for larger documents, fan out into overlapping windows processed
    concurrently and deduplicate on merge.

    Returns (stories, is_news_content, skip_reason).
    """
    if not text or not text.strip():
        return [], False, "The document appears to be empty."

    if len(text) <= WINDOW_SIZE:
        return _normalize_chunk(
            text, source_label, anthropic_key,
            link_hint=link_hint,
            allow_full_doc_fallback=True,
        )

    chunks = _make_chunks(text)
    truncated = len(chunks) > MAX_CHUNKS
    if truncated:
        chunks = chunks[:MAX_CHUNKS]

    logger.info(
        "Chunked normalization for %s: %d chunks (text=%d chars, window=%d, overlap=%d)",
        source_label, len(chunks), len(text), WINDOW_SIZE, WINDOW_OVERLAP,
    )

    # Run chunks concurrently. Failed chunks are logged and skipped; we only
    # raise if every chunk fails.
    results: list[Optional[Tuple[list[Story], bool, str]]] = [None] * len(chunks)

    def _process(idx: int) -> Optional[Tuple[list[Story], bool, str]]:
        if idx > 0:
            time.sleep(INTER_CHUNK_PAUSE_SECONDS)
        try:
            return _normalize_chunk(
                chunks[idx], source_label, anthropic_key,
                link_hint=link_hint,
                allow_full_doc_fallback=False,
            )
        except IngestError as e:
            logger.warning(
                "Chunk %d/%d failed for %s: %s",
                idx + 1, len(chunks), source_label, e,
            )
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=NORMALIZE_CONCURRENCY) as ex:
        futures = {ex.submit(_process, i): i for i in range(len(chunks))}
        for fut in concurrent.futures.as_completed(futures):
            results[futures[fut]] = fut.result()

    failed = sum(1 for r in results if r is None)
    if failed == len(chunks):
        raise IngestError(
            f"{source_label}: all {len(chunks)} chunks failed during normalization."
        )

    # Merge: deduplicate by (title, body prefix). When the same story shows up
    # in two overlapping chunks, keep the one with the longer body — that's
    # the copy where the chunk window contained the full article rather than
    # clipping at the edge.
    merged: dict[str, Story] = {}
    skip_reasons: list[str] = []

    for result in results:
        if result is None:
            continue
        chunk_stories, _, chunk_skip = result
        if chunk_skip and not chunk_stories:
            skip_reasons.append(chunk_skip)
        for story in chunk_stories:
            key = _dedup_key(story)
            existing = merged.get(key)
            if existing is None or len(story.content) > len(existing.content):
                merged[key] = story

    stories = list(merged.values())

    if not stories:
        return [], False, (
            skip_reasons[0] if skip_reasons
            else "No news stories found in this document."
        )

    note_parts: list[str] = []
    if failed:
        note_parts.append(
            f"{failed} of {len(chunks)} chunks failed; partial results."
        )
    if truncated:
        note_parts.append(
            f"Document is very large; processed only the first {MAX_CHUNKS} chunks "
            f"(~{MAX_CHUNKS * (WINDOW_SIZE - WINDOW_OVERLAP) // 1000}k chars)."
        )

    return stories, True, " ".join(note_parts)


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED API
# ─────────────────────────────────────────────────────────────────────────────


def ingest_file(filename: str, raw: bytes, anthropic_key: str) -> IngestedSource:
    """Run both stages on an uploaded file. Never raises; failure is reported
    via excluded/skip_reason/extract_error on the returned IngestedSource."""
    source = IngestedSource(source_label=filename, kind="file")

    try:
        text = extract_text(filename, raw)
    except IngestError as e:
        source.excluded = True
        source.extract_error = str(e)
        source.skip_reason = "Failed to extract text from this file."
        return source

    source.char_count = len(text)
    # Chunked normalization processes the whole text up to MAX_CHUNKS windows;
    # anything beyond that is dropped at merge time.
    source.truncated = len(text) > MAX_CHUNKS * (WINDOW_SIZE - WINDOW_OVERLAP)

    if not text.strip():
        source.excluded = True
        source.skip_reason = "The file appears to be empty or contains no readable text."
        return source

    try:
        stories, is_news, skip_reason = normalize(text, filename, anthropic_key)
    except IngestError as e:
        source.excluded = True
        source.extract_error = str(e)
        source.skip_reason = "Normalization failed."
        return source

    source.stories = stories
    if not is_news:
        source.excluded = True
        source.skip_reason = skip_reason or "This document does not appear to contain news stories."
    return source


def ingest_url(url: str, anthropic_key: str) -> IngestedSource:
    """Fetch a URL, then run both stages. Same failure semantics as ingest_file."""
    source = IngestedSource(source_label=url, kind="url")

    try:
        filename, raw = fetch_url(url)
    except IngestError as e:
        source.excluded = True
        source.extract_error = str(e)
        source.skip_reason = "Failed to fetch this URL."
        return source

    try:
        text = extract_text(filename, raw)
    except IngestError as e:
        source.excluded = True
        source.extract_error = str(e)
        source.skip_reason = "Failed to extract text from the fetched URL."
        return source

    source.char_count = len(text)
    # Chunked normalization processes the whole text up to MAX_CHUNKS windows;
    # anything beyond that is dropped at merge time.
    source.truncated = len(text) > MAX_CHUNKS * (WINDOW_SIZE - WINDOW_OVERLAP)

    if not text.strip():
        source.excluded = True
        source.skip_reason = "The fetched page contains no readable text."
        return source

    try:
        stories, is_news, skip_reason = normalize(
            text, url, anthropic_key, link_hint=url,
        )
    except IngestError as e:
        source.excluded = True
        source.extract_error = str(e)
        source.skip_reason = "Normalization failed."
        return source

    source.stories = stories
    if not is_news:
        source.excluded = True
        source.skip_reason = skip_reason or "This page does not appear to contain news stories."
    return source
